//! Integration tests for `InstanceBvh` — the scene-level BVH over
//! reachable node-mesh instances, layered above the per-primitive
//! `Bvh` and the per-mesh `Mesh::intersect_ray`.
//!
//! Validation strategy:
//! * Build a `Scene3D` with one or more node-mesh instances at known
//!   transforms.
//! * Confirm `InstanceBvh::build` produces the expected number of
//!   instances + a sensible tree shape (`leaf_count` /
//!   `node_count` / `bounds()`).
//! * Cross-validate `InstanceBvh::intersect_ray` against
//!   `Scene3D::intersect_ray` on identical scenes and rays — the
//!   hit `t` and front-face must agree; the `NodeId` /
//!   `primitive_index` can tie-break differently when two
//!   instances strictly tie on `t`.
//! * Cross-validate `InstanceBvh::any_ray_intersection` against
//!   `Scene3D::any_ray_intersection` — the boolean answer is
//!   order-invariant and must match on every input ray.
//! * Exercise the singular-transform / detached-node / shared-child
//!   / empty-scene / cycle robustness paths the same way the
//!   `Scene3D::intersect_ray` integration tests do.

use oxideav_mesh3d::{
    InstanceBvh, Mesh, Node, NodeId, Primitive, Ray, Scene3D, Topology, Transform,
};

const EPS_T: f32 = 1e-4;

fn unit_cube_mesh() -> Mesh {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    p.indices = Some(oxideav_mesh3d::Indices::U32(vec![
        0, 2, 1, 0, 3, 2, // -Z (CCW from -Z)
        4, 5, 6, 4, 6, 7, // +Z
        0, 4, 7, 0, 7, 3, // -X
        1, 2, 6, 1, 6, 5, // +X
        0, 1, 5, 0, 5, 4, // -Y
        3, 7, 6, 3, 6, 2, // +Y
    ]));
    Mesh::new(Some("cube".to_owned())).with_primitive(p)
}

/// Builds a scene of `n` axis-aligned cubes along +X at `spacing`
/// units apart — cube `i` occupies `[i*spacing, i*spacing+1] x [0,1]^2`.
fn grid_scene(n: usize, spacing: f32) -> Scene3D {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(unit_cube_mesh());
    for ix in 0..n {
        let t = Transform::Trs {
            translation: [ix as f32 * spacing, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        };
        let nid = s.add_node(Node::new().with_transform(t).with_mesh(mid));
        s.add_root(nid);
    }
    s
}

#[test]
fn empty_scene_no_bvh() {
    let s = Scene3D::new();
    assert!(InstanceBvh::build(&s).is_none());
    assert!(s.build_instance_bvh().is_none());
}

#[test]
fn single_instance_one_leaf() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(unit_cube_mesh());
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    let b = s.build_instance_bvh().unwrap();
    assert_eq!(b.instance_count(), 1);
    assert_eq!(b.leaf_count(), 1);
    assert_eq!(b.node_count(), 1);
    assert!(b.nodes[0].is_leaf());
    let root = b.bounds().unwrap();
    assert_eq!(root.min, [0.0, 0.0, 0.0]);
    assert_eq!(root.max, [1.0, 1.0, 1.0]);
}

#[test]
fn grid_4_instances_single_leaf() {
    // 4 == LEAF_THRESHOLD, so the build stops at the root leaf.
    let s = grid_scene(4, 3.0);
    let b = s.build_instance_bvh().unwrap();
    assert_eq!(b.instance_count(), 4);
    assert_eq!(b.node_count(), 1);
    assert_eq!(b.leaf_count(), 1);
}

#[test]
fn grid_64_instances_balanced_tree() {
    // Past the leaf threshold — the tree must have interior nodes,
    // and the leaf count must equal ceil(64 / LEAF_THRESHOLD).
    let s = grid_scene(64, 3.0);
    let b = s.build_instance_bvh().unwrap();
    assert_eq!(b.instance_count(), 64);
    assert!(b.leaf_count() > 1);
    assert!(b.node_count() > b.leaf_count());
    // Root bounds span every cube.
    let root = b.bounds().unwrap();
    assert!((root.min[0] - 0.0).abs() < EPS_T);
    // Last cube at ix=63 occupies [189, 190].
    assert!((root.max[0] - 190.0).abs() < EPS_T);
}

#[test]
fn intersect_matches_scene_walk_on_grid_x_rays() {
    let s = grid_scene(16, 3.0);
    let b = s.build_instance_bvh().unwrap();
    for iy in 0..5 {
        let y = -1.0 + 0.5 * iy as f32;
        let r = Ray::new([-1.0, y, 0.5], [1.0, 0.0, 0.0]);
        let scene_hit = s.intersect_ray(r, f32::INFINITY);
        let bvh_hit = b.intersect_ray(&s, r, f32::INFINITY);
        match (scene_hit, bvh_hit) {
            (None, None) => {}
            (Some(sh), Some(bh)) => {
                assert!(
                    (sh.hit.t - bh.hit.t).abs() < EPS_T,
                    "t mismatch at y={y}: scene={sh:?} bvh={bh:?}"
                );
                assert_eq!(sh.hit.front_face, bh.hit.front_face);
            }
            (a, b) => panic!("mismatch at y={y}: scene={a:?} bvh={b:?}"),
        }
    }
}

#[test]
fn intersect_matches_scene_walk_on_grid_z_rays() {
    // Shoot +Z rays through the y=0.5 plane of the grid; every cube
    // is hit at z=0.0 from the [-1, ?] origin.
    let s = grid_scene(8, 3.0);
    let b = s.build_instance_bvh().unwrap();
    for ix in 0..16 {
        let x = -1.0 + 1.0 * ix as f32;
        let r = Ray::new([x, 0.5, -1.0], [0.0, 0.0, 1.0]);
        let scene_hit = s.intersect_ray(r, f32::INFINITY);
        let bvh_hit = b.intersect_ray(&s, r, f32::INFINITY);
        match (scene_hit, bvh_hit) {
            (None, None) => {}
            (Some(sh), Some(bh)) => {
                assert!((sh.hit.t - bh.hit.t).abs() < EPS_T);
                assert_eq!(sh.hit.front_face, bh.hit.front_face);
            }
            (a, b) => panic!("mismatch at x={x}: scene={a:?} bvh={b:?}"),
        }
    }
}

#[test]
fn miss_returns_none() {
    let s = grid_scene(8, 3.0);
    let b = s.build_instance_bvh().unwrap();
    // y=5 misses every cube which lives at y=[0,1].
    let r = Ray::new([-1.0, 5.0, 0.5], [1.0, 0.0, 0.0]);
    assert!(b.intersect_ray(&s, r, f32::INFINITY).is_none());
}

#[test]
fn t_max_culls_all_hits() {
    let s = grid_scene(4, 3.0);
    let b = s.build_instance_bvh().unwrap();
    let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    // First face at t=1.0; tight t_max ahead of it culls.
    assert!(b.intersect_ray(&s, r, 0.5).is_none());
    // Loose t_max captures the first cube.
    assert!(b.intersect_ray(&s, r, 1.5).is_some());
}

#[test]
fn nearest_instance_wins_on_grid() {
    // Closest cube along +X must be the one at ix=0; the BVH walk
    // can't claim a farther one even if it visits the leaf first.
    let s = grid_scene(16, 3.0);
    let b = s.build_instance_bvh().unwrap();
    let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let hit = b.intersect_ray(&s, r, f32::INFINITY).unwrap();
    // Cube 0 -X face at world t=1.0.
    assert!((hit.hit.t - 1.0).abs() < EPS_T);
}

#[test]
fn any_intersection_short_circuits() {
    let s = grid_scene(8, 3.0);
    let b = s.build_instance_bvh().unwrap();
    let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!(b.any_ray_intersection(&s, r, f32::INFINITY));
    // t_max ahead of the first cube — no hit.
    assert!(!b.any_ray_intersection(&s, r, 0.5));
}

#[test]
fn any_intersection_agrees_with_scene_walk() {
    let s = grid_scene(8, 3.0);
    let b = s.build_instance_bvh().unwrap();
    for iy in 0..5 {
        for ix in 0..5 {
            let x = -1.0 + 0.5 * ix as f32;
            let y = -1.0 + 0.5 * iy as f32;
            let r = Ray::new([x, y, -1.0], [0.0, 0.0, 1.0]);
            assert_eq!(
                b.any_ray_intersection(&s, r, f32::INFINITY),
                s.any_ray_intersection(r, f32::INFINITY),
                "mismatch at x={x} y={y}"
            );
        }
    }
}

#[test]
fn detached_node_excluded() {
    // A node with a mesh but not on the roots forest must not
    // contribute an instance.
    let mut s = Scene3D::new();
    let mid = s.add_mesh(unit_cube_mesh());
    let _detached = s.add_node(Node::new().with_mesh(mid));
    let attached = s.add_node(Node::new().with_mesh(mid));
    s.add_root(attached);
    let b = s.build_instance_bvh().unwrap();
    assert_eq!(b.instance_count(), 1);
    assert_eq!(b.instances[0].node, attached);
}

#[test]
fn singular_transform_skipped() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(unit_cube_mesh());
    let bad = s.add_node(
        Node::new()
            .with_transform(Transform::Trs {
                translation: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.0, 1.0, 1.0],
            })
            .with_mesh(mid),
    );
    s.add_root(bad);
    // No good instances at all → None.
    assert!(s.build_instance_bvh().is_none());
}

#[test]
fn cycle_visited_once() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(unit_cube_mesh());
    let a = s.add_node(Node::new().with_mesh(mid));
    let b = s.add_node(Node::new());
    // b -> a -> b cycle via children list.
    if let Some(an) = s.node_mut(a) {
        an.children.push(b);
    }
    if let Some(bn) = s.node_mut(b) {
        bn.children.push(a);
    }
    s.add_root(a);
    let bvh = s.build_instance_bvh().unwrap();
    // Cycle didn't double-count a; b has no mesh so it doesn't
    // contribute either.
    assert_eq!(bvh.instance_count(), 1);
    assert_eq!(bvh.instances[0].node, a);
}

#[test]
fn rotated_instance_world_aabb_grows() {
    // 45 deg rotation about Y of a unit cube widens the AABB on the
    // XZ plane. The cached `bounds` must reflect the rotated extent.
    let s = {
        let mut s = Scene3D::new();
        let mid = s.add_mesh(unit_cube_mesh());
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let t = Transform::Trs {
            translation: [0.0, 0.0, 0.0],
            // Quaternion for 45 deg about Y: (0, sin(22.5), 0, cos(22.5))
            rotation: [
                0.0,
                (22.5_f32.to_radians()).sin(),
                0.0,
                (22.5_f32.to_radians()).cos(),
            ],
            scale: [1.0, 1.0, 1.0],
        };
        let nid = s.add_node(Node::new().with_transform(t).with_mesh(mid));
        s.add_root(nid);
        let _ = half;
        s
    };
    let b = s.build_instance_bvh().unwrap();
    assert_eq!(b.instance_count(), 1);
    let inst = &b.instances[0];
    let size = inst.bounds.size();
    // Rotated cube footprint on X/Z must be larger than the original
    // unit extent (sqrt(2) ≈ 1.414 for the diagonal of the unit
    // square).
    assert!(size[0] > 1.0 && size[0] < 2.0);
    assert!(size[2] > 1.0 && size[2] < 2.0);
    assert!((size[1] - 1.0).abs() < EPS_T);
}

#[test]
fn translated_instance_intersection_at_expected_t() {
    // Cube translated to x = 10 along +X. Ray from origin along +X
    // hits at t = 10 (the -X face after translation).
    let mut s = Scene3D::new();
    let mid = s.add_mesh(unit_cube_mesh());
    let t = Transform::Trs {
        translation: [10.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    };
    let nid = s.add_node(Node::new().with_transform(t).with_mesh(mid));
    s.add_root(nid);
    let b = s.build_instance_bvh().unwrap();
    let r = Ray::new([0.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let hit = b.intersect_ray(&s, r, f32::INFINITY).unwrap();
    assert!((hit.hit.t - 10.0).abs() < EPS_T);
    assert!(hit.hit.front_face);
}

#[test]
fn instance_count_matches_reachable_meshed_node_count() {
    let s = grid_scene(13, 3.0);
    let b = s.build_instance_bvh().unwrap();
    assert_eq!(b.instance_count(), 13);
}

#[test]
fn leaf_threshold_matches_per_primitive_bvh() {
    use oxideav_mesh3d::Bvh;
    assert_eq!(InstanceBvh::LEAF_THRESHOLD, Bvh::LEAF_THRESHOLD);
}

#[test]
fn cross_validate_against_scene_walk_grid_negative_x() {
    // Shoot -X rays from x=+50 back through the grid; nearest cube
    // is the highest-ix one.
    let s = grid_scene(8, 3.0);
    let b = s.build_instance_bvh().unwrap();
    let r = Ray::new([50.0, 0.5, 0.5], [-1.0, 0.0, 0.0]);
    let scene_hit = s.intersect_ray(r, f32::INFINITY).unwrap();
    let bvh_hit = b.intersect_ray(&s, r, f32::INFINITY).unwrap();
    assert!((scene_hit.hit.t - bvh_hit.hit.t).abs() < EPS_T);
}

#[test]
fn cross_validate_many_rays_diagonal() {
    // 16x16 ray sweep across the grid at oblique angles —
    // closest-hit must match across the two walks.
    let s = grid_scene(16, 3.0);
    let b = s.build_instance_bvh().unwrap();
    for iy in 0..16 {
        for ix in 0..16 {
            let ox = -2.0 + 0.5 * ix as f32;
            let oy = -2.0 + 0.25 * iy as f32;
            let r = Ray::new([ox, oy, -1.0], [0.3, 0.1, 1.0]);
            let scene_hit = s.intersect_ray(r, f32::INFINITY);
            let bvh_hit = b.intersect_ray(&s, r, f32::INFINITY);
            match (scene_hit, bvh_hit) {
                (None, None) => {}
                (Some(sh), Some(bh)) => {
                    assert!(
                        (sh.hit.t - bh.hit.t).abs() < 1e-3,
                        "t mismatch ix={ix} iy={iy}: scene t={} bvh t={}",
                        sh.hit.t,
                        bh.hit.t
                    );
                }
                (a, b) => panic!("ix={ix} iy={iy}: scene={a:?} bvh={b:?}"),
            }
        }
    }
}

#[test]
fn shared_child_one_instance() {
    // A node shared under two parents resolves via the first parent
    // (same first-parent rule as `world_node_transforms`). The
    // instance is gathered once, not twice.
    let mut s = Scene3D::new();
    let mid = s.add_mesh(unit_cube_mesh());
    let shared = s.add_node(Node::new().with_mesh(mid));
    let p1 = s.add_node(Node::new());
    let p2 = s.add_node(Node::new());
    if let Some(p1n) = s.node_mut(p1) {
        p1n.children.push(shared);
    }
    if let Some(p2n) = s.node_mut(p2) {
        p2n.children.push(shared);
    }
    s.add_root(p1);
    s.add_root(p2);
    let b = s.build_instance_bvh().unwrap();
    assert_eq!(b.instance_count(), 1);
    assert_eq!(b.instances[0].node, shared);
}

#[test]
fn build_after_node_add_is_pure() {
    // Building does not mutate the scene; a second build returns
    // a tree with the same shape.
    let s = grid_scene(20, 3.0);
    let b1 = s.build_instance_bvh().unwrap();
    let b2 = s.build_instance_bvh().unwrap();
    assert_eq!(b1.instance_count(), b2.instance_count());
    assert_eq!(b1.node_count(), b2.node_count());
    assert_eq!(b1.leaf_count(), b2.leaf_count());
}

#[test]
fn empty_mesh_arena_no_bvh() {
    // Nodes exist but no meshes → no instances.
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    assert!(s.build_instance_bvh().is_none());
}

#[test]
fn intersect_dispatches_to_correct_node() {
    // Two cubes at different y, ray aimed at the second one — the
    // returned NodeId must be the second cube's, not the first.
    let mut s = Scene3D::new();
    let mid = s.add_mesh(unit_cube_mesh());
    let n0 = s.add_node(
        Node::new()
            .with_transform(Transform::Trs {
                translation: [0.0, 10.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            })
            .with_mesh(mid),
    );
    let n1 = s.add_node(
        Node::new()
            .with_transform(Transform::Trs {
                translation: [0.0, -10.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            })
            .with_mesh(mid),
    );
    s.add_root(n0);
    s.add_root(n1);
    let b = s.build_instance_bvh().unwrap();
    // Ray through y=-9.5 hits the second cube only.
    let r = Ray::new([-1.0, -9.5, 0.5], [1.0, 0.0, 0.0]);
    let hit = b.intersect_ray(&s, r, f32::INFINITY).unwrap();
    assert_eq!(hit.node, n1);
    // Ray through y=10.5 hits the first cube only.
    let r2 = Ray::new([-1.0, 10.5, 0.5], [1.0, 0.0, 0.0]);
    let hit2 = b.intersect_ray(&s, r2, f32::INFINITY).unwrap();
    assert_eq!(hit2.node, n0);
}

#[test]
fn deterministic_build() {
    // Same scene → same tree. The build's permutation depends only
    // on the gather order + centroid coordinates.
    let s = grid_scene(10, 3.0);
    let b1 = s.build_instance_bvh().unwrap();
    let b2 = s.build_instance_bvh().unwrap();
    assert_eq!(b1.nodes.len(), b2.nodes.len());
    for (n1, n2) in b1.nodes.iter().zip(b2.nodes.iter()) {
        assert_eq!(n1.instance_count, n2.instance_count);
        assert_eq!(n1.left_or_first, n2.left_or_first);
        assert_eq!(n1.right_child, n2.right_child);
    }
    for (i1, i2) in b1.instances.iter().zip(b2.instances.iter()) {
        assert_eq!(i1.node, i2.node);
        assert_eq!(i1.mesh, i2.mesh);
    }
}

#[test]
fn cached_world_inv_matches_recomputed() {
    // The instance's cached `world_inv` must be the inverse of its
    // `world` matrix — composing them gives identity (up to fp eps).
    let s = grid_scene(4, 3.0);
    let b = s.build_instance_bvh().unwrap();
    for inst in &b.instances {
        // (world * world_inv)[i][j] should be identity.
        let mut product = [[0.0f32; 4]; 4];
        for (i, row) in product.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += inst.world[i][k] * inst.world_inv[k][j];
                }
                *slot = sum;
            }
        }
        for (i, row) in product.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-4,
                    "product[{i}][{j}] = {val} (expected {expected})"
                );
            }
        }
    }
}

#[test]
fn root_bounds_contain_every_instance() {
    let s = grid_scene(12, 3.0);
    let b = s.build_instance_bvh().unwrap();
    let root = b.bounds().unwrap();
    for inst in &b.instances {
        for axis in 0..3 {
            assert!(root.min[axis] <= inst.bounds.min[axis] + EPS_T);
            assert!(root.max[axis] >= inst.bounds.max[axis] - EPS_T);
        }
    }
}

#[test]
fn bvh_size_bounded_by_instance_count_relation() {
    // For a flat binary tree where every leaf holds at most
    // LEAF_THRESHOLD instances, the leaf count is in
    // [ceil(N/T), N]. The node count is <= 2 * leaf_count - 1 for
    // a strictly binary tree (one root, two children per interior).
    let s = grid_scene(20, 3.0);
    let b = s.build_instance_bvh().unwrap();
    let n = b.instance_count();
    let t = InstanceBvh::LEAF_THRESHOLD;
    assert!(b.leaf_count() >= n.div_ceil(t));
    assert!(b.leaf_count() <= n);
    // Strict binary tree invariant — same as `<= 2L - 1`,
    // rewritten to keep clippy's int_plus_one lint happy.
    assert!(b.node_count() < 2 * b.leaf_count());
}

/// Sanity check: NodeId is exported as expected.
#[test]
fn node_id_type_visible() {
    let zero = NodeId(0);
    assert_eq!(zero.0, 0);
}
