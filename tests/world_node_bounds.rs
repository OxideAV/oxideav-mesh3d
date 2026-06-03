//! Tests for [`Scene3D::world_node_bounds`] — per-node world-space
//! axis-aligned bounding box snapshot, indexed by [`NodeId`].
//!
//! Covers reachability, ancestor-chain composition, cycle guarding,
//! shared-instance single-resolution, determinism, transform-rebound
//! tightness, and the cross-check that the per-slot union equals the
//! single-shot [`Scene3D::bounding_box`] reduction.

use oxideav_mesh3d::{
    BoundingBox, Mesh, MeshId, Node, NodeId, Primitive, Scene3D, Topology, Transform,
};

const TOL: f32 = 1e-4;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOL + TOL * a.abs().max(b.abs())
}

fn bbox_close(a: BoundingBox, b: BoundingBox) -> bool {
    approx_eq(a.min[0], b.min[0])
        && approx_eq(a.min[1], b.min[1])
        && approx_eq(a.min[2], b.min[2])
        && approx_eq(a.max[0], b.max[0])
        && approx_eq(a.max[1], b.max[1])
        && approx_eq(a.max[2], b.max[2])
}

fn translation_only(t: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: t,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn scale_only(s: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: s,
    }
}

/// Unit cube at the origin: corners at [0,0,0] .. [1,1,1]. The local
/// AABB is exactly `min=[0,0,0]` .. `max=[1,1,1]`.
fn unit_cube_mesh() -> Mesh {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    Mesh::new(Some("cube".to_owned())).with_primitive(prim)
}

/// Centred unit cube at the origin: [-0.5,-0.5,-0.5] .. [0.5,0.5,0.5]
/// — useful for confirming pure rotations stay tight around the
/// origin without translation contamination.
fn centred_unit_cube_mesh() -> Mesh {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ];
    Mesh::new(Some("centred".to_owned())).with_primitive(prim)
}

/// Single triangle in the XY plane (Z=0). Useful for confirming a
/// flat mesh in Z still produces a finite (zero-thickness) AABB.
fn flat_triangle_mesh() -> Mesh {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
    Mesh::new(Some("flat".to_owned())).with_primitive(prim)
}

// ---- Empty / trivial scenes ----------------------------------------------

#[test]
fn empty_scene_returns_empty_vec() {
    let scene = Scene3D::new();
    let bounds = scene.world_node_bounds();
    assert!(bounds.is_empty());
}

#[test]
fn nodes_with_no_roots_are_all_none() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    scene.add_node(Node::new().with_mesh(mesh));
    scene.add_node(Node::new());
    scene.add_node(Node::new().with_mesh(mesh));
    let bounds = scene.world_node_bounds();
    assert_eq!(bounds.len(), 3);
    assert!(bounds.iter().all(|b| b.is_none()));
}

#[test]
fn reachable_node_without_mesh_is_none() {
    let mut scene = Scene3D::new();
    let n = scene.add_node(Node::new());
    scene.add_root(n);
    let bounds = scene.world_node_bounds();
    assert_eq!(bounds.len(), 1);
    assert!(bounds[0].is_none());
}

#[test]
fn reachable_node_with_empty_mesh_is_none() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(Mesh::new(Some("empty".to_owned())));
    let n = scene.add_node(Node::new().with_mesh(mesh));
    scene.add_root(n);
    let bounds = scene.world_node_bounds();
    assert_eq!(bounds.len(), 1);
    assert!(bounds[0].is_none());
}

#[test]
fn out_of_range_mesh_id_is_none() {
    // Attach a mesh id that doesn't index any meshes.
    let mut scene = Scene3D::new();
    let n = scene.add_node(Node::new().with_mesh(MeshId(7)));
    scene.add_root(n);
    let bounds = scene.world_node_bounds();
    assert_eq!(bounds.len(), 1);
    assert!(bounds[0].is_none());
}

// ---- Identity-transform pass-through -------------------------------------

#[test]
fn single_root_identity_passes_local_aabb_through() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let n = scene.add_node(Node::new().with_mesh(mesh));
    scene.add_root(n);

    let bounds = scene.world_node_bounds();
    let got = bounds[n.0 as usize].unwrap();
    assert!(bbox_close(
        got,
        BoundingBox {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        }
    ));
}

// ---- Translation -------------------------------------------------

#[test]
fn translation_only_shifts_aabb() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let n = scene.add_node(
        Node::new()
            .with_transform(translation_only([10.0, -2.0, 0.5]))
            .with_mesh(mesh),
    );
    scene.add_root(n);

    let bounds = scene.world_node_bounds();
    let got = bounds[n.0 as usize].unwrap();
    assert!(bbox_close(
        got,
        BoundingBox {
            min: [10.0, -2.0, 0.5],
            max: [11.0, -1.0, 1.5],
        }
    ));
}

// ---- Scale -------------------------------------------------------

#[test]
fn uniform_scale_widens_aabb() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(centred_unit_cube_mesh());
    let n = scene.add_node(
        Node::new()
            .with_transform(scale_only([2.0, 2.0, 2.0]))
            .with_mesh(mesh),
    );
    scene.add_root(n);

    let bounds = scene.world_node_bounds();
    let got = bounds[n.0 as usize].unwrap();
    assert!(bbox_close(
        got,
        BoundingBox {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        }
    ));
}

#[test]
fn non_uniform_scale_widens_per_axis() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let n = scene.add_node(
        Node::new()
            .with_transform(scale_only([3.0, 1.0, 5.0]))
            .with_mesh(mesh),
    );
    scene.add_root(n);

    let bounds = scene.world_node_bounds();
    let got = bounds[n.0 as usize].unwrap();
    assert!(bbox_close(
        got,
        BoundingBox {
            min: [0.0, 0.0, 0.0],
            max: [3.0, 1.0, 5.0],
        }
    ));
}

// ---- Rotation ----------------------------------------------------

#[test]
fn quarter_turn_around_z_swaps_xy_extents_for_centred_cube() {
    // 90deg around +Z: (x, y, z) -> (-y, x, z). For a centred unit
    // cube [-0.5, 0.5]³ the AABB stays [-0.5, 0.5]³ — the symmetric
    // box is rotation-invariant under axis-aligned 90deg rotations.
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(centred_unit_cube_mesh());
    let half = std::f32::consts::FRAC_1_SQRT_2; // sin(45deg) = cos(45deg)
    let rot = Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, half, half], // 90deg around +Z (xyzw)
        scale: [1.0, 1.0, 1.0],
    };
    let n = scene.add_node(Node::new().with_transform(rot).with_mesh(mesh));
    scene.add_root(n);

    let bounds = scene.world_node_bounds();
    let got = bounds[n.0 as usize].unwrap();
    assert!(bbox_close(
        got,
        BoundingBox {
            min: [-0.5, -0.5, -0.5],
            max: [0.5, 0.5, 0.5],
        }
    ));
}

#[test]
fn forty_five_degree_rotation_widens_centred_cube_aabb() {
    // 45deg around +Z spins a unit cube; the AABB widens from [-0.5,
    // 0.5] in X and Y to ±0.5·sqrt(2) along each axis, while Z stays
    // ±0.5.
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(centred_unit_cube_mesh());
    let half_angle = std::f32::consts::PI / 8.0; // half of 45deg
    let rot = Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, half_angle.sin(), half_angle.cos()],
        scale: [1.0, 1.0, 1.0],
    };
    let n = scene.add_node(Node::new().with_transform(rot).with_mesh(mesh));
    scene.add_root(n);

    let bounds = scene.world_node_bounds();
    let got = bounds[n.0 as usize].unwrap();
    let half = 0.5 * std::f32::consts::SQRT_2;
    assert!(bbox_close(
        got,
        BoundingBox {
            min: [-half, -half, -0.5],
            max: [half, half, 0.5],
        }
    ));
}

// ---- Ancestor-chain composition ------------------------------------------

#[test]
fn child_inherits_parent_translation() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let parent = scene.add_node(Node::new().with_transform(translation_only([10.0, 0.0, 0.0])));
    let child = scene.add_node(
        Node::new()
            .with_transform(translation_only([1.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    scene.nodes[parent.0 as usize].children.push(child);
    scene.add_root(parent);

    let bounds = scene.world_node_bounds();
    assert!(bounds[parent.0 as usize].is_none()); // no mesh on parent
    let cb = bounds[child.0 as usize].unwrap();
    assert!(bbox_close(
        cb,
        BoundingBox {
            min: [11.0, 0.0, 0.0],
            max: [12.0, 1.0, 1.0],
        }
    ));
}

#[test]
fn parent_scale_modulates_child_translation_and_size() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let parent = scene.add_node(Node::new().with_transform(scale_only([10.0, 10.0, 10.0])));
    let child = scene.add_node(
        Node::new()
            .with_transform(translation_only([1.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    scene.nodes[parent.0 as usize].children.push(child);
    scene.add_root(parent);

    let bounds = scene.world_node_bounds();
    let cb = bounds[child.0 as usize].unwrap();
    // Cube at local +(1,0,0), scaled by 10 → AABB [10,0,0]..[20,10,10].
    assert!(bbox_close(
        cb,
        BoundingBox {
            min: [10.0, 0.0, 0.0],
            max: [20.0, 10.0, 10.0],
        }
    ));
}

// ---- Matrix variant ----------------------------------------------------

#[test]
fn matrix_transform_variant_is_honoured() {
    let m: [[f32; 4]; 4] = [
        [2.0, 0.0, 0.0, 5.0],
        [0.0, 3.0, 0.0, -1.0],
        [0.0, 0.0, 4.0, 7.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let n = scene.add_node(
        Node::new()
            .with_transform(Transform::Matrix(m))
            .with_mesh(mesh),
    );
    scene.add_root(n);
    let bounds = scene.world_node_bounds();
    let b = bounds[n.0 as usize].unwrap();
    assert!(bbox_close(
        b,
        BoundingBox {
            min: [5.0, -1.0, 7.0],
            max: [7.0, 2.0, 11.0],
        }
    ));
}

// ---- Flat / degenerate meshes ---------------------------------------------

#[test]
fn flat_triangle_produces_zero_thickness_z_aabb() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(flat_triangle_mesh());
    let n = scene.add_node(Node::new().with_mesh(mesh));
    scene.add_root(n);
    let bounds = scene.world_node_bounds();
    let b = bounds[n.0 as usize].unwrap();
    assert!(approx_eq(b.min[2], 0.0));
    assert!(approx_eq(b.max[2], 0.0));
    // The triangle is also tight in X / Y.
    assert!(approx_eq(b.min[0], 0.0));
    assert!(approx_eq(b.max[0], 2.0));
    assert!(approx_eq(b.min[1], 0.0));
    assert!(approx_eq(b.max[1], 3.0));
}

// ---- Forest with multiple roots ------------------------------------------

#[test]
fn multiple_roots_each_resolve_locally() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let r0 = scene.add_node(
        Node::new()
            .with_transform(translation_only([100.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    let r1 = scene.add_node(
        Node::new()
            .with_transform(translation_only([0.0, 200.0, 0.0]))
            .with_mesh(mesh),
    );
    scene.add_root(r0);
    scene.add_root(r1);

    let bounds = scene.world_node_bounds();
    let b0 = bounds[r0.0 as usize].unwrap();
    let b1 = bounds[r1.0 as usize].unwrap();
    assert!(approx_eq(b0.min[0], 100.0));
    assert!(approx_eq(b0.max[0], 101.0));
    assert!(approx_eq(b1.min[1], 200.0));
    assert!(approx_eq(b1.max[1], 201.0));
}

// ---- Cycle guarding -----------------------------------------------

#[test]
fn cycle_self_descendant_is_visited_once() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let a = scene.add_node(
        Node::new()
            .with_transform(translation_only([1.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    // a -> a (cycle back into itself).
    scene.nodes[a.0 as usize].children.push(a);
    scene.add_root(a);
    let bounds = scene.world_node_bounds();
    assert_eq!(bounds.len(), 1);
    let b = bounds[a.0 as usize].unwrap();
    assert!(bbox_close(
        b,
        BoundingBox {
            min: [1.0, 0.0, 0.0],
            max: [2.0, 1.0, 1.0],
        }
    ));
}

#[test]
fn three_node_cycle_resolves_via_first_arrival() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let a = scene.add_node(
        Node::new()
            .with_transform(translation_only([10.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    let b = scene.add_node(
        Node::new()
            .with_transform(translation_only([0.0, 1.0, 0.0]))
            .with_mesh(mesh),
    );
    let c = scene.add_node(
        Node::new()
            .with_transform(translation_only([0.0, 0.0, 1.0]))
            .with_mesh(mesh),
    );
    scene.nodes[a.0 as usize].children.push(b);
    scene.nodes[b.0 as usize].children.push(c);
    // Close the cycle: c → a.
    scene.nodes[c.0 as usize].children.push(a);
    scene.add_root(a);
    let bounds = scene.world_node_bounds();
    // a/b/c each get a single resolved AABB; no panic on cycle close.
    assert!(bounds[a.0 as usize].is_some());
    assert!(bounds[b.0 as usize].is_some());
    assert!(bounds[c.0 as usize].is_some());
}

// ---- Shared-instance / first-parent contract -----------------------------

#[test]
fn shared_child_resolves_via_first_parent_chain() {
    // Two roots, both list `shared` as a child; world_node_bounds is
    // per-NodeId, not per-instance, so only the first-parent chain
    // resolves the slot. (Per-instance world AABBs need an explicit
    // instance-list side-channel, per the doc contract.)
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let r0 = scene.add_node(Node::new().with_transform(translation_only([100.0, 0.0, 0.0])));
    let r1 = scene.add_node(Node::new().with_transform(translation_only([0.0, 100.0, 0.0])));
    let shared = scene.add_node(Node::new().with_mesh(mesh));
    scene.nodes[r0.0 as usize].children.push(shared);
    scene.nodes[r1.0 as usize].children.push(shared);
    scene.add_root(r0);
    scene.add_root(r1);

    let bounds = scene.world_node_bounds();
    let b = bounds[shared.0 as usize].unwrap();
    // First-parent chain is r0 → translated +100 in X.
    assert!(bbox_close(
        b,
        BoundingBox {
            min: [100.0, 0.0, 0.0],
            max: [101.0, 1.0, 1.0],
        }
    ));
}

// ---- Out-of-range NodeId entries skipped ---------------------------------

#[test]
fn out_of_range_child_ids_are_skipped() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let n = scene.add_node(Node::new().with_mesh(mesh));
    // Phantom child id beyond node count.
    scene.nodes[n.0 as usize].children.push(NodeId(99));
    scene.add_root(n);
    let bounds = scene.world_node_bounds();
    assert_eq!(bounds.len(), 1);
    assert!(bounds[0].is_some());
}

#[test]
fn out_of_range_root_ids_are_skipped() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let n = scene.add_node(Node::new().with_mesh(mesh));
    scene.add_root(NodeId(42));
    scene.add_root(n);
    let bounds = scene.world_node_bounds();
    assert!(bounds[n.0 as usize].is_some());
}

// ---- Length contract -------------------------------------------------

#[test]
fn output_length_equals_node_count() {
    let mut scene = Scene3D::new();
    for _ in 0..7 {
        scene.add_node(Node::new());
    }
    let bounds = scene.world_node_bounds();
    assert_eq!(bounds.len(), 7);
}

// ---- Cross-check: per-slot union equals Scene3D::bounding_box -----------

#[test]
fn union_of_slots_matches_scene_bounding_box() {
    // A scene with several reachable mesh nodes and one detached
    // node — the union of `world_node_bounds`'s `Some` slots should
    // equal `Scene3D::bounding_box`.
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let a = scene.add_node(
        Node::new()
            .with_transform(translation_only([5.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    let b = scene.add_node(
        Node::new()
            .with_transform(translation_only([0.0, 7.0, 0.0]))
            .with_mesh(mesh),
    );
    let c = scene.add_node(
        Node::new()
            .with_transform(translation_only([-3.0, -3.0, 9.0]))
            .with_mesh(mesh),
    );
    // Detached fourth node — not in roots; must not contribute.
    scene.add_node(
        Node::new()
            .with_transform(translation_only([1000.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    scene.add_root(a);
    scene.add_root(b);
    scene.add_root(c);

    let bounds = scene.world_node_bounds();
    let union_slot = bounds
        .iter()
        .filter_map(|b| *b)
        .reduce(BoundingBox::union)
        .unwrap();
    let single_shot = scene.bounding_box().unwrap();
    assert!(
        bbox_close(union_slot, single_shot),
        "union {:?} != bounding_box {:?}",
        union_slot,
        single_shot
    );
}

// ---- Determinism ---------------------------------------------------

#[test]
fn repeated_calls_produce_identical_output() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let parent = scene.add_node(Node::new().with_transform(translation_only([10.0, 20.0, 30.0])));
    let child0 = scene.add_node(
        Node::new()
            .with_transform(translation_only([1.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    let child1 = scene.add_node(
        Node::new()
            .with_transform(scale_only([2.0, 2.0, 2.0]))
            .with_mesh(mesh),
    );
    scene.nodes[parent.0 as usize].children.push(child0);
    scene.nodes[parent.0 as usize].children.push(child1);
    scene.add_root(parent);

    let a = scene.world_node_bounds();
    let b = scene.world_node_bounds();
    assert_eq!(a.len(), b.len());
    for (ai, bi) in a.iter().zip(b.iter()) {
        assert_eq!(ai.is_some(), bi.is_some());
        if let (Some(x), Some(y)) = (ai, bi) {
            assert!(bbox_close(*x, *y));
        }
    }
}

// ---- Cross-check with world_node_transforms -----------------------

#[test]
fn slot_matches_local_aabb_transformed_by_world_matrix() {
    // The contract: `world_node_bounds[i]` should equal
    // `meshes[i].bounding_box().transform(world_node_transforms[i])`
    // for every reachable node carrying a mesh.
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let parent = scene.add_node(Node::new().with_transform(scale_only([2.0, 3.0, 5.0])));
    let child = scene.add_node(
        Node::new()
            .with_transform(translation_only([0.5, 0.5, 0.0]))
            .with_mesh(mesh),
    );
    scene.nodes[parent.0 as usize].children.push(child);
    scene.add_root(parent);

    let bounds = scene.world_node_bounds();
    let xforms = scene.world_node_transforms();
    let local = scene.meshes[0].bounding_box().unwrap();
    let want = local.transform(xforms[child.0 as usize].unwrap());
    assert!(bbox_close(bounds[child.0 as usize].unwrap(), want));
}

// ---- Detached subtree skipped -------------------------------------------

#[test]
fn detached_subtree_is_not_visited() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let detached_a = scene.add_node(
        Node::new()
            .with_transform(translation_only([1000.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    let detached_b = scene.add_node(
        Node::new()
            .with_transform(translation_only([0.0, 1000.0, 0.0]))
            .with_mesh(mesh),
    );
    scene.nodes[detached_a.0 as usize].children.push(detached_b);
    let attached = scene.add_node(Node::new().with_mesh(mesh));
    scene.add_root(attached);

    let bounds = scene.world_node_bounds();
    assert!(bounds[attached.0 as usize].is_some());
    assert!(bounds[detached_a.0 as usize].is_none());
    assert!(bounds[detached_b.0 as usize].is_none());
}

// ---- Scene with no mesh at all --------------------------------------

#[test]
fn scene_with_zero_meshes_yields_all_none() {
    let mut scene = Scene3D::new();
    let n = scene.add_node(Node::new().with_mesh(MeshId(0)));
    scene.add_root(n);
    let bounds = scene.world_node_bounds();
    assert_eq!(bounds.len(), 1);
    assert!(bounds[0].is_none());
}

// ---- Ray pre-pass use case (illustrative) ------------------------------

#[test]
fn instance_aabb_acts_as_ray_prefilter() {
    // The headline use case: a scene-level ray cast first asks each
    // slot's AABB whether it can possibly contain a hit, descending
    // into Mesh::intersect_ray only when so. We do the AABB-level
    // prefilter here and check it identifies the same instance the
    // scene-level closest-hit query reports.
    use oxideav_mesh3d::Ray;

    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let a = scene.add_node(
        Node::new()
            .with_transform(translation_only([0.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    let b = scene.add_node(
        Node::new()
            .with_transform(translation_only([10.0, 0.0, 0.0]))
            .with_mesh(mesh),
    );
    let c = scene.add_node(
        Node::new()
            .with_transform(translation_only([0.0, 10.0, 0.0]))
            .with_mesh(mesh),
    );
    scene.add_root(a);
    scene.add_root(b);
    scene.add_root(c);

    let bounds = scene.world_node_bounds();
    // Ray going down the +X axis through node a's box and then b's.
    let ray = Ray::new([-5.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let mut prefiltered: Vec<NodeId> = Vec::new();
    for (i, slot) in bounds.iter().enumerate() {
        if let Some(box_) = slot {
            if box_.intersect_ray(ray, f32::INFINITY).is_some() {
                prefiltered.push(NodeId(i as u32));
            }
        }
    }
    // a and b are on the +X line; c is off in +Y.
    assert!(prefiltered.contains(&a));
    assert!(prefiltered.contains(&b));
    assert!(!prefiltered.contains(&c));
}

// ---- Memory + length sanity for big-ish scenes --------------------------

#[test]
fn many_nodes_resolve_without_panic() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(unit_cube_mesh());
    let n = 128;
    let mut prev: Option<NodeId> = None;
    let mut root: Option<NodeId> = None;
    for _ in 0..n {
        let here = scene.add_node(
            Node::new()
                .with_transform(translation_only([1.0, 0.0, 0.0]))
                .with_mesh(mesh),
        );
        if let Some(p) = prev {
            scene.nodes[p.0 as usize].children.push(here);
        } else {
            root = Some(here);
        }
        prev = Some(here);
    }
    scene.add_root(root.unwrap());
    let bounds = scene.world_node_bounds();
    assert_eq!(bounds.len(), n);
    assert!(bounds.iter().all(|b| b.is_some()));
    // Last node should be translated by n in X.
    let last = bounds[n - 1].unwrap();
    assert!(approx_eq(last.min[0], n as f32));
    assert!(approx_eq(last.max[0], n as f32 + 1.0));
}
