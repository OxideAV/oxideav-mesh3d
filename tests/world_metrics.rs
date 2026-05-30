//! Tests for transform-aware scene-level aggregate metrics:
//! [`Scene3D::world_surface_area`], [`Scene3D::world_signed_volume`],
//! [`Scene3D::world_volume`], and the per-primitive helper
//! [`Primitive::world_surface_area`].
//!
//! Every closed-form expected value below is derived from elementary
//! affine-geometry rules: under a linear transform `M`, a triangle's
//! area scales by `|cross(M·E1, M·E2)| / |cross(E1, E2)|` (uniform
//! scale `s` ⇒ factor `s²`; non-uniform diagonal scale depends on
//! triangle orientation), and a closed-surface enclosed volume scales
//! by `det(M_3x3)`.
//!
//! Cube-builder helpers are inlined so this file is self-contained
//! and does not depend on test-helper modules from sibling test
//! binaries.

use oxideav_mesh3d::{Mesh, Node, NodeId, Primitive, Scene3D, Topology, Transform};

const TOL: f64 = 1e-8;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOL + TOL * a.abs().max(b.abs())
}

/// Canonical CCW-from-outside unit cube. Local surface area = 6,
/// local signed volume = 1.
fn unit_cube_ccw() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // +Z face
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        // -Z face
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        // +X face
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        // -X face
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        // +Y face
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // -Y face
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    p
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn translation(t: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: t,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn scale(s: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: s,
    }
}

// ──────────────────────────────────────────────────────────────────
// Primitive::world_surface_area
// ──────────────────────────────────────────────────────────────────

#[test]
fn primitive_world_area_identity_matches_local() {
    let cube = unit_cube_ccw();
    assert!(approx_eq(
        cube.world_surface_area(IDENTITY),
        cube.surface_area()
    ));
    assert!(approx_eq(cube.world_surface_area(IDENTITY), 6.0));
}

#[test]
fn primitive_world_area_pure_translation_invariant() {
    let cube = unit_cube_ccw();
    let m = [
        [1.0, 0.0, 0.0, 100.0],
        [0.0, 1.0, 0.0, -50.0],
        [0.0, 0.0, 1.0, 7.5],
        [0.0, 0.0, 0.0, 1.0],
    ];
    assert!(approx_eq(cube.world_surface_area(m), 6.0));
}

#[test]
fn primitive_world_area_uniform_scale_squares() {
    let cube = unit_cube_ccw();
    // Uniform scale of 3.0 — area scales by 9.0 (3² per face).
    let m = [
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    assert!(approx_eq(cube.world_surface_area(m), 6.0 * 9.0));
}

#[test]
fn primitive_world_area_nonuniform_scale_axis_orientation_sensitive() {
    // For a 2×3×4 stretched cube the world surface area is
    //   2 · (a·b + b·c + a·c)  where (a,b,c) = (2,3,4)
    //   = 2 · (6 + 12 + 8) = 52.
    // A uniform-determinant scaling would predict the wrong number,
    // so this explicitly checks the per-triangle path.
    let cube = unit_cube_ccw();
    let m = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let want = 2.0 * (2.0 * 3.0 + 3.0 * 4.0 + 2.0 * 4.0);
    assert!(
        approx_eq(cube.world_surface_area(m), want),
        "got {} want {}",
        cube.world_surface_area(m),
        want
    );
}

#[test]
fn primitive_world_area_non_triangle_topology_zero() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(approx_eq(p.world_surface_area(IDENTITY), 0.0));
}

#[test]
fn primitive_world_area_empty_is_zero() {
    let p = Primitive::new(Topology::Triangles);
    assert!(approx_eq(p.world_surface_area(IDENTITY), 0.0));
}

#[test]
fn primitive_world_area_finite_on_nan_matrix() {
    let cube = unit_cube_ccw();
    let m = [
        [f32::NAN, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    // Every triangle's edge math produces NaN → contributes 0; total
    // stays finite (and is in fact 0, but the contract is "finite").
    let v = cube.world_surface_area(m);
    assert!(v.is_finite(), "got {v}");
}

#[test]
fn primitive_world_area_mirror_scale_unsigned_unchanged() {
    // Mirroring along one axis (negative scale) preserves area magnitude.
    let cube = unit_cube_ccw();
    let m = [
        [-2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    // Uniform-magnitude scale 2 — area scales by 4 regardless of mirror.
    assert!(approx_eq(cube.world_surface_area(m), 6.0 * 4.0));
}

// ──────────────────────────────────────────────────────────────────
// Scene3D::world_surface_area
// ──────────────────────────────────────────────────────────────────

#[test]
fn scene_world_area_empty_is_zero() {
    let scene = Scene3D::new();
    assert!(approx_eq(scene.world_surface_area(), 0.0));
}

#[test]
fn scene_world_area_unreachable_mesh_contributes_zero() {
    // Mesh exists in the scene's arena but no node references it.
    let mut scene = Scene3D::new();
    scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    assert!(approx_eq(scene.world_surface_area(), 0.0));
}

#[test]
fn scene_world_area_identity_root_matches_local() {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(Node::new().with_mesh(mid));
    scene.add_root(n);
    assert!(approx_eq(scene.world_surface_area(), 6.0));
}

#[test]
fn scene_world_area_node_scale_squares_area() {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(
        Node::new()
            .with_transform(scale([5.0, 5.0, 5.0]))
            .with_mesh(mid),
    );
    scene.add_root(n);
    // Uniform scale 5 — surface area scales 25× → 150.
    assert!(approx_eq(scene.world_surface_area(), 6.0 * 25.0));
}

#[test]
fn scene_world_area_two_instances_sum() {
    // Same mesh referenced by two separate nodes. The world total is
    // 2× the per-instance area; the resource-level Scene3D::surface_area
    // by contrast still reports 6.0 (mesh-resource total).
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n0 = scene.add_node(Node::new().with_mesh(mid));
    let n1 = scene.add_node(
        Node::new()
            .with_transform(translation([10.0, 0.0, 0.0]))
            .with_mesh(mid),
    );
    scene.add_root(n0);
    scene.add_root(n1);
    assert!(approx_eq(scene.world_surface_area(), 12.0));
    // Resource-level total stays at one mesh × 6.
    assert!(approx_eq(scene.surface_area(), 6.0));
}

#[test]
fn scene_world_area_ancestor_scale_chain_multiplies() {
    // Parent scale 2, child scale 3 — net scale 6 → area factor 36.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let parent = scene.add_node(Node::new().with_transform(scale([2.0, 2.0, 2.0])));
    let child = scene.add_node(
        Node::new()
            .with_transform(scale([3.0, 3.0, 3.0]))
            .with_mesh(mid),
    );
    scene.nodes[parent.0 as usize].children.push(child);
    scene.add_root(parent);
    assert!(approx_eq(scene.world_surface_area(), 6.0 * 36.0));
}

#[test]
fn scene_world_area_cycle_visits_once() {
    // A → B → A cycle, only B carries the mesh. Even with the back-edge,
    // the contribution is counted exactly once.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let a = scene.add_node(Node::new());
    let b = scene.add_node(Node::new().with_mesh(mid));
    scene.nodes[a.0 as usize].children.push(b);
    scene.nodes[b.0 as usize].children.push(a);
    scene.add_root(a);
    assert!(approx_eq(scene.world_surface_area(), 6.0));
}

#[test]
fn scene_world_area_detached_node_skipped() {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let _detached = scene.add_node(Node::new().with_mesh(mid));
    let live = scene.add_node(Node::new());
    scene.add_root(live);
    // The detached node is never reached → its mesh contributes 0.
    assert!(approx_eq(scene.world_surface_area(), 0.0));
}

#[test]
fn scene_world_area_out_of_range_mesh_id_is_skipped() {
    let mut scene = Scene3D::new();
    let n = scene.add_node(Node::new().with_mesh(oxideav_mesh3d::MeshId(99)));
    scene.add_root(n);
    assert!(approx_eq(scene.world_surface_area(), 0.0));
}

// ──────────────────────────────────────────────────────────────────
// Scene3D::world_signed_volume / world_volume
// ──────────────────────────────────────────────────────────────────

#[test]
fn scene_world_volume_empty_is_zero() {
    let scene = Scene3D::new();
    assert!(approx_eq(scene.world_signed_volume(), 0.0));
    assert!(approx_eq(scene.world_volume(), 0.0));
}

#[test]
fn scene_world_volume_identity_root_matches_local() {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(Node::new().with_mesh(mid));
    scene.add_root(n);
    assert!(approx_eq(scene.world_signed_volume(), 1.0));
    assert!(approx_eq(scene.world_volume(), 1.0));
}

#[test]
fn scene_world_volume_uniform_scale_cubes() {
    // Uniform scale 4 — volume scales by 64.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(
        Node::new()
            .with_transform(scale([4.0, 4.0, 4.0]))
            .with_mesh(mid),
    );
    scene.add_root(n);
    assert!(approx_eq(scene.world_signed_volume(), 64.0));
    assert!(approx_eq(scene.world_volume(), 64.0));
}

#[test]
fn scene_world_volume_nonuniform_scale_is_determinant_product() {
    // Scale (2, 3, 5) — det = 30; signed volume = 30 · 1 = 30.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(
        Node::new()
            .with_transform(scale([2.0, 3.0, 5.0]))
            .with_mesh(mid),
    );
    scene.add_root(n);
    assert!(approx_eq(scene.world_signed_volume(), 30.0));
    assert!(approx_eq(scene.world_volume(), 30.0));
}

#[test]
fn scene_world_volume_mirror_flips_sign_but_volume_unsigned() {
    // Single-axis mirror (scale -1 on x) flips winding → signed sign flips.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(
        Node::new()
            .with_transform(scale([-1.0, 1.0, 1.0]))
            .with_mesh(mid),
    );
    scene.add_root(n);
    assert!(approx_eq(scene.world_signed_volume(), -1.0));
    assert!(approx_eq(scene.world_volume(), 1.0));
}

#[test]
fn scene_world_volume_double_mirror_keeps_sign() {
    // Two mirrors → product of dets is +1, sign stays positive.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(
        Node::new()
            .with_transform(scale([-1.0, -1.0, 1.0]))
            .with_mesh(mid),
    );
    scene.add_root(n);
    assert!(approx_eq(scene.world_signed_volume(), 1.0));
}

#[test]
fn scene_world_volume_pure_translation_invariant_for_closed_mesh() {
    // Translation must not change a closed-surface volume.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(
        Node::new()
            .with_transform(translation([100.0, -50.0, 7.5]))
            .with_mesh(mid),
    );
    scene.add_root(n);
    assert!(approx_eq(scene.world_signed_volume(), 1.0));
}

#[test]
fn scene_world_volume_two_instances_sum() {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n0 = scene.add_node(Node::new().with_mesh(mid));
    let n1 = scene.add_node(
        Node::new()
            .with_transform(translation([10.0, 0.0, 0.0]))
            .with_mesh(mid),
    );
    scene.add_root(n0);
    scene.add_root(n1);
    assert!(approx_eq(scene.world_signed_volume(), 2.0));
    // Resource-level baseline reports one cube.
    assert!(approx_eq(scene.signed_volume(), 1.0));
}

#[test]
fn scene_world_volume_mirror_plus_unmirrored_cancel_in_signed_sum() {
    // Two instances, one mirrored. The signed sum cancels to ~0;
    // the unsigned helper reports that cancelled magnitude, NOT the
    // per-instance |det|·|V| total (matching the documented contract).
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n0 = scene.add_node(Node::new().with_mesh(mid));
    let n1 = scene.add_node(
        Node::new()
            .with_transform(scale([-1.0, 1.0, 1.0]))
            .with_mesh(mid),
    );
    scene.add_root(n0);
    scene.add_root(n1);
    assert!(approx_eq(scene.world_signed_volume(), 0.0));
    assert!(approx_eq(scene.world_volume(), 0.0));
}

#[test]
fn scene_world_volume_ancestor_scale_chain_multiplies_determinant() {
    // Parent scale (2, 2, 2) → det 8; child scale (3, 1, 1) → det 3;
    // net 24 × unit cube volume = 24.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let parent = scene.add_node(Node::new().with_transform(scale([2.0, 2.0, 2.0])));
    let child = scene.add_node(
        Node::new()
            .with_transform(scale([3.0, 1.0, 1.0]))
            .with_mesh(mid),
    );
    scene.nodes[parent.0 as usize].children.push(child);
    scene.add_root(parent);
    assert!(approx_eq(scene.world_signed_volume(), 24.0));
}

#[test]
fn scene_world_volume_cycle_visits_once() {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let a = scene.add_node(Node::new());
    let b = scene.add_node(Node::new().with_mesh(mid));
    scene.nodes[a.0 as usize].children.push(b);
    scene.nodes[b.0 as usize].children.push(a);
    scene.add_root(a);
    assert!(approx_eq(scene.world_signed_volume(), 1.0));
}

#[test]
fn scene_world_volume_detached_node_skipped() {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let _detached = scene.add_node(Node::new().with_mesh(mid));
    let live = scene.add_node(Node::new());
    scene.add_root(live);
    assert!(approx_eq(scene.world_signed_volume(), 0.0));
}

#[test]
fn scene_world_volume_matrix_transform_variant_passes_through() {
    // A bare 4x4 affine matrix (no TRS decomposition) carrying a
    // scale-and-translation must still produce the correct det.
    let m: [[f32; 4]; 4] = [
        [2.0, 0.0, 0.0, 100.0],
        [0.0, 3.0, 0.0, -1.0],
        [0.0, 0.0, 4.0, 7.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(
        Node::new()
            .with_transform(Transform::Matrix(m))
            .with_mesh(mid),
    );
    scene.add_root(n);
    // det = 24; translation column doesn't affect closed-mesh volume.
    assert!(approx_eq(scene.world_signed_volume(), 24.0));
}

#[test]
fn scene_world_area_matrix_transform_variant_uniform() {
    // Same scenario as above but for area: uniform scale 2 → factor 4.
    let m: [[f32; 4]; 4] = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(
        Node::new()
            .with_transform(Transform::Matrix(m))
            .with_mesh(mid),
    );
    scene.add_root(n);
    assert!(approx_eq(scene.world_surface_area(), 6.0 * 4.0));
}

#[test]
fn scene_world_volume_shared_instance_resolves_once() {
    // Same shared-instance convention as world_node_transforms:
    // a node referenced by two parents is visited once, via the first
    // parent. The mesh contribution is therefore counted once, not twice.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let r0 = scene.add_node(Node::new().with_transform(scale([2.0, 2.0, 2.0])));
    let r1 = scene.add_node(Node::new().with_transform(scale([3.0, 3.0, 3.0])));
    let shared = scene.add_node(Node::new().with_mesh(mid));
    scene.nodes[r0.0 as usize].children.push(shared);
    scene.nodes[r1.0 as usize].children.push(shared);
    scene.add_root(r0);
    scene.add_root(r1);
    // r0 is pushed first → shared resolves via r0's chain (det 8).
    assert!(approx_eq(scene.world_signed_volume(), 8.0));
}

#[test]
fn scene_world_area_shared_instance_resolves_once() {
    // Mirror of the volume case for the area helper.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let r0 = scene.add_node(Node::new().with_transform(scale([2.0, 2.0, 2.0])));
    let r1 = scene.add_node(Node::new().with_transform(scale([3.0, 3.0, 3.0])));
    let shared = scene.add_node(Node::new().with_mesh(mid));
    scene.nodes[r0.0 as usize].children.push(shared);
    scene.nodes[r1.0 as usize].children.push(shared);
    scene.add_root(r0);
    scene.add_root(r1);
    // r0's uniform scale 2 → area factor 4 → 6 × 4 = 24.
    assert!(approx_eq(scene.world_surface_area(), 24.0));
}

#[test]
fn scene_world_volume_out_of_range_root_skipped() {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let n = scene.add_node(Node::new().with_mesh(mid));
    scene.add_root(NodeId(99));
    scene.add_root(n);
    assert!(approx_eq(scene.world_signed_volume(), 1.0));
}

#[test]
fn scene_world_volume_is_deterministic_across_calls() {
    // The DFS order is fixed; repeated calls must produce identical
    // bit-for-bit f64 values.
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(unit_cube_ccw()));
    let parent = scene.add_node(Node::new().with_transform(scale([2.0, 3.0, 5.0])));
    let child = scene.add_node(
        Node::new()
            .with_transform(scale([0.5, 0.5, 0.5]))
            .with_mesh(mid),
    );
    scene.nodes[parent.0 as usize].children.push(child);
    scene.add_root(parent);
    let a = scene.world_signed_volume();
    let b = scene.world_signed_volume();
    assert_eq!(a.to_bits(), b.to_bits());
}
