//! Tests for the transform-aware volume-weighted centroid:
//! [`Primitive::world_volume_centroid`],
//! [`Primitive::world_signed_volume`],
//! [`Mesh::world_volume_centroid`], and
//! [`Scene3D::world_volume_centroid`].
//!
//! Every closed-form expected value below is derived from elementary
//! geometry. The unit-cube CCW closed solid has signed volume +1 with
//! centre of mass at (0.5, 0.5, 0.5) — see `tests/volume_centroid.rs`
//! for the per-tet derivation (Mirtich, JGT 1996; Cha & Chen, ICIP
//! 2001). Under an invertible affine `M = [M_3 | t]` mapping a closed
//! two-manifold mesh, the post-transform signed volume is
//! `det(M_3) · V_local` (closed-mesh translation cancellation in the
//! origin-anchored tet sum) and the post-transform centre of mass is
//! `M · C_local = M_3 · C_local + t` (additivity of the volume
//! integral over the mapped body).

use oxideav_mesh3d::{Mesh, Node, Primitive, Scene3D, Topology, Transform};

fn approx_eq3(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn translation_matrix(t: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn scale_matrix(s: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [s[0], 0.0, 0.0, 0.0],
        [0.0, s[1], 0.0, 0.0],
        [0.0, 0.0, s[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn translation_node(t: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: t,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn scale_node(s: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: s,
    }
}

/// CCW-from-outside unit cube spanning [0, 1]³ (12 triangles, closed
/// two-manifold). Same vertex layout as `tests/volume_centroid.rs`'s
/// `unit_cube_ccw`.
fn unit_cube_ccw() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // +Z
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        // -Z
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        // +X
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        // -X
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        // +Y
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // -Y
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    p
}

// ──────────────────────────────────────────────────────────────────
// Primitive::world_volume_centroid + world_signed_volume
// ──────────────────────────────────────────────────────────────────

/// Identity world matrix reproduces the local centroid bit-for-bit.
#[test]
fn primitive_world_volume_centroid_identity_matches_local() {
    let p = unit_cube_ccw();
    let local = p.volume_centroid().unwrap();
    let world = p.world_volume_centroid(IDENTITY).unwrap();
    assert!(approx_eq3(local, world, 1e-12));
    assert!(approx_eq3(world, [0.5, 0.5, 0.5], 1e-12));
}

/// Identity matrix world signed volume matches the local signed
/// volume (closed mesh, unit cube → +1).
#[test]
fn primitive_world_signed_volume_identity_matches_local() {
    let p = unit_cube_ccw();
    assert!((p.world_signed_volume(IDENTITY) - p.signed_volume()).abs() < 1e-12);
    assert!((p.world_signed_volume(IDENTITY) - 1.0).abs() < 1e-12);
}

/// Pure-translation world matrix: post-transform centroid equals
/// local centroid plus the translation column.
#[test]
fn primitive_world_volume_centroid_pure_translation_equivariant() {
    let p = unit_cube_ccw();
    let local = p.volume_centroid().unwrap();
    let t = [10.0_f32, -7.0, 3.5];
    let m = translation_matrix(t);
    let world = p.world_volume_centroid(m).unwrap();
    let expected = [
        local[0] + t[0] as f64,
        local[1] + t[1] as f64,
        local[2] + t[2] as f64,
    ];
    assert!(approx_eq3(world, expected, 1e-6), "got {:?}", world);
}

/// Pure translation of a closed mesh: world signed volume equals the
/// local signed volume (no det / no translation enters for a closed
/// surface; the boundary terms cancel pairwise).
#[test]
fn primitive_world_signed_volume_pure_translation_unchanged_for_closed_mesh() {
    let p = unit_cube_ccw();
    let m = translation_matrix([100.0, -200.0, 50.0]);
    let world = p.world_signed_volume(m);
    let local = p.signed_volume();
    assert!((world - local).abs() < 1e-6, "got {} vs {}", world, local);
    assert!((world - 1.0).abs() < 1e-6);
}

/// Uniform scale around the origin: the centroid scales from the
/// origin by the same factor.
#[test]
fn primitive_world_volume_centroid_uniform_scale_scales_centroid() {
    let p = unit_cube_ccw();
    let m = scale_matrix([3.0, 3.0, 3.0]);
    let world = p.world_volume_centroid(m).unwrap();
    let expected = [0.5 * 3.0, 0.5 * 3.0, 0.5 * 3.0];
    assert!(approx_eq3(world, expected, 1e-12), "got {:?}", world);
}

/// Uniform scale by `s` cubes the signed volume.
#[test]
fn primitive_world_signed_volume_uniform_scale_cubes() {
    let p = unit_cube_ccw();
    let m = scale_matrix([2.0, 2.0, 2.0]);
    let world = p.world_signed_volume(m);
    assert!((world - 8.0).abs() < 1e-9, "got {}", world);
}

/// Non-uniform scale `(sx, sy, sz)`: each axis scales independently
/// — the centroid `(0.5, 0.5, 0.5)` maps to `(0.5·sx, 0.5·sy, 0.5·sz)`.
#[test]
fn primitive_world_volume_centroid_non_uniform_scale() {
    let p = unit_cube_ccw();
    let m = scale_matrix([2.0, 3.0, 5.0]);
    let world = p.world_volume_centroid(m).unwrap();
    let expected = [1.0, 1.5, 2.5];
    assert!(approx_eq3(world, expected, 1e-12), "got {:?}", world);
}

/// Non-uniform scale signed volume is `sx · sy · sz` (det of diagonal).
#[test]
fn primitive_world_signed_volume_non_uniform_scale_is_det() {
    let p = unit_cube_ccw();
    let m = scale_matrix([2.0, 3.0, 5.0]);
    let world = p.world_signed_volume(m);
    assert!((world - 30.0).abs() < 1e-9, "got {}", world);
}

/// Mirror scale on one axis flips the signed volume's sign.
#[test]
fn primitive_world_signed_volume_mirror_flips_sign() {
    let p = unit_cube_ccw();
    let m = scale_matrix([-1.0, 1.0, 1.0]);
    let world = p.world_signed_volume(m);
    assert!((world + 1.0).abs() < 1e-9, "got {}", world);
}

/// Mirror scale on the x axis maps the centroid `(0.5, 0.5, 0.5)` to
/// `(-0.5, 0.5, 0.5)`. The signed volume flips but the
/// volume-weighted centroid is sign-invariant (both numerator and
/// denominator scale by the same `det`).
#[test]
fn primitive_world_volume_centroid_mirror_x() {
    let p = unit_cube_ccw();
    let m = scale_matrix([-1.0, 1.0, 1.0]);
    let world = p.world_volume_centroid(m).unwrap();
    assert!(
        approx_eq3(world, [-0.5, 0.5, 0.5], 1e-12),
        "got {:?}",
        world
    );
}

/// Combined scale + translation: centroid maps to
/// `M_3 · C_local + t`.
#[test]
fn primitive_world_volume_centroid_scale_then_translate() {
    let p = unit_cube_ccw();
    // M = T(t) * S(s) applied as a single 4x4 (row-major col-vector):
    // upper-left 3x3 = diag(s), translation column = t.
    let m: [[f32; 4]; 4] = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 3.0, 0.0, -5.0],
        [0.0, 0.0, 5.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let world = p.world_volume_centroid(m).unwrap();
    let expected = [2.0 * 0.5 + 10.0, 3.0 * 0.5 - 5.0, 5.0 * 0.5 + 2.0];
    assert!(approx_eq3(world, expected, 1e-6), "got {:?}", world);
}

/// 180-degree rotation about the Z axis maps `(0.5, 0.5, 0.5)` to
/// `(-0.5, -0.5, 0.5)`.
#[test]
fn primitive_world_volume_centroid_z_rotation_180() {
    let p = unit_cube_ccw();
    let m: [[f32; 4]; 4] = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let world = p.world_volume_centroid(m).unwrap();
    assert!(
        approx_eq3(world, [-0.5, -0.5, 0.5], 1e-12),
        "got {:?}",
        world
    );
}

/// A 180-degree rotation has determinant +1 — signed volume is
/// unchanged.
#[test]
fn primitive_world_signed_volume_rotation_unchanged() {
    let p = unit_cube_ccw();
    let m: [[f32; 4]; 4] = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let world = p.world_signed_volume(m);
    assert!((world - 1.0).abs() < 1e-9, "got {}", world);
}

/// `[0, 0, 0]` scale collapses every tet to zero signed volume and
/// the centroid is `None`.
#[test]
fn primitive_world_volume_centroid_zero_scale_is_none() {
    let p = unit_cube_ccw();
    let m = scale_matrix([0.0, 0.0, 0.0]);
    assert!(p.world_volume_centroid(m).is_none());
}

/// `[1, 1, 0]` flattens the cube to a slab of zero signed volume.
#[test]
fn primitive_world_volume_centroid_partial_collapse_is_none() {
    let p = unit_cube_ccw();
    let m = scale_matrix([1.0, 1.0, 0.0]);
    assert!(p.world_volume_centroid(m).is_none());
}

/// Non-triangle topology returns `None` regardless of transform.
#[test]
fn primitive_world_volume_centroid_lines_is_none() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(p.world_volume_centroid(IDENTITY).is_none());
}

#[test]
fn primitive_world_volume_centroid_points_is_none() {
    let mut p = Primitive::new(Topology::Points);
    p.positions = vec![[0.0, 0.0, 0.0]];
    assert!(p.world_volume_centroid(IDENTITY).is_none());
}

#[test]
fn primitive_world_volume_centroid_empty_is_none() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.world_volume_centroid(IDENTITY).is_none());
}

/// NaN coords are skipped triangle by triangle — finite tris carry
/// the result.
#[test]
fn primitive_world_volume_centroid_nan_coord_skipped() {
    let mut p = unit_cube_ccw();
    // Wipe one corner of one triangle to NaN — should reduce the sum
    // but still produce a finite, off-centre centroid.
    p.positions[0] = [f32::NAN, 0.0, 1.0];
    let c = p.world_volume_centroid(IDENTITY).expect("rest is finite");
    assert!(c[0].is_finite() && c[1].is_finite() && c[2].is_finite());
}

/// Out-of-range indices skipped silently.
#[test]
fn primitive_world_volume_centroid_oob_index_skipped() {
    let mut p = unit_cube_ccw();
    // Force an indexed topology with a single out-of-range triangle
    // plus the existing geometry: build a fresh primitive that
    // explicitly references the cube positions plus a bogus index.
    let n = p.positions.len() as u32;
    p.indices = Some(oxideav_mesh3d::Indices::U32(vec![
        // First triangle: out of range.
        n,
        n + 1,
        n + 2,
        // Second + third triangle: valid (-Y face from the cube).
        30,
        31,
        32,
        33,
        34,
        35,
    ]));
    // Two triangles spanning corners of a face — non-closed; we only
    // verify the helper returns finite output (no crash on OOB).
    let c = p.world_volume_centroid(IDENTITY);
    // -Y face triangles share corner (0, 0, 0) so the per-tet
    // signed-volume sum may be zero — accept either Some(finite) or
    // None.
    if let Some(v) = c {
        assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
    }
}

/// Translation-invariance for a closed mesh: `world_volume_centroid`
/// under translation t shifts the centroid by t — agrees with
/// post-divide form `M · C_local`.
#[test]
fn primitive_world_volume_centroid_matches_post_divide_formula() {
    let p = unit_cube_ccw();
    let m: [[f32; 4]; 4] = [
        [2.0, 0.0, 0.0, 7.0],
        [0.0, 3.0, 0.0, -1.0],
        [0.0, 0.0, 4.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let local = p.volume_centroid().unwrap();
    let expected = [
        2.0 * local[0] + 7.0,
        3.0 * local[1] - 1.0,
        4.0 * local[2] + 0.5,
    ];
    let world = p.world_volume_centroid(m).unwrap();
    assert!(approx_eq3(world, expected, 1e-6), "got {:?}", world);
}

// ──────────────────────────────────────────────────────────────────
// Mesh::world_volume_centroid
// ──────────────────────────────────────────────────────────────────

/// Single-primitive mesh: pass-through (local helper agrees).
#[test]
fn mesh_world_volume_centroid_passthrough() {
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let c = m.world_volume_centroid(IDENTITY).unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12));
}

/// Empty mesh → None.
#[test]
fn mesh_world_volume_centroid_empty_is_none() {
    let m = Mesh::new(Some("empty".to_string()));
    assert!(m.world_volume_centroid(IDENTITY).is_none());
}

/// Mesh with all-degenerate primitives → None.
#[test]
fn mesh_world_volume_centroid_all_degenerate_is_none() {
    let mut m = Mesh::new(Some("degen".to_string()));
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    m = m.with_primitive(p);
    assert!(m.world_volume_centroid(IDENTITY).is_none());
}

/// Two equal-weight cubes at (0, 0, 0) and at (10, 0, 0): centroid
/// is the midpoint of their centres of mass in world space when the
/// world transform is identity, because the second cube's positions
/// are already at (10, 0, 0).
#[test]
fn mesh_world_volume_centroid_two_cubes_midpoint() {
    let cube_a = unit_cube_ccw();
    let mut cube_b = unit_cube_ccw();
    for v in &mut cube_b.positions {
        v[0] += 10.0;
    }
    let mut m = Mesh::new(Some("two".to_string()));
    m = m.with_primitive(cube_a).with_primitive(cube_b);
    let c = m.world_volume_centroid(IDENTITY).unwrap();
    // Both cubes have signed volume +1; equal-weight average of
    // (0.5, 0.5, 0.5) and (10.5, 0.5, 0.5).
    assert!(approx_eq3(c, [5.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

/// Translation moves the mesh-level centroid by the same vector.
#[test]
fn mesh_world_volume_centroid_translation_shifts_result() {
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let t = [3.0_f32, -2.0, 7.5];
    let c = m.world_volume_centroid(translation_matrix(t)).unwrap();
    assert!(approx_eq3(c, [0.5 + 3.0, 0.5 - 2.0, 0.5 + 7.5], 1e-6));
}

/// Two cubes with one mirrored: the mirror flips the signed volume of
/// the second cube while keeping its centroid at the same place. The
/// signed-volume-weighted recombination cancels them.
#[test]
fn mesh_world_volume_centroid_cancelling_shells_is_none() {
    let cube_a = unit_cube_ccw();
    // Mirror cube b in place by reversing every triangle's winding so
    // its signed volume is -1 with the same centroid.
    let mut cube_b = unit_cube_ccw();
    for tri in cube_b.positions.chunks_mut(3) {
        tri.swap(1, 2);
    }
    let mut m = Mesh::new(Some("cancel".to_string()));
    m = m.with_primitive(cube_a).with_primitive(cube_b);
    // sum_v = +1 + (-1) = 0 → None.
    assert!(m.world_volume_centroid(IDENTITY).is_none());
}

// ──────────────────────────────────────────────────────────────────
// Scene3D::world_volume_centroid
// ──────────────────────────────────────────────────────────────────

/// Empty scene → None.
#[test]
fn scene_world_volume_centroid_empty_is_none() {
    let s = Scene3D::new();
    assert!(s.world_volume_centroid().is_none());
}

/// Scene with a mesh but no roots → None.
#[test]
fn scene_world_volume_centroid_no_roots_is_none() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let _ = s.add_mesh(m);
    assert!(s.world_volume_centroid().is_none());
}

/// Identity-transform single node: matches the resource-level
/// `volume_centroid`.
#[test]
fn scene_world_volume_centroid_single_identity_matches_local() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let mid = s.add_mesh(m);
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    let c = s.world_volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

/// Translated node: centroid shifts by the translation.
#[test]
fn scene_world_volume_centroid_translation_shifts() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let mid = s.add_mesh(m);
    let nid = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_node([10.0, -5.0, 2.0])),
    );
    s.add_root(nid);
    let c = s.world_volume_centroid().unwrap();
    assert!(
        approx_eq3(c, [0.5 + 10.0, 0.5 - 5.0, 0.5 + 2.0], 1e-6),
        "got {:?}",
        c
    );
}

/// Scaled node: centroid scales from the origin by the same factor.
#[test]
fn scene_world_volume_centroid_uniform_scale() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let mid = s.add_mesh(m);
    let nid = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(scale_node([4.0; 3])),
    );
    s.add_root(nid);
    let c = s.world_volume_centroid().unwrap();
    assert!(approx_eq3(c, [2.0, 2.0, 2.0], 1e-12), "got {:?}", c);
}

/// Two instances of the same mesh at different translations: the
/// scene-level centroid is the volume-weighted midpoint of the per-
/// instance centroids. Both instances have the same (+1) signed
/// volume so the recombination is the simple midpoint of their
/// world centres.
#[test]
fn scene_world_volume_centroid_two_instances_midpoint() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let mid = s.add_mesh(m);
    let n0 = s.add_node(Node::new().with_mesh(mid).with_transform(IDENTITY_TRS));
    let n1 = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_node([10.0, 0.0, 0.0])),
    );
    s.add_root(n0);
    s.add_root(n1);
    let c = s.world_volume_centroid().unwrap();
    // World centres: (0.5, 0.5, 0.5) and (10.5, 0.5, 0.5) → midpoint
    // (5.5, 0.5, 0.5).
    assert!(approx_eq3(c, [5.5, 0.5, 0.5], 1e-6), "got {:?}", c);
}

/// Scene-level helper instances meshes per-node (vs resource-level
/// [`Scene3D::volume_centroid`] which walks each mesh once
/// regardless of instance count). Two-instance scene vs the
/// resource-level total must differ — the per-instance result must
/// account for the second translated copy.
#[test]
fn scene_world_volume_centroid_per_instance_not_per_resource() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let mid = s.add_mesh(m);
    let n0 = s.add_node(Node::new().with_mesh(mid).with_transform(IDENTITY_TRS));
    let n1 = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_node([10.0, 0.0, 0.0])),
    );
    s.add_root(n0);
    s.add_root(n1);
    let world = s.world_volume_centroid().unwrap();
    let resource = s.volume_centroid().unwrap();
    // Resource-level walks the mesh once → (0.5, 0.5, 0.5). The
    // per-instance walk adds the translated copy.
    assert!(approx_eq3(resource, [0.5, 0.5, 0.5], 1e-12));
    assert!(approx_eq3(world, [5.5, 0.5, 0.5], 1e-6));
    assert!(!approx_eq3(world, resource, 1e-3));
}

/// Unreachable mesh (no node references it) contributes nothing.
#[test]
fn scene_world_volume_centroid_unreachable_mesh_skipped() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let _mid = s.add_mesh(m);
    // Add a node that does NOT carry a mesh, mark it as root.
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    // No reachable mesh → None.
    assert!(s.world_volume_centroid().is_none());
}

/// Two reachable meshes at world centres `(0.5, 0.5, 0.5)` and
/// `(2.5, 0.5, 0.5)` (the second is a 2x-scaled cube whose centre is
/// at the (1.0, 0.5, 0.5) post-translation... wait, careful.)
///
/// We construct: mesh A is a cube at local origin (signed_volume = 1,
/// centroid (0.5, 0.5, 0.5)); mesh B is the same cube translated in
/// the node by (5, 0, 0). World centres: (0.5, 0.5, 0.5) and
/// (5.5, 0.5, 0.5). Both signed-volume +1 → average x = 3.0.
#[test]
fn scene_world_volume_centroid_two_meshes() {
    let mut s = Scene3D::new();
    let mut a = Mesh::new(Some("a".to_string()));
    a = a.with_primitive(unit_cube_ccw());
    let aid = s.add_mesh(a);
    let mut b = Mesh::new(Some("b".to_string()));
    b = b.with_primitive(unit_cube_ccw());
    let bid = s.add_mesh(b);
    let na = s.add_node(Node::new().with_mesh(aid).with_transform(IDENTITY_TRS));
    let nb = s.add_node(
        Node::new()
            .with_mesh(bid)
            .with_transform(translation_node([5.0, 0.0, 0.0])),
    );
    s.add_root(na);
    s.add_root(nb);
    let c = s.world_volume_centroid().unwrap();
    assert!(approx_eq3(c, [3.0, 0.5, 0.5], 1e-6), "got {:?}", c);
}

/// Nested transforms: child node inherits parent translation.
#[test]
fn scene_world_volume_centroid_nested_transforms() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let mid = s.add_mesh(m);
    // Root translates by (10, 0, 0); child carries the mesh and
    // additionally translates by (0, 5, 0).
    let child = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_node([0.0, 5.0, 0.0])),
    );
    let mut root = Node::new().with_transform(translation_node([10.0, 0.0, 0.0]));
    root.children.push(child);
    let rid = s.add_node(root);
    s.add_root(rid);
    let c = s.world_volume_centroid().unwrap();
    // World centre of the cube: (0.5 + 0 + 10, 0.5 + 5 + 0, 0.5).
    assert!(approx_eq3(c, [10.5, 5.5, 0.5], 1e-6), "got {:?}", c);
}

/// Cancelling mirror instance: a cube at (0, 0, 0) and a mirrored
/// (x-flipped) copy at the same place — signed volumes +1 and -1
/// cancel.
#[test]
fn scene_world_volume_centroid_mirrored_instance_cancels() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let mid = s.add_mesh(m);
    let n0 = s.add_node(Node::new().with_mesh(mid));
    // Mirrored copy: scale x by -1 then translate by (1, 0, 0) so the
    // cube re-occupies the same [0, 1]³ region.
    let m_mirror: [[f32; 4]; 4] = [
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let n1 = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(Transform::Matrix(m_mirror)),
    );
    s.add_root(n0);
    s.add_root(n1);
    // sum_v = +1 + (-1) = 0 → None.
    assert!(s.world_volume_centroid().is_none());
}

/// Scene with a single mesh under a chain that includes a non-mesh
/// intermediate node: the world transform still composes the
/// intermediate's translation.
#[test]
fn scene_world_volume_centroid_intermediate_pure_xform_node() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let mid = s.add_mesh(m);
    let leaf = s.add_node(Node::new().with_mesh(mid));
    let mut inter = Node::new().with_transform(scale_node([2.0, 1.0, 1.0]));
    inter.children.push(leaf);
    let iid = s.add_node(inter);
    let mut root = Node::new().with_transform(translation_node([3.0, 0.0, 0.0]));
    root.children.push(iid);
    let rid = s.add_node(root);
    s.add_root(rid);
    let c = s.world_volume_centroid().unwrap();
    // Local centroid (0.5, 0.5, 0.5) → scale(2,1,1) → (1.0, 0.5, 0.5)
    // → translate(3, 0, 0) → (4.0, 0.5, 0.5).
    assert!(approx_eq3(c, [4.0, 0.5, 0.5], 1e-6), "got {:?}", c);
}

/// Sanity: scene-level world_volume_centroid is independent of the
/// scene-graph cycle guard's reachability rules — a mesh attached to
/// an unreachable node (NOT a root and NOT a child of any root) is
/// not visited and so the answer matches the all-reachable case.
#[test]
fn scene_world_volume_centroid_unreachable_node_skipped() {
    let mut s = Scene3D::new();
    let mut m = Mesh::new(Some("cube".to_string()));
    m = m.with_primitive(unit_cube_ccw());
    let mid = s.add_mesh(m);
    let reachable = s.add_node(Node::new().with_mesh(mid));
    let _unreachable = s.add_node(Node::new().with_mesh(mid));
    s.add_root(reachable);
    let c = s.world_volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

const IDENTITY_TRS: Transform = Transform::Trs {
    translation: [0.0; 3],
    rotation: [0.0, 0.0, 0.0, 1.0],
    scale: [1.0, 1.0, 1.0],
};
