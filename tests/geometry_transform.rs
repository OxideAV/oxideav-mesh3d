//! Primitive/Mesh::transformed + reverse_winding coverage: point vs
//! normal transform rules, non-uniform-scale perpendicularity, mirror
//! handedness, and winding reversal.

use oxideav_mesh3d::{Indices, Mesh, Primitive, Topology};

fn translate(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn scale(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [0.0, 0.0, z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// A single triangle in the xy-plane with an explicit +z normal.
fn tri_with_normal() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2]));
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    p.tangents = Some(vec![[1.0, 0.0, 0.0, 1.0]; 3]);
    p
}

#[test]
fn translation_moves_points_not_normals() {
    let prim = tri_with_normal();
    let out = prim.transformed(translate(5.0, -2.0, 3.0));
    assert_eq!(out.positions[0], [5.0, -2.0, 3.0]);
    assert_eq!(out.positions[1], [6.0, -2.0, 3.0]);
    // Normals are unaffected by translation.
    for n in out.normals.unwrap() {
        assert!((n[2] - 1.0).abs() < 1e-6);
        assert!(n[0].abs() < 1e-6 && n[1].abs() < 1e-6);
    }
}

#[test]
fn uniform_scale_keeps_normals_unit_and_oriented() {
    let prim = tri_with_normal();
    let out = prim.transformed(scale(3.0, 3.0, 3.0));
    assert_eq!(out.positions[1], [3.0, 0.0, 0.0]);
    for n in out.normals.unwrap() {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-6, "normal not unit: {len}");
        assert!((n[2] - 1.0).abs() < 1e-6);
    }
}

#[test]
fn non_uniform_scale_keeps_normal_perpendicular() {
    // Tilt the triangle out of the xy-plane so a non-uniform scale would
    // shear a naively-transformed normal off the surface. The
    // inverse-transpose rule must keep the stored normal equal to the
    // geometric face normal of the transformed triangle.
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    prim.indices = Some(Indices::U32(vec![0, 1, 2]));
    prim.normals = Some(prim.compute_normals());

    let m = scale(2.0, 1.0, 0.5);
    let out = prim.transformed(m);

    // Geometric face normal of the transformed triangle, freshly
    // computed from the moved positions.
    let geo = out.compute_normals();
    let stored = out.normals.unwrap();
    for i in 0..3 {
        // The inverse-transpose-transformed stored normal should align
        // with the actual transformed-surface normal (dot ≈ 1).
        let d = stored[i][0] * geo[i][0] + stored[i][1] * geo[i][1] + stored[i][2] * geo[i][2];
        assert!(d > 0.999, "normal not perpendicular to surface: dot {d}");
    }
}

#[test]
fn rotation_preserves_edge_lengths() {
    // 90° about z: x→y, y→-x.
    let rot = [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let prim = tri_with_normal();
    let out = prim.transformed(rot);
    // Vertex (1,0,0) rotates to (0,1,0).
    assert!((out.positions[1][0]).abs() < 1e-6);
    assert!((out.positions[1][1] - 1.0).abs() < 1e-6);
    // Surface area is rotation-invariant.
    assert!((out.surface_area() - prim.surface_area()).abs() < 1e-6);
}

#[test]
fn mirror_flips_tangent_handedness() {
    // Negative-determinant transform (flip x): handedness w must invert.
    let prim = tri_with_normal();
    let mirror = scale(-1.0, 1.0, 1.0);
    let out = prim.transformed(mirror);
    for t in out.tangents.unwrap() {
        assert!((t[3] + 1.0).abs() < 1e-6, "w not flipped: {}", t[3]);
    }
}

#[test]
fn singular_linear_part_leaves_normals_but_moves_points() {
    // Collapse z to 0: linear part is singular (det 0). Positions still
    // transform; normals are left as-is (finite fallback).
    let prim = tri_with_normal();
    let out = prim.transformed(scale(1.0, 1.0, 0.0));
    // z collapsed.
    assert!(out.positions.iter().all(|p| p[2].abs() < 1e-6));
    // Normals untouched (still +z).
    for n in out.normals.unwrap() {
        assert!((n[2] - 1.0).abs() < 1e-6);
    }
}

#[test]
fn mesh_transformed_applies_to_every_primitive() {
    let mesh = Mesh::new(Some("m".to_owned()))
        .with_primitive(tri_with_normal())
        .with_primitive(tri_with_normal());
    let out = mesh.transformed(translate(10.0, 0.0, 0.0));
    assert_eq!(out.primitives.len(), 2);
    for prim in &out.primitives {
        assert_eq!(prim.positions[0], [10.0, 0.0, 0.0]);
    }
    assert_eq!(out.name.as_deref(), Some("m"));
}

#[test]
fn reverse_winding_flips_face_orientation() {
    let prim = tri_with_normal();
    let before = prim.compute_normals()[0];
    let rev = prim.reverse_winding();
    let after = rev.compute_normals()[0];
    // The geometric face normal flips sign.
    assert!((before[2] - 1.0).abs() < 1e-6);
    assert!(
        (after[2] + 1.0).abs() < 1e-6,
        "winding not reversed: {after:?}"
    );
    // Stored normals are negated too.
    for n in rev.normals.unwrap() {
        assert!((n[2] + 1.0).abs() < 1e-6);
    }
}

#[test]
fn reverse_winding_inverts_tangent_w() {
    let prim = tri_with_normal();
    let rev = prim.reverse_winding();
    for t in rev.tangents.unwrap() {
        assert!((t[3] + 1.0).abs() < 1e-6);
    }
}

#[test]
fn transformed_preserves_topology_and_indices() {
    let prim = tri_with_normal();
    let out = prim.transformed(scale(2.0, 2.0, 2.0));
    assert_eq!(out.topology, Topology::Triangles);
    assert_eq!(out.triangle_indices(), prim.triangle_indices());
}

// --- morph deltas under transform (round 401) --------------------------

/// The triangle plus one morph target with position/normal/tangent
/// deltas of known non-unit length.
fn tri_with_morph() -> Primitive {
    let mut p = tri_with_normal();
    p.targets = vec![oxideav_mesh3d::MorphTarget {
        position: Some(vec![[0.0, 0.0, 2.0]; 3]),
        normal: Some(vec![[0.5, 0.0, 0.0]; 3]),
        tangent: Some(vec![[0.0, 3.0, 0.0]; 3]),
    }];
    p
}

#[test]
fn morph_position_deltas_rotate_with_the_linear_part_not_the_translation() {
    // Deltas are displacements: translation must not leak in.
    let out = tri_with_morph().transformed(translate(10.0, 20.0, 30.0));
    let d = out.targets[0].position.as_ref().unwrap();
    assert_eq!(d[0], [0.0, 0.0, 2.0], "translation-invariant");

    // +90° about X (column-vector): ẑ → -ŷ... check: R·(0,0,2) with
    // rows [[1,0,0],[0,0,-1],[0,1,0]] gives (0, -2, 0).
    let rot_x_90 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let out = tri_with_morph().transformed(rot_x_90);
    let d = out.targets[0].position.as_ref().unwrap();
    for v in d {
        assert!(
            (v[0]).abs() < 1e-6 && (v[1] + 2.0).abs() < 1e-6 && v[2].abs() < 1e-6,
            "delta should rotate to (0,-2,0): {v:?}"
        );
    }
}

#[test]
fn morph_delta_lengths_are_not_renormalised() {
    // Scale x2: position delta doubles (amplitude is data, not a unit
    // direction); normal delta transforms by inverse-transpose
    // (halves); tangent delta doubles.
    let out = tri_with_morph().transformed(scale(2.0, 2.0, 2.0));
    let t = &out.targets[0];
    assert_eq!(t.position.as_ref().unwrap()[0], [0.0, 0.0, 4.0]);
    assert_eq!(t.normal.as_ref().unwrap()[0], [0.25, 0.0, 0.0]);
    assert_eq!(t.tangent.as_ref().unwrap()[0], [0.0, 6.0, 0.0]);
}

#[test]
fn morph_then_transform_commutes_on_positions() {
    // M·(p + Σ wᵢdᵢ) == M·p + Σ wᵢ(L·dᵢ) exactly for affine M.
    let prim = tri_with_morph();
    let m = scale(2.0, 0.5, 3.0);
    let w = [0.7f32];

    // morph → transform
    let morphed = prim.apply_morph_weights(&w);
    let mut morphed_prim = prim.clone();
    morphed_prim.positions = morphed.positions;
    morphed_prim.targets = Vec::new();
    let a = morphed_prim.transformed(m);

    // transform → morph
    let b = prim.transformed(m).apply_morph_weights(&w);

    for (pa, pb) in a.positions.iter().zip(b.positions.iter()) {
        for i in 0..3 {
            assert!((pa[i] - pb[i]).abs() < 1e-5, "{pa:?} vs {pb:?}");
        }
    }
}

#[test]
fn singular_transform_keeps_normal_deltas_but_moves_position_deltas() {
    let out = tri_with_morph().transformed(scale(0.0, 1.0, 1.0));
    let t = &out.targets[0];
    assert_eq!(
        t.normal.as_ref().unwrap()[0],
        [0.5, 0.0, 0.0],
        "no inverse-transpose exists: delta kept"
    );
    assert_eq!(
        t.position.as_ref().unwrap()[0],
        [0.0, 0.0, 2.0],
        "z-delta survives the x-collapse"
    );
}
