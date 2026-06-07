//! Tests for `Primitive::volume_centroid`, `Mesh::volume_centroid`,
//! and `Scene3D::volume_centroid`.
//!
//! Every closed-form expected value below is derived from elementary
//! geometry — the signed-volume-weighted average of per-tetrahedron
//! centroids, with each surface triangle `(P_a, P_b, P_c)` mapped to
//! the origin-anchored tetrahedron whose centroid is
//! `(P_a + P_b + P_c) / 4` and signed volume is
//! `(P_a · (P_b × P_c)) / 6`. The same derivation supports the
//! `Primitive::signed_volume` reduction (Cha & Chen, ICIP 2001;
//! Mirtich, JGT 1996).

use oxideav_mesh3d::{Indices, Mesh, Primitive, Scene3D, Topology};

fn approx_eq3(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
}

/// CCW (front-facing-outward) unit cube spanning [0, 1]³ with 12
/// triangles. Centre of mass sits at (0.5, 0.5, 0.5). Same winding
/// convention as `tests/volume.rs::unit_cube_ccw` (signed volume = +1).
fn unit_cube_ccw() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // +Z face (top, normal +Z) — CCW seen from +Z.
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        // -Z face (bottom, normal -Z) — CCW seen from -Z.
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        // +X face (right, normal +X) — CCW seen from +X.
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        // -X face (left, normal -X) — CCW seen from -X.
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        // +Y face (back, normal +Y) — CCW seen from +Y.
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // -Y face (front, normal -Y) — CCW seen from -Y.
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    p
}

/// Same cube wound CW-from-outside (inside-out). Same centre of mass.
fn unit_cube_cw() -> Primitive {
    let mut p = unit_cube_ccw();
    // Reverse winding of every triangle by swapping the last two
    // corners.
    for tri in p.positions.chunks_mut(3) {
        tri.swap(1, 2);
    }
    p
}

/// Canonical tetrahedron with corners at the origin and three unit
/// edges along the axes. Volume = 1/6. The four corners are
/// (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1); the corner-mean
/// centroid is (0.25, 0.25, 0.25).
fn axis_tetrahedron() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // Bottom: O, B, A (normal -Z).
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        // Front: O, A, C (normal -Y).
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        // Left: O, C, B (normal -X).
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        // Slanted face: A, B, C (normal in +,+,+).
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    p
}

// ---- Primitive-level tests ----

/// Unit cube spans [0, 1]³; the centre of mass is the cube's centre.
#[test]
fn unit_cube_centroid_is_centre() {
    let p = unit_cube_ccw();
    let c = p.volume_centroid().expect("closed solid");
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

/// Sign invariance: the same cube wound CW-from-outside (inside-out)
/// produces a negative signed volume but the same centre of mass.
#[test]
fn cw_cube_centroid_matches_ccw() {
    let ccw = unit_cube_ccw();
    let cw = unit_cube_cw();
    // Sanity: signed volumes are negatives of each other.
    assert!((ccw.signed_volume() - 1.0).abs() < 1e-12);
    assert!((cw.signed_volume() + 1.0).abs() < 1e-12);
    let c_ccw = ccw.volume_centroid().unwrap();
    let c_cw = cw.volume_centroid().unwrap();
    assert!(
        approx_eq3(c_ccw, c_cw, 1e-12),
        "ccw={:?} cw={:?}",
        c_ccw,
        c_cw
    );
}

/// Translating every vertex by a constant moves the centre of mass by
/// the same vector (translation equivariance for closed surfaces).
#[test]
fn centroid_translation_equivariant() {
    let p0 = unit_cube_ccw();
    let c0 = p0.volume_centroid().unwrap();
    let mut p1 = unit_cube_ccw();
    let t = [10.0, -7.0, 3.5];
    for v in &mut p1.positions {
        v[0] += t[0];
        v[1] += t[1];
        v[2] += t[2];
    }
    let c1 = p1.volume_centroid().unwrap();
    let expected = [
        c0[0] + t[0] as f64,
        c0[1] + t[1] as f64,
        c0[2] + t[2] as f64,
    ];
    assert!(
        approx_eq3(c1, expected, 1e-10),
        "c0={:?} c1={:?} expected={:?}",
        c0,
        c1,
        expected
    );
}

/// Scaling the cube non-uniformly moves the centroid to the centre of
/// the scaled box.
#[test]
fn non_uniform_scale_centroid_at_box_centre() {
    let mut p = unit_cube_ccw();
    for v in &mut p.positions {
        v[0] *= 4.0;
        v[1] *= 2.0;
        v[2] *= 8.0;
    }
    let c = p.volume_centroid().unwrap();
    assert!(approx_eq3(c, [2.0, 1.0, 4.0], 1e-9), "got {:?}", c);
}

/// The axis-tetrahedron centroid equals the corner mean
/// `(O + A + B + C) / 4 = (0.25, 0.25, 0.25)`.
#[test]
fn axis_tetrahedron_centroid() {
    let p = axis_tetrahedron();
    // Signed volume sanity: V = 1/6.
    assert!((p.signed_volume() - 1.0 / 6.0).abs() < 1e-12);
    let c = p.volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.25, 0.25, 0.25], 1e-12), "got {:?}", c);
}

/// Cube made via an index buffer agrees with the unindexed version.
///
/// Corner indexing:
///   0 = (0,0,0)  1 = (1,0,0)  2 = (1,1,0)  3 = (0,1,0)
///   4 = (0,0,1)  5 = (1,0,1)  6 = (1,1,1)  7 = (0,1,1)
///
/// Winding mirrors `unit_cube_ccw` (CCW seen from outside on every
/// face, signed volume = +1).
#[test]
fn indexed_cube_matches_unindexed() {
    let unindexed = unit_cube_ccw();
    let mut indexed = Primitive::new(Topology::Triangles);
    indexed.positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [1.0, 1.0, 0.0], // 2
        [0.0, 1.0, 0.0], // 3
        [0.0, 0.0, 1.0], // 4
        [1.0, 0.0, 1.0], // 5
        [1.0, 1.0, 1.0], // 6
        [0.0, 1.0, 1.0], // 7
    ];
    indexed.indices = Some(Indices::U16(vec![
        // +Z face (top), CCW seen from +Z: (4,5,6), (4,6,7).
        4, 5, 6, 4, 6, 7, // -Z face (bottom), CCW seen from -Z: (3,2,1), (3,1,0).
        3, 2, 1, 3, 1, 0, // +X face, CCW seen from +X: (1,2,6), (1,6,5).
        1, 2, 6, 1, 6, 5, // -X face, CCW seen from -X: (4,7,3), (4,3,0).
        4, 7, 3, 4, 3, 0, // +Y face, CCW seen from +Y: (2,3,7), (2,7,6).
        2, 3, 7, 2, 7, 6, // -Y face, CCW seen from -Y: (0,1,5), (0,5,4).
        0, 1, 5, 0, 5, 4,
    ]));
    // Cross-check signed_volume too so any winding error surfaces as a
    // sign mismatch rather than a centroid drift.
    let sv_u = unindexed.signed_volume();
    let sv_i = indexed.signed_volume();
    assert!(
        (sv_u - sv_i).abs() < 1e-12,
        "signed_volume mismatch u={} i={}",
        sv_u,
        sv_i
    );
    let cu = unindexed.volume_centroid().unwrap();
    let ci = indexed.volume_centroid().unwrap();
    assert!(approx_eq3(cu, ci, 1e-12), "u={:?} i={:?}", cu, ci);
    assert!(approx_eq3(ci, [0.5, 0.5, 0.5], 1e-12));
}

/// `TriangleStrip` topology integrates through `triangle_indices`. A
/// stripped tetrahedron is unlikely to be closed; instead validate via
/// the equivalent flat-triangle answer on a strip-encoded fan that
/// degenerates back to one triangle.
#[test]
fn single_triangle_strip_signed_volume_zero_returns_none() {
    let mut p = Primitive::new(Topology::TriangleStrip);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    // One triangle → flat sheet → signed volume in `f64` is zero (the
    // scalar triple product `0 · (P_b × P_c)` is zero because `P_a` is
    // the origin) → None.
    assert!(p.volume_centroid().is_none());
}

/// A single triangle whose three corners are collinear (zero area)
/// has signed volume zero, so the volume centroid is undefined.
#[test]
fn collinear_triangle_returns_none() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    assert!(p.volume_centroid().is_none());
    // Sanity: signed volume is also zero (the per-tet triple product
    // is zero because the three corners are collinear).
    assert_eq!(p.signed_volume(), 0.0);
}

/// Lines / points / linestrip / lineloop / linelist topologies return
/// `None` (no triangles → no signed volume → no centroid).
#[test]
fn non_triangle_topologies_return_none() {
    let topologies = [
        Topology::Lines,
        Topology::LineStrip,
        Topology::LineLoop,
        Topology::Points,
    ];
    for t in topologies {
        let mut p = Primitive::new(t);
        p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(
            p.volume_centroid().is_none(),
            "{:?} should give None",
            p.topology
        );
    }
}

/// Empty positions return `None`.
#[test]
fn empty_positions_return_none() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.volume_centroid().is_none());
}

/// A primitive with every triangle degenerate (coincident corners)
/// returns `None` because every per-tet signed volume is zero.
#[test]
fn all_degenerate_returns_none() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[1.0, 1.0, 1.0]; 9];
    assert!(p.volume_centroid().is_none());
}

/// Out-of-range index entries are silently skipped — the remaining
/// in-range triangles drive the answer.
#[test]
fn out_of_range_indices_skipped() {
    let cube = unit_cube_ccw();
    let mut indexed = Primitive::new(Topology::Triangles);
    indexed.positions = cube.positions.clone();
    let mut idx = Vec::with_capacity(3 + cube.positions.len());
    // 0xFFFF is one past the cube's 36 vertices.
    idx.push(0xFFFFu32);
    idx.push(0xFFFFu32);
    idx.push(0xFFFFu32);
    for i in 0..cube.positions.len() {
        idx.push(i as u32);
    }
    indexed.indices = Some(Indices::U32(idx));
    let c = indexed.volume_centroid().unwrap();
    // The extra out-of-range triple is skipped; the rest is the cube.
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
    // Sanity: signed_volume is unchanged.
    assert!((indexed.signed_volume() - cube.signed_volume()).abs() < 1e-12);
}

/// A NaN coordinate in one corner produces non-finite per-triangle
/// arithmetic; that triangle is skipped silently and the remaining
/// faces still drive the answer.
#[test]
fn nan_positions_skipped() {
    let mut p = unit_cube_ccw();
    // Replace one isolated face with NaN coordinates.
    p.positions[0] = [f32::NAN, 0.0, 1.0];
    // The result should still be finite (other faces still work).
    if let Some(c) = p.volume_centroid() {
        assert!(c[0].is_finite() && c[1].is_finite() && c[2].is_finite());
    }
    // Either Some(finite) or None depending on how many faces survive
    // — what matters is no panic and no NaN output.
}

/// A cube centred at the origin has its centre of mass at the origin
/// (translation equivariance dropping any origin dependence).
#[test]
fn centred_cube_centroid_at_origin() {
    let mut p = unit_cube_ccw();
    for v in &mut p.positions {
        v[0] -= 0.5;
        v[1] -= 0.5;
        v[2] -= 0.5;
    }
    let c = p.volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.0, 0.0, 0.0], 1e-12), "got {:?}", c);
}

/// A scaled-up cube far from the origin still gives the correct
/// centre of mass — proves no origin dependence.
#[test]
fn distant_cube_centroid() {
    let mut p = unit_cube_ccw();
    for v in &mut p.positions {
        v[0] = v[0] * 2.0 + 1000.0;
        v[1] = v[1] * 3.0 - 500.0;
        v[2] = v[2] * 5.0 + 250.0;
    }
    let c = p.volume_centroid().unwrap();
    let expected = [1000.0 + 1.0, -500.0 + 1.5, 250.0 + 2.5];
    assert!(approx_eq3(c, expected, 1e-6), "got {:?}", c);
}

/// `TriangleStrip` for a closed strip (a small ribbon) — verify
/// topology integration. A flat strip has signed volume zero.
#[test]
fn flat_triangle_strip_returns_none() {
    let mut p = Primitive::new(Topology::TriangleStrip);
    // Flat zigzag in XY-plane.
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    // All triangles share Z = 0, so signed volume is zero.
    assert!(p.volume_centroid().is_none());
}

/// `TriangleFan` topology integration. A flat fan has signed volume
/// zero → None.
#[test]
fn flat_triangle_fan_returns_none() {
    let mut p = Primitive::new(Topology::TriangleFan);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    assert!(p.volume_centroid().is_none());
}

/// Cube centroid lies strictly inside the bounding box for a convex
/// closed shape.
#[test]
fn centroid_inside_bounding_box() {
    let p = unit_cube_ccw();
    let bb = p.bounding_box().unwrap();
    let c = p.volume_centroid().unwrap();
    assert!(c[0] >= bb.min[0] as f64 && c[0] <= bb.max[0] as f64);
    assert!(c[1] >= bb.min[1] as f64 && c[1] <= bb.max[1] as f64);
    assert!(c[2] >= bb.min[2] as f64 && c[2] <= bb.max[2] as f64);
}

/// For a uniform-density convex closed mesh whose centroid equals the
/// bounding-box centre (the unit cube), `volume_centroid` and
/// `surface_centroid` must agree at the centre.
#[test]
fn cube_surface_and_volume_centroids_coincide() {
    let p = unit_cube_ccw();
    let cv = p.volume_centroid().unwrap();
    let cs = p.surface_centroid().unwrap();
    assert!(approx_eq3(cv, cs, 1e-12), "v={:?} s={:?}", cv, cs);
}

/// For a non-cube (the axis-tetrahedron), the surface centroid and the
/// volume centroid disagree — surface centroid is pulled toward the
/// large slanted face; volume centroid is the corner mean.
#[test]
fn tetrahedron_surface_and_volume_centroids_differ() {
    let p = axis_tetrahedron();
    let cv = p.volume_centroid().unwrap();
    let cs = p.surface_centroid().unwrap();
    // Volume centroid is corner mean (0.25, 0.25, 0.25).
    assert!(approx_eq3(cv, [0.25, 0.25, 0.25], 1e-12));
    // Surface centroid is pulled toward the slanted face (away from
    // the origin) — it's NOT at (0.25, ...).
    assert!(
        !approx_eq3(cv, cs, 1e-3),
        "expected divergence got v={:?} s={:?}",
        cv,
        cs
    );
    // All three components agree by symmetry.
    assert!((cs[0] - cs[1]).abs() < 1e-12 && (cs[1] - cs[2]).abs() < 1e-12);
}

// ---- Mesh-level tests ----

#[test]
fn mesh_single_primitive_passthrough() {
    let p = unit_cube_ccw();
    let mesh = Mesh::new(Some("cube".to_owned())).with_primitive(p);
    let c = mesh.volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

#[test]
fn mesh_empty_returns_none() {
    let mesh = Mesh::new(Some("empty".to_owned()));
    assert!(mesh.volume_centroid().is_none());
}

/// A mesh with one degenerate primitive (zero signed volume) and one
/// good cube — the good cube's centroid wins.
#[test]
fn mesh_degenerate_primitive_skipped() {
    let mut bad = Primitive::new(Topology::Triangles);
    bad.positions = vec![[5.0, 5.0, 5.0]; 3];
    let good = unit_cube_ccw();
    let mesh = Mesh::new(Some("mixed".to_owned()))
        .with_primitive(bad)
        .with_primitive(good);
    let c = mesh.volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

/// Two equal-volume cubes — centroid is the midpoint.
#[test]
fn mesh_two_equal_volume_cubes_midpoint() {
    let p1 = unit_cube_ccw();
    let mut p2 = unit_cube_ccw();
    // p2 shifted to [10, 11]³.
    for v in &mut p2.positions {
        v[0] += 10.0;
        v[1] += 10.0;
        v[2] += 10.0;
    }
    let mesh = Mesh::new(Some("two".to_owned()))
        .with_primitive(p1)
        .with_primitive(p2);
    let c = mesh.volume_centroid().unwrap();
    // p1 centroid (0.5, 0.5, 0.5); p2 centroid (10.5, 10.5, 10.5);
    // equal volumes → midpoint (5.5, 5.5, 5.5).
    assert!(approx_eq3(c, [5.5, 5.5, 5.5], 1e-10), "got {:?}", c);
}

/// Unequal-volume cubes — the larger one pulls harder.
#[test]
fn mesh_unequal_volume_weighting() {
    // p1 is a 1x1x1 cube at origin (V = 1, centroid at (0.5, 0.5, 0.5)).
    let p1 = unit_cube_ccw();
    // p2 is a 2x2x2 cube at (10, 0, 0) (V = 8, centroid at (11, 1, 1)).
    let mut p2 = unit_cube_ccw();
    for v in &mut p2.positions {
        v[0] = v[0] * 2.0 + 10.0;
        v[1] *= 2.0;
        v[2] *= 2.0;
    }
    let mesh = Mesh::new(Some("two".to_owned()))
        .with_primitive(p1)
        .with_primitive(p2);
    let c = mesh.volume_centroid().unwrap();
    // Volume-weighted: (1·0.5 + 8·11)/(1+8) = 88.5/9 = 9.833...; for
    // Y and Z: (1·0.5 + 8·1)/9 = 8.5/9 = 0.9444....
    let expected_x = (1.0 * 0.5 + 8.0 * 11.0) / 9.0;
    let expected_y = (1.0 * 0.5 + 8.0 * 1.0) / 9.0;
    assert!(
        approx_eq3(c, [expected_x, expected_y, expected_y], 1e-9),
        "got {:?} expected ({}, {}, {})",
        c,
        expected_x,
        expected_y,
        expected_y
    );
}

/// A mesh whose every primitive cancels (one CCW cube + one identical
/// CW cube) reports zero signed volume → None.
#[test]
fn mesh_cancelling_shells_return_none() {
    let p_in = unit_cube_ccw();
    let p_out = unit_cube_cw();
    let mesh = Mesh::new(Some("cancel".to_owned()))
        .with_primitive(p_in)
        .with_primitive(p_out);
    assert!(mesh.volume_centroid().is_none());
}

/// Mesh with one non-triangle primitive (lines) and one cube — the
/// non-triangle primitive contributes nothing.
#[test]
fn mesh_lines_primitive_skipped() {
    let mut lines = Primitive::new(Topology::Lines);
    lines.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let cube = unit_cube_ccw();
    let mesh = Mesh::new(Some("mixed".to_owned()))
        .with_primitive(lines)
        .with_primitive(cube);
    let c = mesh.volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

// ---- Scene3D-level tests ----

#[test]
fn scene_empty_returns_none() {
    let s = Scene3D::new();
    assert!(s.volume_centroid().is_none());
}

#[test]
fn scene_single_mesh_passthrough() {
    let cube = Mesh::new(Some("c".to_owned())).with_primitive(unit_cube_ccw());
    let mut s = Scene3D::new();
    s.add_mesh(cube);
    let c = s.volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

/// Two equal-volume meshes — centroid is the midpoint.
#[test]
fn scene_two_equal_volume_meshes_midpoint() {
    let cube1 = Mesh::new(Some("a".to_owned())).with_primitive(unit_cube_ccw());
    let mut p2 = unit_cube_ccw();
    for v in &mut p2.positions {
        v[0] += 100.0;
    }
    let cube2 = Mesh::new(Some("b".to_owned())).with_primitive(p2);
    let mut s = Scene3D::new();
    s.add_mesh(cube1);
    s.add_mesh(cube2);
    let c = s.volume_centroid().unwrap();
    assert!(approx_eq3(c, [50.5, 0.5, 0.5], 1e-10), "got {:?}", c);
}

/// Scene walks meshes once, not node instances: instancing the same
/// mesh under multiple nodes doesn't change `scene.volume_centroid()`.
#[test]
fn scene_centroid_walks_meshes_not_nodes() {
    use oxideav_mesh3d::{Node, Transform};
    let cube = Mesh::new(Some("c".to_owned())).with_primitive(unit_cube_ccw());
    let mut s = Scene3D::new();
    let mid = s.add_mesh(cube);
    let n0 = s.add_node(Node::new().with_mesh(mid));
    let mut translated = Node::new().with_mesh(mid);
    translated.transform = Transform::Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [100.0, 100.0, 0.0, 1.0],
    ]);
    let n1 = s.add_node(translated);
    s.add_root(n0);
    s.add_root(n1);
    let c = s.volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

/// Scene with one degenerate mesh + one cube — the cube wins.
#[test]
fn scene_degenerate_mesh_skipped() {
    let mut bad_p = Primitive::new(Topology::Triangles);
    bad_p.positions = vec![[5.0, 5.0, 5.0]; 3];
    let bad_mesh = Mesh::new(Some("bad".to_owned())).with_primitive(bad_p);
    let good_mesh = Mesh::new(Some("good".to_owned())).with_primitive(unit_cube_ccw());
    let mut s = Scene3D::new();
    s.add_mesh(bad_mesh);
    s.add_mesh(good_mesh);
    let c = s.volume_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

/// Scene centroid components are always finite for finite input.
#[test]
fn scene_centroid_components_finite() {
    let mesh = Mesh::new(Some("c".to_owned())).with_primitive(unit_cube_ccw());
    let mut s = Scene3D::new();
    s.add_mesh(mesh);
    let c = s.volume_centroid().unwrap();
    assert!(c[0].is_finite() && c[1].is_finite() && c[2].is_finite());
}

/// Three equal-volume cubes at axis-aligned corners — the
/// volume-weighted centroid is the geometric centre of the three
/// (each cube contributes its own (0.5,0.5,0.5) plus the offset).
#[test]
fn scene_three_equal_volume_cubes() {
    let origins = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]];
    let mut s = Scene3D::new();
    for o in origins {
        let mut p = unit_cube_ccw();
        for v in &mut p.positions {
            v[0] += o[0];
            v[1] += o[1];
            v[2] += o[2];
        }
        s.add_mesh(Mesh::new(None::<String>).with_primitive(p));
    }
    let c = s.volume_centroid().unwrap();
    // Each cube centroid: (0.5+o.x, 0.5+o.y, 0.5+o.z); equal weights
    // → average: (0.5+10/3, 0.5+10/3, 0.5+0) = (3.833..., 3.833..., 0.5).
    let expected = [0.5 + 10.0 / 3.0, 0.5 + 10.0 / 3.0, 0.5];
    assert!(approx_eq3(c, expected, 1e-9), "got {:?}", c);
}

/// Cross-check: a single cube's `Scene3D::volume_centroid` matches its
/// `Primitive::volume_centroid` and its `Scene3D::surface_centroid`
/// (uniform-density convex symmetric shape).
#[test]
fn scene_cube_cross_check() {
    let p = unit_cube_ccw();
    let cp = p.volume_centroid().unwrap();
    let mesh = Mesh::new(Some("c".to_owned())).with_primitive(p);
    let mut s = Scene3D::new();
    s.add_mesh(mesh);
    let cs = s.volume_centroid().unwrap();
    let cs_surf = s.surface_centroid().unwrap();
    assert!(approx_eq3(cp, cs, 1e-12));
    assert!(approx_eq3(cp, cs_surf, 1e-12));
}
