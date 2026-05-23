//! Tests for [`Primitive::compute_normals`] — area-weighted smooth
//! per-vertex normal recomputation.
//!
//! The scheme accumulates each triangle's un-normalised face normal
//! `(P[b]-P[a]) × (P[c]-P[a])` into its three vertices and normalises
//! the per-vertex sum. Because the cross-product magnitude equals twice
//! the triangle area, the accumulation is area-weighted automatically
//! (the textbook smooth-shading recomputation — Gouraud 1971; Foley,
//! van Dam et al.). A format decoder runs this when the wire stream
//! omits normals (OBJ without `vn`, glTF without `NORMAL`).

use oxideav_mesh3d::{Indices, Primitive, Topology};

const EPS: f32 = 1e-6;

fn assert_vec3_close(a: [f32; 3], b: [f32; 3]) {
    for k in 0..3 {
        assert!(
            (a[k] - b[k]).abs() < EPS,
            "component {k}: {a:?} vs {b:?} (diff {})",
            (a[k] - b[k]).abs()
        );
    }
}

fn is_unit(v: [f32; 3]) -> bool {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (len - 1.0).abs() < 1e-5
}

// ---- Single flat triangle ------------------------------------------------

#[test]
fn single_ccw_triangle_in_xy_plane_normal_is_plus_z() {
    // CCW winding in the XY plane => normal +Z (right-handed).
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let n = p.compute_normals();
    assert_eq!(n.len(), 3);
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, 1.0]);
        assert!(is_unit(*v));
    }
}

#[test]
fn single_cw_triangle_flips_normal_to_minus_z() {
    // Reverse winding => normal flips to -Z.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    let n = p.compute_normals();
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, -1.0]);
    }
}

#[test]
fn triangle_in_xz_plane_normal_is_minus_y() {
    // a=(0,0,0) b=(1,0,0) c=(0,0,1): u=+X, v=+Z, u×v = (0,-1,0).
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let n = p.compute_normals();
    for v in &n {
        assert_vec3_close(*v, [0.0, -1.0, 0.0]);
    }
}

#[test]
fn tilted_triangle_normal_is_unit_and_correct_direction() {
    // a=(0,0,0) b=(1,0,0) c=(0,1,1): u=+X, v=(0,1,1).
    // u×v = (0*1-0*1, 0*0-1*1, 1*1-0*0) = (0,-1,1) normalised.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
    let n = p.compute_normals();
    let inv = 1.0 / 2.0f32.sqrt();
    for v in &n {
        assert_vec3_close(*v, [0.0, -inv, inv]);
        assert!(is_unit(*v));
    }
}

// ---- Output shape contract -----------------------------------------------

#[test]
fn output_length_always_matches_positions() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = (0..9).map(|i| [i as f32, 0.0, 0.0]).collect();
    assert_eq!(p.compute_normals().len(), 9);
}

#[test]
fn empty_primitive_yields_empty_normals() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.compute_normals().is_empty());
}

#[test]
fn unreferenced_vertex_gets_fallback_normal() {
    // 4 positions, but only the first 3 form a triangle. Vertex 3 is
    // unreferenced => fallback [0,0,1].
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [9.0, 9.0, 9.0],
    ];
    p.indices = Some(Indices::U16(vec![0, 1, 2]));
    let n = p.compute_normals();
    assert_eq!(n.len(), 4);
    assert_vec3_close(n[0], [0.0, 0.0, 1.0]);
    assert_vec3_close(n[3], [0.0, 0.0, 1.0]); // fallback, not zero
}

// ---- Area weighting ------------------------------------------------------

#[test]
fn area_weighting_large_face_dominates_shared_vertex() {
    // Two triangles meeting at the shared vertex 0, lying in different
    // planes. The bigger triangle's face normal should dominate the
    // shared vertex normal (area-weighted average).
    //
    // Tri A (small, XY plane, +Z normal):
    //   v0=(0,0,0) v1=(1,0,0) v2=(0,1,0)
    // Tri B (large, XZ plane, -Y normal, 100x the area):
    //   v0=(0,0,0) v3=(10,0,0) v4=(0,0,10)  -> CCW gives -Y
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],  // 0 shared
        [1.0, 0.0, 0.0],  // 1
        [0.0, 1.0, 0.0],  // 2
        [10.0, 0.0, 0.0], // 3
        [0.0, 0.0, 10.0], // 4
    ];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 3, 4]));
    let n = p.compute_normals();

    // Tri A area cross = (0,0,1); tri B area cross = (0,-100,0).
    // Shared vertex sum = (0,-100,1), normalised ~ (0,-0.99995,0.0099995).
    assert!(is_unit(n[0]));
    assert!(
        n[0][1] < -0.99,
        "large -Y face must dominate shared vertex, got {:?}",
        n[0]
    );
    assert!(
        n[0][2] > 0.0 && n[0][2] < 0.02,
        "small +Z component, got {:?}",
        n[0]
    );

    // Vertex 2 only touches tri A => pure +Z.
    assert_vec3_close(n[2], [0.0, 0.0, 1.0]);
    // Vertex 4 only touches tri B => pure -Y.
    assert_vec3_close(n[4], [0.0, -1.0, 0.0]);
}

#[test]
fn coplanar_neighbours_give_identical_normals() {
    // Two triangles forming a flat quad in the XY plane share an edge.
    // Every vertex normal must be exactly +Z regardless of which/how
    // many faces touch it.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
    let n = p.compute_normals();
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, 1.0]);
    }
}

// ---- Topology integration (strip/fan via triangle_indices) ---------------

#[test]
fn triangle_strip_alternating_winding_consistent_normals() {
    // A flat strip across the XY plane: alternating winding is handled
    // by triangle_indices, so every face must come out +Z and every
    // vertex normal +Z.
    let mut p = Primitive::new(Topology::TriangleStrip);
    // Quad strip: (0,0)-(1,0)-(0,1)-(1,1) -> tri0=(0,1,2), tri1=(1,3,2).
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let n = p.compute_normals();
    assert_eq!(n.len(), 4);
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, 1.0]);
    }
}

#[test]
fn triangle_fan_shares_anchor_consistent_normals() {
    // Fan around anchor (0,0) in the XY plane.
    let mut p = Primitive::new(Topology::TriangleFan);
    p.positions = vec![
        [0.0, 0.0, 0.0], // anchor
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let n = p.compute_normals();
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, 1.0]);
    }
}

#[test]
fn non_triangle_topology_all_fallback() {
    // Lines contribute no faces => every normal is the fallback.
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ];
    let n = p.compute_normals();
    assert_eq!(n.len(), 4);
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, 1.0]);
    }
}

#[test]
fn points_topology_all_fallback() {
    let mut p = Primitive::new(Topology::Points);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let n = p.compute_normals();
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, 1.0]);
    }
}

// ---- Degenerate + robustness ---------------------------------------------

#[test]
fn degenerate_collinear_triangle_falls_back() {
    // Three collinear points => zero cross product => fallback.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let n = p.compute_normals();
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, 1.0]);
    }
}

#[test]
fn coincident_vertices_triangle_falls_back() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[5.0, 5.0, 5.0]; 3];
    let n = p.compute_normals();
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, 1.0]);
    }
}

#[test]
fn degenerate_face_does_not_corrupt_good_face_at_shared_vertex() {
    // Vertex 0 is shared by a good triangle (+Z) and a degenerate one.
    // The degenerate face adds zero => vertex 0 stays +Z.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0], // 0 shared
        [1.0, 0.0, 0.0], // 1
        [0.0, 1.0, 0.0], // 2
        [5.0, 0.0, 0.0], // 3 collinear w/ 0 and below
        [9.0, 0.0, 0.0], // 4 collinear -> degenerate face (0,3,4)
    ];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 3, 4]));
    let n = p.compute_normals();
    assert_vec3_close(n[0], [0.0, 0.0, 1.0]);
}

#[test]
fn out_of_range_index_is_skipped_not_panic() {
    // Index 99 has no position; that face is skipped. Vertex 0 still
    // gets the contribution from the valid face.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 1, 99]));
    let n = p.compute_normals();
    assert_eq!(n.len(), 3);
    assert_vec3_close(n[0], [0.0, 0.0, 1.0]);
}

#[test]
fn nan_position_face_skipped_vertex_falls_back() {
    // A face touching a NaN vertex produces a non-finite normal and is
    // skipped; the NaN vertex (touched only by that face) falls back.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [f32::NAN, 1.0, 0.0]];
    let n = p.compute_normals();
    assert_eq!(n.len(), 3);
    for v in &n {
        assert_vec3_close(*v, [0.0, 0.0, 1.0]);
    }
}

// ---- U16 vs U32 index parity ---------------------------------------------

#[test]
fn u16_and_u32_indices_produce_identical_normals() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let mut a = Primitive::new(Topology::Triangles);
    a.positions = positions.clone();
    a.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));

    let mut b = Primitive::new(Topology::Triangles);
    b.positions = positions;
    b.indices = Some(Indices::U32(vec![0, 1, 2, 0, 2, 3]));

    let na = a.compute_normals();
    let nb = b.compute_normals();
    assert_eq!(na.len(), nb.len());
    for (x, y) in na.iter().zip(nb.iter()) {
        assert_vec3_close(*x, *y);
    }
}

// ---- Round-trips with assignment + de-stripping --------------------------

#[test]
fn computed_normals_assignable_to_normals_field() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.normals = Some(p.compute_normals());
    let n = p.normals.as_ref().unwrap();
    assert_eq!(n.len(), p.positions.len());
    assert_vec3_close(n[0], [0.0, 0.0, 1.0]);
}

#[test]
fn normals_invariant_under_to_triangle_list() {
    // De-stripping must not change the geometry, so recomputed normals
    // before and after to_triangle_list must match per-vertex.
    let mut p = Primitive::new(Topology::TriangleStrip);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let before = p.compute_normals();
    let after = p.to_triangle_list().compute_normals();
    assert_eq!(before.len(), after.len());
    for (x, y) in before.iter().zip(after.iter()) {
        assert_vec3_close(*x, *y);
    }
}

#[test]
fn cube_corner_normals_point_outward_diagonally() {
    // Three faces of a unit cube meeting at the origin corner, each
    // wound CCW as seen from outside. The shared corner's smooth normal
    // should point roughly along the (-,-,-) diagonal away from the
    // cube interior (which sits in the +,+,+ octant here we orient so
    // outward is consistent). We just assert it is a unit vector with
    // all three components the same sign and non-trivial magnitude.
    let mut p = Primitive::new(Topology::Triangles);
    // Corner C=(0,0,0). Neighbours along +X,+Y,+Z.
    let c = [0.0, 0.0, 0.0];
    let x = [1.0, 0.0, 0.0];
    let y = [0.0, 1.0, 0.0];
    let z = [0.0, 0.0, 1.0];
    p.positions = vec![c, x, y, z];
    // Three faces around the corner, each CCW viewed from the -,-,-
    // outside so their normals lean negative on each axis.
    p.indices = Some(Indices::U16(vec![
        0, 2, 1, // corner, +Y, +X  (XY face, normal -Z)
        0, 3, 2, // corner, +Z, +Y  (YZ face, normal -X)
        0, 1, 3, // corner, +X, +Z  (XZ face, normal -Y)
    ]));
    let n = p.compute_normals();
    assert!(is_unit(n[0]));
    // Sum of (-Z)+(-X)+(-Y) face normals => (-1,-1,-1)/sqrt(3).
    let inv = 1.0 / 3.0f32.sqrt();
    assert_vec3_close(n[0], [-inv, -inv, -inv]);
}
