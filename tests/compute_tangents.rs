//! Tests for [`Primitive::compute_tangents`] — per-vertex tangent-space
//! basis vectors from positions + UVs + normals.
//!
//! Closed-form derivation of `T = ( Δv2·E1 - Δv1·E2 ) / det` /
//! `B = (-Δu2·E1 + Δu1·E2 ) / det` (inverse of the 2×2 UV-delta linear
//! system; see Lengyel, "Computing Tangent Space Basis Vectors for an
//! Arbitrary Mesh" 2001, and Akenine-Möller, Haines & Hoffman,
//! *Real-Time Rendering*, "Normal Mapping" chapter). Per-vertex
//! accumulation is area-weighted by `|det|`, then Gram-Schmidt
//! orthonormalised against `N`, with handedness
//! `w = sign((N × T) · B)` such that the renderer reconstructs
//! `B = w * (N × T)` (glTF 2.0 §3.7.2.1).

use oxideav_mesh3d::{Indices, Primitive, Topology};

const EPS: f32 = 1e-5;

fn assert_vec3_close(a: [f32; 3], b: [f32; 3]) {
    for k in 0..3 {
        assert!(
            (a[k] - b[k]).abs() < EPS,
            "component {k}: {a:?} vs {b:?} (diff {})",
            (a[k] - b[k]).abs()
        );
    }
}

fn is_unit3(v: [f32; 3]) -> bool {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (len - 1.0).abs() < 1e-4
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Construct a unit-square primitive in the XY plane with axis-aligned
/// UVs `(0,0)..(1,1)`. Expected per-vertex tangent: `+X`, bitangent:
/// `+Y` (V grows with +Y, so B is +Y), normal `+Z` => handedness `+1`.
fn unit_square_xy() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
    p
}

// ---- Axis-aligned reference cases ----------------------------------------

#[test]
fn unit_square_xy_axis_aligned_uvs_yield_plus_x_tangent_plus_one_handedness() {
    let p = unit_square_xy();
    let t = p.compute_tangents(0).expect("present");
    assert_eq!(t.len(), 4);
    for v in &t {
        assert_vec3_close([v[0], v[1], v[2]], [1.0, 0.0, 0.0]);
        assert!(is_unit3([v[0], v[1], v[2]]));
        assert_eq!(v[3], 1.0);
    }
}

#[test]
fn flipped_v_axis_yields_negative_handedness() {
    // Same square, but V grows downward: UVs (0,1),(1,1),(1,0),(0,0).
    // Now B points in -Y; with N = +Z, the canonical (N × T) = +Y,
    // so dot((N×T), B) < 0 => w = -1.
    let mut p = unit_square_xy();
    p.uvs = vec![vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]];
    let t = p.compute_tangents(0).expect("present");
    for v in &t {
        // Tangent still +X (U axis is unchanged).
        assert_vec3_close([v[0], v[1], v[2]], [1.0, 0.0, 0.0]);
        assert_eq!(v[3], -1.0, "flipped V should mirror handedness");
    }
}

#[test]
fn flipped_u_axis_yields_negative_x_tangent() {
    // UVs grow with -X in object space.
    let mut p = unit_square_xy();
    p.uvs = vec![vec![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]];
    let t = p.compute_tangents(0).expect("present");
    for v in &t {
        assert_vec3_close([v[0], v[1], v[2]], [-1.0, 0.0, 0.0]);
        // U-flip + V unchanged => mirrored => -1.
        assert_eq!(v[3], -1.0);
    }
}

#[test]
fn square_in_xz_plane_tangent_in_xz_plane() {
    // Square in the XZ plane. U grows with +X, V grows with +Z. The
    // CCW winding (0,1,2)(0,2,3) gives geometric N = -Y, so we feed
    // that as the per-vertex normal. Expected per-vertex tangent
    // T = +X (parallel to U direction in object space, and already
    // perpendicular to N).
    //
    // Handedness check (renderer side): B = w * (N × T).
    //   N × T = (0,-1,0) × (1,0,0) = (0,0,1)
    // The data's UV says B = +Z, so dot((N×T), B) = +1 => w = +1.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    p.normals = Some(vec![[0.0, -1.0, 0.0]; 4]);
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
    let t = p.compute_tangents(0).expect("present");
    for v in &t {
        assert_vec3_close([v[0], v[1], v[2]], [1.0, 0.0, 0.0]);
        assert!(is_unit3([v[0], v[1], v[2]]));
        assert_eq!(v[3], 1.0);
    }
}

// ---- Orthonormality contract --------------------------------------------

#[test]
fn output_tangent_is_unit_length() {
    let p = unit_square_xy();
    let t = p.compute_tangents(0).expect("present");
    for v in &t {
        assert!(is_unit3([v[0], v[1], v[2]]));
    }
}

#[test]
fn output_tangent_is_orthogonal_to_normal() {
    // Gram-Schmidt step must make T perpendicular to N.
    let p = unit_square_xy();
    let t = p.compute_tangents(0).expect("present");
    let n = p.normals.as_ref().unwrap();
    for (tv, nv) in t.iter().zip(n.iter()) {
        let d = dot3([tv[0], tv[1], tv[2]], *nv);
        assert!(d.abs() < EPS, "T·N must be ~0 after Gram-Schmidt, got {d}");
    }
}

#[test]
fn handedness_is_exactly_plus_or_minus_one() {
    let p = unit_square_xy();
    let t = p.compute_tangents(0).expect("present");
    for v in &t {
        assert!(v[3] == 1.0 || v[3] == -1.0, "w must be ±1.0, got {}", v[3]);
    }
}

#[test]
fn reconstructed_bitangent_b_eq_w_cross_n_t_is_unit_and_orthogonal() {
    // Renderer-side reconstruction: B = w * (N × T). For our square
    // with axis-aligned UVs, B should come out exactly +Y.
    let p = unit_square_xy();
    let t = p.compute_tangents(0).expect("present");
    let n = p.normals.as_ref().unwrap();
    for (tv, nv) in t.iter().zip(n.iter()) {
        let cross = [
            nv[1] * tv[2] - nv[2] * tv[1],
            nv[2] * tv[0] - nv[0] * tv[2],
            nv[0] * tv[1] - nv[1] * tv[0],
        ];
        let b = [tv[3] * cross[0], tv[3] * cross[1], tv[3] * cross[2]];
        assert_vec3_close(b, [0.0, 1.0, 0.0]);
        assert!(is_unit3(b));
        // Bitangent orthogonal to both N and T.
        assert!(dot3(b, *nv).abs() < EPS);
        assert!(dot3(b, [tv[0], tv[1], tv[2]]).abs() < EPS);
    }
}

// ---- None-returns / missing inputs --------------------------------------

#[test]
fn missing_normals_returns_none() {
    let mut p = unit_square_xy();
    p.normals = None;
    assert!(p.compute_tangents(0).is_none());
}

#[test]
fn missing_uvs_returns_none() {
    let mut p = unit_square_xy();
    p.uvs.clear();
    assert!(p.compute_tangents(0).is_none());
}

#[test]
fn out_of_range_uv_set_returns_none() {
    let p = unit_square_xy();
    assert!(p.compute_tangents(5).is_none());
}

#[test]
fn empty_primitive_returns_none() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.compute_tangents(0).is_none());
}

#[test]
fn mismatched_normals_length_returns_none() {
    let mut p = unit_square_xy();
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]); // wrong length
    assert!(p.compute_tangents(0).is_none());
}

#[test]
fn mismatched_uvs_length_returns_none() {
    let mut p = unit_square_xy();
    p.uvs = vec![vec![[0.0, 0.0]; 2]]; // wrong length
    assert!(p.compute_tangents(0).is_none());
}

// ---- Output shape contract ----------------------------------------------

#[test]
fn output_length_always_matches_positions() {
    let p = unit_square_xy();
    let t = p.compute_tangents(0).expect("present");
    assert_eq!(t.len(), p.positions.len());
}

#[test]
fn assignable_to_tangents_field() {
    let mut p = unit_square_xy();
    p.tangents = Some(p.compute_tangents(0).expect("present"));
    let t = p.tangents.as_ref().unwrap();
    assert_eq!(t.len(), p.positions.len());
    assert_vec3_close([t[0][0], t[0][1], t[0][2]], [1.0, 0.0, 0.0]);
}

// ---- Topology integration (strip / fan via triangle_indices) -------------

#[test]
fn triangle_strip_produces_same_tangents_as_triangles() {
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    strip.normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
    strip.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]];
    let t = strip.compute_tangents(0).expect("present");
    for v in &t {
        assert_vec3_close([v[0], v[1], v[2]], [1.0, 0.0, 0.0]);
        assert_eq!(v[3], 1.0);
    }
}

#[test]
fn non_triangle_topology_all_fallback() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 2]);
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0]]];
    let t = p.compute_tangents(0).expect("present");
    assert_eq!(t.len(), 2);
    for v in &t {
        // Fallback is [1, 0, 0, 1].
        assert_vec3_close([v[0], v[1], v[2]], [1.0, 0.0, 0.0]);
        assert_eq!(v[3], 1.0);
    }
}

// ---- Degenerate UV cases -------------------------------------------------

#[test]
fn collinear_uvs_skip_face_and_fall_back() {
    // All three UVs collinear (in fact identical) => det = 0 => the
    // face contributes nothing => each vertex falls back to
    // [1, 0, 0, 1].
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    p.uvs = vec![vec![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]];
    let t = p.compute_tangents(0).expect("present");
    for v in &t {
        assert_vec3_close([v[0], v[1], v[2]], [1.0, 0.0, 0.0]);
        assert_eq!(v[3], 1.0);
    }
}

#[test]
fn unreferenced_vertex_gets_fallback_tangent() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [9.0, 9.0, 9.0],
    ];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]];
    p.indices = Some(Indices::U16(vec![0, 1, 2]));
    let t = p.compute_tangents(0).expect("present");
    assert_eq!(t.len(), 4);
    // Vertex 3 is unreferenced => fallback.
    assert_vec3_close([t[3][0], t[3][1], t[3][2]], [1.0, 0.0, 0.0]);
    assert_eq!(t[3][3], 1.0);
}

#[test]
fn out_of_range_index_skipped_not_panic() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 1, 99]));
    let t = p.compute_tangents(0).expect("present");
    // First face still produces good tangents.
    assert_vec3_close([t[0][0], t[0][1], t[0][2]], [1.0, 0.0, 0.0]);
}

#[test]
fn nan_uv_face_skipped_vertex_falls_back() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    p.uvs = vec![vec![[0.0, 0.0], [f32::NAN, 0.0], [0.0, 1.0]]];
    let t = p.compute_tangents(0).expect("present");
    assert_eq!(t.len(), 3);
    for v in &t {
        assert_vec3_close([v[0], v[1], v[2]], [1.0, 0.0, 0.0]);
        assert_eq!(v[3], 1.0);
    }
}

#[test]
fn tangent_parallel_to_normal_falls_back() {
    // Construct a case where T_sum and N are intentionally parallel:
    // the Gram-Schmidt T' = T - (T·N)N collapses to zero => fallback.
    // Easiest: supply normals that are NOT perpendicular to the
    // surface. Take the unit XY square but feed normals = +X. Then
    // T_sum = +X (from UVs) is parallel to N => T' = 0 => fallback.
    let mut p = unit_square_xy();
    p.normals = Some(vec![[1.0, 0.0, 0.0]; 4]);
    let t = p.compute_tangents(0).expect("present");
    for v in &t {
        // Fallback values.
        assert_vec3_close([v[0], v[1], v[2]], [1.0, 0.0, 0.0]);
        assert_eq!(v[3], 1.0);
    }
}

// ---- Multi-UV-set selection ---------------------------------------------

#[test]
fn uv_set_index_selects_correct_channel() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    // UV set 0: U=+X (T=+X). UV set 1: U=+Y (T=+Y, B=-X).
    p.uvs = vec![
        vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    ];
    let t0 = p.compute_tangents(0).expect("present");
    let t1 = p.compute_tangents(1).expect("present");
    assert_vec3_close([t0[0][0], t0[0][1], t0[0][2]], [1.0, 0.0, 0.0]);
    assert_vec3_close([t1[0][0], t1[0][1], t1[0][2]], [0.0, 1.0, 0.0]);
}

// ---- U16 vs U32 index parity --------------------------------------------

#[test]
fn u16_and_u32_indices_produce_identical_tangents() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
    let uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]];

    let mut a = Primitive::new(Topology::Triangles);
    a.positions = positions.clone();
    a.normals = normals.clone();
    a.uvs = uvs.clone();
    a.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));

    let mut b = Primitive::new(Topology::Triangles);
    b.positions = positions;
    b.normals = normals;
    b.uvs = uvs;
    b.indices = Some(Indices::U32(vec![0, 1, 2, 0, 2, 3]));

    let ta = a.compute_tangents(0).expect("present");
    let tb = b.compute_tangents(0).expect("present");
    assert_eq!(ta.len(), tb.len());
    for (x, y) in ta.iter().zip(tb.iter()) {
        for k in 0..4 {
            assert!((x[k] - y[k]).abs() < EPS, "{x:?} vs {y:?}");
        }
    }
}

// ---- Area weighting + UV-flip mixed mesh --------------------------------

#[test]
fn shared_vertex_with_consistent_uv_orientation_keeps_plus_one_handedness() {
    // Quad split into two triangles, both with non-mirrored UVs. The
    // shared vertices should still come out w=+1.
    let p = unit_square_xy();
    let t = p.compute_tangents(0).expect("present");
    for v in &t {
        assert_eq!(v[3], 1.0);
    }
}

#[test]
fn tangent_is_pure_function_of_inputs() {
    // Calling twice with the same inputs yields the same output
    // (no hidden state).
    let p = unit_square_xy();
    let a = p.compute_tangents(0).expect("present");
    let b = p.compute_tangents(0).expect("present");
    assert_eq!(a, b);
}

#[test]
fn does_not_mutate_self() {
    let p = unit_square_xy();
    let positions_before = p.positions.clone();
    let normals_before = p.normals.clone();
    let uvs_before = p.uvs.clone();
    let tangents_before = p.tangents.clone();
    let _ = p.compute_tangents(0);
    assert_eq!(p.positions, positions_before);
    assert_eq!(p.normals, normals_before);
    assert_eq!(p.uvs, uvs_before);
    assert_eq!(p.tangents, tangents_before);
}
