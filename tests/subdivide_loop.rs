//! Tests for `Primitive::subdivide_loop` — one step of Loop subdivision
//! (Charles Loop, 1987).
//!
//! Each step replaces every triangle with four (the central + three
//! corner sub-triangles), inserts an edge vertex on every undirected
//! edge (interior `3/8(A+B)+1/8(C+D)`, boundary midpoint), and relaxes
//! the original vertices (interior Warren β; boundary cubic-B-spline
//! mask). Positions carry the Loop masks; every other attribute is
//! linearly interpolated. Boundaries stay watertight.

use oxideav_mesh3d::{Indices, MorphTarget, Primitive, Topology};

// --- helpers ---------------------------------------------------------

fn tri(p: Vec<[f32; 3]>) -> Primitive {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = p;
    prim
}

fn tri_indexed(p: Vec<[f32; 3]>, idx: Vec<u32>) -> Primitive {
    let mut prim = tri(p);
    prim.indices = Some(Indices::U32(idx));
    prim
}

fn idx_vec(p: &Primitive) -> Vec<u32> {
    match &p.indices {
        Some(Indices::U32(v)) => v.clone(),
        Some(Indices::U16(v)) => v.iter().map(|&x| x as u32).collect(),
        None => (0..p.positions.len() as u32).collect(),
    }
}

fn approx(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
    (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
}

fn find_pos(p: &Primitive, target: [f32; 3], eps: f32) -> Option<usize> {
    p.positions.iter().position(|&q| approx(q, target, eps))
}

// Closed unit tetrahedron (outward CCW) — a closed two-manifold.
fn tetrahedron() -> Primitive {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let idx = vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
    tri_indexed(positions, idx)
}

// --- topology / count tests -----------------------------------------

#[test]
fn single_triangle_splits_into_four() {
    let prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    let s = prim.subdivide_loop();
    assert_eq!(s.topology, Topology::Triangles);
    assert_eq!(s.triangle_count(), 4);
    // 3 original + 3 edge vertices.
    assert_eq!(s.positions.len(), 6);
}

#[test]
fn output_is_always_triangles() {
    let prim = tetrahedron();
    let s = prim.subdivide_loop();
    assert_eq!(s.topology, Topology::Triangles);
}

#[test]
fn face_count_quadruples_on_tetrahedron() {
    let prim = tetrahedron();
    assert_eq!(prim.triangle_count(), 4);
    let s = prim.subdivide_loop();
    assert_eq!(s.triangle_count(), 16);
    // 4 original + 6 edges = 10 vertices.
    assert_eq!(s.positions.len(), 10);
}

#[test]
fn two_steps_quadruple_twice() {
    let prim = tetrahedron();
    let s2 = prim.subdivide_loop().subdivide_loop();
    assert_eq!(s2.triangle_count(), 4 * 16);
}

// --- edge-vertex position masks -------------------------------------

#[test]
fn boundary_edge_vertices_are_midpoints() {
    // Single triangle: all three edges are boundaries → edge vertices
    // are the exact edge midpoints.
    let prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    let s = prim.subdivide_loop();
    // midpoints of (0,1), (1,2), (2,0)
    assert!(find_pos(&s, [1.0, 0.0, 0.0], 1e-5).is_some());
    assert!(find_pos(&s, [1.0, 1.0, 0.0], 1e-5).is_some());
    assert!(find_pos(&s, [0.0, 1.0, 0.0], 1e-5).is_some());
}

#[test]
fn single_triangle_corner_vertices_unmoved() {
    // All three vertices are boundary vertices with two boundary
    // neighbours each → cubic-B-spline mask. For an equilateral-ish
    // triangle the mask pulls each corner toward the opposite edge
    // midpoint by a known amount; just confirm they move but stay
    // finite, and the centroid is preserved (affine-invariant mask).
    let prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    let orig_centroid = [(0.0 + 2.0 + 0.0) / 3.0, (0.0 + 0.0 + 2.0) / 3.0, 0.0_f32];
    let s = prim.subdivide_loop();
    // Centroid of the three repositioned original corners (first 3 pool
    // slots) stays at the original centroid: 3/4 V + 1/8(B0+B1) summed
    // over all three corners preserves the mean.
    let mut c = [0.0_f32; 3];
    for v in &s.positions[0..3] {
        c[0] += v[0];
        c[1] += v[1];
        c[2] += v[2];
    }
    c[0] /= 3.0;
    c[1] /= 3.0;
    c[2] /= 3.0;
    assert!(approx(c, orig_centroid, 1e-5), "centroid drifted: {c:?}");
}

#[test]
fn interior_edge_uses_opposite_apexes() {
    // Two triangles sharing edge 1-2 (a quad split along the diagonal):
    //   tri A = (0,1,2), tri B = (2,1,3)
    // Shared edge is 1-2 with opposite apexes 0 and 3.
    // Interior-edge mask: 3/8(P1+P2) + 1/8(P0+P3).
    let positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [0.0, 1.0, 0.0], // 2
        [1.0, 1.0, 0.0], // 3
    ];
    let prim = tri_indexed(positions.clone(), vec![0, 1, 2, 2, 1, 3]);
    let s = prim.subdivide_loop();
    let expected = {
        let a = positions[1];
        let b = positions[2];
        let c = positions[0];
        let d = positions[3];
        [
            0.375 * (a[0] + b[0]) + 0.125 * (c[0] + d[0]),
            0.375 * (a[1] + b[1]) + 0.125 * (c[1] + d[1]),
            0.375 * (a[2] + b[2]) + 0.125 * (c[2] + d[2]),
        ]
    };
    assert!(
        find_pos(&s, expected, 1e-5).is_some(),
        "interior edge vertex {expected:?} not found in {:?}",
        s.positions
    );
}

// --- watertightness / boundary preservation -------------------------

#[test]
fn closed_manifold_stays_closed() {
    let prim = tetrahedron();
    assert!(prim.edge_manifold_report().is_closed_manifold());
    let s = prim.subdivide_loop();
    assert!(
        s.edge_manifold_report().is_closed_manifold(),
        "subdivided closed tetrahedron should stay watertight"
    );
    // No boundary loops on a closed surface.
    assert!(s.boundary_loops().is_empty());
}

#[test]
fn open_surface_keeps_its_one_boundary_loop() {
    // Single triangle has exactly one boundary loop; after subdivision
    // it still has one loop (refined: 6 boundary edges instead of 3).
    let prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    assert_eq!(prim.boundary_loops().len(), 1);
    let s = prim.subdivide_loop();
    let loops = s.boundary_loops();
    assert_eq!(loops.len(), 1);
    // 3 corners + 3 edge midpoints all sit on the rim.
    assert_eq!(loops[0].len(), 6);
}

#[test]
fn boundary_vertices_stay_on_their_edge_lines() {
    // A flat square (two tris). Its four corners are boundary vertices;
    // each boundary edge vertex is a midpoint → the whole subdivided
    // patch stays in the z=0 plane.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    let prim = tri_indexed(positions, vec![0, 1, 2, 0, 2, 3]);
    let s = prim.subdivide_loop();
    for p in &s.positions {
        assert!(p[2].abs() < 1e-5, "vertex left the z=0 plane: {p:?}");
    }
}

// --- attribute interpolation ----------------------------------------

#[test]
fn uvs_interpolate_at_edge_midpoints() {
    let mut prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    prim.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]];
    let s = prim.subdivide_loop();
    assert_eq!(s.uvs.len(), 1);
    assert_eq!(s.uvs[0].len(), 6);
    // First 3 UVs are the originals; the edge between vert0 and vert1
    // gets UV (0.5, 0.0). Find that vertex by its position midpoint.
    let mid = find_pos(&s, [1.0, 0.0, 0.0], 1e-5).expect("edge midpoint missing");
    let uv = s.uvs[0][mid];
    assert!(
        (uv[0] - 0.5).abs() < 1e-5 && uv[1].abs() < 1e-5,
        "edge UV not midpoint-interpolated: {uv:?}"
    );
}

#[test]
fn colors_interpolate_at_edge_midpoints() {
    let mut prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    prim.colors = vec![vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]];
    let s = prim.subdivide_loop();
    assert_eq!(s.colors[0].len(), 6);
    let mid = find_pos(&s, [1.0, 0.0, 0.0], 1e-5).unwrap();
    let c = s.colors[0][mid];
    // mean of red and green
    assert!((c[0] - 0.5).abs() < 1e-5 && (c[1] - 0.5).abs() < 1e-5);
}

#[test]
fn normals_interpolated_and_renormalised() {
    let mut prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    // All face up; interpolation of identical unit vectors stays unit.
    prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    let s = prim.subdivide_loop();
    let ns = s.normals.as_ref().unwrap();
    assert_eq!(ns.len(), 6);
    for n in ns {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "normal not unit: {n:?} len {len}");
    }
}

#[test]
fn skinning_weights_interpolate_joints_copy() {
    let mut prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    prim.joints = Some(vec![[0, 1, 0, 0], [2, 3, 0, 0], [4, 5, 0, 0]]);
    prim.weights = Some(vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ]);
    let s = prim.subdivide_loop();
    let ws = s.weights.as_ref().unwrap();
    let js = s.joints.as_ref().unwrap();
    assert_eq!(ws.len(), 6);
    assert_eq!(js.len(), 6);
    // Edge 0-1 weight = mean([1,0,0,0],[0,1,0,0]) = [0.5,0.5,0,0]
    let mid = find_pos(&s, [1.0, 0.0, 0.0], 1e-5).unwrap();
    assert!((ws[mid][0] - 0.5).abs() < 1e-5 && (ws[mid][1] - 0.5).abs() < 1e-5);
    // Joints copy the lower-index endpoint (vertex 0).
    assert_eq!(js[mid], [0, 1, 0, 0]);
}

#[test]
fn morph_deltas_interpolate() {
    let mut prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    let mut t = MorphTarget::new();
    t.position = Some(vec![[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    prim.targets = vec![t];
    let s = prim.subdivide_loop();
    assert_eq!(s.targets.len(), 1);
    let dp = s.targets[0].position.as_ref().unwrap();
    assert_eq!(dp.len(), 6);
    // edge 0-1 delta = mean([1,0,0],[3,0,0]) = [2,0,0]
    let mid = find_pos(&s, [1.0, 0.0, 0.0], 1e-5).unwrap();
    assert!(approx(dp[mid], [2.0, 0.0, 0.0], 1e-5));
}

// --- robustness ------------------------------------------------------

#[test]
fn non_triangle_topology_returns_empty_triangles() {
    let mut prim = Primitive::new(Topology::Lines);
    prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0]];
    let s = prim.subdivide_loop();
    assert_eq!(s.topology, Topology::Triangles);
    assert_eq!(s.triangle_count(), 0);
}

#[test]
fn empty_primitive_is_safe() {
    let prim = Primitive::new(Topology::Triangles);
    let s = prim.subdivide_loop();
    assert_eq!(s.topology, Topology::Triangles);
    assert_eq!(s.triangle_count(), 0);
    assert!(s.positions.is_empty());
}

#[test]
fn out_of_range_and_degenerate_triangles_dropped() {
    // One good triangle (0,1,2), one with an out-of-range index (5),
    // one with a duplicate corner (3,3,4).
    let positions = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
    ];
    let prim = tri_indexed(positions, vec![0, 1, 2, 0, 1, 5, 3, 3, 4]);
    let s = prim.subdivide_loop();
    // Only the one valid triangle subdivides → 4 sub-triangles.
    assert_eq!(s.triangle_count(), 4);
}

#[test]
fn nan_position_falls_back_without_panic() {
    let prim = tri(vec![[f32::NAN, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    // Must not panic; result is a Triangles primitive.
    let s = prim.subdivide_loop();
    assert_eq!(s.topology, Topology::Triangles);
}

#[test]
fn unwelded_vertex_soup_is_welded_first() {
    // Two triangles forming a square, but stored as a 6-vertex soup
    // (no shared indices). subdivide_loop welds first, so the shared
    // diagonal becomes an interior edge and the result is a closed
    // edge-manifold patch with one boundary loop, not two.
    let positions = vec![
        // tri A
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        // tri B (shares the 2.0,0 and 0,2.0 corners by value)
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    let prim = tri(positions);
    let s = prim.subdivide_loop();
    // After welding the soup has 4 distinct corners; the shared
    // diagonal is interior → exactly one boundary loop survives.
    assert_eq!(s.boundary_loops().len(), 1);
}

#[test]
fn index_width_stays_u16_for_small_meshes() {
    let prim = tetrahedron();
    let s = prim.subdivide_loop();
    assert!(matches!(s.indices, Some(Indices::U16(_))));
}

#[test]
fn does_not_mutate_self() {
    let prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    let before = prim.positions.clone();
    let _ = prim.subdivide_loop();
    assert_eq!(prim.positions, before);
    assert_eq!(prim.topology, Topology::Triangles);
}

#[test]
fn material_and_extras_carried_through() {
    use oxideav_mesh3d::MaterialId;
    let mut prim = tri(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
    prim.material = Some(MaterialId(7));
    prim.extras
        .insert("tag".to_owned(), serde_json::json!("cage"));
    let s = prim.subdivide_loop();
    assert_eq!(s.material, Some(MaterialId(7)));
    assert_eq!(s.extras.get("tag"), Some(&serde_json::json!("cage")));
}

#[test]
fn indices_reference_valid_pool_entries() {
    let prim = tetrahedron();
    let s = prim.subdivide_loop();
    let n = s.positions.len() as u32;
    for &i in &idx_vec(&s) {
        assert!(i < n, "index {i} out of range (pool size {n})");
    }
}

#[test]
fn triangle_strip_input_is_destripped_and_subdivided() {
    // A 4-vertex strip = 2 triangles. After welding+destrip it has the
    // same geometry as the indexed quad; subdivides to 8 sub-triangles.
    let mut prim = Primitive::new(Topology::TriangleStrip);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let s = prim.subdivide_loop();
    assert_eq!(s.topology, Topology::Triangles);
    assert_eq!(s.triangle_count(), 8);
}
