//! Tests for `Primitive::boundary_loops` — chaining the loose boundary
//! edges of `Primitive::boundary_edges` end-to-end into ordered,
//! winding-consistent vertex loops.
//!
//! Each loop is one hole / crack / open rim, listed as the ordered
//! vertex-pool indices walked along the boundary in the surface's
//! winding-consistent direction, rotated to start at the loop's smallest
//! vertex index, with the loop list sorted ascending. A closed
//! two-manifold has no boundary loops.

use oxideav_mesh3d::{Indices, Primitive, Topology};

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

// A unit tetrahedron: closed two-manifold → zero boundary loops.
fn tetrahedron() -> Primitive {
    let positions = vec![
        [0.0, 0.0, 0.0], // v0
        [1.0, 0.0, 0.0], // v1
        [0.0, 1.0, 0.0], // v2
        [0.0, 0.0, 1.0], // v3
    ];
    let indices = vec![
        0, 2, 1, // bottom
        0, 1, 3, // front
        0, 3, 2, // left
        1, 2, 3, // far
    ];
    tri_indexed(positions, indices)
}

// A quad split into two triangles sharing the v1–v2 diagonal:
// v0 v1
// v2 v3   →  (0,1,2) and (1,3,2). One boundary loop of four vertices.
fn open_quad() -> Primitive {
    let positions = vec![
        [0.0, 0.0, 0.0], // v0
        [1.0, 0.0, 0.0], // v1
        [0.0, 1.0, 0.0], // v2
        [1.0, 1.0, 0.0], // v3
    ];
    tri_indexed(positions, vec![0, 1, 2, 1, 3, 2])
}

// Walk a loop's consecutive (a,b) edges, returned as canonical
// `[min, max]` pairs, sorted — for comparison against `boundary_edges`.
fn loop_edge_set(l: &[u32]) -> Vec<[u32; 2]> {
    let mut out = Vec::new();
    for i in 0..l.len() {
        let a = l[i];
        let b = l[(i + 1) % l.len()];
        out.push(if a < b { [a, b] } else { [b, a] });
    }
    out.sort_unstable();
    out
}

// =====================================================================
// Closed meshes & empties: no loops.
// =====================================================================

#[test]
fn closed_tetrahedron_has_no_loops() {
    let prim = tetrahedron();
    assert!(prim.boundary_loops().is_empty());
    assert!(prim.edge_manifold_report().is_closed_manifold());
}

#[test]
fn empty_primitive_has_no_loops() {
    let prim = Primitive::new(Topology::Triangles);
    assert!(prim.boundary_loops().is_empty());
}

#[test]
fn points_topology_has_no_loops() {
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert!(prim.boundary_loops().is_empty());
}

#[test]
fn lines_topology_has_no_loops() {
    let mut prim = Primitive::new(Topology::Lines);
    prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0]];
    assert!(prim.boundary_loops().is_empty());
}

// =====================================================================
// Single open triangle → one three-vertex loop.
// =====================================================================

#[test]
fn single_triangle_one_loop_of_three() {
    let prim = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let loops = prim.boundary_loops();
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].len(), 3);
}

#[test]
fn single_triangle_loop_is_zero_one_two() {
    // CCW (0,1,2): directed half-edges 0→1, 1→2, 2→0 chain to [0,1,2].
    let prim = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    assert_eq!(prim.boundary_loops(), vec![vec![0, 1, 2]]);
}

#[test]
fn single_triangle_loop_edges_match_boundary_edges() {
    let prim = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let loops = prim.boundary_loops();
    assert_eq!(loop_edge_set(&loops[0]), prim.boundary_edges());
}

#[test]
fn single_triangle_indexed_uses_pool_indices() {
    let mut positions = vec![[0.0; 3]; 10];
    positions[5] = [0.0, 0.0, 0.0];
    positions[7] = [1.0, 0.0, 0.0];
    positions[9] = [0.0, 1.0, 0.0];
    let prim = tri_indexed(positions, vec![5, 7, 9]);
    // Loop rotates to start at smallest vertex (5): [5, 7, 9].
    assert_eq!(prim.boundary_loops(), vec![vec![5, 7, 9]]);
}

// =====================================================================
// Two-triangle open quad → one loop of four.
// =====================================================================

#[test]
fn open_quad_one_loop_of_four() {
    let prim = open_quad();
    let loops = prim.boundary_loops();
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].len(), 4);
}

#[test]
fn open_quad_loop_starts_at_smallest_vertex() {
    let prim = open_quad();
    let loops = prim.boundary_loops();
    assert_eq!(loops[0][0], 0);
}

#[test]
fn open_quad_loop_edges_match_boundary_edges() {
    let prim = open_quad();
    let loops = prim.boundary_loops();
    assert_eq!(loops.len(), 1);
    assert_eq!(loop_edge_set(&loops[0]), prim.boundary_edges());
}

#[test]
fn open_quad_loop_visits_all_four_rim_vertices() {
    let prim = open_quad();
    let mut verts = prim.boundary_loops()[0].clone();
    verts.sort_unstable();
    assert_eq!(verts, vec![0, 1, 2, 3]);
}

// =====================================================================
// A square hole in the middle of a larger surface → two loops.
// =====================================================================

// A genuine square-hole annulus: an outer square ring of triangles with
// the centre quad missing. Outer corners 0..3, inner corners 4..7.
//
//   0 --- 1
//   | 4-5 |
//   | 7-6 |
//   3 --- 2
fn annulus_square_hole() -> Primitive {
    let positions = vec![
        [0.0, 0.0, 0.0], // 0 outer TL
        [3.0, 0.0, 0.0], // 1 outer TR
        [3.0, 3.0, 0.0], // 2 outer BR
        [0.0, 3.0, 0.0], // 3 outer BL
        [1.0, 1.0, 0.0], // 4 inner TL
        [2.0, 1.0, 0.0], // 5 inner TR
        [2.0, 2.0, 0.0], // 6 inner BR
        [1.0, 2.0, 0.0], // 7 inner BL
    ];
    // Four trapezoidal sides, each split into two triangles, CCW so the
    // outer boundary winds one way and the inner hole the other.
    let indices = vec![
        // top side: 0,1,5,4
        0, 1, 4, 1, 5, 4, //
        // right side: 1,2,6,5
        1, 2, 5, 2, 6, 5, //
        // bottom side: 2,3,7,6
        2, 3, 6, 3, 7, 6, //
        // left side: 3,0,4,7
        3, 0, 7, 0, 4, 7, //
    ];
    tri_indexed(positions, indices)
}

#[test]
fn annulus_has_two_loops() {
    let prim = annulus_square_hole();
    let loops = prim.boundary_loops();
    assert_eq!(loops.len(), 2, "outer rim + inner hole");
}

#[test]
fn annulus_loops_partition_boundary_edges() {
    let prim = annulus_square_hole();
    let loops = prim.boundary_loops();
    let mut covered: Vec<[u32; 2]> = loops.iter().flat_map(|l| loop_edge_set(l)).collect();
    covered.sort_unstable();
    let mut expected = prim.boundary_edges();
    expected.sort_unstable();
    assert_eq!(covered, expected);
}

#[test]
fn annulus_outer_loop_has_four_vertices() {
    let prim = annulus_square_hole();
    let loops = prim.boundary_loops();
    // Outer loop = corners 0,1,2,3; inner = 4,5,6,7.
    let outer: Vec<u32> = loops.iter().find(|l| l.contains(&0)).unwrap().clone();
    let mut o = outer.clone();
    o.sort_unstable();
    assert_eq!(o, vec![0, 1, 2, 3]);
}

#[test]
fn annulus_inner_loop_has_four_vertices() {
    let prim = annulus_square_hole();
    let loops = prim.boundary_loops();
    let inner: Vec<u32> = loops.iter().find(|l| l.contains(&4)).unwrap().clone();
    let mut i = inner.clone();
    i.sort_unstable();
    assert_eq!(i, vec![4, 5, 6, 7]);
}

#[test]
fn annulus_total_boundary_vertices_is_eight() {
    let prim = annulus_square_hole();
    let total: usize = prim.boundary_loops().iter().map(|l| l.len()).sum();
    assert_eq!(total, 8);
}

// =====================================================================
// Two disjoint open triangles → two independent loops.
// =====================================================================

#[test]
fn two_disjoint_triangles_two_loops() {
    let positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [0.0, 1.0, 0.0], // 2
        [5.0, 0.0, 0.0], // 3
        [6.0, 0.0, 0.0], // 4
        [5.0, 1.0, 0.0], // 5
    ];
    let prim = tri_indexed(positions, vec![0, 1, 2, 3, 4, 5]);
    let loops = prim.boundary_loops();
    assert_eq!(loops.len(), 2);
    assert_eq!(loops[0], vec![0, 1, 2]);
    assert_eq!(loops[1], vec![3, 4, 5]);
}

// =====================================================================
// Determinism & ordering.
// =====================================================================

#[test]
fn loop_list_is_sorted_ascending() {
    let positions = vec![
        [5.0, 0.0, 0.0], // 0  (far triangle indices on purpose)
        [6.0, 0.0, 0.0], // 1
        [5.0, 1.0, 0.0], // 2
        [0.0, 0.0, 0.0], // 3
        [1.0, 0.0, 0.0], // 4
        [0.0, 1.0, 0.0], // 5
    ];
    // First triangle uses high indices, second uses low — the output
    // must still come out sorted by smallest vertex.
    let prim = tri_indexed(positions, vec![0, 1, 2, 3, 4, 5]);
    let loops = prim.boundary_loops();
    assert_eq!(loops[0][0], 0);
    assert_eq!(loops[1][0], 3);
    assert!(loops[0] < loops[1]);
}

#[test]
fn result_is_deterministic_across_calls() {
    let prim = annulus_square_hole();
    let a = prim.boundary_loops();
    let b = prim.boundary_loops();
    assert_eq!(a, b);
}

#[test]
fn loop_rotation_is_seed_independent() {
    // Same triangle, but indexed so the implicit walk would start at a
    // non-minimal vertex; the loop must still rotate to its smallest.
    let mut positions = vec![[0.0; 3]; 6];
    positions[2] = [0.0, 0.0, 0.0];
    positions[4] = [1.0, 0.0, 0.0];
    positions[1] = [0.0, 1.0, 0.0];
    let prim = tri_indexed(positions, vec![2, 4, 1]);
    // Directed edges 2→4, 4→1, 1→2. Rotated to start at min (1): [1,2,4].
    assert_eq!(prim.boundary_loops(), vec![vec![1, 2, 4]]);
}

// =====================================================================
// Winding direction.
// =====================================================================

#[test]
fn loop_direction_follows_triangle_winding() {
    // CCW triangle 0,1,2 → loop walks 0→1→2 (the surface is on the left).
    let prim = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    assert_eq!(prim.boundary_loops(), vec![vec![0, 1, 2]]);
}

#[test]
fn reversed_winding_reverses_loop() {
    // CW triangle 0,2,1 → directed edges 0→2, 2→1, 1→0 → [0,2,1].
    let prim = tri_indexed(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 2, 1],
    );
    assert_eq!(prim.boundary_loops(), vec![vec![0, 2, 1]]);
}

// =====================================================================
// Robustness: degenerate / out-of-range triangles excluded whole.
// =====================================================================

#[test]
fn out_of_range_triangle_excluded() {
    // First triangle valid (0,1,2); second references vertex 9 which
    // doesn't exist → excluded whole, contributing no boundary.
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let prim = tri_indexed(positions, vec![0, 1, 2, 0, 1, 9]);
    assert_eq!(prim.boundary_loops(), vec![vec![0, 1, 2]]);
}

#[test]
fn duplicate_corner_triangle_excluded() {
    // Second triangle has a duplicate corner (0,0,1) → excluded whole.
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let prim = tri_indexed(positions, vec![0, 1, 2, 0, 0, 1]);
    assert_eq!(prim.boundary_loops(), vec![vec![0, 1, 2]]);
}

#[test]
fn collinear_triangle_still_contributes_topologically() {
    // A geometrically degenerate (collinear) triangle is still a valid
    // topological triangle by index — `boundary_loops` is index-based,
    // not position-based, matching `boundary_edges`. Its three edges are
    // boundary just like a non-degenerate triangle's.
    let prim = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    assert_eq!(prim.boundary_loops(), vec![vec![0, 1, 2]]);
}

// =====================================================================
// Strip & fan topology feed in via triangle_indices.
// =====================================================================

#[test]
fn triangle_strip_open_fan_produces_loops() {
    // A 4-vertex strip → triangles (0,1,2) and (2,1,3) (alternating
    // winding). Its boundary loops must cover the same edges as
    // `boundary_edges`.
    let mut prim = Primitive::new(Topology::TriangleStrip);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let loops = prim.boundary_loops();
    let mut covered: Vec<[u32; 2]> = loops.iter().flat_map(|l| loop_edge_set(l)).collect();
    covered.sort_unstable();
    covered.dedup();
    assert_eq!(covered, prim.boundary_edges());
}

#[test]
fn triangle_fan_produces_loops() {
    // A fan anchored at 0 over rim 1,2,3 → triangles (0,1,2) (0,2,3).
    let mut prim = Primitive::new(Topology::TriangleFan);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let loops = prim.boundary_loops();
    assert_eq!(loops.len(), 1);
    let mut covered: Vec<[u32; 2]> = loop_edge_set(&loops[0]);
    covered.sort_unstable();
    assert_eq!(covered, prim.boundary_edges());
}

// =====================================================================
// Cross-checks against the existing aggregate / extractor surface.
// =====================================================================

#[test]
fn loop_edge_count_matches_boundary_edge_count() {
    let prim = annulus_square_hole();
    let loop_edges: usize = prim.boundary_loops().iter().map(|l| l.len()).sum();
    assert_eq!(loop_edges, prim.boundary_edges().len());
    assert_eq!(loop_edges, prim.edge_manifold_report().boundary_edge_count);
}

#[test]
fn closed_manifold_loop_count_is_zero() {
    let prim = tetrahedron();
    assert!(prim.boundary_loops().is_empty());
    assert_eq!(prim.edge_manifold_report().boundary_edge_count, 0);
}

#[test]
fn every_loop_vertex_is_a_boundary_endpoint() {
    let prim = annulus_square_hole();
    let edges = prim.boundary_edges();
    let mut endpoints: Vec<u32> = edges.iter().flat_map(|e| [e[0], e[1]]).collect();
    endpoints.sort_unstable();
    endpoints.dedup();
    for l in prim.boundary_loops() {
        for v in l {
            assert!(endpoints.binary_search(&v).is_ok());
        }
    }
}

#[test]
fn pure_does_not_mutate_primitive() {
    let prim = open_quad();
    let before = prim.clone();
    let _ = prim.boundary_loops();
    assert_eq!(prim.positions, before.positions);
    assert_eq!(prim.indices, before.indices);
}
