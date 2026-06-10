//! Tests for `Primitive::boundary_edges` — the detection-only
//! extractor counterpart to `EdgeManifoldReport::boundary_edge_count`.
//!
//! A boundary edge is an undirected triangle edge used by exactly one
//! triangle: a hole, a crack, or the open rim of a non-closed surface.
//! A closed two-manifold mesh has none. The extractor mirrors the edge
//! bucketing of `edge_manifold_report` (same out-of-range /
//! duplicate-corner exclusion, same `triangle_indices` topology feed)
//! and returns the canonical `[min, max]` vertex-index pairs sorted
//! ascending so the output is deterministic.

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

// A unit tetrahedron: closed two-manifold, 4 triangles, 4 vertices,
// 6 edges all shared by exactly two faces → zero boundary edges.
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
// v2 v3   →  (0,1,2) and (1,3,2). The shared diagonal (1,2) is interior;
// the four outer edges (0,1) (0,2) (1,3) (2,3) are boundary.
fn open_quad() -> Primitive {
    let positions = vec![
        [0.0, 0.0, 0.0], // v0
        [1.0, 0.0, 0.0], // v1
        [0.0, 1.0, 0.0], // v2
        [1.0, 1.0, 0.0], // v3
    ];
    tri_indexed(positions, vec![0, 1, 2, 1, 3, 2])
}

// --- closed mesh: no boundary ---------------------------------------

#[test]
fn closed_tetrahedron_has_no_boundary_edges() {
    let prim = tetrahedron();
    assert!(prim.boundary_edges().is_empty());
    // Cross-check with the aggregate report.
    assert!(prim.edge_manifold_report().is_closed_manifold());
}

#[test]
fn empty_primitive_returns_empty() {
    let prim = Primitive::new(Topology::Triangles);
    assert!(prim.boundary_edges().is_empty());
}

// --- single open triangle: three boundary edges ---------------------

#[test]
fn single_triangle_all_three_edges_are_boundary() {
    let prim = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    // Non-indexed → implicit order 0,1,2. Edges (0,1) (1,2) (0,2).
    assert_eq!(prim.boundary_edges(), vec![[0, 1], [0, 2], [1, 2]]);
}

#[test]
fn single_triangle_indexed_uses_pool_indices() {
    // The triangle references pool slots 5,7,9 (positions padded out).
    let mut positions = vec![[0.0; 3]; 10];
    positions[5] = [0.0, 0.0, 0.0];
    positions[7] = [1.0, 0.0, 0.0];
    positions[9] = [0.0, 1.0, 0.0];
    let prim = tri_indexed(positions, vec![5, 7, 9]);
    assert_eq!(prim.boundary_edges(), vec![[5, 7], [5, 9], [7, 9]]);
}

// --- two-triangle open quad -----------------------------------------

#[test]
fn open_quad_returns_only_the_four_rim_edges() {
    let prim = open_quad();
    // Diagonal (1,2) is interior; rim = (0,1) (0,2) (1,3) (2,3).
    assert_eq!(prim.boundary_edges(), vec![[0, 1], [0, 2], [1, 3], [2, 3]]);
}

#[test]
fn open_quad_count_matches_report() {
    let prim = open_quad();
    let report = prim.edge_manifold_report();
    assert_eq!(prim.boundary_edges().len(), report.boundary_edge_count);
    assert_eq!(report.boundary_edge_count, 4);
    assert_eq!(report.manifold_interior_edge_count, 1);
}

// --- output ordering & canonical form -------------------------------

#[test]
fn output_is_sorted_ascending() {
    let prim = open_quad();
    let edges = prim.boundary_edges();
    let mut sorted = edges.clone();
    sorted.sort_unstable();
    assert_eq!(edges, sorted, "output must be deterministically sorted");
}

#[test]
fn each_edge_is_min_max_ordered() {
    let prim = open_quad();
    for [a, b] in prim.boundary_edges() {
        assert!(a < b, "edge [{a}, {b}] must be ascending");
    }
}

#[test]
fn winding_direction_does_not_change_canonical_edge() {
    // Same single triangle but with reversed winding (0,2,1). The
    // undirected boundary edges are identical.
    let fwd = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let rev = tri_indexed(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 2, 1],
    );
    assert_eq!(fwd.boundary_edges(), rev.boundary_edges());
}

// --- closed mesh with one hole --------------------------------------

#[test]
fn tetrahedron_missing_one_face_exposes_that_faces_rim() {
    // Drop the "far" face (1,2,3) from the tetrahedron. Its three edges
    // (1,2) (1,3) (2,3) were each shared by two faces; removing one use
    // leaves them at count 1 → boundary. All other edges stay interior.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let prim = tri_indexed(
        positions,
        vec![
            0, 2, 1, // bottom
            0, 1, 3, // front
            0, 3, 2, // left
        ], // far face omitted
    );
    assert_eq!(prim.boundary_edges(), vec![[1, 2], [1, 3], [2, 3]]);
}

// --- non-manifold edges are NOT boundary ----------------------------

#[test]
fn non_manifold_edge_used_thrice_is_not_boundary() {
    // Three triangles all sharing edge (0,1): a "book spine". The shared
    // edge has use count 3 (non-manifold), so it is excluded. The six
    // outer edges each appear once → boundary.
    let positions = vec![
        [0.0, 0.0, 0.0],  // v0
        [1.0, 0.0, 0.0],  // v1
        [0.0, 1.0, 0.0],  // v2
        [0.0, 0.0, 1.0],  // v3
        [0.0, -1.0, 0.0], // v4
    ];
    let prim = tri_indexed(positions, vec![0, 1, 2, 0, 1, 3, 0, 1, 4]);
    let edges = prim.boundary_edges();
    // (0,1) must NOT appear (use count 3).
    assert!(!edges.contains(&[0, 1]));
    // The six fin edges appear once each.
    assert_eq!(edges, vec![[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]]);
    let report = prim.edge_manifold_report();
    assert_eq!(report.non_manifold_edge_count, 1);
    assert_eq!(edges.len(), report.boundary_edge_count);
}

// --- robustness: skipped triangles ----------------------------------

#[test]
fn out_of_range_triangle_is_excluded_whole() {
    // First triangle valid (boundary), second references slot 99.
    let prim = tri_indexed(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 1, 2, 0, 1, 99],
    );
    // Only the first triangle's three edges feed in.
    assert_eq!(prim.boundary_edges(), vec![[0, 1], [0, 2], [1, 2]]);
}

#[test]
fn duplicate_corner_triangle_is_excluded_whole() {
    // Triangle (0,1,1) has a zero-length edge → excluded. A second valid
    // triangle (0,1,2) provides the boundary set.
    let prim = tri_indexed(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 1, 1, 0, 1, 2],
    );
    assert_eq!(prim.boundary_edges(), vec![[0, 1], [0, 2], [1, 2]]);
}

#[test]
fn excluded_neighbour_does_not_close_a_seam() {
    // Two triangles share edge (0,1). One of them additionally has a
    // duplicate corner making it degenerate-by-index, so it is dropped
    // whole — the surviving triangle's (0,1) reverts to a boundary edge
    // rather than being (wrongly) counted as interior.
    let prim = tri_indexed(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![
            0, 1, 2, // valid
            0, 1, 1, // degenerate-by-index → dropped
        ],
    );
    // (0,1) is a boundary edge of the single surviving triangle.
    assert!(prim.boundary_edges().contains(&[0, 1]));
    assert_eq!(prim.boundary_edges(), vec![[0, 1], [0, 2], [1, 2]]);
}

// --- topology gating -------------------------------------------------

#[test]
fn lines_topology_returns_empty() {
    let mut prim = Primitive::new(Topology::Lines);
    prim.positions = vec![[0.0; 3], [1.0; 3], [2.0; 3], [3.0; 3]];
    assert!(prim.boundary_edges().is_empty());
}

#[test]
fn points_topology_returns_empty() {
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![[0.0; 3], [1.0; 3], [2.0; 3]];
    assert!(prim.boundary_edges().is_empty());
}

#[test]
fn triangle_strip_feeds_in_via_triangle_indices() {
    // A 4-vertex strip = 2 triangles: (0,1,2) and (2,1,3) (alternating
    // winding). They share edge (1,2); the rim has four boundary edges.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let mut prim = Primitive::new(Topology::TriangleStrip);
    prim.positions = positions;
    prim.indices = Some(Indices::U32(vec![0, 1, 2, 3]));
    // Shared diagonal is (1,2); boundary rim = (0,1) (0,2) (1,3) (2,3).
    assert_eq!(prim.boundary_edges(), vec![[0, 1], [0, 2], [1, 3], [2, 3]]);
}

#[test]
fn triangle_fan_feeds_in_via_triangle_indices() {
    // A 4-vertex fan around v0: triangles (0,1,2) and (0,2,3). They share
    // edge (0,2); boundary = (0,1) (1,2) (0,3) (2,3).
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let mut prim = Primitive::new(Topology::TriangleFan);
    prim.positions = positions;
    prim.indices = Some(Indices::U32(vec![0, 1, 2, 3]));
    assert_eq!(prim.boundary_edges(), vec![[0, 1], [0, 3], [1, 2], [2, 3]]);
}

// --- index-width independence ---------------------------------------

#[test]
fn u16_and_u32_indices_agree() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut u16p = tri(positions.clone());
    u16p.indices = Some(Indices::U16(vec![0, 1, 2]));
    let u32p = tri_indexed(positions, vec![0, 1, 2]);
    assert_eq!(u16p.boundary_edges(), u32p.boundary_edges());
}

// --- purity ----------------------------------------------------------

#[test]
fn does_not_mutate_self() {
    let prim = open_quad();
    let positions_before = prim.positions.clone();
    // Two calls on the same &self must agree (pure, idempotent).
    let first = prim.boundary_edges();
    let second = prim.boundary_edges();
    assert_eq!(first, second);
    assert_eq!(prim.positions, positions_before);
}

// --- weld interaction (documented caveat) ---------------------------

#[test]
fn positionally_coincident_unwelded_seam_reads_as_two_boundaries() {
    // Two triangles meeting along a seam but with DISTINCT indices at
    // the coincident corners (an unwelded vertex soup). Topology is by
    // index, so the seam reads as two boundary edges until welded.
    let positions = vec![
        // triangle A
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        // triangle B — corners 3,4 coincide with A's 1,2 positionally
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let prim = tri_indexed(positions, vec![0, 1, 2, 3, 4, 5]);
    // Each triangle contributes three boundary edges; nothing is shared
    // by index → six boundary edges.
    assert_eq!(prim.boundary_edges().len(), 6);
    // After welding the two soups into a shared pool, the seam edge
    // becomes interior and the boundary drops.
    let welded = prim.weld_vertices();
    assert!(welded.boundary_edges().len() < 6);
}
