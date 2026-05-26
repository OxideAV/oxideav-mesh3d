//! Tests for mesh-validity invariants:
//!
//! * `Primitive::degenerate_triangles` — detection of zero-area /
//!   collinear / coincident triangles.
//! * `Primitive::edge_manifold_report` + `EdgeManifoldReport` —
//!   classification of undirected triangle edges into boundary
//!   (used once), manifold-interior (used twice), and non-manifold
//!   (used three or more times), plus the `is_closed_manifold` shortcut.
//!
//! These pin the geometric/topological invariants the STL spec's
//! vertex-to-vertex rule demands and that a renderer needs to compute
//! a smooth normal at every vertex. They are detection-only — repair
//! / welding / pruning are handled by the existing
//! `weld_vertices` / `to_triangle_list` helpers.

use oxideav_mesh3d::{EdgeManifoldReport, Indices, Primitive, Topology};

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

// A unit tetrahedron (closed two-manifold, 4 triangles, 4 vertices,
// 6 edges all shared by exactly two faces). Outward-facing winding.
fn tetrahedron() -> Primitive {
    let positions = vec![
        [0.0, 0.0, 0.0], // v0
        [1.0, 0.0, 0.0], // v1
        [0.0, 1.0, 0.0], // v2
        [0.0, 0.0, 1.0], // v3
    ];
    // 4 outward-facing triangles. Each (i, j, k) is CCW from outside.
    let indices = vec![
        0, 2, 1, // bottom (v0, v2, v1) — y=0 face viewed from -z (out)
        0, 1, 3, // front (v0, v1, v3)
        0, 3, 2, // left  (v0, v3, v2)
        1, 2, 3, // far/top
    ];
    tri_indexed(positions, indices)
}

// --- degenerate_triangles --------------------------------------------

#[test]
fn degenerate_empty_primitive_returns_empty() {
    let prim = Primitive::new(Topology::Triangles);
    assert!(prim.degenerate_triangles().is_empty());
}

#[test]
fn degenerate_one_well_formed_triangle_returns_empty() {
    let prim = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    assert_eq!(prim.degenerate_triangles(), Vec::<usize>::new());
}

#[test]
fn degenerate_collinear_triangle_detected() {
    // Three points on the x-axis.
    let prim = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    assert_eq!(prim.degenerate_triangles(), vec![0]);
}

#[test]
fn degenerate_coincident_two_corners_detected() {
    // Two corners share a position.
    let prim = tri(vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    assert_eq!(prim.degenerate_triangles(), vec![0]);
}

#[test]
fn degenerate_all_three_coincident_detected() {
    let prim = tri(vec![[1.5, 2.5, 3.5]; 3]);
    assert_eq!(prim.degenerate_triangles(), vec![0]);
}

#[test]
fn degenerate_mixed_returns_only_bad_indices() {
    // Triangle 0: valid.       Triangle 1: collinear.    Triangle 2: valid.
    let prim = tri(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]);
    assert_eq!(prim.degenerate_triangles(), vec![1]);
}

#[test]
fn degenerate_out_of_range_index_reported() {
    // Three positions but the third index points at slot 9.
    let prim = tri_indexed(
        vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 1, 9],
    );
    assert_eq!(prim.degenerate_triangles(), vec![0]);
}

#[test]
fn degenerate_nan_face_reported() {
    let prim = tri(vec![[0.0, 0.0, 0.0], [f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    assert_eq!(prim.degenerate_triangles(), vec![0]);
}

#[test]
fn degenerate_lines_topology_returns_empty() {
    let mut prim = Primitive::new(Topology::Lines);
    prim.positions = vec![[0.0; 3], [1.0; 3], [2.0; 3], [3.0; 3]];
    assert!(prim.degenerate_triangles().is_empty());
}

#[test]
fn degenerate_points_topology_returns_empty() {
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![[0.0; 3]; 4];
    assert!(prim.degenerate_triangles().is_empty());
}

#[test]
fn degenerate_strip_alternating_winding_does_not_affect_detection() {
    // Strip on a single line — every triangle is collinear and
    // degenerate, regardless of strip winding.
    let mut prim = Primitive::new(Topology::TriangleStrip);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ];
    // 2 triangles from 4 strip verts.
    assert_eq!(prim.degenerate_triangles().len(), 2);
}

#[test]
fn degenerate_fan_one_corner_collapses() {
    // Fan with anchor v0; the second strip-corner is coincident with anchor.
    let mut prim = Primitive::new(Topology::TriangleFan);
    prim.positions = vec![
        [0.0, 0.0, 0.0], // anchor
        [0.0, 0.0, 0.0], // coincident with anchor → tri 0 degenerate
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], // tri 1 well-formed
    ];
    let bad = prim.degenerate_triangles();
    assert!(bad.contains(&0), "first fan triangle should be degenerate");
    assert!(!bad.contains(&1), "second fan triangle should be valid");
}

#[test]
fn degenerate_indexed_collinear_detected() {
    // Vertex pool has a fourth point that creates a collinear triangle.
    let prim = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        vec![
            0, 1, 2, // valid
            0, 1, 3, // collinear on x-axis
        ],
    );
    assert_eq!(prim.degenerate_triangles(), vec![1]);
}

#[test]
fn degenerate_zero_area_via_two_coincident_indices() {
    // Index buffer references the same vertex twice — produces a
    // zero-length edge, hence a zero cross product.
    let prim = tri_indexed(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 1, 1],
    );
    assert_eq!(prim.degenerate_triangles(), vec![0]);
}

#[test]
fn degenerate_tetrahedron_clean() {
    assert!(tetrahedron().degenerate_triangles().is_empty());
}

#[test]
fn degenerate_almost_but_not_quite_collinear_not_reported() {
    // Tiny offset — cross product is non-zero (just very small), so
    // the triangle is reported as valid (no epsilon thresholding).
    let prim = tri(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 1.0e-30_f32, 0.0],
    ]);
    assert!(prim.degenerate_triangles().is_empty());
}

// --- edge_manifold_report --------------------------------------------

#[test]
fn manifold_empty_primitive_all_zero() {
    let r = Primitive::new(Topology::Triangles).edge_manifold_report();
    assert_eq!(r, EdgeManifoldReport::default());
    assert_eq!(r.total_edge_count, 0);
    assert_eq!(r.boundary_edge_count, 0);
    assert_eq!(r.manifold_interior_edge_count, 0);
    assert_eq!(r.non_manifold_edge_count, 0);
    assert_eq!(r.max_edge_use, 0);
    assert!(!r.is_closed_manifold(), "empty primitive isn't closed");
}

#[test]
fn manifold_single_triangle_has_three_boundary_edges() {
    let prim = tri(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let r = prim.edge_manifold_report();
    assert_eq!(r.total_edge_count, 3);
    assert_eq!(r.boundary_edge_count, 3);
    assert_eq!(r.manifold_interior_edge_count, 0);
    assert_eq!(r.non_manifold_edge_count, 0);
    assert_eq!(r.max_edge_use, 1);
    assert!(!r.is_closed_manifold());
}

#[test]
fn manifold_two_triangles_sharing_one_edge_is_open_quad() {
    // A quad split into two triangles. Shared diagonal edge is used
    // by two faces → interior; the four outer edges are boundary.
    let prim = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![0, 1, 2, 0, 2, 3],
    );
    let r = prim.edge_manifold_report();
    assert_eq!(r.total_edge_count, 5);
    assert_eq!(r.boundary_edge_count, 4);
    assert_eq!(r.manifold_interior_edge_count, 1);
    assert_eq!(r.non_manifold_edge_count, 0);
    assert_eq!(r.max_edge_use, 2);
    assert!(!r.is_closed_manifold(), "open surface isn't closed");
}

#[test]
fn manifold_tetrahedron_is_closed_two_manifold() {
    let r = tetrahedron().edge_manifold_report();
    // 4 triangles, 4 vertices, 6 edges, every edge shared by exactly 2 faces.
    assert_eq!(r.total_edge_count, 6, "tetrahedron has 6 edges");
    assert_eq!(r.boundary_edge_count, 0);
    assert_eq!(r.manifold_interior_edge_count, 6);
    assert_eq!(r.non_manifold_edge_count, 0);
    assert_eq!(r.max_edge_use, 2);
    assert!(r.is_closed_manifold(), "tetrahedron is closed manifold");
}

#[test]
fn manifold_three_triangles_sharing_one_edge_is_non_manifold() {
    // "Book-spine" / fan-of-three around a single shared edge (v0-v1).
    // Each of v2, v3, v4 sits opposite the spine.
    let prim = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],  // v0
            [1.0, 0.0, 0.0],  // v1
            [0.5, 1.0, 0.0],  // v2 — page 1
            [0.5, -1.0, 0.0], // v3 — page 2
            [0.5, 0.0, 1.0],  // v4 — page 3 (out of plane)
        ],
        vec![0, 1, 2, 0, 1, 3, 0, 1, 4],
    );
    let r = prim.edge_manifold_report();
    // The spine edge (0,1) is used by all 3 triangles; the 6 outer
    // edges are used once each.
    assert_eq!(r.total_edge_count, 7);
    assert_eq!(r.boundary_edge_count, 6);
    assert_eq!(r.manifold_interior_edge_count, 0);
    assert_eq!(r.non_manifold_edge_count, 1);
    assert_eq!(r.max_edge_use, 3);
    assert!(!r.is_closed_manifold());
}

#[test]
fn manifold_strip_topology_shares_edges() {
    // Triangle strip on a flat ribbon: 4 vertices → 2 triangles
    // sharing the interior diagonal.
    let mut prim = Primitive::new(Topology::TriangleStrip);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let r = prim.edge_manifold_report();
    assert_eq!(r.total_edge_count, 5);
    assert_eq!(r.manifold_interior_edge_count, 1);
    assert_eq!(r.boundary_edge_count, 4);
    assert_eq!(r.max_edge_use, 2);
    assert!(!r.is_closed_manifold());
}

#[test]
fn manifold_lines_topology_yields_empty_report() {
    let mut prim = Primitive::new(Topology::Lines);
    prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
    let r = prim.edge_manifold_report();
    assert_eq!(r, EdgeManifoldReport::default());
    assert!(!r.is_closed_manifold());
}

#[test]
fn manifold_points_topology_yields_empty_report() {
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![[0.0; 3]; 4];
    let r = prim.edge_manifold_report();
    assert_eq!(r, EdgeManifoldReport::default());
}

#[test]
fn manifold_degenerate_by_index_triangle_excluded() {
    // Two valid triangles plus one with a duplicate corner index.
    // The duplicate-corner triangle is excluded entirely so its
    // bogus edges don't pollute the count.
    let prim = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![
            0, 1, 2, 0, 2, 3, 0, 1, 1, // degenerate-by-index: skipped
        ],
    );
    let r = prim.edge_manifold_report();
    // Same five-edge quad-split report as without the bogus row.
    assert_eq!(r.total_edge_count, 5);
    assert_eq!(r.manifold_interior_edge_count, 1);
    assert_eq!(r.boundary_edge_count, 4);
}

#[test]
fn manifold_out_of_range_triangle_excluded() {
    // Second triangle references an out-of-range vertex; the whole
    // triangle is excluded so the report describes only the valid one.
    let prim = tri_indexed(
        vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 1, 2, 0, 1, 99],
    );
    let r = prim.edge_manifold_report();
    assert_eq!(r.total_edge_count, 3);
    assert_eq!(r.boundary_edge_count, 3);
    assert_eq!(r.manifold_interior_edge_count, 0);
}

#[test]
fn manifold_winding_direction_does_not_affect_count() {
    // Two triangles sharing edge (1,2), but the second triangle uses
    // the edge in the opposite winding direction (2,1) instead of
    // (1,2). Undirected lookup should still find the shared edge.
    let prim = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![
            0, 1, 2, // CCW: edges (0,1) (1,2) (2,0)
            3, 2, 1, // CCW: edges (3,2) (2,1) (1,3)  — shares (1,2)
        ],
    );
    let r = prim.edge_manifold_report();
    assert_eq!(
        r.manifold_interior_edge_count, 1,
        "(1,2) is shared regardless of winding"
    );
    assert_eq!(r.boundary_edge_count, 4);
    assert_eq!(r.total_edge_count, 5);
}

#[test]
fn manifold_index_comparison_by_id_not_position() {
    // Two triangles with **positionally coincident but indexed
    // distinctly** vertices: indices 0..2 form one triangle, indices
    // 3..5 form a positional duplicate but with separate vertex IDs.
    // Topology by index → six boundary edges, no sharing.
    let prim = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![0, 1, 2, 3, 4, 5],
    );
    let r = prim.edge_manifold_report();
    assert_eq!(r.total_edge_count, 6);
    assert_eq!(r.boundary_edge_count, 6);
    assert_eq!(r.manifold_interior_edge_count, 0);
    // weld first → both positional duplicates collapse to one pool
    // entry each, so both triangles point at the same {0, 1, 2} index
    // set. The two faces now share all three edges → 3 interior edges,
    // 0 boundary. (Strictly this is also non-manifold-ish — two faces
    // back-to-back at every edge — but it lands in the "use count 2"
    // bucket, which is the documented bucket boundary.)
    let welded = prim.weld_vertices();
    let r2 = welded.edge_manifold_report();
    assert_eq!(
        r2.total_edge_count, 3,
        "after weld the duplicate corners merge"
    );
    assert_eq!(r2.boundary_edge_count, 0);
    assert_eq!(r2.manifold_interior_edge_count, 3);
}

#[test]
fn manifold_max_edge_use_reports_worst_case() {
    // Four triangles all sharing the (0,1) spine.
    let prim = tri_indexed(
        vec![
            [0.0, 0.0, 0.0], // v0
            [1.0, 0.0, 0.0], // v1
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 0.0, 1.0],
            [0.5, 0.0, -1.0],
        ],
        vec![0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5],
    );
    let r = prim.edge_manifold_report();
    assert_eq!(r.max_edge_use, 4, "spine (0,1) used by all 4 faces");
    assert_eq!(r.non_manifold_edge_count, 1);
}

#[test]
fn manifold_report_default_is_all_zero() {
    let r = EdgeManifoldReport::default();
    assert_eq!(r.total_edge_count, 0);
    assert_eq!(r.boundary_edge_count, 0);
    assert_eq!(r.manifold_interior_edge_count, 0);
    assert_eq!(r.non_manifold_edge_count, 0);
    assert_eq!(r.max_edge_use, 0);
    assert!(!r.is_closed_manifold(), "default is not closed");
}

#[test]
fn manifold_report_is_copy() {
    fn takes_copy<T: Copy>(_: T) {}
    let r = tetrahedron().edge_manifold_report();
    takes_copy(r);
    let _r2 = r; // copy, not move
    assert!(r.is_closed_manifold());
}

#[test]
fn manifold_report_sum_invariant() {
    // Across any primitive: boundary + interior + non_manifold == total.
    let prim = tetrahedron();
    let r = prim.edge_manifold_report();
    assert_eq!(
        r.boundary_edge_count + r.manifold_interior_edge_count + r.non_manifold_edge_count,
        r.total_edge_count,
        "bucket sums must equal total"
    );
}

#[test]
fn manifold_strip_three_triangles_two_interior_edges() {
    // 5-vertex triangle strip → 3 triangles.
    let mut prim = Primitive::new(Topology::TriangleStrip);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    let r = prim.edge_manifold_report();
    // 3 triangles × 3 edges = 9 edge-incidences. Two interior seams,
    // so unique edges = 9 - 2 = 7.
    assert_eq!(r.total_edge_count, 7);
    assert_eq!(r.manifold_interior_edge_count, 2);
    assert_eq!(r.boundary_edge_count, 5);
    assert_eq!(r.non_manifold_edge_count, 0);
    assert_eq!(r.max_edge_use, 2);
    assert!(!r.is_closed_manifold());
}

#[test]
fn manifold_after_weld_open_quad_remains_open() {
    // Welding should not change topology of an already-shared mesh.
    let prim = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![0, 1, 2, 0, 2, 3],
    );
    let before = prim.edge_manifold_report();
    let after = prim.weld_vertices().edge_manifold_report();
    assert_eq!(before, after);
}
