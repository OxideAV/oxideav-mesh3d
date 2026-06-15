//! Tests for `Primitive::orient_consistent` — winding-consistency
//! flood-fill across the face-dual adjacency graph.
//!
//! Two edge-adjacent triangles are *consistently* wound iff they
//! traverse their shared undirected edge in opposite directions (glTF
//! 2.0 §3.7.2.1 CCW front-face convention: each interior edge of a
//! coherently-oriented manifold is crossed once each way). Every
//! expected value below comes from hand-tracing which corners each
//! triangle lists and which shared edges agree or disagree. The seed of
//! each edge-connected component is the lowest-indexed valid triangle
//! and keeps its input winding; the rest are brought into agreement.

use oxideav_mesh3d::{Indices, OrientationReport, Primitive, Topology};

// ── Fixtures ───────────────────────────────────────────────────────

/// Two triangles of a unit quad, ALREADY consistently wound. Triangle
/// 0 = (0,1,2) walks the diagonal 1→2; triangle 1 = (2,1,3) walks it
/// 2→1 (opposite) — coherent, no flip needed.
fn quad_consistent() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 2, 1, 3]));
    p
}

/// Same quad but the second triangle is wound the WRONG way:
/// (1,2,3) walks the shared diagonal 1→2 — the SAME direction as
/// triangle 0's 1→2, so it disagrees and must flip to (1,3,2).
fn quad_inconsistent() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 1, 2, 3]));
    p
}

/// A closed tetrahedron, all four faces wound CCW-from-outside (a
/// classic outward-consistent solid). Vertices 0..3.
fn tetra_consistent() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    // Outward winding: base then three sides.
    p.indices = Some(Indices::U32(vec![
        0, 2, 1, // base (z=0), CCW seen from -z (outside)
        0, 1, 3, // side
        1, 2, 3, // side
        2, 0, 3, // side
    ]));
    p
}

// ── Tests ──────────────────────────────────────────────────────────

#[test]
fn already_consistent_no_flips() {
    let (faces, report) = quad_consistent().orient_consistent();
    assert_eq!(report.flipped_count, 0);
    assert_eq!(report.component_count, 1);
    assert!(!report.non_orientable);
    // Output is the input verbatim.
    assert_eq!(faces, vec![[0, 1, 2], [2, 1, 3]]);
}

#[test]
fn inconsistent_neighbour_flipped() {
    let (faces, report) = quad_inconsistent().orient_consistent();
    assert_eq!(report.flipped_count, 1);
    assert_eq!(report.component_count, 1);
    assert!(!report.non_orientable);
    // Seed (triangle 0) untouched; neighbour flipped [1,2,3] -> [1,3,2].
    assert_eq!(faces[0], [0, 1, 2]);
    assert_eq!(faces[1], [1, 3, 2]);
    // After the flip the shared diagonal 1↔2 is now traversed in
    // opposite directions: triangle 0 walks 1→2, triangle 1 walks 2→1.
}

#[test]
fn doc_example_matches() {
    // Mirrors the rustdoc example on `orient_consistent`.
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    prim.indices = Some(Indices::U32(vec![0, 1, 2, 1, 2, 3]));
    let (faces, report) = prim.orient_consistent();
    assert_eq!(report.flipped_count, 1);
    assert_eq!(report.component_count, 1);
    assert!(!report.non_orientable);
    assert_eq!(faces[1], [1, 3, 2]);
}

#[test]
fn tetra_already_consistent() {
    let (faces, report) = tetra_consistent().orient_consistent();
    assert_eq!(report.flipped_count, 0);
    assert_eq!(report.component_count, 1);
    assert!(!report.non_orientable);
    // Verbatim.
    assert_eq!(faces, vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]);
}

#[test]
fn tetra_one_bad_face_repaired() {
    let mut p = tetra_consistent();
    // Corrupt the last face's winding: (2,0,3) -> (0,2,3).
    p.indices = Some(Indices::U32(vec![
        0, 2, 1, //
        0, 1, 3, //
        1, 2, 3, //
        0, 2, 3, // reversed
    ]));
    let (faces, report) = p.orient_consistent();
    // Exactly one face needed flipping; the result equals the known
    // all-consistent tetra (the repaired face matches face 3 above).
    assert_eq!(report.flipped_count, 1);
    assert_eq!(report.component_count, 1);
    assert!(!report.non_orientable);
    // The flipped face [0,2,3] -> [0,3,2], which shares each edge with a
    // neighbour the opposite way. Verify by re-running: now stable.
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = p.positions.clone();
    let mut flat = Vec::new();
    for f in &faces {
        flat.extend_from_slice(f);
    }
    p2.indices = Some(Indices::U32(flat));
    let (_, report2) = p2.orient_consistent();
    assert_eq!(
        report2.flipped_count, 0,
        "second pass must be a fixed point"
    );
}

#[test]
fn signed_volume_sign_consistent_after_orient() {
    // A consistently-oriented closed solid has a well-defined signed
    // volume sign; flipping one face breaks it, and orient_consistent
    // restores a coherent (single-sign-contributing) winding so the
    // magnitude matches the clean tetra.
    let clean = tetra_consistent();
    let clean_vol = clean.volume();
    assert!(clean_vol > 0.0);

    let mut bad = tetra_consistent();
    bad.indices = Some(Indices::U32(vec![0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 2, 3]));
    let (faces, _) = bad.orient_consistent();
    let mut fixed = Primitive::new(Topology::Triangles);
    fixed.positions = bad.positions.clone();
    let mut flat = Vec::new();
    for f in &faces {
        flat.extend_from_slice(f);
    }
    fixed.indices = Some(Indices::U32(flat));
    // |volume| is winding-magnitude invariant only when coherent.
    assert!((fixed.volume() - clean_vol).abs() < 1e-9);
}

#[test]
fn two_components_each_seeded_independently() {
    // Two disjoint quads (no shared vertices). Component 0 already
    // consistent; component 1 has a bad face. Each is seeded from its
    // own lowest-index triangle.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // quad A
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        // quad B (offset)
        [5.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [5.0, 1.0, 0.0],
        [6.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![
        0, 1, 2, 2, 1, 3, // A: consistent
        4, 5, 6, 5, 6, 7, // B: face 3 wrong (5→6 same dir as 5→6)
    ]));
    let (faces, report) = p.orient_consistent();
    assert_eq!(report.component_count, 2);
    assert_eq!(report.flipped_count, 1);
    assert!(!report.non_orientable);
    // A untouched.
    assert_eq!(faces[0], [0, 1, 2]);
    assert_eq!(faces[1], [2, 1, 3]);
    // B's seed untouched, its neighbour flipped.
    assert_eq!(faces[2], [4, 5, 6]);
    assert_eq!(faces[3], [5, 7, 6]);
}

#[test]
fn non_triangle_topology_empty() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0; 3], [1.0, 0.0, 0.0]];
    let (faces, report) = p.orient_consistent();
    assert!(faces.is_empty());
    assert_eq!(report, OrientationReport::default());
}

#[test]
fn empty_primitive_default_report() {
    let p = Primitive::new(Topology::Triangles);
    let (faces, report) = p.orient_consistent();
    assert!(faces.is_empty());
    assert_eq!(report, OrientationReport::default());
}

#[test]
fn invalid_triangle_kept_verbatim() {
    // One good triangle + one out-of-range triangle. The bad one keeps
    // its slot untouched and never participates.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 99]));
    let (faces, report) = p.orient_consistent();
    // Only the valid triangle forms a (single-face) component.
    assert_eq!(report.component_count, 1);
    assert_eq!(report.flipped_count, 0);
    assert!(!report.non_orientable);
    // Bad triangle slot is verbatim.
    assert_eq!(faces[0], [0, 1, 2]);
    assert_eq!(faces[1], [0, 1, 99]);
}

#[test]
fn boundary_only_single_face_no_constraint() {
    // A lone triangle has only boundary edges — nothing to agree with.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let (faces, report) = p.orient_consistent();
    assert_eq!(report.flipped_count, 0);
    assert_eq!(report.component_count, 1);
    assert!(!report.non_orientable);
    assert_eq!(faces, vec![[0, 1, 2]]);
}

#[test]
fn non_manifold_edge_unconstrained() {
    // Three triangles fanning a shared edge (0,1) — a non-manifold
    // edge (3 users). It carries NO orientation constraint, so the
    // three faces are not linked across it; each is its own component
    // unless they share other clean edges. Here they share only the
    // non-manifold edge, so 3 components, no flips.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 3, 0, 1, 4]));
    let (faces, report) = p.orient_consistent();
    assert_eq!(report.component_count, 3);
    assert_eq!(report.flipped_count, 0);
    assert!(!report.non_orientable);
    assert_eq!(faces, vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]]);
}

#[test]
fn strip_feeds_through_consistently() {
    // A triangle strip is already alternating-wound by triangle_indices,
    // so it is internally consistent — no flips, one component.
    let mut p = Primitive::new(Topology::TriangleStrip);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    // implicit indices 0,1,2,3 -> tris (0,1,2) and (1,3,2)
    let (faces, report) = p.orient_consistent();
    assert_eq!(report.component_count, 1);
    assert_eq!(report.flipped_count, 0);
    assert!(!report.non_orientable);
    assert_eq!(faces, vec![[0, 1, 2], [1, 3, 2]]);
}
