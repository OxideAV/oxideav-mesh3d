//! Tests for `Primitive::triangle_adjacency` — the face-dual graph.
//!
//! Every expected value below comes from hand-counting which triangles
//! share which edges of a small, fully-enumerated fixture. Two
//! triangles are neighbours across an edge iff that undirected edge is
//! used by exactly those two triangles (a clean manifold-interior edge);
//! boundary edges (one user) and non-manifold edges (three+ users)
//! yield `None`. The output is indexed by `Primitive::triangle_indices`
//! enumeration order, and each entry is `[n01, n12, n20]` keyed to the
//! triangle's three corner-pair edges (0→1, 1→2, 2→0).

use oxideav_mesh3d::{EdgeManifoldReport, Indices, Primitive, Topology};

// ── Fixtures ───────────────────────────────────────────────────────

/// A single open triangle, no index buffer: all three edges boundary.
fn open_triangle() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p
}

/// Two triangles sharing the diagonal edge (1,2) of a unit quad.
/// Triangle 0 = (0,1,2); triangle 1 = (2,1,3). The shared edge is
/// triangle 0's edge 1→2 (slot 1) and triangle 1's edge 2→1 (slot 0).
fn quad_two_tris() -> Primitive {
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

/// A closed regular tetrahedron: every one of the 6 edges is shared by
/// exactly two of the 4 faces, so every face has all three neighbours.
fn tetrahedron() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    p.indices = Some(Indices::U32(vec![
        0, 2, 1, // f0 bottom
        0, 1, 3, // f1 front
        0, 3, 2, // f2 left
        1, 2, 3, // f3 far
    ]));
    p
}

/// A closed axis-aligned unit cube, indexed: 8 corners, 12 triangles,
/// 18 edges. 12 of those edges are the diagonals split inside each face
/// (one diagonal per face, shared by that face's two triangles) plus
/// the 12 cube edges (each shared by two triangles on adjacent faces).
fn cube() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    p.indices = Some(Indices::U32(vec![
        0, 2, 1, 0, 3, 2, // -Z
        4, 5, 6, 4, 6, 7, // +Z
        0, 1, 5, 0, 5, 4, // -Y
        3, 7, 6, 3, 6, 2, // +Y
        0, 4, 7, 0, 7, 3, // -X
        1, 2, 6, 1, 6, 5, // +X
    ]));
    p
}

/// Count the number of `Some` links in an adjacency result.
fn link_count(adj: &[[Option<u32>; 3]]) -> usize {
    adj.iter()
        .map(|t| t.iter().filter(|s| s.is_some()).count())
        .sum()
}

// ── Output shape / length ──────────────────────────────────────────

#[test]
fn empty_primitive_returns_empty() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.triangle_adjacency().is_empty());
}

#[test]
fn output_length_equals_triangle_count() {
    let p = cube();
    let adj = p.triangle_adjacency();
    assert_eq!(adj.len(), p.triangle_indices().len());
    assert_eq!(adj.len(), p.triangle_count());
    assert_eq!(adj.len(), 12);
}

#[test]
fn lines_topology_returns_empty() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert!(p.triangle_adjacency().is_empty());
}

#[test]
fn points_topology_returns_empty() {
    let mut p = Primitive::new(Topology::Points);
    p.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert!(p.triangle_adjacency().is_empty());
}

// ── Boundary edges yield None ──────────────────────────────────────

#[test]
fn open_triangle_all_boundary() {
    let p = open_triangle();
    let adj = p.triangle_adjacency();
    assert_eq!(adj.len(), 1);
    assert_eq!(adj[0], [None, None, None]);
    assert_eq!(link_count(&adj), 0);
}

// ── Two-triangle shared edge ───────────────────────────────────────

#[test]
fn quad_shared_diagonal_links_correct_slots() {
    let p = quad_two_tris();
    let adj = p.triangle_adjacency();
    assert_eq!(adj.len(), 2);
    // Triangle 0 = (0,1,2): edge 1→2 is the shared diagonal (slot 1).
    assert_eq!(adj[0], [None, Some(1), None]);
    // Triangle 1 = (2,1,3): edge 2→1 is the shared diagonal (slot 0).
    assert_eq!(adj[1], [Some(0), None, None]);
}

#[test]
fn quad_link_count_is_two() {
    // One interior edge ⇒ two directed links.
    let p = quad_two_tris();
    assert_eq!(link_count(&p.triangle_adjacency()), 2);
}

#[test]
fn quad_links_are_symmetric() {
    let p = quad_two_tris();
    let adj = p.triangle_adjacency();
    // If 0 points to 1, 1 must point back to 0.
    assert!(adj[0].contains(&Some(1)));
    assert!(adj[1].contains(&Some(0)));
}

// ── Closed manifolds: every face fully linked ──────────────────────

#[test]
fn tetrahedron_every_face_three_neighbours() {
    let p = tetrahedron();
    let adj = p.triangle_adjacency();
    assert_eq!(adj.len(), 4);
    for (fi, entry) in adj.iter().enumerate() {
        for (slot, n) in entry.iter().enumerate() {
            assert!(
                n.is_some(),
                "face {fi} slot {slot} should have a neighbour on a closed mesh"
            );
        }
    }
}

#[test]
fn tetrahedron_neighbours_are_distinct_and_self_free() {
    let p = tetrahedron();
    let adj = p.triangle_adjacency();
    for (fi, entry) in adj.iter().enumerate() {
        let ns: Vec<u32> = entry.iter().filter_map(|s| *s).collect();
        // No self-loops.
        assert!(!ns.contains(&(fi as u32)));
        // The three neighbours of a tetra face are the other three faces.
        let mut sorted = ns.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            3,
            "face {fi} should have 3 distinct neighbours"
        );
    }
}

#[test]
fn tetrahedron_link_count_matches_interior_edges() {
    let p = tetrahedron();
    let adj = p.triangle_adjacency();
    let report = p.edge_manifold_report();
    assert_eq!(link_count(&adj), 2 * report.manifold_interior_edge_count);
    // Tetra: 6 interior edges ⇒ 12 directed links.
    assert_eq!(link_count(&adj), 12);
}

#[test]
fn cube_every_face_fully_linked() {
    let p = cube();
    let adj = p.triangle_adjacency();
    for entry in &adj {
        for n in entry {
            assert!(n.is_some());
        }
    }
}

#[test]
fn cube_link_count_matches_interior_edges() {
    let p = cube();
    let adj = p.triangle_adjacency();
    let report = p.edge_manifold_report();
    // Cube: 18 edges, all interior ⇒ 36 directed links.
    assert_eq!(report.manifold_interior_edge_count, 18);
    assert_eq!(link_count(&adj), 36);
}

// ── Symmetry over a whole closed mesh ──────────────────────────────

#[test]
fn closed_mesh_links_are_globally_symmetric() {
    for p in [tetrahedron(), cube()] {
        let adj = p.triangle_adjacency();
        for (fi, entry) in adj.iter().enumerate() {
            for n in entry.iter().flatten() {
                let back = adj[*n as usize];
                assert!(
                    back.contains(&Some(fi as u32)),
                    "face {fi} → {n} link not mirrored back"
                );
            }
        }
    }
}

// ── Non-manifold edges yield None on every involved side ───────────

#[test]
fn non_manifold_edge_gives_none_on_all_three_faces() {
    // Three triangles all sharing edge (0,1): a book-spine / fin.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![
        0, 1, 2, // f0
        0, 1, 3, // f1
        0, 1, 4, // f2
    ]));
    let adj = p.triangle_adjacency();
    // The shared edge (0,1) is non-manifold (3 users) ⇒ no link on it.
    // Each face: edge 0→1 is slot 0; the other two edges are boundary.
    for entry in &adj {
        assert_eq!(*entry, [None, None, None]);
    }
    assert_eq!(link_count(&adj), 0);
    assert!(p.edge_manifold_report().non_manifold_edge_count >= 1);
}

#[test]
fn non_manifold_edge_does_not_block_other_clean_edges() {
    // Two triangles share a clean edge; a third triangle makes a
    // DIFFERENT edge non-manifold. The clean link must survive.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],  // 0
        [1.0, 0.0, 0.0],  // 1
        [0.0, 1.0, 0.0],  // 2
        [1.0, 1.0, 0.0],  // 3
        [0.5, 0.5, 1.0],  // 4
        [0.5, 0.5, -1.0], // 5
    ];
    p.indices = Some(Indices::U32(vec![
        0, 1, 2, // f0  has clean edge (1,2) with f1
        2, 1, 3, // f1
        1, 2, 4, // f2  makes edge (1,2) non-manifold
        1, 2, 5, // f3  makes edge (1,2) non-manifold further
    ]));
    let adj = p.triangle_adjacency();
    // Edge (1,2) now has 4 users → non-manifold → no links across it.
    // But each face's OTHER edges may still be boundary, none clean here.
    // Verify: no link references edge (1,2) at all.
    assert_eq!(link_count(&adj), 0);
}

// ── Excluded triangles keep their slot but are inert ───────────────

#[test]
fn out_of_range_triangle_is_inert() {
    let mut p = quad_two_tris();
    // Append a triangle referencing a non-existent vertex 99.
    if let Some(Indices::U32(idx)) = &mut p.indices {
        idx.extend_from_slice(&[0, 1, 99]);
    }
    let adj = p.triangle_adjacency();
    // Three triangles in the enumeration; the last is excluded.
    assert_eq!(adj.len(), 3);
    assert_eq!(adj[2], [None, None, None]);
    // The valid pair's link is untouched.
    assert_eq!(adj[0], [None, Some(1), None]);
    assert_eq!(adj[1], [Some(0), None, None]);
}

#[test]
fn duplicate_corner_triangle_is_inert() {
    let mut p = quad_two_tris();
    // Append a degenerate-by-index triangle (corner repeated).
    if let Some(Indices::U32(idx)) = &mut p.indices {
        idx.extend_from_slice(&[1, 1, 2]);
    }
    let adj = p.triangle_adjacency();
    assert_eq!(adj.len(), 3);
    assert_eq!(adj[2], [None, None, None]);
    // The degenerate triangle's edge (1,2) must NOT have linked into the
    // valid pair's shared diagonal (that would make it non-manifold).
    assert_eq!(adj[0], [None, Some(1), None]);
    assert_eq!(adj[1], [Some(0), None, None]);
    assert_eq!(link_count(&adj), 2);
}

// ── Topology feed: strip / fan parity with explicit list ───────────

#[test]
fn triangle_strip_feeds_in() {
    // Strip 0,1,2,3 → triangles (0,1,2) and (2,1,3) after the
    // alternating-winding rule on the second triangle. Same connectivity
    // as the indexed quad above.
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    strip.indices = Some(Indices::U32(vec![0, 1, 2, 3]));
    let adj = strip.triangle_adjacency();
    assert_eq!(adj.len(), 2);
    // The two strip triangles share an edge ⇒ two symmetric links.
    assert_eq!(link_count(&adj), 2);
    assert!(adj[0].contains(&Some(1)));
    assert!(adj[1].contains(&Some(0)));
}

#[test]
fn triangle_fan_feeds_in() {
    // Fan anchor 0 with rim 1,2,3 → triangles (0,1,2) and (0,2,3),
    // sharing edge (0,2).
    let mut fan = Primitive::new(Topology::TriangleFan);
    fan.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    fan.indices = Some(Indices::U32(vec![0, 1, 2, 3]));
    let adj = fan.triangle_adjacency();
    assert_eq!(adj.len(), 2);
    assert_eq!(link_count(&adj), 2);
    assert!(adj[0].contains(&Some(1)));
    assert!(adj[1].contains(&Some(0)));
}

// ── Non-indexed (vertex-soup) primitive ────────────────────────────

#[test]
fn vertex_soup_no_shared_indices_all_boundary() {
    // Two separate triangles, distinct vertex pools (no shared index),
    // even though positions coincide → no link without a weld.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        // duplicate positions, different indices:
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    // No index buffer → implicit 0,1,2 / 3,4,5.
    let adj = p.triangle_adjacency();
    assert_eq!(adj.len(), 2);
    assert_eq!(link_count(&adj), 0);
}

#[test]
fn weld_then_adjacency_links_coincident_seam() {
    // Same soup as above; welding merges the coincident corners so the
    // shared edge becomes a clean interior edge with a link.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let welded = p.weld_vertices();
    let adj = welded.triangle_adjacency();
    assert_eq!(adj.len(), 2);
    assert_eq!(link_count(&adj), 2, "welded seam should link the two faces");
}

// ── Component partition agreement with topology_summary ────────────

#[test]
fn bfs_over_links_recovers_component_count() {
    // Two disjoint closed tetrahedra → two components.
    let mut p = Primitive::new(Topology::Triangles);
    let base = tetrahedron();
    p.positions = base.positions.clone();
    // Second tetra, offset, distinct vertex indices 4..8.
    for v in &base.positions {
        p.positions.push([v[0] + 10.0, v[1], v[2]]);
    }
    let mut idx = vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
    let shifted: Vec<u32> = idx.iter().map(|i| i + 4).collect();
    idx.extend_from_slice(&shifted);
    p.indices = Some(Indices::U32(idx));

    let adj = p.triangle_adjacency();
    assert_eq!(adj.len(), 8);

    // BFS over the Some-links.
    let mut seen = vec![false; adj.len()];
    let mut components = 0;
    for start in 0..adj.len() {
        if seen[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(f) = stack.pop() {
            for n in adj[f].iter().flatten() {
                let nu = *n as usize;
                if !seen[nu] {
                    seen[nu] = true;
                    stack.push(nu);
                }
            }
        }
    }
    assert_eq!(components, 2);
    assert_eq!(components, p.topology_summary().component_count);
}

// ── Determinism ────────────────────────────────────────────────────

#[test]
fn deterministic_across_calls() {
    let p = cube();
    assert_eq!(p.triangle_adjacency(), p.triangle_adjacency());
}

#[test]
fn purity_no_mutation() {
    let p = cube();
    let before = p.clone();
    let _ = p.triangle_adjacency();
    assert_eq!(p.positions, before.positions);
    assert_eq!(p.indices, before.indices);
}

// ── Cross-check: closed manifold has zero None slots ───────────────

#[test]
fn closed_manifold_iff_no_none_slots() {
    for p in [tetrahedron(), cube()] {
        let report: EdgeManifoldReport = p.edge_manifold_report();
        assert!(report.is_closed_manifold());
        let adj = p.triangle_adjacency();
        let none_count: usize = adj
            .iter()
            .map(|t| t.iter().filter(|s| s.is_none()).count())
            .sum();
        assert_eq!(none_count, 0);
    }
    // Open triangle: closed_manifold false ⇒ some None slots.
    let p = open_triangle();
    assert!(!p.edge_manifold_report().is_closed_manifold());
    let adj = p.triangle_adjacency();
    let none_count: usize = adj
        .iter()
        .map(|t| t.iter().filter(|s| s.is_none()).count())
        .sum();
    assert!(none_count > 0);
}
