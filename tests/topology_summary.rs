//! Tests for `Primitive::topology_summary` and the `TopologySummary`
//! report it produces.
//!
//! Every expected value below comes from elementary combinatorial
//! topology: the Euler characteristic `χ = V − E + F` of a triangulated
//! surface, and the genus relation `χ = 2 − 2g` for a single closed
//! orientable two-manifold. Fixtures are built by hand or welded down
//! from a vertex soup so the vertex / edge / face counts are exactly
//! the textbook values.

use oxideav_mesh3d::{Indices, Primitive, Topology, TopologySummary};

// ── Fixtures ───────────────────────────────────────────────────────

/// A single open triangle (vertex soup, no index buffer): V=3, E=3,
/// F=1, one component, one boundary loop, not closed.
fn open_triangle() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p
}

/// A closed regular tetrahedron — the simplest closed two-manifold.
/// V=4, E=6, F=4 ⇒ χ = 4 − 6 + 4 = 2 ⇒ genus 0. Wound CCW outward.
fn tetrahedron() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    // Four faces, each wound CCW as seen from outside the solid.
    p.indices = Some(Indices::U32(vec![
        0, 2, 1, // bottom (-Z-ish)
        0, 1, 3, // front
        0, 3, 2, // left
        1, 2, 3, // far
    ]));
    p
}

/// A closed axis-aligned unit cube as an *indexed* mesh: 8 distinct
/// corners, 12 triangles. After construction V=8, E=18, F=12 ⇒
/// χ = 8 − 18 + 12 = 2 ⇒ genus 0.
fn cube() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [1.0, 1.0, 0.0], // 2
        [0.0, 1.0, 0.0], // 3
        [0.0, 0.0, 1.0], // 4
        [1.0, 0.0, 1.0], // 5
        [1.0, 1.0, 1.0], // 6
        [0.0, 1.0, 1.0], // 7
    ];
    // Each face two triangles, wound CCW outward.
    p.indices = Some(Indices::U32(vec![
        // -Z (bottom)
        0, 2, 1, 0, 3, 2, // +Z (top)
        4, 5, 6, 4, 6, 7, // -Y (front)
        0, 1, 5, 0, 5, 4, // +Y (back)
        3, 7, 6, 3, 6, 2, // -X (left)
        0, 4, 7, 0, 7, 3, // +X (right)
        1, 2, 6, 1, 6, 5,
    ]));
    p
}

/// A closed torus tessellated as an `nu × nv` grid wrapped in both
/// directions. Every grid vertex is distinct (V = nu·nv), every quad is
/// two triangles (F = 2·nu·nv), and a quad grid wrapped on a torus has
/// E = 3·nu·nv (each vertex owns one right, one up, one diagonal edge).
/// Euler: V − E + F = nu·nv − 3·nu·nv + 2·nu·nv = 0 ⇒ genus 1.
fn torus(nu: usize, nv: usize, r_major: f32, r_minor: f32) -> Primitive {
    use std::f32::consts::TAU;
    let mut p = Primitive::new(Topology::Triangles);
    let idx = |u: usize, v: usize| ((u % nu) * nv + (v % nv)) as u32;
    for u in 0..nu {
        let au = TAU * u as f32 / nu as f32;
        for v in 0..nv {
            let av = TAU * v as f32 / nv as f32;
            let x = (r_major + r_minor * av.cos()) * au.cos();
            let y = (r_major + r_minor * av.cos()) * au.sin();
            let z = r_minor * av.sin();
            p.positions.push([x, y, z]);
        }
    }
    let mut tris: Vec<u32> = Vec::with_capacity(nu * nv * 6);
    for u in 0..nu {
        for v in 0..nv {
            let a = idx(u, v);
            let b = idx(u + 1, v);
            let c = idx(u + 1, v + 1);
            let d = idx(u, v + 1);
            tris.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    p.indices = Some(Indices::U32(tris));
    p
}

// ── Empty / non-triangle ───────────────────────────────────────────

#[test]
fn empty_primitive_all_zero() {
    let p = Primitive::new(Topology::Triangles);
    let t = p.topology_summary();
    assert_eq!(t, TopologySummary::default());
    assert_eq!(t.genus, None);
}

#[test]
fn lines_topology_all_zero() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
    let t = p.topology_summary();
    assert_eq!(t.face_count, 0);
    assert_eq!(t.vertex_count, 0);
    assert_eq!(t.edge_count, 0);
    assert_eq!(t.component_count, 0);
    assert_eq!(t.genus, None);
}

#[test]
fn points_topology_all_zero() {
    let mut p = Primitive::new(Topology::Points);
    p.positions = vec![[0.0; 3]; 10];
    assert_eq!(p.topology_summary(), TopologySummary::default());
}

// ── Open triangle ──────────────────────────────────────────────────

#[test]
fn open_triangle_counts() {
    let t = open_triangle().topology_summary();
    assert_eq!(t.vertex_count, 3);
    assert_eq!(t.edge_count, 3);
    assert_eq!(t.face_count, 1);
    assert_eq!(t.euler_characteristic, 1); // 3 - 3 + 1
    assert_eq!(t.component_count, 1);
    assert_eq!(t.boundary_loop_count, 1);
    assert_eq!(t.genus, None); // open patch
}

/// A two-triangle quad shares one diagonal edge: V=4, E=5, F=2, χ=1.
#[test]
fn quad_two_triangles() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 2, 3]));
    let t = p.topology_summary();
    assert_eq!(t.vertex_count, 4);
    assert_eq!(t.edge_count, 5); // 4 rim + 1 shared diagonal
    assert_eq!(t.face_count, 2);
    assert_eq!(t.euler_characteristic, 1);
    assert_eq!(t.component_count, 1);
    assert_eq!(t.boundary_loop_count, 1); // the four rim edges chain into one loop
    assert_eq!(t.genus, None);
}

// ── Closed genus-0 surfaces ────────────────────────────────────────

#[test]
fn tetrahedron_is_genus_zero() {
    let t = tetrahedron().topology_summary();
    assert_eq!(t.vertex_count, 4);
    assert_eq!(t.edge_count, 6);
    assert_eq!(t.face_count, 4);
    assert_eq!(t.euler_characteristic, 2);
    assert_eq!(t.component_count, 1);
    assert_eq!(t.boundary_loop_count, 0);
    assert_eq!(t.genus, Some(0));
}

#[test]
fn cube_is_genus_zero() {
    let t = cube().topology_summary();
    assert_eq!(t.vertex_count, 8);
    assert_eq!(t.edge_count, 18);
    assert_eq!(t.face_count, 12);
    assert_eq!(t.euler_characteristic, 2);
    assert_eq!(t.component_count, 1);
    assert_eq!(t.boundary_loop_count, 0);
    assert_eq!(t.genus, Some(0));
}

/// A vertex-soup cube (36 positions, no index buffer) is *not* a
/// manifold by vertex index — each corner is duplicated, so every edge
/// is used by exactly one triangle. Welding first collapses it to the
/// 8-corner indexed cube; genus then resolves.
#[test]
fn cube_soup_needs_weld_for_genus() {
    let mut p = Primitive::new(Topology::Triangles);
    // Build 12 triangles' worth of duplicated corners from the indexed cube.
    let c = cube();
    let pos = &c.positions;
    if let Some(Indices::U32(idx)) = &c.indices {
        for &i in idx {
            p.positions.push(pos[i as usize]);
        }
    }
    // As a soup, every vertex is distinct by index → all boundary edges.
    let soup = p.topology_summary();
    assert_eq!(soup.face_count, 12);
    assert_eq!(soup.vertex_count, 36);
    assert_eq!(soup.genus, None);
    assert!(
        soup.component_count > 1,
        "soup facets share no edges by index"
    );

    // Welding merges the 36 corners back to 8 distinct vertices.
    let welded = p.weld_vertices().topology_summary();
    assert_eq!(welded.vertex_count, 8);
    assert_eq!(welded.edge_count, 18);
    assert_eq!(welded.face_count, 12);
    assert_eq!(welded.euler_characteristic, 2);
    assert_eq!(welded.component_count, 1);
    assert_eq!(welded.genus, Some(0));
}

// ── Closed genus-1 surface ─────────────────────────────────────────

#[test]
fn torus_is_genus_one() {
    let t = torus(8, 6, 3.0, 1.0).topology_summary();
    // V = 8*6 = 48, F = 2*48 = 96, E = 3*48 = 144.
    assert_eq!(t.vertex_count, 48);
    assert_eq!(t.face_count, 96);
    assert_eq!(t.edge_count, 144);
    assert_eq!(t.euler_characteristic, 0); // 48 - 144 + 96
    assert_eq!(t.component_count, 1);
    assert_eq!(t.boundary_loop_count, 0);
    assert_eq!(t.genus, Some(1));
}

#[test]
fn larger_torus_still_genus_one() {
    let t = torus(20, 12, 5.0, 1.5).topology_summary();
    assert_eq!(t.euler_characteristic, 0);
    assert_eq!(t.genus, Some(1));
    assert_eq!(t.component_count, 1);
}

// ── Connected components ───────────────────────────────────────────

#[test]
fn two_disjoint_tetrahedra_two_components() {
    let mut p = tetrahedron();
    let base = p.positions.len() as u32;
    // Append a second tetra translated far away with fresh indices.
    let second = tetrahedron();
    for v in &second.positions {
        p.positions.push([v[0] + 100.0, v[1], v[2]]);
    }
    if let (Some(Indices::U32(a)), Some(Indices::U32(b))) = (&mut p.indices, &second.indices) {
        for &i in b {
            a.push(i + base);
        }
    }
    let t = p.topology_summary();
    assert_eq!(t.face_count, 8);
    assert_eq!(t.vertex_count, 8);
    assert_eq!(t.component_count, 2);
    // Two closed components ⇒ not a single closed manifold ⇒ no genus.
    assert_eq!(t.genus, None);
    // χ of two disjoint spheres is 2 + 2 = 4.
    assert_eq!(t.euler_characteristic, 4);
}

/// Two triangles touching only at a single shared vertex are *not*
/// edge-connected — `component_count` must be 2 (vertex adjacency is
/// not facet adjacency).
#[test]
fn vertex_touching_is_two_components() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0], // 0 shared apex
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 3, 4]));
    let t = p.topology_summary();
    assert_eq!(t.face_count, 2);
    assert_eq!(t.component_count, 2);
}

// ── Open surface: a disc patch with one hole ───────────────────────

/// A square annulus (outer 4 corners, inner hole of 4 corners) — open
/// surface with two boundary loops. Genus stays `None` (not closed).
#[test]
fn annulus_two_boundary_loops_no_genus() {
    // Outer corners 0..4 (a 2x2 square), inner corners 4..8 (a centred
    // 1x1 square). Eight triangles around the ring.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0], // 0 outer
        [2.0, 0.0, 0.0], // 1
        [2.0, 2.0, 0.0], // 2
        [0.0, 2.0, 0.0], // 3
        [0.5, 0.5, 0.0], // 4 inner
        [1.5, 0.5, 0.0], // 5
        [1.5, 1.5, 0.0], // 6
        [0.5, 1.5, 0.0], // 7
    ];
    p.indices = Some(Indices::U32(vec![
        0, 1, 5, 0, 5, 4, // bottom strip
        1, 2, 6, 1, 6, 5, // right strip
        2, 3, 7, 2, 7, 6, // top strip
        3, 0, 4, 3, 4, 7, // left strip
    ]));
    let t = p.topology_summary();
    assert_eq!(t.face_count, 8);
    assert_eq!(t.component_count, 1);
    assert_eq!(t.boundary_loop_count, 2); // outer rim + inner hole
    assert_eq!(t.genus, None);
    // V=8, F=8, E=16 (8 outer/inner rim + 8 spokes/diagonals).
    assert_eq!(t.vertex_count, 8);
}

// ── Robustness: out-of-range and duplicate-corner exclusion ────────

#[test]
fn out_of_range_face_excluded() {
    let mut p = tetrahedron();
    // Append a triangle referencing a vertex that does not exist.
    if let Some(Indices::U32(idx)) = &mut p.indices {
        idx.extend_from_slice(&[0, 1, 999]);
    }
    let t = p.topology_summary();
    // The bogus face is dropped → still a clean tetra.
    assert_eq!(t.face_count, 4);
    assert_eq!(t.genus, Some(0));
}

#[test]
fn duplicate_corner_face_excluded() {
    let mut p = tetrahedron();
    if let Some(Indices::U32(idx)) = &mut p.indices {
        idx.extend_from_slice(&[1, 1, 2]); // zero-length edge
    }
    let t = p.topology_summary();
    assert_eq!(t.face_count, 4);
    assert_eq!(t.genus, Some(0));
}

#[test]
fn nan_position_still_counted_combinatorially() {
    // topology_summary is combinatorial — a NaN position does not change
    // the index-level connectivity. (Unlike degenerate_triangles, which
    // inspects geometry.)
    let mut p = tetrahedron();
    p.positions[0] = [f32::NAN, 0.0, 0.0];
    let t = p.topology_summary();
    assert_eq!(t.face_count, 4);
    assert_eq!(t.vertex_count, 4);
    assert_eq!(t.genus, Some(0));
}

#[test]
fn unreferenced_vertices_do_not_inflate_v() {
    let mut p = tetrahedron();
    // Add 100 stray pool vertices no triangle references.
    for _ in 0..100 {
        p.positions.push([42.0, 42.0, 42.0]);
    }
    let t = p.topology_summary();
    assert_eq!(t.vertex_count, 4, "V counts referenced vertices only");
    assert_eq!(t.genus, Some(0));
}

// ── Open mesh: a half-open cube (one face removed) ─────────────────

/// Remove one face's two triangles from the cube — it becomes an open
/// box. χ shifts and genus is `None` (boundary present).
#[test]
fn cube_with_hole_is_open() {
    let mut p = cube();
    if let Some(Indices::U32(idx)) = &mut p.indices {
        // Drop the last face (two triangles = 6 indices).
        idx.truncate(idx.len() - 6);
    }
    let t = p.topology_summary();
    assert_eq!(t.face_count, 10);
    assert_eq!(t.component_count, 1);
    assert!(t.boundary_loop_count >= 1);
    assert_eq!(t.genus, None);
    // χ of a disc-topology open box is 1.
    assert_eq!(t.euler_characteristic, 1);
}

// ── Non-manifold: edge shared by three triangles ──────────────────

#[test]
fn non_manifold_edge_no_genus() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],  // 0
        [1.0, 0.0, 0.0],  // 1 — shared edge 0-1
        [0.0, 1.0, 0.0],  // 2
        [0.0, -1.0, 0.0], // 3
        [0.0, 0.0, 1.0],  // 4
    ];
    // Three triangles all share edge (0,1) — a "book spine".
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 3, 0, 1, 4]));
    let t = p.topology_summary();
    assert_eq!(t.face_count, 3);
    assert_eq!(t.component_count, 1); // edge-connected via (0,1)
    assert_eq!(t.genus, None); // not a clean two-manifold
}

// ── Topology feed: strip and fan parity with a triangle list ───────

#[test]
fn strip_and_list_agree() {
    // A 4-vertex strip = 2 triangles; same connectivity as the quad.
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
    let st = strip.topology_summary();
    assert_eq!(st.face_count, 2);
    assert_eq!(st.vertex_count, 4);
    assert_eq!(st.edge_count, 5);
    assert_eq!(st.euler_characteristic, 1);
    assert_eq!(st.component_count, 1);
}

#[test]
fn fan_counts() {
    // A 5-vertex fan = 3 triangles all sharing the anchor vertex 0.
    let mut fan = Primitive::new(Topology::TriangleFan);
    fan.positions = vec![
        [0.0, 0.0, 0.0], // anchor
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let t = fan.topology_summary();
    assert_eq!(t.face_count, 3);
    assert_eq!(t.vertex_count, 5);
    // E: 3 fan spokes shared + 1 anchor-to-first + 1 anchor-to-last + 3 rim = 7.
    assert_eq!(t.edge_count, 7);
    assert_eq!(t.euler_characteristic, 1); // 5 - 7 + 3
    assert_eq!(t.component_count, 1);
    assert_eq!(t.boundary_loop_count, 1);
}

// ── Determinism ────────────────────────────────────────────────────

#[test]
fn summary_is_deterministic_and_pure() {
    let p = torus(10, 8, 4.0, 1.0);
    let a = p.topology_summary();
    let b = p.topology_summary();
    assert_eq!(a, b);
    // Purity: a second call still produces the same primitive's result;
    // no fields mutated (Copy struct equality already implies this).
    assert_eq!(p.topology_summary(), a);
}

#[test]
fn copy_and_default_traits() {
    let t = cube().topology_summary();
    let copy = t; // Copy
    assert_eq!(copy, t);
    let d = TopologySummary::default();
    assert_eq!(d.genus, None);
    assert_eq!(d.euler_characteristic, 0);
}
