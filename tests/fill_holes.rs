//! Tests for `Primitive::fill_holes` — capping every boundary loop of a
//! surface with an ear-clip patch to close its holes / cracks / open rims.
//!
//! The patch triangles reference existing pool vertices (no new vertices),
//! are wound to cross every boundary edge opposite to the loop traversal
//! (so the front-face normal agrees with the surrounding surface), and are
//! appended to a de-stripped copy of the input. A closed two-manifold or a
//! non-triangle topology is returned unchanged.

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

fn idx_vec(p: &Primitive) -> Vec<u32> {
    match &p.indices {
        Some(Indices::U32(v)) => v.clone(),
        Some(Indices::U16(v)) => v.iter().map(|&x| x as u32).collect(),
        None => (0..p.positions.len() as u32).collect(),
    }
}

// Closed unit tetrahedron (outward CCW) → no boundary loops.
fn tetrahedron() -> Primitive {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![
        0, 2, 1, // bottom
        0, 1, 3, // front
        0, 3, 2, // left
        1, 2, 3, // far
    ];
    tri_indexed(positions, indices)
}

// Closed axis-aligned unit cube [0,1]^3, outward CCW, 12 triangles.
fn unit_cube() -> Primitive {
    let positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [1.0, 1.0, 0.0], // 2
        [0.0, 1.0, 0.0], // 3
        [0.0, 0.0, 1.0], // 4
        [1.0, 0.0, 1.0], // 5
        [1.0, 1.0, 1.0], // 6
        [0.0, 1.0, 1.0], // 7
    ];
    // Each face two triangles, outward-facing (CCW seen from outside).
    let indices = vec![
        // -z (bottom), normal (0,0,-1): CW in xy seen from +z
        0, 2, 1, 0, 3, 2, //
        // +z (top), normal (0,0,1)
        4, 5, 6, 4, 6, 7, //
        // -y (front), normal (0,-1,0)
        0, 1, 5, 0, 5, 4, //
        // +y (back), normal (0,1,0)
        3, 7, 6, 3, 6, 2, //
        // -x (left), normal (-1,0,0)
        0, 4, 7, 0, 7, 3, //
        // +x (right), normal (1,0,0)
        1, 2, 6, 1, 6, 5, //
    ];
    tri_indexed(positions, indices)
}

// --- closed / no-op paths -------------------------------------------

#[test]
fn closed_tetrahedron_unchanged() {
    let p = tetrahedron();
    let filled = p.fill_holes();
    // De-stripped form of the same surface: same triangle set, no caps.
    assert_eq!(filled.topology, Topology::Triangles);
    assert_eq!(filled.triangle_count(), p.triangle_count());
    assert_eq!(idx_vec(&filled), idx_vec(&p.to_triangle_list()));
}

#[test]
fn closed_cube_unchanged() {
    let p = unit_cube();
    let filled = p.fill_holes();
    assert_eq!(filled.triangle_count(), 12);
    assert!((filled.signed_volume() - 1.0).abs() < 1e-6);
}

#[test]
fn empty_primitive_returns_empty() {
    let p = Primitive::new(Topology::Triangles);
    let filled = p.fill_holes();
    assert_eq!(filled.triangle_count(), 0);
    assert_eq!(filled.topology, Topology::Triangles);
}

#[test]
fn lines_topology_returns_no_caps() {
    let mut p = tri(vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    p.topology = Topology::Lines;
    let filled = p.fill_holes();
    assert_eq!(filled.triangle_count(), 0);
}

// --- single planar hole ---------------------------------------------

#[test]
fn open_quad_single_loop_gets_capped() {
    // Two triangles forming a square with a four-vertex boundary loop.
    // Filling adds two triangles (a 4-gon → 2 ears).
    let p = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        vec![0, 1, 2, 1, 3, 2],
    );
    assert_eq!(p.boundary_loops().len(), 1);
    let filled = p.fill_holes();
    // Original 2 + cap 2.
    assert_eq!(filled.triangle_count(), 4);
    // The cap is on the other side of the same square; the filled
    // surface is a (degenerate, zero-thickness) closed shell whose
    // signed volume is zero but whose boundary is now empty.
    assert!(filled.boundary_edges().is_empty());
}

#[test]
fn square_hole_in_plane_is_filled() {
    // A 3x3 planar grid of quads with the centre quad removed leaves a
    // square hole. `fill_holes` caps EVERY boundary loop — a free-floating
    // patch has no intrinsic "outer" rim, so both the inner hole and the
    // outer rim are filled, closing the surface entirely.
    let mut positions = Vec::new();
    for j in 0..4 {
        for i in 0..4 {
            positions.push([i as f32, j as f32, 0.0]);
        }
    }
    let vid = |i: u32, j: u32| j * 4 + i;
    let mut indices = Vec::new();
    for j in 0..3u32 {
        for i in 0..3u32 {
            if i == 1 && j == 1 {
                continue; // hole in the centre quad
            }
            let (a, b, c, d) = (vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1));
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let p = tri_indexed(positions, indices);
    // Two boundary loops before fill: outer rim + inner square hole.
    assert_eq!(p.boundary_loops().len(), 2);
    let filled = p.fill_holes();
    // Both loops capped → no boundary edges remain.
    assert!(filled.boundary_edges().is_empty());
    assert_eq!(filled.boundary_loops().len(), 0);
}

// --- non-planar hole: cube with a face removed ----------------------

#[test]
fn cube_missing_face_refills_to_closed_manifold() {
    // Cube with its +z face's two triangles dropped → a square open hole.
    let full = unit_cube();
    let mut idx = idx_vec(&full);
    // +z face triangles are the 2nd pair: indices 6..12 (4,5,6,4,6,7).
    idx.drain(6..12);
    let p = tri_indexed(full.positions.clone(), idx);

    assert_eq!(p.triangle_count(), 10);
    assert_eq!(p.boundary_loops().len(), 1, "one square rim where +z was");
    assert!(!p.edge_manifold_report().is_closed_manifold());

    let filled = p.fill_holes();
    assert_eq!(filled.triangle_count(), 12, "10 walls + 2 cap ears");
    assert!(filled.boundary_edges().is_empty(), "watertight again");
    assert!(filled.edge_manifold_report().is_closed_manifold());
    // The +z cap restores the unit volume with the right (outward) sign.
    assert!(
        (filled.signed_volume() - 1.0).abs() < 1e-5,
        "vol = {}",
        filled.signed_volume()
    );
}

#[test]
fn cap_winding_is_outward_for_tilted_hole() {
    // A tetrahedron with its top face (1,2,3) removed, then refilled. The
    // cap must restore the original outward winding, so signed_volume
    // returns to the closed tetra's positive value.
    let full = tetrahedron();
    let closed_vol = full.signed_volume();
    assert!(closed_vol > 0.0);

    let mut idx = idx_vec(&full);
    idx.drain(9..12); // drop the "far" face 1,2,3
    let open = tri_indexed(full.positions.clone(), idx);
    assert_eq!(open.boundary_loops().len(), 1);

    let filled = open.fill_holes();
    assert!(filled.boundary_edges().is_empty());
    assert!(filled.edge_manifold_report().is_closed_manifold());
    assert!(
        (filled.signed_volume() - closed_vol).abs() < 1e-6,
        "refilled vol {} != original {}",
        filled.signed_volume(),
        closed_vol
    );
}

// --- pentagon hole (>4 gon → multiple ears) -------------------------

#[test]
fn pentagon_band_both_loops_fill() {
    // A pentagonal disc made of a 5-triangle fan around a centre, with
    // the fan removed leaving just... build a true pentagon hole: a fan
    // ring of 5 outer verts + a centre, drop the centre triangles so the
    // pentagon boundary remains, then fill.
    let mut positions = vec![[0.0f32, 0.0, 0.0]]; // 0 = centre (unused after drop)
    for k in 0..5 {
        let a = std::f32::consts::TAU * (k as f32) / 5.0;
        positions.push([a.cos(), a.sin(), 0.0]);
    }
    // Outer ring as a single boundary loop via a degenerate "rim" of
    // line-less triangles: instead, build the pentagon as a hole inside a
    // bigger frame is complex; use the fan and drop centre fan to expose
    // the rim. Simplest: a pentagon boundary is the loop 1,2,3,4,5 if we
    // make a surface that only touches those edges. Build a thin outer
    // band: pentagon verts (1..=5) + scaled-out verts (6..=10), banded.
    for k in 0..5 {
        let a = std::f32::consts::TAU * (k as f32) / 5.0;
        positions.push([2.0 * a.cos(), 2.0 * a.sin(), 0.0]);
    }
    let inner = |k: u32| 1 + k % 5; // 1..=5
    let outer = |k: u32| 6 + k % 5; // 6..=10
    let mut indices = Vec::new();
    for k in 0..5u32 {
        let (i0, i1) = (inner(k), inner(k + 1));
        let (o0, o1) = (outer(k), outer(k + 1));
        indices.extend_from_slice(&[i0, o0, o1, i0, o1, i1]);
    }
    let p = tri_indexed(positions, indices);
    // Two loops: inner pentagon hole + outer pentagon rim.
    assert_eq!(p.boundary_loops().len(), 2);
    let filled = p.fill_holes();
    // Band had 10 tris; each pentagon (5 verts) caps with 3 ears, both
    // loops filled → 10 + 3 + 3 = 16, no boundary left.
    assert_eq!(filled.triangle_count(), 16);
    assert_eq!(filled.boundary_loops().len(), 0);
}

// --- attribute / topology preservation ------------------------------

#[test]
fn cap_references_existing_vertices_only() {
    let p = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        vec![0, 1, 2, 1, 3, 2],
    );
    let filled = p.fill_holes();
    // No new vertices: positions length is unchanged.
    assert_eq!(filled.positions.len(), p.positions.len());
    // Every index is in range.
    let n = filled.positions.len() as u32;
    assert!(idx_vec(&filled).iter().all(|&i| i < n));
}

#[test]
fn strip_topology_feeds_in() {
    // A triangle strip forming two triangles (a square) → de-stripped and
    // capped exactly like the indexed quad.
    let mut p = tri(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]);
    p.topology = Topology::TriangleStrip;
    let filled = p.fill_holes();
    assert_eq!(filled.topology, Topology::Triangles);
    assert!(filled.boundary_edges().is_empty());
    assert_eq!(filled.triangle_count(), 4);
}

#[test]
fn does_not_mutate_self() {
    let p = unit_cube();
    let before_tris = p.triangle_count();
    let _ = p.fill_holes();
    assert_eq!(p.triangle_count(), before_tris);
}

#[test]
fn out_of_range_loop_vertex_is_skipped_without_panic() {
    // Craft an open quad, then poke an out-of-range index into the buffer
    // alongside it. boundary detection excludes the bad triangle, so the
    // good rim still fills; the call must not panic.
    let p = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        vec![0, 1, 2, 1, 3, 2, 0, 99, 1],
    );
    let filled = p.fill_holes();
    // Must produce a valid index buffer (no panic); count is at least the
    // de-stripped original.
    assert!(filled.triangle_count() >= p.to_triangle_list().triangle_count());
}

#[test]
fn nan_loop_vertex_loop_is_skipped() {
    // An open quad whose v3 is NaN: boundary_loops still finds the rim,
    // but the projection bails (non-finite vertex) and that loop is
    // skipped — no panic, no cap.
    let p = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [f32::NAN, 1.0, 0.0],
        ],
        vec![0, 1, 2, 1, 3, 2],
    );
    let filled = p.fill_holes();
    // No cap added (loop had a NaN vertex), so still just the 2 originals.
    assert_eq!(filled.triangle_count(), 2);
}

#[test]
fn idempotent_after_fill() {
    // Filling a watertight result again changes nothing.
    let open = {
        let full = unit_cube();
        let mut idx = idx_vec(&full);
        idx.drain(6..12);
        tri_indexed(full.positions.clone(), idx)
    };
    let once = open.fill_holes();
    let twice = once.fill_holes();
    assert_eq!(once.triangle_count(), twice.triangle_count());
    assert_eq!(idx_vec(&once), idx_vec(&twice));
}
