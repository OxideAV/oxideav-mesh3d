//! Quadric-error-metric edge-collapse simplification
//! (`Primitive::simplify_quadric`).

use oxideav_mesh3d::{Indices, Primitive, Topology};

/// Build an indexed `Triangles` primitive.
fn tri_prim(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = positions;
    p.indices = Some(Indices::U32(indices));
    p
}

/// A subdivided unit grid in the XY plane: `(n+1)²` vertices, `2n²`
/// triangles. Flat → simplification should be near-lossless.
fn grid(n: usize) -> Primitive {
    let mut pos = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            pos.push([i as f32, j as f32, 0.0]);
        }
    }
    let row = n + 1;
    let mut idx = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = (j * row + i) as u32;
            let b = (j * row + i + 1) as u32;
            let c = ((j + 1) * row + i) as u32;
            let d = ((j + 1) * row + i + 1) as u32;
            idx.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    tri_prim(pos, idx)
}

/// Unit-ish closed octahedron (8 faces, 6 vertices), CCW outward.
fn octahedron() -> Primitive {
    let pos = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    // +X=0 -X=1 +Y=2 -Y=3 +Z=4 -Z=5
    let idx = vec![
        4, 0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, // top cap
        5, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, // bottom cap
    ];
    tri_prim(pos, idx)
}

fn live_tris(p: &Primitive) -> usize {
    p.triangle_count()
}

#[test]
fn empty_and_degenerate_yield_empty() {
    let empty = Primitive::new(Topology::Triangles);
    assert_eq!(live_tris(&empty.simplify_quadric(10)), 0);

    // Non-triangle topology.
    let mut pts = Primitive::new(Topology::Points);
    pts.positions = vec![[0.0; 3], [1.0; 3]];
    assert_eq!(live_tris(&pts.simplify_quadric(0)), 0);

    // A single degenerate (collinear) triangle.
    let degen = tri_prim(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![0, 1, 2],
    );
    assert_eq!(live_tris(&degen.simplify_quadric(0)), 0);
}

#[test]
fn target_above_input_returns_welded_input() {
    let g = grid(2); // 8 triangles
    let out = g.simplify_quadric(1000);
    assert_eq!(live_tris(&out), live_tris(&g));
    assert_eq!(out.topology, Topology::Triangles);
}

#[test]
fn grid_simplifies_to_target() {
    let g = grid(8); // 128 triangles, 81 verts, flat plane
    let before = live_tris(&g);
    assert_eq!(before, 128);
    let out = g.simplify_quadric(20);
    let after = live_tris(&out);
    assert!(
        (2..=30).contains(&after),
        "expected near-target reduction, got {after}"
    );
    // A flat plane simplifies cheaply; the result must stay planar (all
    // z == 0) and within the original XY bounds.
    for p in &out.positions {
        assert!(p[2].abs() < 1e-4, "plane should stay flat, z={}", p[2]);
        assert!((-0.01..=8.01).contains(&p[0]));
        assert!((-0.01..=8.01).contains(&p[1]));
    }
}

#[test]
fn flat_plane_collapses_to_few_triangles_cheaply() {
    // A flat plane has zero quadric error everywhere, so it should reduce
    // all the way to the target without fold-over rejection.
    let g = grid(6); // 72 tris
    let out = g.simplify_quadric(2);
    assert!(
        live_tris(&out) <= 12,
        "flat plane should reduce hard, got {}",
        live_tris(&out)
    );
}

#[test]
fn closed_manifold_stays_closed_and_bounded() {
    let oct = octahedron();
    assert_eq!(live_tris(&oct), 8);
    let out = oct.simplify_quadric(4);
    // A closed octahedron cannot drop below 4 faces; it should be at or
    // above the target and remain finite + bounded.
    let after = live_tris(&out);
    assert!((4..=8).contains(&after), "got {after}");
    for p in &out.positions {
        for c in p {
            assert!(c.is_finite());
            assert!(c.abs() <= 1.5, "stayed near the unit octahedron");
        }
    }
}

#[test]
fn output_indices_in_range() {
    let g = grid(5);
    let out = g.simplify_quadric(10);
    let n = out.positions.len() as u32;
    for [a, b, c] in out.triangle_indices() {
        assert!(a < n && b < n && c < n);
        assert!(a != b && b != c && a != c, "no degenerate face emitted");
    }
}

#[test]
fn no_unreferenced_vertices() {
    let g = grid(5);
    let out = g.simplify_quadric(8);
    let n = out.positions.len();
    let mut used = vec![false; n];
    for [a, b, c] in out.triangle_indices() {
        used[a as usize] = true;
        used[b as usize] = true;
        used[c as usize] = true;
    }
    assert!(used.iter().all(|&u| u), "every pooled vertex is referenced");
}

#[test]
fn does_not_mutate_self() {
    let g = grid(4);
    let pos_before = g.positions.clone();
    let count_before = live_tris(&g);
    let _ = g.simplify_quadric(4);
    assert_eq!(g.positions, pos_before);
    assert_eq!(live_tris(&g), count_before);
}

#[test]
fn attributes_carry_through() {
    // Grid with normals + one UV set + colours. After simplification the
    // attribute buffers must match the new vertex count.
    let mut g = grid(5);
    let nv = g.positions.len();
    g.normals = Some(vec![[0.0, 0.0, 1.0]; nv]);
    g.uvs = vec![(0..nv).map(|i| [i as f32 * 0.01, 0.0]).collect()];
    g.colors = vec![vec![[1.0, 0.0, 0.0, 1.0]; nv]];

    let out = g.simplify_quadric(10);
    let on = out.positions.len();
    let normals = out.normals.as_ref().expect("normals carried");
    assert_eq!(normals.len(), on);
    // Flat-plane normals stay +Z unit.
    for nrm in normals {
        assert!((nrm[2] - 1.0).abs() < 1e-3, "normal renormalised to +Z");
    }
    assert_eq!(out.uvs.len(), 1);
    assert_eq!(out.uvs[0].len(), on);
    assert_eq!(out.colors.len(), 1);
    assert_eq!(out.colors[0].len(), on);
}

#[test]
fn weights_renormalise_to_one() {
    let mut g = grid(4);
    let nv = g.positions.len();
    g.joints = Some(vec![[0, 1, 2, 3]; nv]);
    g.weights = Some(vec![[0.25, 0.25, 0.25, 0.25]; nv]);
    let out = g.simplify_quadric(6);
    let w = out.weights.as_ref().expect("weights carried");
    let on = out.positions.len();
    assert_eq!(w.len(), on);
    assert_eq!(out.joints.as_ref().unwrap().len(), on);
    for row in w {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "weights sum to 1, got {s}");
    }
}

#[test]
fn target_zero_reduces_as_far_as_legal() {
    let g = grid(4); // 32 tris, an open patch
    let out = g.simplify_quadric(0);
    // An open quad patch reduces to at least 2 triangles (the boundary
    // lock keeps the four corners + the seam). Should be far below input.
    let after = live_tris(&out);
    assert!(after < 32, "should reduce, got {after}");
    assert!(
        after >= 2,
        "boundary lock keeps a minimal patch, got {after}"
    );
}

#[test]
fn handles_triangle_strip_input() {
    // A strip of 4 triangles in a flat row.
    let mut p = Primitive::new(Topology::TriangleStrip);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 3, 4, 5]));
    let out = p.simplify_quadric(2);
    assert_eq!(out.topology, Topology::Triangles);
    assert!(live_tris(&out) <= 4);
    for p in &out.positions {
        assert!(p[2].abs() < 1e-4);
    }
}

#[test]
fn non_finite_positions_dropped() {
    // A grid with one NaN vertex referenced by some triangles; those are
    // excluded, the rest simplify cleanly without panicking or producing
    // NaNs.
    let mut g = grid(4);
    g.positions[5] = [f32::NAN, 0.0, 0.0];
    let out = g.simplify_quadric(6);
    for p in &out.positions {
        for c in p {
            assert!(c.is_finite(), "no NaN propagated to output");
        }
    }
}

#[test]
fn idempotent_at_full_target() {
    // Simplifying to exactly the welded triangle count is a no-op on
    // count and keeps the geometry within bounds.
    let oct = octahedron();
    let out = oct.simplify_quadric(8);
    assert_eq!(live_tris(&out), 8);
}

#[test]
fn bunny_like_bumpy_grid_preserves_shape() {
    // A grid with a single raised vertex (a tent). Simplification should
    // keep the apex region distinct from the flat skirt: the bounding box
    // height must be preserved within tolerance because collapsing the
    // apex away costs a lot of quadric error.
    let n = 6;
    let mut g = grid(n);
    let row = n + 1;
    let apex = n / 2 * row + n / 2;
    g.positions[apex][2] = 3.0;
    let bb_before = g.bounding_box().unwrap();
    let out = g.simplify_quadric(20);
    let bb_after = out.bounding_box().unwrap();
    // The apex height should largely survive (within 1 unit) — the metric
    // protects the high-curvature peak.
    assert!(
        bb_after.max[2] > bb_before.max[2] - 1.2,
        "apex height {} dropped too far from {}",
        bb_after.max[2],
        bb_before.max[2]
    );
}

// --- error-bounded variant -------------------------------------------

#[test]
fn error_bound_zero_only_collapses_planar() {
    // A flat grid: every collapse is zero-error, so a 0.0 bound still
    // reduces it (coplanar merges change nothing geometrically). The
    // result must stay perfectly planar.
    let g = grid(6); // 72 tris, flat
    let out = g.simplify_quadric_error(0.0);
    assert!(out.triangle_count() < 72, "planar grid should reduce");
    for p in &out.positions {
        assert!(p[2].abs() < 1e-4, "stayed flat, z={}", p[2]);
    }
}

#[test]
fn error_bound_zero_preserves_curved_features() {
    // A grid with a raised apex: a 0.0 error bound must NOT collapse the
    // apex (that costs > 0), so the peak height is preserved exactly.
    let n = 6;
    let mut g = grid(n);
    let row = n + 1;
    let apex = n / 2 * row + n / 2;
    g.positions[apex][2] = 3.0;
    let out = g.simplify_quadric_error(0.0);
    let bb = out.bounding_box().unwrap();
    assert!(
        (bb.max[2] - 3.0).abs() < 1e-3,
        "zero-error bound must keep the apex, got {}",
        bb.max[2]
    );
}

#[test]
fn error_bound_monotone_in_budget() {
    // Larger error budgets never yield MORE triangles than smaller ones.
    let n = 8;
    let mut g = grid(n);
    // A gentle bump so collapses carry a range of costs.
    let row = n + 1;
    for j in 0..=n {
        for i in 0..=n {
            let dx = i as f32 - n as f32 / 2.0;
            let dy = j as f32 - n as f32 / 2.0;
            g.positions[j * row + i][2] = (-(dx * dx + dy * dy) / 8.0).exp();
        }
    }
    let mut last = usize::MAX;
    for &budget in &[0.0_f64, 0.001, 0.01, 0.1, 1.0, 100.0] {
        let c = g.simplify_quadric_error(budget).triangle_count();
        assert!(
            c <= last,
            "budget {budget} gave {c} tris, more than the previous {last}"
        );
        last = c;
    }
}

#[test]
fn error_bound_infinite_matches_target_zero() {
    // An unbounded error budget reduces exactly as far as the legality
    // guards allow — the same floor as simplify_quadric(0).
    let oct = octahedron();
    let a = oct.simplify_quadric_error(f64::INFINITY).triangle_count();
    let b = oct.simplify_quadric(0).triangle_count();
    assert_eq!(a, b);

    // Negative / NaN budgets are treated as "no bound".
    let c = oct.simplify_quadric_error(-1.0).triangle_count();
    let d = oct.simplify_quadric_error(f64::NAN).triangle_count();
    assert_eq!(c, a);
    assert_eq!(d, a);
}

#[test]
fn error_bound_empty_input_yields_empty() {
    let empty = Primitive::new(Topology::Triangles);
    assert_eq!(empty.simplify_quadric_error(1.0).triangle_count(), 0);
}

#[test]
fn error_bound_output_well_formed() {
    let g = grid(6);
    let out = g.simplify_quadric_error(0.5);
    let n = out.positions.len() as u32;
    for [a, b, c] in out.triangle_indices() {
        assert!(a < n && b < n && c < n);
        assert!(a != b && b != c && a != c);
    }
}
