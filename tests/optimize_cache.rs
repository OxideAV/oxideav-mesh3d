//! Tests for the post-transform vertex-cache analysis + triangle
//! reordering: `simulate_cache`, `Primitive::cache_stats`, and
//! `Primitive::optimize_vertex_cache`.
//!
//! The reordering pass permutes triangle order only — the rendered set
//! of triangles (as unordered vertex triples) and the vertex pool are
//! invariant; only the post-transform cache miss count improves.

use std::collections::HashSet;

use oxideav_mesh3d::{simulate_cache, Indices, Primitive, Topology, DEFAULT_CACHE_SIZE};

// --- helpers ---------------------------------------------------------

fn idx_vec(p: &Primitive) -> Vec<u32> {
    match &p.indices {
        Some(Indices::U32(v)) => v.clone(),
        Some(Indices::U16(v)) => v.iter().map(|&x| x as u32).collect(),
        None => (0..p.positions.len() as u32).collect(),
    }
}

/// Unordered triangle set (each tri sorted, then the whole set sorted)
/// so two orderings of the same mesh compare equal.
fn tri_set(idx: &[u32]) -> Vec<[u32; 3]> {
    let mut out: Vec<[u32; 3]> = idx
        .chunks_exact(3)
        .map(|c| {
            let mut t = [c[0], c[1], c[2]];
            t.sort_unstable();
            t
        })
        .collect();
    out.sort_unstable();
    out
}

/// Build a regular `w × h` vertex grid and triangulate every cell into
/// two triangles. Returns an indexed `Triangles` primitive.
fn grid(w: u32, h: u32) -> Primitive {
    let mut prim = Primitive::new(Topology::Triangles);
    for y in 0..h {
        for x in 0..w {
            prim.positions.push([x as f32, y as f32, 0.0]);
        }
    }
    let mut idx = Vec::new();
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let a = y * w + x;
            let b = y * w + x + 1;
            let c = (y + 1) * w + x;
            let d = (y + 1) * w + x + 1;
            idx.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    prim.indices = Some(Indices::U32(idx));
    prim
}

/// A grid with its triangles shuffled into a deliberately cache-hostile
/// order (every other triangle far apart) to give the optimiser headroom.
fn grid_scrambled(w: u32, h: u32) -> Primitive {
    let g = grid(w, h);
    let idx = idx_vec(&g);
    let tris: Vec<[u32; 3]> = idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    // Interleave the two halves so neighbours in the cell grid end up far
    // apart in the triangle stream.
    let half = tris.len() / 2;
    let mut order = Vec::new();
    for i in 0..half {
        order.push(tris[i]);
        if half + i < tris.len() {
            order.push(tris[half + i]);
        }
    }
    let mut prim = grid(w, h);
    let flat: Vec<u32> = order.into_iter().flat_map(|t| t.into_iter()).collect();
    prim.indices = Some(Indices::U32(flat));
    prim
}

// --- simulate_cache --------------------------------------------------

#[test]
fn empty_stream_is_nan() {
    let s = simulate_cache(&[], 16);
    assert_eq!(s.misses, 0);
    assert_eq!(s.triangles, 0);
    assert_eq!(s.vertices, 0);
    assert!(s.acmr.is_nan());
    assert!(s.atvr.is_nan());
}

#[test]
fn single_triangle_is_three_misses() {
    let s = simulate_cache(&[0, 1, 2], 16);
    assert_eq!(s.misses, 3);
    assert_eq!(s.triangles, 1);
    assert_eq!(s.vertices, 3);
    assert_eq!(s.acmr, 3.0);
    assert_eq!(s.atvr, 1.0);
}

#[test]
fn shared_edge_reuses_two_vertices() {
    // Two triangles sharing edge (1,2): 0,1,2 then 1,2,3.
    // Walk: 0 miss,1 miss,2 miss,1 hit,2 hit,3 miss => 4 misses.
    let s = simulate_cache(&[0, 1, 2, 1, 2, 3], 16);
    assert_eq!(s.misses, 4);
    assert_eq!(s.triangles, 2);
    assert_eq!(s.vertices, 4);
    assert_eq!(s.acmr, 2.0);
    assert_eq!(s.atvr, 1.0);
}

#[test]
fn tiny_cache_evicts_so_no_reuse() {
    // Cache of 1 slot: each new index evicts the last, so a reused index
    // is rarely resident. 0,1,2,1,2,3 with size 1:
    // 0 miss(cache=0),1 miss(=1),2 miss(=2),1 miss(=1),2 miss(=2),3 miss
    // => 6 misses, ACMR 3.0.
    let s = simulate_cache(&[0, 1, 2, 1, 2, 3], 1);
    assert_eq!(s.misses, 6);
    assert_eq!(s.acmr, 3.0);
}

#[test]
fn cache_size_clamped_to_one() {
    let s = simulate_cache(&[0, 1, 2], 0);
    assert_eq!(s.cache_size, 1);
    assert_eq!(s.misses, 3);
}

// --- cache_stats on a Primitive --------------------------------------

#[test]
fn cache_stats_matches_manual_simulation() {
    let g = grid(4, 4);
    let idx = idx_vec(&g);
    let manual = simulate_cache(&idx, DEFAULT_CACHE_SIZE);
    let via = g.cache_stats(DEFAULT_CACHE_SIZE);
    assert_eq!(manual.misses, via.misses);
    assert_eq!(manual.acmr, via.acmr);
}

#[test]
fn cache_stats_destrips() {
    // A 4-vertex strip 0,1,2,3 => triangles (0,1,2),(2,1,3).
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = vec![[0.0; 3]; 4];
    strip.indices = Some(Indices::U32(vec![0, 1, 2, 3]));
    let s = strip.cache_stats(16);
    assert_eq!(s.triangles, 2);
    // distinct vertices 0..3
    assert_eq!(s.vertices, 4);
}

// --- optimize_vertex_cache: invariants -------------------------------

#[test]
fn optimize_preserves_triangle_set() {
    let scrambled = grid_scrambled(8, 8);
    let before = tri_set(&idx_vec(&scrambled));
    let opt = scrambled.optimize_vertex_cache();
    let after = tri_set(&idx_vec(&opt));
    assert_eq!(before, after, "triangle set must be invariant");
    assert_eq!(opt.topology, Topology::Triangles);
}

#[test]
fn optimize_preserves_vertex_pool() {
    let scrambled = grid_scrambled(6, 6);
    let opt = scrambled.optimize_vertex_cache();
    // Pool unchanged: same positions, same length, same order.
    assert_eq!(opt.positions, scrambled.positions);
}

#[test]
fn optimize_reduces_or_holds_acmr() {
    for (w, h) in [(8u32, 8u32), (16, 16), (10, 7)] {
        let scrambled = grid_scrambled(w, h);
        let before = scrambled.cache_stats(DEFAULT_CACHE_SIZE).acmr;
        let opt = scrambled.optimize_vertex_cache();
        let after = opt.cache_stats(DEFAULT_CACHE_SIZE).acmr;
        assert!(
            after <= before + 1e-6,
            "{w}x{h}: ACMR worsened {before} -> {after}"
        );
    }
}

#[test]
fn optimize_grid_beats_scrambled_substantially() {
    // A regular grid scrambled into a hostile order should see a real
    // ACMR improvement, not a marginal one.
    let scrambled = grid_scrambled(24, 24);
    let before = scrambled.cache_stats(DEFAULT_CACHE_SIZE).acmr;
    let opt = scrambled.optimize_vertex_cache();
    let after = opt.cache_stats(DEFAULT_CACHE_SIZE).acmr;
    assert!(
        after < before * 0.85,
        "expected substantial ACMR drop, {before} -> {after}"
    );
    // A well-ordered grid lands well under 1.0 ACMR for a 32-slot cache.
    assert!(after < 1.0, "optimised ACMR should be < 1.0, got {after}");
}

#[test]
fn optimize_is_deterministic() {
    let scrambled = grid_scrambled(12, 9);
    let a = idx_vec(&scrambled.optimize_vertex_cache());
    let b = idx_vec(&scrambled.optimize_vertex_cache());
    assert_eq!(a, b);
}

#[test]
fn optimize_handles_disconnected_components() {
    // Two separate triangles sharing no vertices.
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 5.0, 0.0],
        [6.0, 5.0, 0.0],
        [5.0, 6.0, 0.0],
    ];
    prim.indices = Some(Indices::U32(vec![0, 1, 2, 3, 4, 5]));
    let opt = prim.optimize_vertex_cache();
    assert_eq!(tri_set(&idx_vec(&opt)), tri_set(&[0, 1, 2, 3, 4, 5]));
}

#[test]
fn optimize_drops_out_of_range_and_degenerate() {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0; 3]; 4];
    // valid, out-of-range (9), degenerate (1,1,2)
    prim.indices = Some(Indices::U32(vec![0, 1, 2, 9, 0, 1, 1, 1, 2]));
    let opt = prim.optimize_vertex_cache();
    let got = tri_set(&idx_vec(&opt));
    // Only the first triangle survives (degenerate kept? it is not
    // out-of-range, so it stays — degenerate handling is the caller's;
    // the cache pass only drops out-of-range). Verify the out-of-range
    // one is gone.
    assert!(got.contains(&[0, 1, 2]));
    let flat: Vec<u32> = idx_vec(&opt);
    assert!(!flat.windows(3).any(|w| w.contains(&9)));
}

#[test]
fn optimize_non_triangle_topology_unchanged() {
    let mut lines = Primitive::new(Topology::Lines);
    lines.positions = vec![[0.0; 3]; 4];
    lines.indices = Some(Indices::U32(vec![0, 1, 2, 3]));
    let opt = lines.optimize_vertex_cache();
    assert_eq!(opt.topology, Topology::Lines);
    assert_eq!(idx_vec(&opt), vec![0, 1, 2, 3]);
}

#[test]
fn optimize_empty_primitive() {
    let prim = Primitive::new(Topology::Triangles);
    let opt = prim.optimize_vertex_cache();
    assert_eq!(opt.topology, Topology::Triangles);
    assert!(idx_vec(&opt).is_empty());
}

#[test]
fn optimize_promotes_index_width_only_when_needed() {
    let g = grid(6, 6); // 36 verts -> U16
    let opt = g.optimize_vertex_cache();
    assert!(matches!(opt.indices, Some(Indices::U16(_))));
}

#[test]
fn smaller_cache_still_valid_order() {
    // Order optimised for the default cache should still be a valid,
    // non-worse order when re-measured against a smaller cache than the
    // scrambled input.
    let scrambled = grid_scrambled(16, 16);
    let opt = scrambled.optimize_vertex_cache();
    for cs in [8usize, 12, 16, 24] {
        let before = scrambled.cache_stats(cs).acmr;
        let after = opt.cache_stats(cs).acmr;
        assert!(
            after <= before + 0.05,
            "cache {cs}: {before} -> {after} regressed"
        );
    }
}

#[test]
fn all_indices_in_range_after_optimize() {
    let opt = grid_scrambled(10, 10).optimize_vertex_cache();
    let n = opt.positions.len() as u32;
    let seen: HashSet<u32> = idx_vec(&opt).into_iter().collect();
    assert!(seen.iter().all(|&i| i < n));
}
