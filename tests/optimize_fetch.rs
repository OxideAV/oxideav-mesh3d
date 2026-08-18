//! Tests for `Primitive::optimize_vertex_fetch` — vertex-pool
//! linearisation into index-stream first-appearance order.
//!
//! The pass relabels the vertex pool; the rendered geometry (the set of
//! drawn vertex *tuples*, resolved through the index buffer) is invariant.

use oxideav_mesh3d::{Indices, MorphTarget, Primitive, Topology};

// --- helpers ---------------------------------------------------------

fn idx_vec(p: &Primitive) -> Vec<u32> {
    match &p.indices {
        Some(Indices::U32(v)) => v.clone(),
        Some(Indices::U16(v)) => v.iter().map(|&x| x as u32).collect(),
        None => (0..p.positions.len() as u32).collect(),
    }
}

/// Resolve the drawn position tuples (one per index), so two pool
/// orderings of the same draw compare equal.
fn drawn_positions(p: &Primitive) -> Vec<[f32; 3]> {
    idx_vec(p)
        .iter()
        .map(|&i| p.positions[i as usize])
        .collect()
}

/// "Fetch scatter" metric: sum of absolute index deltas between
/// consecutive references. A perfectly linear first-touch order keeps
/// this small; a scattered order inflates it.
fn fetch_scatter(idx: &[u32]) -> u64 {
    idx.windows(2)
        .map(|w| (w[0] as i64 - w[1] as i64).unsigned_abs())
        .sum()
}

fn pos(x: f32) -> [f32; 3] {
    [x, 0.0, 0.0]
}

// --- basic relabel ---------------------------------------------------

#[test]
fn relabels_into_first_use_order() {
    // Pool 0..5, but the draw touches them as 4,2,0 then 2,0,5.
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = (0..6).map(|i| pos(i as f32)).collect();
    prim.indices = Some(Indices::U32(vec![4, 2, 0, 2, 0, 5]));
    let opt = prim.optimize_vertex_fetch();

    // First-touch order is 4,2,0,5 -> output slots 0,1,2,3. Unreferenced
    // 1 and 3 append after, original order: slots 4,5.
    // New index stream: 0,1,2,1,2,3.
    assert_eq!(idx_vec(&opt), vec![0, 1, 2, 1, 2, 3]);
    // Slot 0 must hold old vertex 4's position.
    assert_eq!(opt.positions[0], pos(4.0));
    assert_eq!(opt.positions[1], pos(2.0));
    assert_eq!(opt.positions[2], pos(0.0));
    assert_eq!(opt.positions[3], pos(5.0));
    // Drawn geometry unchanged.
    assert_eq!(drawn_positions(&opt), drawn_positions(&prim));
}

#[test]
fn preserves_unreferenced_vertices() {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = (0..5).map(|i| pos(i as f32)).collect();
    prim.indices = Some(Indices::U32(vec![0, 1, 2]));
    let opt = prim.optimize_vertex_fetch();
    // All 5 positions survive (3 referenced + 2 appended).
    assert_eq!(opt.positions.len(), 5);
    let set: Vec<f32> = opt.positions.iter().map(|p| p[0]).collect();
    for v in 0..5 {
        assert!(set.contains(&(v as f32)), "vertex {v} dropped");
    }
}

#[test]
fn already_linear_is_identity_pool() {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = (0..6).map(|i| pos(i as f32)).collect();
    prim.indices = Some(Indices::U32(vec![0, 1, 2, 3, 4, 5]));
    let opt = prim.optimize_vertex_fetch();
    assert_eq!(opt.positions, prim.positions);
    assert_eq!(idx_vec(&opt), vec![0, 1, 2, 3, 4, 5]);
}

// --- attribute carry -------------------------------------------------

#[test]
fn carries_all_attributes() {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![pos(0.0), pos(1.0), pos(2.0)];
    prim.normals = Some(vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    prim.tangents = Some(vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]);
    prim.uvs = vec![vec![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]];
    prim.colors = vec![vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]];
    prim.joints = Some(vec![[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]);
    prim.weights = Some(vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ]);
    prim.targets = vec![MorphTarget::with_deltas(
        Some(vec![[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        None,
        None,
    )];
    // Draw touches 2,0,1.
    prim.indices = Some(Indices::U32(vec![2, 0, 1]));
    let opt = prim.optimize_vertex_fetch();

    // Output slot 0 = old vertex 2; every attribute must move together.
    assert_eq!(opt.normals.as_ref().unwrap()[0], [0.0, 0.0, 1.0]);
    assert_eq!(opt.tangents.as_ref().unwrap()[0], [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(opt.uvs[0][0], [1.0, 1.0]);
    assert_eq!(opt.colors[0][0], [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(opt.joints.as_ref().unwrap()[0], [8, 9, 10, 11]);
    assert_eq!(opt.weights.as_ref().unwrap()[0], [0.25, 0.25, 0.25, 0.25]);
    assert_eq!(
        opt.targets[0].position.as_ref().unwrap()[0],
        [0.3, 0.0, 0.0]
    );
}

// --- locality improvement --------------------------------------------

#[test]
fn improves_fetch_scatter() {
    // A draw whose first-touch order is badly scattered across a large
    // pool; relabelling must collapse the scatter.
    let mut prim = Primitive::new(Topology::Triangles);
    let n = 300u32;
    prim.positions = (0..n).map(|i| pos(i as f32)).collect();
    // Reference vertices in a scattered pattern.
    let mut idx = Vec::new();
    for k in 0..(n / 3) {
        let a = (k * 97) % n;
        let b = (k * 53 + 7) % n;
        let c = (k * 31 + 13) % n;
        idx.extend_from_slice(&[a, b, c]);
    }
    prim.indices = Some(Indices::U32(idx));
    let before = fetch_scatter(&idx_vec(&prim));
    let opt = prim.optimize_vertex_fetch();
    let after = fetch_scatter(&idx_vec(&opt));
    assert!(
        after < before,
        "fetch scatter should drop: {before} -> {after}"
    );
    // Drawn geometry invariant.
    assert_eq!(drawn_positions(&opt), drawn_positions(&prim));
}

// --- topology + edge cases -------------------------------------------

#[test]
fn keeps_topology_for_strip() {
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = (0..5).map(|i| pos(i as f32)).collect();
    strip.indices = Some(Indices::U32(vec![2, 3, 4, 0, 1]));
    let opt = strip.optimize_vertex_fetch();
    assert_eq!(opt.topology, Topology::TriangleStrip);
    // First-touch 2,3,4,0,1 -> 0,1,2,3,4; stream relabels to 0,1,2,3,4.
    assert_eq!(idx_vec(&opt), vec![0, 1, 2, 3, 4]);
    assert_eq!(opt.positions[0], pos(2.0));
}

#[test]
fn non_indexed_materialises_explicit_order() {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = (0..6).map(|i| pos(i as f32)).collect();
    // No index buffer: implicit 0..6.
    let opt = prim.optimize_vertex_fetch();
    assert_eq!(idx_vec(&opt), vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(opt.positions, prim.positions);
}

#[test]
fn drops_out_of_range_references() {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = (0..3).map(|i| pos(i as f32)).collect();
    prim.indices = Some(Indices::U32(vec![0, 1, 2, 9, 9, 9]));
    let opt = prim.optimize_vertex_fetch();
    // Out-of-range triangle dropped; valid one relabelled.
    assert_eq!(idx_vec(&opt), vec![0, 1, 2]);
    assert_eq!(opt.positions.len(), 3);
}

#[test]
fn empty_primitive_roundtrips() {
    let prim = Primitive::new(Topology::Triangles);
    let opt = prim.optimize_vertex_fetch();
    assert!(opt.positions.is_empty());
    assert!(idx_vec(&opt).is_empty());
}

#[test]
fn is_deterministic() {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = (0..50).map(|i| pos(i as f32)).collect();
    let mut idx = Vec::new();
    for k in 0..16u32 {
        idx.extend_from_slice(&[(k * 7) % 50, (k * 3 + 1) % 50, (k * 11 + 2) % 50]);
    }
    prim.indices = Some(Indices::U32(idx));
    let a = idx_vec(&prim.optimize_vertex_fetch());
    let b = idx_vec(&prim.optimize_vertex_fetch());
    assert_eq!(a, b);
}

// --- pipeline composition: cache then fetch --------------------------

#[test]
fn cache_then_fetch_is_lossless_and_local() {
    // Build a scrambled grid, run both passes, verify the rendered
    // triangle set is invariant and the fetch order is near-linear.
    let mut prim = Primitive::new(Topology::Triangles);
    let (w, h) = (12u32, 12u32);
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
    // Reverse the triangle stream to scramble draw order.
    let tris: Vec<[u32; 3]> = idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let scrambled: Vec<u32> = tris.iter().rev().flat_map(|t| t.iter().copied()).collect();
    prim.indices = Some(Indices::U32(scrambled));

    // Reference resolved tuples before optimisation.
    let want = {
        let mut v: Vec<[u32; 3]> = idx_vec(&prim)
            .chunks_exact(3)
            .map(|c| {
                // resolve to position-tuple identity via position x,y
                let p0 = prim.positions[c[0] as usize];
                let p1 = prim.positions[c[1] as usize];
                let p2 = prim.positions[c[2] as usize];
                let mut t = [
                    (p0[0] as u32) * 100 + p0[1] as u32,
                    (p1[0] as u32) * 100 + p1[1] as u32,
                    (p2[0] as u32) * 100 + p2[1] as u32,
                ];
                t.sort_unstable();
                t
            })
            .collect();
        v.sort_unstable();
        v
    };

    let opt = prim.optimize_vertex_cache().optimize_vertex_fetch();

    let got = {
        let mut v: Vec<[u32; 3]> = idx_vec(&opt)
            .chunks_exact(3)
            .map(|c| {
                let p0 = opt.positions[c[0] as usize];
                let p1 = opt.positions[c[1] as usize];
                let p2 = opt.positions[c[2] as usize];
                let mut t = [
                    (p0[0] as u32) * 100 + p0[1] as u32,
                    (p1[0] as u32) * 100 + p1[1] as u32,
                    (p2[0] as u32) * 100 + p2[1] as u32,
                ];
                t.sort_unstable();
                t
            })
            .collect();
        v.sort_unstable();
        v
    };

    assert_eq!(want, got, "geometry must survive the full pipeline");

    // After fetch optimisation the first reference to each vertex is in
    // increasing order, so the max index seen grows monotonically as we
    // walk: every new (first-seen) index equals the running high-water
    // mark + 1.
    let mut hi: i64 = -1;
    for &i in &idx_vec(&opt) {
        let i = i as i64;
        if i > hi {
            assert_eq!(i, hi + 1, "first-use order must be contiguous");
            hi = i;
        }
    }
}
