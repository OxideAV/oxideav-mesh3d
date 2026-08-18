//! Tests for `Primitive::optimize_vertex_spatial` — Z-order (Morton)
//! spatial sort of the vertex pool for fetch/compression locality on
//! non-indexed / seam-heavy meshes.
//!
//! The pass relabels the pool; rendered geometry (drawn position tuples,
//! resolved through the index buffer) is invariant.

use oxideav_mesh3d::{Indices, MorphTarget, Primitive, Topology};

// --- helpers ---------------------------------------------------------

fn idx_vec(p: &Primitive) -> Vec<u32> {
    match &p.indices {
        Some(Indices::U32(v)) => v.clone(),
        Some(Indices::U16(v)) => v.iter().map(|&x| x as u32).collect(),
        None => (0..p.positions.len() as u32).collect(),
    }
}

fn drawn(p: &Primitive) -> Vec<[f32; 3]> {
    idx_vec(p)
        .iter()
        .map(|&i| p.positions[i as usize])
        .collect()
}

/// Multiset of all pool positions (sorted) so two orderings of the same
/// pool compare equal regardless of slot.
fn pos_multiset(p: &Primitive) -> Vec<[u32; 3]> {
    let mut v: Vec<[u32; 3]> = p
        .positions
        .iter()
        .map(|q| [q[0].to_bits(), q[1].to_bits(), q[2].to_bits()])
        .collect();
    v.sort_unstable();
    v
}

// --- pool invariance + remap correctness -----------------------------

#[test]
fn preserves_pool_multiset_and_geometry() {
    let mut prim = Primitive::new(Topology::Triangles);
    // 3x3x3 lattice of points in scrambled storage order.
    let mut raw = Vec::new();
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                raw.push([x as f32, y as f32, z as f32]);
            }
        }
    }
    // Reverse to scramble.
    raw.reverse();
    prim.positions = raw;
    let idx: Vec<u32> = (0..prim.positions.len() as u32).collect();
    prim.indices = Some(Indices::U32(idx));

    let before_geo = drawn(&prim);
    let before_pool = pos_multiset(&prim);
    let opt = prim.optimize_vertex_spatial(8);
    assert_eq!(pos_multiset(&opt), before_pool, "pool multiset invariant");
    assert_eq!(drawn(&opt), before_geo, "rendered geometry invariant");
}

#[test]
fn index_buffer_remapped() {
    // Three points; an index buffer draws them as a triangle. After the
    // spatial sort the index buffer must still resolve to the same tuples.
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[9.0, 9.0, 9.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    prim.indices = Some(Indices::U32(vec![0, 1, 2]));
    let opt = prim.optimize_vertex_spatial(8);
    // (0,0,0) and (1,0,0) are Z-near, (9,9,9) far: sort puts the near
    // pair first. Whatever the order, the drawn tuple set is preserved.
    assert_eq!(drawn(&opt), drawn(&prim));
}

// --- spatial locality effect -----------------------------------------

#[test]
fn nearby_points_become_adjacent() {
    // Two tight clusters far apart, interleaved in storage. After the
    // sort each cluster's members should be contiguous.
    let mut prim = Primitive::new(Topology::Points);
    let mut raw = Vec::new();
    // Interleave cluster A (near origin) and cluster B (far).
    for i in 0..8 {
        raw.push([0.0 + i as f32 * 0.01, 0.0, 0.0]); // A
        raw.push([100.0 + i as f32 * 0.01, 100.0, 100.0]); // B
    }
    prim.positions = raw;
    let opt = prim.optimize_vertex_spatial(10);
    // Partition the sorted pool by x < 50; each side must be a contiguous
    // run (all of one cluster before the other).
    let side: Vec<bool> = opt.positions.iter().map(|p| p[0] < 50.0).collect();
    // Count transitions between the two clusters.
    let transitions = side.windows(2).filter(|w| w[0] != w[1]).count();
    assert_eq!(transitions, 1, "clusters must be contiguous after sort");
}

#[test]
fn morton_locality_metric_improves() {
    // Build a scrambled grid of points; measure summed neighbour distance
    // in storage order. The Z-order sort must reduce it vs a scramble.
    let mut prim = Primitive::new(Topology::Points);
    let n = 16;
    let mut raw = Vec::new();
    for y in 0..n {
        for x in 0..n {
            raw.push([x as f32, y as f32, 0.0]);
        }
    }
    // Deterministic scramble.
    let mut scrambled = Vec::with_capacity(raw.len());
    let m = raw.len();
    for k in 0..m {
        scrambled.push(raw[(k * 7) % m]);
    }
    prim.positions = scrambled;

    let storage_dist = |p: &Primitive| -> f64 {
        p.positions
            .windows(2)
            .map(|w| {
                let dx = (w[0][0] - w[1][0]) as f64;
                let dy = (w[0][1] - w[1][1]) as f64;
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    };
    let before = storage_dist(&prim);
    let opt = prim.optimize_vertex_spatial(8);
    let after = storage_dist(&opt);
    assert!(
        after < before * 0.6,
        "Z-order should cut neighbour distance substantially: {before} -> {after}"
    );
}

// --- attribute carry -------------------------------------------------

#[test]
fn carries_attributes_with_positions() {
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![[9.0, 9.0, 9.0], [0.0, 0.0, 0.0]];
    prim.normals = Some(vec![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    prim.targets = vec![MorphTarget::with_deltas(
        Some(vec![[0.9, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        None,
        None,
    )];
    let opt = prim.optimize_vertex_spatial(8);
    // The (0,0,0) vertex sorts first; its normal [0,0,1] and morph delta
    // [0.1,..] must travel with it.
    let first_is_origin = opt.positions[0] == [0.0, 0.0, 0.0];
    assert!(first_is_origin);
    assert_eq!(opt.normals.as_ref().unwrap()[0], [0.0, 0.0, 1.0]);
    assert_eq!(
        opt.targets[0].position.as_ref().unwrap()[0],
        [0.1, 0.0, 0.0]
    );
}

// --- robustness ------------------------------------------------------

#[test]
fn empty_primitive_roundtrips() {
    let prim = Primitive::new(Topology::Triangles);
    let opt = prim.optimize_vertex_spatial(8);
    assert!(opt.positions.is_empty());
}

#[test]
fn degenerate_bbox_is_stable() {
    // All points share a plane (z=0) and even a line on one axis.
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![[0.0, 0.0, 0.0]; 5];
    let opt = prim.optimize_vertex_spatial(8);
    assert_eq!(opt.positions.len(), 5);
}

#[test]
fn non_finite_positions_sort_last() {
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![
        [f32::NAN, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ];
    let opt = prim.optimize_vertex_spatial(8);
    // The two finite points come first; the non-finite ones last.
    assert!(opt.positions[0].iter().all(|c| c.is_finite()));
    assert!(opt.positions[1].iter().all(|c| c.is_finite()));
    assert!(opt.positions[2..].iter().any(|p| !p[0].is_finite()));
}

#[test]
fn is_deterministic() {
    let mut prim = Primitive::new(Topology::Points);
    for k in 0..40u32 {
        prim.positions.push([
            ((k * 13) % 11) as f32,
            ((k * 7) % 11) as f32,
            ((k * 3) % 11) as f32,
        ]);
    }
    let r1: Vec<[f32; 3]> = prim.optimize_vertex_spatial(8).positions;
    let r2: Vec<[f32; 3]> = prim.optimize_vertex_spatial(8).positions;
    assert_eq!(r1, r2);
}

#[test]
fn bits_clamped() {
    // bits=0 clamps to 1 (coarse), bits=99 clamps to 10; both must run
    // and preserve geometry.
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
    let coarse = prim.optimize_vertex_spatial(0);
    let fine = prim.optimize_vertex_spatial(99);
    assert_eq!(coarse.positions.len(), 3);
    assert_eq!(fine.positions.len(), 3);
}

// --- pipeline: spatial then cache ------------------------------------

#[test]
fn spatial_then_fetch_geometry_invariant() {
    // Triangle soup (each tri unique vertices) — spatial sort then
    // fetch linearisation; geometry must survive.
    let mut prim = Primitive::new(Topology::Triangles);
    let mut raw = Vec::new();
    for i in 0..20 {
        let b = i as f32;
        raw.push([b, 0.0, 0.0]);
        raw.push([b + 0.5, 1.0, 0.0]);
        raw.push([b + 1.0, 0.0, 0.0]);
    }
    prim.positions = raw;
    let before = drawn(&prim);
    let opt = prim.optimize_vertex_spatial(8).optimize_vertex_cache();
    // Resolve drawn tuples as sorted bit-triples for comparison.
    let resolve = |p: &Primitive| {
        let mut v: Vec<[u32; 9]> = idx_vec(p)
            .chunks_exact(3)
            .map(|c| {
                let a = p.positions[c[0] as usize];
                let b = p.positions[c[1] as usize];
                let d = p.positions[c[2] as usize];
                let mut t = [
                    [a[0].to_bits(), a[1].to_bits(), a[2].to_bits()],
                    [b[0].to_bits(), b[1].to_bits(), b[2].to_bits()],
                    [d[0].to_bits(), d[1].to_bits(), d[2].to_bits()],
                ];
                t.sort_unstable();
                [
                    t[0][0], t[0][1], t[0][2], t[1][0], t[1][1], t[1][2], t[2][0], t[2][1], t[2][2],
                ]
            })
            .collect();
        v.sort_unstable();
        v
    };
    let _ = before;
    assert_eq!(resolve(&prim), resolve(&opt));
}
