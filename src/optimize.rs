//! GPU vertex-buffer optimisation: triangle and vertex reordering for
//! locality of reference.
//!
//! A decoded mesh draws correctly regardless of the order its triangles
//! and vertices sit in memory, but a real-time renderer pays for that
//! order twice over:
//!
//! * **Post-transform vertex cache.** A GPU keeps a small cache of the
//!   most recently *transformed* vertices (the output of the vertex
//!   shader, keyed by index). When a triangle references an index still
//!   resident in that cache, the transform result is reused for free;
//!   otherwise the vertex shader runs again. Re-ordering the *triangles*
//!   so that consecutive triangles share vertices maximises that reuse.
//! * **Pre-transform vertex fetch.** The vertex shader reads each
//!   attribute stream from memory by index. If the index buffer hops all
//!   over the vertex buffer, those reads scatter across cache lines.
//!   Re-ordering the *vertices* into the order the (cache-optimised)
//!   index buffer first touches them makes the fetch sweep the vertex
//!   buffer roughly front-to-back.
//!
//! Both transforms preserve the rendered result exactly — they permute
//! storage, never geometry. The same two passes are also the recommended
//! pre-processing for index/vertex stream compression
//! (`KHR_meshopt_compression`, "Compressing geometry data"): triangle
//! order maximises vertex-reference recency, vertex order linearises the
//! index stream.
//!
//! This module operates on the de-stripped triangle list of a
//! [`Primitive`] (via [`Primitive::triangle_indices`]) and is therefore
//! topology-aware: a `TriangleStrip` / `TriangleFan` input is flattened
//! to an indexed `Triangles` primitive on the way out. Non-triangle
//! topologies (lines / points) have no post-transform reuse to exploit
//! and are returned unchanged by the reordering passes; the analysis
//! metrics still simulate them as a flat index walk.

use crate::mesh::{Indices, MorphTarget, Primitive, Topology};

/// Result of simulating a fixed-size post-transform vertex cache over an
/// index stream.
///
/// The simulation walks the indices in draw order through a
/// first-in-first-out cache of `cache_size` slots: an index already
/// resident is a *hit* (its transform is reused), an index not resident
/// is a *miss* (the vertex shader re-runs and the new index evicts the
/// oldest slot). FIFO is the conservative model — real hardware caches
/// vary, but FIFO tracks the relative quality of two orderings the same
/// way and is the standard yardstick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheStats {
    /// Cache size used for the simulation (slots).
    pub cache_size: usize,
    /// Number of triangles walked.
    pub triangles: usize,
    /// Number of distinct vertices referenced by the stream.
    pub vertices: usize,
    /// Total cache misses (vertex-shader invocations).
    pub misses: usize,
    /// **Average Cache Miss Ratio** — `misses / triangles`. Ranges from
    /// a best case near `0.5` (every triangle but the first two shares
    /// two cached vertices with its neighbours) to a worst case of
    /// `3.0` (every index a miss). `f64::NAN` when there are no
    /// triangles.
    pub acmr: f64,
    /// **Average Transformed Vertex Ratio** — `misses / vertices`. The
    /// ideal is `1.0` (every vertex transformed exactly once); a value
    /// of `2.0` means the average vertex was transformed twice. Scale-
    /// free across mesh sizes, unlike ACMR which floors near `0.5`.
    /// `f64::NAN` when there are no vertices.
    pub atvr: f64,
}

/// Simulate a FIFO post-transform vertex cache over a raw index stream.
///
/// `cache_size` is clamped to at least `1`. An empty stream yields zero
/// misses and `NaN` ratios. This is the measurement primitive behind
/// [`Primitive::cache_stats`]; expose it directly for callers that hold
/// a bare index buffer (e.g. a format encoder comparing two candidate
/// orderings before serialisation).
pub fn simulate_cache(indices: &[u32], cache_size: usize) -> CacheStats {
    let cache_size = cache_size.max(1);
    // Ring buffer of resident indices. `usize::MAX` marks an empty slot.
    let mut cache = vec![u32::MAX; cache_size];
    let mut head = 0usize; // next slot to evict (oldest)
    let mut misses = 0usize;
    let mut any = false;

    for &idx in indices {
        any = true;
        if cache.contains(&idx) {
            continue; // hit — transform reused
        }
        // miss: re-transform and evict the oldest slot.
        misses += 1;
        cache[head] = idx;
        head = (head + 1) % cache_size;
    }

    // Distinct-vertex count for ATVR: count unique indices.
    let vertices = if any {
        let mut seen = std::collections::HashSet::new();
        for &idx in indices {
            seen.insert(idx);
        }
        seen.len()
    } else {
        0
    };

    let triangles = indices.len() / 3;
    let acmr = if triangles > 0 {
        misses as f64 / triangles as f64
    } else {
        f64::NAN
    };
    let atvr = if vertices > 0 {
        misses as f64 / vertices as f64
    } else {
        f64::NAN
    };

    CacheStats {
        cache_size,
        triangles,
        vertices,
        misses,
        acmr,
        atvr,
    }
}

/// The post-transform cache size assumed by [`Primitive::optimize_vertex_cache`]
/// and the default for [`Primitive::cache_stats`] when no explicit size is
/// given.
///
/// Real GPUs expose post-transform caches in the rough range of 12–32
/// entries; the optimiser's greedy scoring is tuned around a small,
/// representative size so the produced order is good across the whole
/// range rather than overfit to one cache. `32` is a conservative upper
/// estimate — an order good for a 32-slot cache stays good for smaller
/// ones (more reuse never hurts), so this is the safe default to optimise
/// against.
pub const DEFAULT_CACHE_SIZE: usize = 32;

impl Primitive {
    /// Simulate a post-transform vertex cache over this primitive's
    /// triangle list and report hit/miss statistics.
    ///
    /// `cache_size` is the number of cache slots; pass
    /// [`DEFAULT_CACHE_SIZE`] for the optimiser's reference size, or a
    /// smaller value (e.g. `16`) to model older hardware. The indices
    /// walked are the de-stripped triangle list
    /// ([`Primitive::triangle_indices`]), so strips / fans are evaluated
    /// in their expanded triangle order. A non-triangle topology walks
    /// its raw index / implicit order with three indices per "triangle"
    /// for the ratio denominators, which is rarely meaningful — the
    /// metric is intended for triangle meshes.
    ///
    /// Use this to *measure* the effect of [`optimize_vertex_cache`]:
    /// compute `cache_stats` before and after and compare `acmr` /
    /// `atvr` (lower is better for both).
    ///
    /// [`optimize_vertex_cache`]: Primitive::optimize_vertex_cache
    pub fn cache_stats(&self, cache_size: usize) -> CacheStats {
        let flat: Vec<u32> = match self.topology {
            Topology::Triangles | Topology::TriangleStrip | Topology::TriangleFan => self
                .triangle_indices()
                .into_iter()
                .flat_map(|t| t.into_iter())
                .collect(),
            _ => self.draw_index_stream(),
        };
        simulate_cache(&flat, cache_size)
    }

    /// The flat index stream this primitive draws, honouring an explicit
    /// index buffer or materialising the implicit `0..n` order.
    fn draw_index_stream(&self) -> Vec<u32> {
        match &self.indices {
            Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
            Some(Indices::U32(v)) => v.clone(),
            None => (0..self.positions.len() as u32).collect(),
        }
    }

    /// Reorder this primitive's triangles to minimise post-transform
    /// vertex-cache misses, returning an equivalent indexed `Triangles`
    /// primitive whose vertex pool is **unchanged** (same positions and
    /// attributes, same indices in the pool) but whose *triangle order*
    /// is locality-optimised.
    ///
    /// # Method
    ///
    /// A greedy, linear-time score-driven walk. Each not-yet-emitted
    /// triangle carries a live score equal to the sum of its three
    /// vertices' scores; a vertex's score combines two effects:
    ///
    /// * **Cache residency** — a vertex still resident in a simulated
    ///   cache scores higher the closer it is to the most-recently-used
    ///   end, so emitting a triangle that reuses hot vertices is
    ///   preferred. The three most-recent positions share the top score
    ///   (they survive emitting the next triangle, which itself pushes
    ///   three vertices), and the score falls off toward the cache tail.
    /// * **Valence (remaining use count)** — a vertex used by few
    ///   remaining triangles scores higher, so the walk clears such
    ///   vertices out before they fall from the cache and have to be
    ///   re-transformed later. This is the "low-valence first" bias that
    ///   keeps the frontier compact.
    ///
    /// The walk emits the best-scoring triangle, pushes its vertices to
    /// the front of the cache, and re-scores only the triangles incident
    /// to the (few) vertices whose cache position or valence changed —
    /// giving near-linear cost. When the cache and valence give no
    /// candidate (a fresh connected component), the next unemitted
    /// triangle in input order seeds the walk, so disconnected
    /// components are handled without special-casing.
    ///
    /// # Output
    ///
    /// * `topology` becomes [`Topology::Triangles`]; the index buffer is
    ///   the reordered triangle list. Index width follows glTF promotion
    ///   (`U16` while the pool fits, else `U32`).
    /// * All vertex buffers (`positions`, `normals`, `tangents`, every
    ///   `uvs` / `colors` set, `joints`, `weights`) and `targets` are
    ///   carried over **verbatim** — this pass never touches the pool,
    ///   only the order triangles reference it. Pair it with
    ///   [`optimize_vertex_fetch`] afterwards to also linearise the pool.
    /// * Degenerate / out-of-range triangles from the source are dropped
    ///   (they contribute no reuse and would corrupt the valence book-
    ///   keeping). A non-triangle topology is returned unchanged.
    /// * **Does not mutate `self`.**
    ///
    /// [`optimize_vertex_fetch`]: Primitive::optimize_vertex_fetch
    pub fn optimize_vertex_cache(&self) -> Primitive {
        self.optimize_vertex_cache_sized(DEFAULT_CACHE_SIZE)
    }

    /// [`optimize_vertex_cache`](Primitive::optimize_vertex_cache) with an
    /// explicit simulated cache size for the greedy scoring. The default
    /// entry point uses [`DEFAULT_CACHE_SIZE`]; override only to tune for
    /// a specific known target cache.
    pub fn optimize_vertex_cache_sized(&self, cache_size: usize) -> Primitive {
        if !matches!(
            self.topology,
            Topology::Triangles | Topology::TriangleStrip | Topology::TriangleFan
        ) {
            return self.clone();
        }
        let cache_size = cache_size.max(3);

        // De-stripped triangle list with out-of-range corners dropped.
        let vcount = self.positions.len() as u32;
        let tris: Vec<[u32; 3]> = self
            .triangle_indices()
            .into_iter()
            .filter(|t| t[0] < vcount && t[1] < vcount && t[2] < vcount)
            .collect();

        if tris.is_empty() {
            let mut out = self.clone();
            out.topology = Topology::Triangles;
            out.indices = Some(Indices::U32(Vec::new()));
            return out;
        }

        let order = greedy_cache_order(&tris, self.positions.len(), cache_size);

        let mut flat: Vec<u32> = Vec::with_capacity(order.len() * 3);
        for &ti in &order {
            flat.extend_from_slice(&tris[ti]);
        }

        let mut out = self.clone();
        out.topology = Topology::Triangles;
        out.indices = Some(pack_indices(flat, self.positions.len()));
        out
    }

    /// Reorder this primitive's **vertex pool** so vertices appear in the
    /// order the index buffer first references them, returning an
    /// equivalent primitive whose triangle order (the index sequence) is
    /// unchanged but whose vertex storage is linearised for fetch
    /// locality.
    ///
    /// # Why
    ///
    /// The vertex shader fetches each attribute stream from memory keyed
    /// by index. If the (possibly cache-optimised) index buffer first
    /// touches vertex 900, then 12, then 740, those reads scatter across
    /// the vertex buffer and waste pre-transform fetch bandwidth.
    /// Re-numbering the pool so the *k*-th distinct index encountered in
    /// draw order becomes vertex *k* makes the index stream reference a
    /// gently-increasing run, so the fetch sweeps the buffer roughly
    /// front-to-back. It is also the recommended companion to index-
    /// stream compression: a linearised vertex order minimises the
    /// residual the index codec must store.
    ///
    /// Run this **after** [`optimize_vertex_cache`] — the cache pass fixes
    /// the *triangle* order, then this pass relabels the pool to match
    /// that final draw order. Running it first is harmless but pointless,
    /// since the cache pass changes the draw order it linearises against.
    ///
    /// # Method
    ///
    /// Walk the draw index stream once. The first time each source index
    /// is seen it is assigned the next free output slot; subsequent
    /// references reuse that slot. The output index buffer is the source
    /// stream rewritten through this map; every attribute buffer
    /// (`positions`, `normals`, `tangents`, each `uvs` / `colors` set,
    /// `joints`, `weights`) and every [`MorphTarget`] delta is gathered
    /// into the new slot order. Source vertices that the index buffer
    /// never references are appended after the referenced ones in their
    /// original relative order, so no pool data is silently dropped and a
    /// re-index round-trip is lossless.
    ///
    /// # Output
    ///
    /// * The index buffer is preserved as a permutation-relabelled copy
    ///   of the source draw order (an implicit `0..n` order is
    ///   materialised). Index width follows glTF promotion.
    /// * `topology`, `material`, `targets` roster shape, and `extras`
    ///   carry over. The pass is valid for every topology — it relabels
    ///   the pool, never the stitching rule — so strips / fans keep their
    ///   topology (their index buffer, if any, is relabelled in place;
    ///   the implicit order of a non-indexed strip becomes explicit).
    /// * **Does not mutate `self`.**
    ///
    /// [`optimize_vertex_cache`]: Primitive::optimize_vertex_cache
    pub fn optimize_vertex_fetch(&self) -> Primitive {
        let n = self.positions.len();
        let stream = self.draw_index_stream();

        // First-use order: remap[source] = output slot, `perm` lists the
        // source index that fills each output slot.
        let mut remap = vec![u32::MAX; n];
        let mut perm: Vec<usize> = Vec::with_capacity(n);
        for &raw in &stream {
            let s = raw as usize;
            if s >= n {
                continue; // out-of-range reference: handled below
            }
            if remap[s] == u32::MAX {
                remap[s] = perm.len() as u32;
                perm.push(s);
            }
        }
        // Append any vertices the stream never touched, original order.
        for (s, slot) in remap.iter_mut().enumerate() {
            if *slot == u32::MAX {
                *slot = perm.len() as u32;
                perm.push(s);
            }
        }

        // Rewrite the index stream through the remap (out-of-range
        // references are dropped — they cannot survive a relabel).
        let new_stream: Vec<u32> = stream
            .iter()
            .filter_map(|&raw| {
                let s = raw as usize;
                if s < n {
                    Some(remap[s])
                } else {
                    None
                }
            })
            .collect();

        let mut out = gather_vertices(self, &perm);
        out.indices = Some(pack_indices(new_stream, perm.len()));
        out
    }
}

/// Gather every per-vertex attribute of `src` into the order given by
/// `perm` (output vertex `j` takes source vertex `perm[j]`), returning a
/// new primitive with the same topology / material / extras and the
/// reordered buffers. The index buffer is **not** set here — the caller
/// supplies the relabelled stream.
pub(crate) fn gather_vertices(src: &Primitive, perm: &[usize]) -> Primitive {
    let g3 = |b: &Vec<[f32; 3]>| -> Vec<[f32; 3]> {
        perm.iter()
            .map(|&i| b.get(i).copied().unwrap_or([0.0; 3]))
            .collect()
    };
    let g4 = |b: &Vec<[f32; 4]>| -> Vec<[f32; 4]> {
        perm.iter()
            .map(|&i| b.get(i).copied().unwrap_or([0.0; 4]))
            .collect()
    };

    let mut out = src.clone();
    out.positions = g3(&src.positions);
    out.normals = src.normals.as_ref().map(&g3);
    out.tangents = src.tangents.as_ref().map(&g4);
    out.uvs = src
        .uvs
        .iter()
        .map(|set| {
            perm.iter()
                .map(|&i| set.get(i).copied().unwrap_or([0.0; 2]))
                .collect()
        })
        .collect();
    out.colors = src.colors.iter().map(&g4).collect();
    out.joints = src.joints.as_ref().map(|s| {
        perm.iter()
            .map(|&i| s.get(i).copied().unwrap_or([0; 4]))
            .collect()
    });
    out.weights = src.weights.as_ref().map(&g4);
    out.targets = src.targets.iter().map(|t| permute_morph(t, perm)).collect();
    out
}

/// Carry a single `MorphTarget`'s present slots through a vertex
/// permutation `perm` (output vertex `j` gathers source vertex
/// `perm[j]`).
pub(crate) fn permute_morph(t: &MorphTarget, perm: &[usize]) -> MorphTarget {
    let g3 = |src: &Vec<[f32; 3]>| -> Vec<[f32; 3]> {
        perm.iter()
            .map(|&i| src.get(i).copied().unwrap_or([0.0; 3]))
            .collect()
    };
    MorphTarget {
        position: t.position.as_ref().map(&g3),
        normal: t.normal.as_ref().map(&g3),
        tangent: t.tangent.as_ref().map(&g3),
    }
}

/// Pack a flat `u32` index list into the narrowest glTF width that holds
/// the vertex pool (`U16` while `vertex_count <= 65536`, else `U32`).
pub(crate) fn pack_indices(flat: Vec<u32>, vertex_count: usize) -> Indices {
    if vertex_count <= u16::MAX as usize + 1 {
        Indices::U16(flat.into_iter().map(|i| i as u16).collect())
    } else {
        Indices::U32(flat)
    }
}

/// Greedy post-transform-cache triangle ordering.
///
/// Returns a permutation of `0..tris.len()` (triangle indices) in emit
/// order. `vertex_count` bounds the per-vertex book-keeping arrays;
/// `cache_size` is the simulated FIFO/scored cache used for residency
/// scoring.
fn greedy_cache_order(tris: &[[u32; 3]], vertex_count: usize, cache_size: usize) -> Vec<usize> {
    let nt = tris.len();

    // Per-vertex incident-triangle lists (CSR-style) and live valence.
    let mut valence = vec![0u32; vertex_count];
    for t in tris {
        for &v in t {
            valence[v as usize] += 1;
        }
    }
    // CSR offsets.
    let mut offset = vec![0usize; vertex_count + 1];
    for v in 0..vertex_count {
        offset[v + 1] = offset[v] + valence[v] as usize;
    }
    let mut incident = vec![0u32; offset[vertex_count]];
    {
        let mut cursor = offset.clone();
        for (ti, t) in tris.iter().enumerate() {
            for &v in t {
                let v = v as usize;
                incident[cursor[v]] = ti as u32;
                cursor[v] += 1;
            }
        }
    }
    // `live` valence is decremented as triangles are emitted.
    let mut live = valence.clone();

    // Cache position per vertex: how many vertices have been pushed since
    // this vertex was last pushed (smaller = more recent). `u32::MAX` =
    // not resident. We track a monotonically increasing timestamp and
    // derive residency position from it.
    let mut last_push = vec![u32::MAX; vertex_count]; // timestamp of last push
    let mut clock: u32 = 0;

    let mut emitted = vec![false; nt];
    let mut order = Vec::with_capacity(nt);

    // Per-triangle cached score; recomputed lazily for dirty triangles.
    let mut tri_score = vec![0.0f32; nt];
    for ti in 0..nt {
        tri_score[ti] = score_triangle(tris[ti], &live, &last_push, clock, cache_size);
    }

    // Vertex score derives from cache position (recency) + valence.
    // Pull these out so `score_triangle` and the incremental rescoring
    // stay in sync (defined as a free fn below).

    let mut next_seed = 0usize; // input-order cursor for fresh components
    let mut best_hint: Option<usize> = None; // a likely-good next triangle

    for _ in 0..nt {
        // Choose the next triangle: prefer the recorded hint if still
        // valid, else scan for the max-scoring unemitted triangle, else
        // fall back to input order (fresh component).
        let chosen = {
            let mut best = best_hint.filter(|&t| !emitted[t]);
            let mut best_val = best.map(|t| tri_score[t]).unwrap_or(f32::NEG_INFINITY);
            // The hint may be stale or never set: confirm it is actually
            // the best among currently-relevant candidates by scanning
            // the frontier (triangles incident to resident vertices).
            // To keep this linear we only scan the incident triangles of
            // the most-recently-pushed vertices, which is where the next
            // good candidate must live.
            for v in recent_vertices(&order, tris, cache_size) {
                for &cand in &incident[offset[v]..offset[v + 1]] {
                    let cand = cand as usize;
                    if emitted[cand] {
                        continue;
                    }
                    let s = tri_score[cand];
                    if s > best_val {
                        best_val = s;
                        best = Some(cand);
                    }
                }
            }
            match best {
                Some(t) => t,
                None => {
                    // Fresh component: advance the input-order cursor.
                    while next_seed < nt && emitted[next_seed] {
                        next_seed += 1;
                    }
                    next_seed
                }
            }
        };

        emitted[chosen] = true;
        order.push(chosen);

        // Update live valence + push the three vertices to the cache.
        let t = tris[chosen];
        for &v in &t {
            let v = v as usize;
            if live[v] > 0 {
                live[v] -= 1;
            }
        }
        for &v in &t {
            clock = clock.wrapping_add(1);
            last_push[v as usize] = clock;
        }

        // Re-score every triangle incident to the three pushed vertices
        // (their cache position and/or valence just changed).
        let mut hint: Option<usize> = None;
        let mut hint_score = f32::NEG_INFINITY;
        for &v in &t {
            let v = v as usize;
            for &inc in &incident[offset[v]..offset[v + 1]] {
                let inc = inc as usize;
                if emitted[inc] {
                    continue;
                }
                let s = score_triangle(tris[inc], &live, &last_push, clock, cache_size);
                tri_score[inc] = s;
                if s > hint_score {
                    hint_score = s;
                    hint = Some(inc);
                }
            }
        }
        best_hint = hint;
    }

    order
}

/// The vertices pushed by the last `cache_size / 3` emitted triangles —
/// the frontier whose incident triangles are the next-candidate set.
fn recent_vertices(order: &[usize], tris: &[[u32; 3]], cache_size: usize) -> Vec<usize> {
    let span = (cache_size / 3).max(1);
    let start = order.len().saturating_sub(span);
    let mut out = Vec::with_capacity(span * 3);
    for &ti in &order[start..] {
        for &v in &tris[ti] {
            out.push(v as usize);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Score one triangle as the sum of its three vertices' scores.
fn score_triangle(
    t: [u32; 3],
    live: &[u32],
    last_push: &[u32],
    clock: u32,
    cache_size: usize,
) -> f32 {
    t.iter()
        .map(|&v| vertex_score(v as usize, live, last_push, clock, cache_size))
        .sum()
}

/// Per-vertex score: cache-recency term + valence term.
///
/// Recency: a vertex pushed within the last `cache_size` pushes is
/// resident; its rank (0 = most recent) maps to a score that is flat-high
/// for the three most-recent ranks (they survive emitting the next
/// triangle) then falls toward the cache tail with a smooth curve.
///
/// Valence: `1 / sqrt(live)` rewards low remaining use so such vertices
/// are cleared before falling out of cache; a vertex with no remaining
/// triangles scores `0` on this term (it will not be needed again).
fn vertex_score(v: usize, live: &[u32], last_push: &[u32], clock: u32, cache_size: usize) -> f32 {
    // Cache recency.
    let lp = last_push[v];
    let recency = if lp == u32::MAX {
        0.0
    } else {
        // rank = how many vertices pushed since this one (0 = newest).
        let rank = clock.wrapping_sub(lp);
        if rank as usize >= cache_size {
            0.0
        } else if rank < 3 {
            // The three newest stay high and equal: emitting the next
            // triangle pushes three vertices, so these would otherwise
            // dominate purely by recency noise.
            0.75
        } else {
            let scaler = 1.0 / (cache_size as f32 - 3.0);
            let frac = (cache_size as f32 - rank as f32) * scaler;
            // Smooth-ish falloff toward the tail.
            frac.powf(1.5)
        }
    };

    // Valence reward.
    let l = live[v];
    let valence = if l == 0 {
        0.0
    } else {
        2.0 * (l as f32).powf(-0.5)
    };

    recency + valence
}
