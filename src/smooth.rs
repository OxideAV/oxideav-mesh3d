//! Laplacian and Taubin mesh fairing (position smoothing).
//!
//! Smoothing **relaxes vertex positions toward their one-ring
//! neighbourhood** to remove high-frequency surface noise (scanner
//! jitter, voxel-marching staircasing, coarse-tessellation faceting)
//! while leaving the connectivity untouched. Unlike
//! [`subdivide_loop`](crate::Primitive::subdivide_loop) (which *adds*
//! vertices) or [`simplify_quadric`](crate::Primitive::simplify_quadric)
//! (which *removes* them), a smoothing pass keeps the index buffer
//! byte-for-byte identical and only moves the points — so UV seams,
//! material bindings, and the triangle roster all survive.
//!
//! # The umbrella Laplacian
//!
//! For a vertex `vᵢ` with one-ring neighbour set `N(i)`, the *uniform
//! (umbrella) discrete Laplacian* is the vector from `vᵢ` to the
//! centroid of its neighbours:
//!
//! ```text
//! L(vᵢ) = ( 1/|N(i)| · Σ_{j ∈ N(i)} vⱼ )  −  vᵢ
//! ```
//!
//! It points "inward" toward the local average surface, and its
//! magnitude grows with how far `vᵢ` deviates from that average — i.e.
//! it is large exactly where the surface is noisy. One **Laplacian
//! smoothing step** with factor `λ ∈ [0, 1]` nudges every vertex a
//! fraction of the way along its umbrella:
//!
//! ```text
//! vᵢ' = vᵢ + λ · L(vᵢ)
//! ```
//!
//! All vertices are updated from the *same* pre-step positions (a
//! Jacobi sweep), so the result is independent of vertex ordering.
//! `λ = 1` snaps each interior vertex exactly onto its neighbour
//! centroid in a single step (maximally aggressive); smaller `λ`
//! diffuses more gently and is the usual choice when iterating.
//!
//! # Shrinkage and the Taubin λ\|μ fix
//!
//! Pure Laplacian smoothing is a low-pass filter that also attenuates
//! the *signal*: repeated steps shrink a closed surface toward its
//! centroid (a sphere collapses to a point in the limit). Taubin's
//! insight is to alternate a **positive** shrinking pass (`λ > 0`) with
//! a **negative** un-shrinking pass (`μ < 0`, `|μ| > λ`):
//!
//! ```text
//! shrink:   vᵢ' = vᵢ + λ · L(vᵢ)
//! inflate:  vᵢ' = vᵢ + μ · L(vᵢ)      with  μ < −λ < 0
//! ```
//!
//! The two factors are tuned so the filter's transfer function passes
//! low spatial frequencies near-unchanged while still killing the high
//! ones, giving noise removal with negligible volume loss. A single
//! Taubin *iteration* is one λ pass immediately followed by one μ pass;
//! `n` iterations apply the pair `n` times.
//!
//! # Boundaries and robustness
//!
//! Open meshes have boundary vertices (those touching an edge owned by
//! a single triangle). Diffusing a boundary vertex toward its full
//! one-ring would pull the silhouette inward and erode the outline, so
//! by default boundary vertices are **pinned** (left exactly in place).
//! [`SmoothOptions::preserve_boundary`] can be cleared to smooth them
//! along the surface anyway. A vertex whose smoothed position would be
//! non-finite (an isolated point with an empty one-ring, or arithmetic
//! that produced a NaN) is left at its original location.
//!
//! Both entry points **weld first** (so a per-face triangle soup is
//! seen as a connected surface, matching `subdivide_loop`'s
//! convention), operate on the welded connectivity, and never mutate
//! `self`. Non-triangle / empty input returns an empty `Triangles`
//! primitive, consistent with the other reductions.

use std::collections::{BTreeSet, HashMap};

use crate::mesh::{Indices, Primitive, Topology};

/// Tunables for a smoothing pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothOptions {
    /// Number of relaxation iterations. `0` is a no-op (positions
    /// returned unchanged, after the weld). Laplacian runs this many
    /// single sweeps; Taubin runs this many λ-then-μ pairs.
    pub iterations: u32,
    /// Pin boundary vertices (those on an edge owned by exactly one
    /// triangle, and — defensively — non-manifold edges owned by three
    /// or more) so the silhouette is preserved. Default `true`.
    pub preserve_boundary: bool,
}

impl Default for SmoothOptions {
    fn default() -> Self {
        Self {
            iterations: 1,
            preserve_boundary: true,
        }
    }
}

/// Per-vertex one-ring adjacency plus a boundary flag, derived from the
/// welded triangle connectivity. Built once and reused across every
/// iteration of a multi-step smooth.
struct Adjacency {
    /// `neighbours[i]` = the set of vertices sharing an edge with `i`.
    neighbours: Vec<BTreeSet<u32>>,
    /// `boundary[i]` = `i` touches a non-manifold (count ≠ 2) edge.
    boundary: Vec<bool>,
}

impl Adjacency {
    /// Build the one-ring adjacency and boundary classification from a
    /// triangle list over `n` vertices.
    fn build(tris: &[[u32; 3]], n: usize) -> Self {
        // Undirected edge → owning-triangle count.
        let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
        let key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
        for &[a, b, c] in tris {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *edge_count.entry(key(u, v)).or_insert(0) += 1;
            }
        }
        let mut neighbours: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
        let mut boundary = vec![false; n];
        for (&(a, b), &count) in &edge_count {
            neighbours[a as usize].insert(b);
            neighbours[b as usize].insert(a);
            // count == 1 → open boundary; count >= 3 → non-manifold.
            // Either way the vertex is not interior-manifold; pin it
            // when boundary preservation is requested.
            if count != 2 {
                boundary[a as usize] = true;
                boundary[b as usize] = true;
            }
        }
        Self {
            neighbours,
            boundary,
        }
    }
}

/// Compute the umbrella offset `L(vᵢ) = centroid(N(i)) − vᵢ` for every
/// vertex, returning `[0,0,0]` where the one-ring is empty (the vertex
/// keeps its position). Reads from `pos`, never writes.
fn umbrella(pos: &[[f32; 3]], adj: &Adjacency) -> Vec<[f32; 3]> {
    let n = pos.len();
    let mut out = vec![[0.0f32; 3]; n];
    for i in 0..n {
        let ring = &adj.neighbours[i];
        if ring.is_empty() {
            continue;
        }
        let mut sum = [0.0f64; 3];
        for &j in ring {
            let p = pos[j as usize];
            sum[0] += p[0] as f64;
            sum[1] += p[1] as f64;
            sum[2] += p[2] as f64;
        }
        let inv = 1.0 / ring.len() as f64;
        let centroid = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
        out[i] = [
            (centroid[0] - pos[i][0] as f64) as f32,
            (centroid[1] - pos[i][1] as f64) as f32,
            (centroid[2] - pos[i][2] as f64) as f32,
        ];
    }
    out
}

/// Apply one Jacobi relaxation sweep `vᵢ' = vᵢ + factor · L(vᵢ)` to
/// `pos` in place. Boundary vertices are skipped when `pin_boundary`.
/// A vertex whose result is non-finite is left unchanged.
fn relax(pos: &mut [[f32; 3]], adj: &Adjacency, factor: f32, pin_boundary: bool) {
    let lap = umbrella(pos, adj);
    for i in 0..pos.len() {
        if pin_boundary && adj.boundary[i] {
            continue;
        }
        let candidate = [
            pos[i][0] + factor * lap[i][0],
            pos[i][1] + factor * lap[i][1],
            pos[i][2] + factor * lap[i][2],
        ];
        if candidate.iter().all(|c| c.is_finite()) {
            pos[i] = candidate;
        }
    }
}

/// Reduce `self` to a clean welded indexed `Triangles` primitive, the
/// shared front-end for both smoothers. Returns `None` (caller emits an
/// empty primitive) when there is no valid triangle topology.
fn welded_triangles(prim: &Primitive) -> Option<(Primitive, Vec<[u32; 3]>)> {
    let sn = prim.positions.len();
    if sn == 0 {
        return None;
    }
    let clean_tris: Vec<[u32; 3]> = prim
        .triangle_indices()
        .into_iter()
        .filter(|&[a, b, c]| {
            (a as usize) < sn
                && (b as usize) < sn
                && (c as usize) < sn
                && a != b
                && b != c
                && a != c
        })
        .collect();
    if clean_tris.is_empty() {
        return None;
    }
    let mut clean = prim.to_triangle_list();
    let mut flat: Vec<u32> = Vec::with_capacity(clean_tris.len() * 3);
    for t in &clean_tris {
        flat.extend_from_slice(t);
    }
    clean.indices = Some(Indices::U32(flat));
    let welded = clean.weld_vertices();
    let tris = welded.triangle_indices();
    if tris.is_empty() || welded.positions.is_empty() {
        return None;
    }
    Some((welded, tris))
}

/// An empty welded-shaped `Triangles` primitive carrying `self`'s
/// attribute roster — the canonical "nothing to smooth" return.
fn empty_like(prim: &Primitive) -> Primitive {
    let mut o = prim.to_triangle_list();
    o.topology = Topology::Triangles;
    o.indices = Some(Indices::U32(Vec::new()));
    o
}

impl Primitive {
    /// One or more **uniform Laplacian smoothing** sweeps.
    ///
    /// Each iteration moves every (non-pinned) vertex a fraction
    /// `lambda` of the way toward the centroid of its one-ring
    /// neighbours:  `vᵢ' = vᵢ + λ · (centroid(N(i)) − vᵢ)`. Updates are
    /// computed Jacobi-style from a single snapshot per sweep, so the
    /// result is independent of vertex ordering.
    ///
    /// `lambda` is clamped to `[0, 1]`. `0` (or
    /// `options.iterations == 0`) returns the welded surface with
    /// positions unchanged; `1` snaps each interior vertex exactly onto
    /// its neighbour centroid per step. Boundary vertices are pinned
    /// unless [`SmoothOptions::preserve_boundary`] is cleared.
    ///
    /// Pure Laplacian smoothing shrinks closed surfaces toward their
    /// centroid as iterations accumulate — use
    /// [`smooth_taubin`](Self::smooth_taubin) for shrink-free fairing.
    ///
    /// The connectivity is preserved (the index buffer is the welded
    /// triangle list); only positions move. All other attributes pass
    /// through unchanged from the welded primitive. `self` is not
    /// mutated; non-triangle / empty input yields an empty `Triangles`
    /// primitive.
    pub fn smooth_laplacian(&self, lambda: f32, options: SmoothOptions) -> Primitive {
        let Some((mut welded, tris)) = welded_triangles(self) else {
            return empty_like(self);
        };
        let lambda = lambda.clamp(0.0, 1.0);
        let n = welded.positions.len();
        let adj = Adjacency::build(&tris, n);
        for _ in 0..options.iterations {
            relax(
                &mut welded.positions,
                &adj,
                lambda,
                options.preserve_boundary,
            );
        }
        welded
    }

    /// One or more **Taubin λ\|μ smoothing** iterations — low-pass
    /// fairing without the volume shrinkage of the plain Laplacian.
    ///
    /// Each iteration is two umbrella sweeps: a positive *shrink* pass
    /// with factor `lambda` immediately followed by a negative
    /// *inflate* pass with factor `mu`. For the filter to remove noise
    /// without net shrinkage `mu` must satisfy `mu < −lambda < 0`
    /// (a common pairing is `λ ≈ 0.33`, `μ ≈ −0.34`); the method does
    /// not enforce this so callers can experiment, but a non-negative
    /// `mu` degenerates into extra Laplacian shrinking.
    ///
    /// `lambda` is clamped to `[0, 1]`; `mu` is used as supplied.
    /// Boundary handling, weld-first behaviour, attribute pass-through,
    /// connectivity preservation, and the empty-input contract match
    /// [`smooth_laplacian`](Self::smooth_laplacian). `self` is not
    /// mutated.
    pub fn smooth_taubin(&self, lambda: f32, mu: f32, options: SmoothOptions) -> Primitive {
        let Some((mut welded, tris)) = welded_triangles(self) else {
            return empty_like(self);
        };
        let lambda = lambda.clamp(0.0, 1.0);
        let n = welded.positions.len();
        let adj = Adjacency::build(&tris, n);
        for _ in 0..options.iterations {
            relax(
                &mut welded.positions,
                &adj,
                lambda,
                options.preserve_boundary,
            );
            relax(&mut welded.positions, &adj, mu, options.preserve_boundary);
        }
        welded
    }
}
