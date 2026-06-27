//! Discrete per-vertex surface curvature: Gaussian (angle defect) and
//! mean (cotangent-Laplacian) estimates.
//!
//! Curvature distinguishes flat from creased from rounded regions and
//! is the workhorse signal behind feature-preserving simplification,
//! adaptive remeshing, salient-region selection, and quality metrics
//! for a decoded mesh. This module computes both classical discrete
//! curvature estimates per vertex from the triangle connectivity alone
//! — no normals, UVs, or fitted patches required.
//!
//! All formulas are the standard piecewise-linear discretisations of
//! the smooth differential-geometry quantities, derived from first
//! principles below. They operate on the welded triangle connectivity
//! (so a per-face soup is seen as a surface) and never mutate `self`.
//!
//! # Gaussian curvature — the angle defect
//!
//! The Gauss–Bonnet theorem ties the integral of Gaussian curvature
//! over a surface to its topology. Its pointwise discrete form assigns
//! to an **interior** vertex `i` the *angle defect*: how far the
//! incident triangle corner angles fall short of a full turn.
//!
//! ```text
//! defect(i) = 2π − Σ_{f ∋ i} θ_f(i)
//! ```
//!
//! where `θ_f(i)` is the interior angle of face `f` at vertex `i`. A
//! flat vertex (its incident triangles tile a plane) has angles summing
//! to `2π`, so its defect is 0 — zero Gaussian curvature, as expected.
//! A convex corner (a cube vertex) has angles summing to less than
//! `2π`, giving positive defect; a saddle sums to more than `2π`,
//! giving negative defect. For a **boundary** vertex the surface only
//! wraps a half-turn, so the reference is `π`:
//!
//! ```text
//! defect(i) = π − Σ θ_f(i)        (boundary vertex)
//! ```
//!
//! The *integrated* defect is unitless. Dividing by the vertex's
//! associated surface area (the **mixed Voronoi area**, below) gives a
//! pointwise Gaussian curvature `K` in units of `1/length²`:
//!
//! ```text
//! K(i) = defect(i) / A_mixed(i)
//! ```
//!
//! # Mean curvature — the cotangent Laplacian
//!
//! The discrete Laplace–Beltrami operator applied to the vertex
//! positions yields the *mean-curvature normal*. For a vertex `i` with
//! one-ring neighbours `j`, the cotangent-weighted Laplacian is
//!
//! ```text
//! Δp_i = (1 / 2 A_mixed(i)) · Σ_j (cot α_ij + cot β_ij) (p_j − p_i)
//! ```
//!
//! where `α_ij` and `β_ij` are the two angles *opposite* the edge
//! `(i, j)` in the two triangles sharing it (only one for a boundary
//! edge). The magnitude of this vector is twice the mean curvature:
//!
//! ```text
//! H(i) = ½ · ‖Δp_i‖
//! ```
//!
//! The cotangent weights are the unique linear weights that make the
//! discrete Laplacian agree with the smooth operator to first order on
//! a triangulated surface; they fall out of integrating the gradient of
//! the piecewise-linear hat basis function over each triangle.
//!
//! # Mixed Voronoi area
//!
//! Both estimates normalise by `A_mixed(i)`, the surface area
//! "belonging" to vertex `i`. For a non-obtuse triangle the
//! per-triangle contribution to each of its corners is the local
//! Voronoi area `⅛ (cot α · ‖e_α‖² + cot β · ‖e_β‖²)`; for an obtuse
//! triangle that construction degenerates, so the contribution is the
//! triangle area split as `½` to the obtuse corner and `¼` to each
//! other — the **mixed** rule that keeps every contribution positive.
//! Summing over a vertex's incident triangles partitions the whole
//! surface area across the vertices exactly once.
//!
//! # Robustness
//!
//! Degenerate triangles (zero area, collinear corners) contribute no
//! angles or weights. A vertex with no finite incident area yields a
//! curvature of `0.0` (rather than a division-by-zero infinity) so the
//! output is always finite. Non-triangle / empty input yields an empty
//! result.

use std::collections::HashMap;

use crate::mesh::{Indices, Primitive};

/// Per-vertex discrete curvature estimates for a welded mesh.
///
/// The three vectors are parallel: index `i` of each refers to the
/// `i`th vertex of the **welded** primitive (see
/// [`Primitive::curvature`]). Use [`CurvatureReport::welded`] to recover
/// the position pool the indices refer to.
#[derive(Clone, Debug)]
pub struct CurvatureReport {
    /// Pointwise Gaussian curvature `K = defect / A_mixed` at each
    /// welded vertex (`1/length²`).
    pub gaussian: Vec<f64>,
    /// Pointwise mean curvature `H = ½‖Δp‖` at each welded vertex
    /// (`1/length`), always non-negative (the unsigned magnitude).
    pub mean: Vec<f64>,
    /// Mixed Voronoi area associated with each welded vertex
    /// (`length²`); the normalisation denominator, exposed so callers
    /// can re-derive integrated quantities or area-weight an average.
    pub area: Vec<f64>,
    /// The welded primitive the indices above refer to. Its
    /// `positions[i]` is the vertex whose curvature is `gaussian[i]` /
    /// `mean[i]`.
    pub welded: Primitive,
}

impl CurvatureReport {
    /// Number of welded vertices the report covers.
    pub fn len(&self) -> usize {
        self.gaussian.len()
    }

    /// `true` when the report covers no vertices.
    pub fn is_empty(&self) -> bool {
        self.gaussian.is_empty()
    }

    /// The total **integrated** Gaussian curvature `Σ K(i) · A(i)`,
    /// which equals the total angle defect. By the discrete
    /// Gauss–Bonnet theorem this is `2π · χ` for a closed mesh, where
    /// `χ` is the Euler characteristic — a cheap topological sanity
    /// check independent of the mesh's embedding.
    pub fn total_angle_defect(&self) -> f64 {
        self.gaussian
            .iter()
            .zip(&self.area)
            .map(|(&k, &a)| k * a)
            .sum()
    }
}

/// `cot` of the angle at the apex `a` of the triangle `(a, b, c)`,
/// i.e. the angle subtended at `a` by the opposite edge `(b, c)`.
/// Computed as `cos/sin = (u·v) / ‖u×v‖` with `u = b−a`, `v = c−a`,
/// which is numerically stable and avoids a separate `acos`. Returns
/// `0.0` for a degenerate (zero-area) corner.
fn cotangent(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let cross = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let cross_len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    if cross_len <= 0.0 || !cross_len.is_finite() {
        return 0.0;
    }
    dot / cross_len
}

/// Interior angle (radians) at apex `a` of triangle `(a, b, c)`, via
/// the stable `atan2(‖u×v‖, u·v)` form. `0.0` for a degenerate corner.
fn angle_at(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let cross = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let cross_len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    if cross_len <= 0.0 || !cross_len.is_finite() {
        return 0.0;
    }
    cross_len.atan2(dot)
}

/// Squared length of edge `p → q`.
fn edge_len2(p: [f64; 3], q: [f64; 3]) -> f64 {
    let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// Triangle area from its three corners (cross-product magnitude / 2).
fn tri_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

impl Primitive {
    /// Per-vertex discrete Gaussian and mean curvature.
    ///
    /// The primitive is welded first (so a per-face soup is treated as
    /// a connected surface), then for every welded vertex this computes:
    ///
    /// * the **angle defect** (`2π − Σθ` interior, `π − Σθ` boundary),
    /// * the **mixed Voronoi area** `A_mixed`,
    /// * the **Gaussian curvature** `K = defect / A_mixed`,
    /// * the **mean curvature** `H = ½‖Δp‖` from the cotangent
    ///   Laplacian.
    ///
    /// See the [module documentation](crate::curvature) for the
    /// derivations. Results are returned in a [`CurvatureReport`] whose
    /// vectors are parallel to the welded vertex pool (also carried in
    /// the report). Degenerate triangles contribute nothing; a vertex
    /// with no finite incident area gets `0.0` curvature (never an
    /// infinity). Non-triangle / empty input yields an empty report.
    /// Does not mutate `self`.
    pub fn curvature(&self) -> CurvatureReport {
        // Weld so shared corners are one vertex, then de-strip.
        let welded = self.weld_vertices();
        let tris = welded.triangle_indices();
        let n = welded.positions.len();

        let empty = CurvatureReport {
            gaussian: Vec::new(),
            mean: Vec::new(),
            area: Vec::new(),
            welded: {
                let mut w = welded.clone();
                w.indices = Some(Indices::U32(Vec::new()));
                w
            },
        };
        if n == 0 || tris.is_empty() {
            return empty;
        }

        let pos: Vec<[f64; 3]> = welded
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();

        // Per-vertex accumulators.
        let mut angle_sum = vec![0.0f64; n]; // Σ interior corner angles
        let mut area = vec![0.0f64; n]; // mixed Voronoi area
        let mut lap = vec![[0.0f64; 3]; n]; // Σ w_ij (p_j − p_i)

        // Edge → owning-triangle count, to classify boundary vertices.
        let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
        let ekey = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };

        for &[ia, ib, ic] in &tris {
            let (a, b, c) = (ia as usize, ib as usize, ic as usize);
            if a >= n || b >= n || c >= n {
                continue;
            }
            let (pa, pb, pc) = (pos[a], pos[b], pos[c]);
            let area_f = tri_area(pa, pb, pc);
            if !area_f.is_finite() || area_f <= 0.0 {
                continue; // degenerate face contributes nothing
            }

            // Edge incidence for boundary classification.
            for (u, v) in [(ia, ib), (ib, ic), (ic, ia)] {
                *edge_count.entry(ekey(u, v)).or_insert(0) += 1;
            }

            // Corner angles at each vertex.
            let ang_a = angle_at(pa, pb, pc);
            let ang_b = angle_at(pb, pc, pa);
            let ang_c = angle_at(pc, pa, pb);
            angle_sum[a] += ang_a;
            angle_sum[b] += ang_b;
            angle_sum[c] += ang_c;

            // Cotangents at each corner; the cot at the apex opposite an
            // edge is the weight that edge receives in the Laplacian.
            let cot_a = cotangent(pa, pb, pc); // opposite edge (b,c)
            let cot_b = cotangent(pb, pc, pa); // opposite edge (c,a)
            let cot_c = cotangent(pc, pa, pb); // opposite edge (a,b)

            // Laplacian edge contributions: edge (b,c) weighted by cot_a,
            // (c,a) by cot_b, (a,b) by cot_c. Each edge gets a symmetric
            // (p_other − p_self) term on both endpoints.
            accumulate_edge(&mut lap, &pos, b, c, cot_a);
            accumulate_edge(&mut lap, &pos, c, a, cot_b);
            accumulate_edge(&mut lap, &pos, a, b, cot_c);

            // Mixed Voronoi area distribution.
            let obtuse_at = if ang_a > std::f64::consts::FRAC_PI_2 {
                Some(a)
            } else if ang_b > std::f64::consts::FRAC_PI_2 {
                Some(b)
            } else if ang_c > std::f64::consts::FRAC_PI_2 {
                Some(c)
            } else {
                None
            };
            match obtuse_at {
                None => {
                    // Non-obtuse: per-corner Voronoi area
                    //   A_i = ⅛ Σ (cot of the two angles opposite the two
                    //          edges incident to i) · ‖that edge‖².
                    // Edge (a,b) opposite c; (b,c) opposite a; (c,a) opp b.
                    let l_ab = edge_len2(pa, pb);
                    let l_bc = edge_len2(pb, pc);
                    let l_ca = edge_len2(pc, pa);
                    area[a] += (cot_c * l_ab + cot_b * l_ca) / 8.0;
                    area[b] += (cot_a * l_bc + cot_c * l_ab) / 8.0;
                    area[c] += (cot_b * l_ca + cot_a * l_bc) / 8.0;
                }
                Some(o) => {
                    // Obtuse triangle: ½ area to the obtuse corner, ¼ each
                    // to the other two.
                    for &i in &[a, b, c] {
                        area[i] += if i == o { area_f / 2.0 } else { area_f / 4.0 };
                    }
                }
            }
        }

        // Per-vertex boundary flag: touches an edge owned by ≠ 2 faces.
        let mut boundary = vec![false; n];
        for (&(u, v), &count) in &edge_count {
            if count != 2 {
                boundary[u as usize] = true;
                boundary[v as usize] = true;
            }
        }

        let two_pi = std::f64::consts::TAU;
        let pi = std::f64::consts::PI;
        let mut gaussian = vec![0.0f64; n];
        let mut mean = vec![0.0f64; n];
        for i in 0..n {
            let reference = if boundary[i] { pi } else { two_pi };
            let defect = reference - angle_sum[i];
            if area[i] > 0.0 && area[i].is_finite() {
                gaussian[i] = defect / area[i];
                // ‖Δp‖ = ‖(1/2A) Σ w (p_j − p_i)‖; H = ½‖Δp‖.
                let l = lap[i];
                let mag = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
                mean[i] = 0.5 * mag / (2.0 * area[i]);
            } else {
                gaussian[i] = 0.0;
                mean[i] = 0.0;
            }
        }

        let mut out_welded = welded.clone();
        out_welded.indices = Some(Indices::U32(
            tris.iter().flat_map(|t| t.iter().copied()).collect(),
        ));

        CurvatureReport {
            gaussian,
            mean,
            area,
            welded: out_welded,
        }
    }
}

/// Add the symmetric cotangent-weighted Laplacian contribution for edge
/// `(i, j)` to both endpoints: `lap[i] += w (p_j − p_i)` and
/// `lap[j] += w (p_i − p_j)`.
fn accumulate_edge(lap: &mut [[f64; 3]], pos: &[[f64; 3]], i: usize, j: usize, w: f64) {
    if !w.is_finite() {
        return;
    }
    let pi = pos[i];
    let pj = pos[j];
    for k in 0..3 {
        lap[i][k] += w * (pj[k] - pi[k]);
        lap[j][k] += w * (pi[k] - pj[k]);
    }
}
