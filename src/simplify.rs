//! Quadric-error-metric (QEM) edge-collapse mesh simplification.
//!
//! Where [`Primitive::simplify_cluster`](crate::Primitive::simplify_cluster)
//! quantises vertices onto a regular grid — robust but
//! position-snapping — this module performs **error-optimal** decimation:
//! it removes geometry in the order that perturbs the surface least,
//! measured by a per-vertex quadratic error form. It is the high-quality
//! counterpart of the clustering reducer and the reduction dual of
//! [`Primitive::subdivide_loop`](crate::Primitive::subdivide_loop).
//!
//! # The quadric error metric
//!
//! Every triangle defines a plane. Write that plane as the unit-normal
//! form `n·x + d = 0` with `‖n‖ = 1` and pack the coefficients into a
//! column 4-vector `p = (a, b, c, d)ᵀ` where `(a, b, c) = n`. The
//! **squared distance** from an arbitrary point `x = (x, y, z)` to that
//! plane is `(p·x̃)²` with `x̃ = (x, y, z, 1)ᵀ`, and that quadratic form
//! is the symmetric `4 × 4` matrix `Kp = p · pᵀ`, the *fundamental error
//! quadric* of the plane.
//!
//! Assign to each vertex the quadric `Q = Σ Kp` summed over the planes
//! of its incident triangles. Then `vᵀ Q v` (with `v` homogeneous) is
//! the **sum of squared distances** from the candidate position `v` to
//! all those planes — a single number that says how far a proposed new
//! position drifts from the surface the vertex used to belong to. A
//! freshly decoded mesh has every vertex sitting exactly on its incident
//! planes, so each starts with `vᵀ Q v = 0`.
//!
//! # Edge collapse
//!
//! The atomic operation is the **half-edge collapse** `(i, j) → v̄`: the
//! two endpoints of an edge are merged into one vertex `v̄`, the two
//! triangles sharing that edge vanish, and every other triangle that
//! referenced `i` or `j` now references `v̄`. The merged vertex inherits
//! the summed quadric `Q̄ = Qᵢ + Qⱼ`, and its **optimal position** is the
//! `v̄` minimising `v̄ᵀ Q̄ v̄`. Setting the gradient to zero gives a `3 × 3`
//! linear system in the `Q̄` sub-block; when that block is invertible the
//! solve places `v̄` at the error-minimising point (often *off* the
//! original edge, letting a flat region slide its sole survivor to the
//! best spot). When it is singular — a flat or symmetric neighbourhood —
//! the code falls back to the cheaper of the two endpoints and the
//! midpoint. The **collapse cost** is the residual `v̄ᵀ Q̄ v̄` at that
//! chosen `v̄`.
//!
//! # Greedy schedule
//!
//! All collapsible edges are costed and pushed into a min-heap. The
//! cheapest legal collapse is applied, the affected vertices' quadrics
//! and the costs of every edge in their one-ring are recomputed, and the
//! updated edges are re-pushed. Stale heap entries are skipped with a
//! per-vertex version stamp (lazy deletion) so no expensive heap removal
//! is needed. Collapses that would be illegal are rejected up front:
//!
//! * **Fold-over guard.** A collapse that would flip any surviving
//!   incident triangle past edge-on (its normal reverses, or it
//!   degenerates to a sliver) is skipped — this is what keeps the
//!   simplified surface from self-intersecting.
//! * **Boundary lock.** A vertex on an open boundary may only collapse
//!   along the boundary, never into the interior, so open seams keep
//!   their silhouette. Boundary edges additionally carry a perpendicular
//!   constraint quadric so the seam itself resists shrinking.
//! * **Non-manifold edges** (shared by ≠ 2 triangles, after the boundary
//!   treatment) and edges whose collapse would merge two already-adjacent
//!   vertices into a non-simple fan are left intact.
//!
//! The schedule stops when the live triangle count reaches the requested
//! target or no legal collapse remains. The whole pass is roughly
//! `O(E log E)`.
//!
//! # Attributes
//!
//! The metric drives **position** only. Every other attribute on the
//! merged vertex (`normals`, `tangents`, `uvs`, `colors`, `weights`,
//! morph deltas) is the linear blend of the two endpoints at the
//! collapse's barycentric split — the fraction of the optimal `v̄` along
//! the original edge — matching the interpolation conventions used by
//! `subdivide_loop`: normals / tangent-`xyz` are renormalised, the
//! tangent handedness `w` and the categorical `joints` quad take the
//! surviving (lower-index) endpoint, and `weights` are renormalised to
//! sum 1.

use crate::mesh::{Indices, Primitive, Topology};
use std::collections::BinaryHeap;

/// A symmetric `4 × 4` quadric stored as its 10 distinct coefficients in
/// row-major upper-triangular order:
/// `[a00, a01, a02, a03, a11, a12, a13, a22, a23, a33]`.
#[derive(Clone, Copy, Debug, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    /// Fundamental error quadric `p · pᵀ` of a plane `p = (a, b, c, d)`.
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Quadric([
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        ])
    }

    fn add(&self, o: &Quadric) -> Quadric {
        let mut q = self.0;
        for (k, slot) in q.iter_mut().enumerate() {
            *slot += o.0[k];
        }
        Quadric(q)
    }

    /// Evaluate `vᵀ Q v` for the homogeneous point `(x, y, z, 1)`.
    fn eval(&self, x: f64, y: f64, z: f64) -> f64 {
        let q = &self.0;
        // a00 x² + a11 y² + a22 z² + a33
        //  + 2(a01 xy + a02 xz + a03 x + a12 yz + a13 y + a23 z)
        q[0] * x * x
            + q[4] * y * y
            + q[7] * z * z
            + q[9]
            + 2.0 * (q[1] * x * y + q[2] * x * z + q[3] * x + q[5] * y * z + q[6] * y + q[8] * z)
    }

    /// Solve for the position minimising `vᵀ Q v` by inverting the upper
    /// `3 × 3` block. Returns `None` when that block is singular (flat or
    /// symmetric neighbourhood).
    fn optimal_position(&self) -> Option<[f64; 3]> {
        let q = &self.0;
        // The 3×3 sub-block A and the right-hand side b = -(a03, a13, a23):
        //   A = [[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]]
        let (a00, a01, a02) = (q[0], q[1], q[2]);
        let (a11, a12) = (q[4], q[5]);
        let a22 = q[7];
        let (b0, b1, b2) = (-q[3], -q[6], -q[8]);

        // Cofactors / determinant of the symmetric 3×3.
        let c00 = a11 * a22 - a12 * a12;
        let c01 = a02 * a12 - a01 * a22;
        let c02 = a01 * a12 - a02 * a11;
        let det = a00 * c00 + a01 * c01 + a02 * c02;
        if !det.is_finite() || det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        // Remaining cofactors (symmetric).
        let c11 = a00 * a22 - a02 * a02;
        let c12 = a02 * a01 - a00 * a12;
        let c22 = a00 * a11 - a01 * a01;
        // x = A⁻¹ b, A⁻¹ = adj/det with adj symmetric = the cofactor matrix.
        let x = (c00 * b0 + c01 * b1 + c02 * b2) * inv_det;
        let y = (c01 * b0 + c11 * b1 + c12 * b2) * inv_det;
        let z = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det;
        if x.is_finite() && y.is_finite() && z.is_finite() {
            Some([x, y, z])
        } else {
            None
        }
    }
}

/// A heap entry: a pending collapse of the undirected edge `(u, v)` with
/// its computed `cost`, plus the version stamps of both endpoints at the
/// time the cost was computed (for lazy invalidation).
#[derive(Clone, Copy, Debug)]
struct Candidate {
    cost: f64,
    u: u32,
    v: u32,
    vu: u32,
    vv: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, o: &Self) -> bool {
        self.cost == o.cost
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap; invert so the smallest cost pops
        // first. NaN costs sort last (never popped before finite ones).
        o.cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Less)
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Unit face normal (f64) of the triangle `(p0, p1, p2)`; `None` if
/// degenerate.
fn face_normal(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> Option<[f64; 3]> {
    let n = cross(sub(p1, p0), sub(p2, p0));
    let len = norm(n);
    if len.is_finite() && len > 1e-20 {
        Some([n[0] / len, n[1] / len, n[2] / len])
    } else {
        None
    }
}

impl Primitive {
    /// Simplify this mesh to approximately `target_triangles` triangles
    /// by greedy quadric-error edge collapse, preserving the surface
    /// shape far more faithfully than the grid clustering of
    /// [`Primitive::simplify_cluster`].
    ///
    /// The algorithm assigns each vertex the sum of the *fundamental
    /// error quadrics* of its incident triangle planes, then repeatedly
    /// collapses the edge whose removal adds the least squared-distance
    /// error to the surface, re-positioning the merged vertex at the
    /// error-minimising point (see the [module docs](crate::simplify)).
    /// Collapses that would flip a triangle (fold-over), cross an open
    /// boundary into the interior, or touch a non-manifold edge are
    /// rejected, so the output stays a clean, crack-free, same-silhouette
    /// approximation.
    ///
    /// # Target
    ///
    /// `target_triangles` is the desired live triangle count. The reducer
    /// collapses edges until the count reaches it or no legal collapse
    /// remains, whichever comes first — so the result has **at least**
    /// `target_triangles` triangles when the mesh's topology blocks
    /// further reduction (a closed manifold cannot go below 4). A target
    /// `≥` the input triangle count returns the input welded and cleaned
    /// (no collapses). A target of `0` reduces as far as the legality
    /// guards allow.
    ///
    /// # Attributes
    ///
    /// Position is driven by the metric; every other attribute on a
    /// merged vertex is the linear blend of the collapsed edge's two
    /// endpoints at the optimal split fraction (normals / tangent-`xyz`
    /// renormalised, tangent `w` and `joints` from the lower-index
    /// endpoint, `weights` renormalised to sum 1), matching
    /// [`Primitive::subdivide_loop`]'s interpolation rules. Morph deltas
    /// blend per target.
    ///
    /// # Output
    ///
    /// * An indexed [`Topology::Triangles`] primitive over the surviving
    ///   vertices, carrying `material`, the `targets` roster, and `extras`
    ///   from `self`; index width follows the crate convention
    ///   ([`Indices::U16`] while the pool fits, else [`Indices::U32`]).
    ///   Vertices left unreferenced by any surviving triangle are pruned.
    /// * Same triangle feed and exclusion rules as the rest of the
    ///   toolkit: `Triangles` / `TriangleStrip` (alternating winding) /
    ///   `TriangleFan` feed through [`Primitive::triangle_indices`];
    ///   out-of-range / duplicate-corner triangles, non-finite vertex
    ///   positions, non-triangle topologies, and empty primitives all
    ///   yield an empty indexed `Triangles` primitive.
    /// * **Does not mutate `self`.**
    pub fn simplify_quadric(&self, target_triangles: usize) -> Primitive {
        self.simplify_qem(target_triangles, f64::INFINITY)
    }

    /// Simplify by greedy quadric-error edge collapse until the next
    /// cheapest legal collapse would add more than `max_error`
    /// squared-distance error to the surface — an **error-bounded** budget
    /// instead of [`Primitive::simplify_quadric`]'s triangle-count budget.
    ///
    /// `max_error` is a threshold on the collapse cost `v̄ᵀ Q̄ v̄`, i.e. the
    /// sum of squared distances from the merged vertex to the planes it
    /// replaced (units: squared model-space length). It scales with the
    /// model's coordinate magnitude — a value of `0.0` performs only
    /// zero-cost collapses (coplanar / colinear merges that change nothing
    /// geometrically), and larger values trade fidelity for fewer
    /// triangles. The pass stops as soon as the cheapest remaining legal
    /// collapse exceeds the bound, so the surface never deviates more than
    /// the budget allows at any single step. A non-finite or negative
    /// `max_error` is treated as "no bound" (reduces as far as the
    /// legality guards permit, like `simplify_quadric(0)`).
    ///
    /// All other behaviour — the metric, the fold-over / boundary /
    /// non-manifold legality guards, attribute blending, output shape, and
    /// `self`-immutability — matches [`Primitive::simplify_quadric`].
    pub fn simplify_quadric_error(&self, max_error: f64) -> Primitive {
        let bound = if max_error.is_finite() && max_error >= 0.0 {
            max_error
        } else {
            f64::INFINITY
        };
        // Error-bounded reduction has no triangle floor of its own; the
        // legality guards provide the only lower bound.
        self.simplify_qem(0, bound)
    }

    /// Shared core: collapse until the live triangle count reaches
    /// `target_triangles` **or** the cheapest legal collapse would exceed
    /// `max_error`, whichever binds first.
    fn simplify_qem(&self, target_triangles: usize, max_error: f64) -> Primitive {
        // Empty skeleton carrying self's attribute shape (material /
        // extras / targets roster), Triangles topology, empty index.
        let empty_out = {
            let mut o = self.to_triangle_list();
            o.positions.clear();
            o.normals = o.normals.as_ref().map(|_| Vec::new());
            o.tangents = o.tangents.as_ref().map(|_| Vec::new());
            for s in &mut o.uvs {
                s.clear();
            }
            for s in &mut o.colors {
                s.clear();
            }
            o.joints = o.joints.as_ref().map(|_| Vec::new());
            o.weights = o.weights.as_ref().map(|_| Vec::new());
            for t in &mut o.targets {
                *t = t.map_buffers(|_| Vec::new());
            }
            o.indices = Some(Indices::U16(Vec::new()));
            o
        };

        let sn = self.positions.len();
        let raw_tris: Vec<[u32; 3]> = self
            .triangle_indices()
            .into_iter()
            .filter(|&[a, b, c]| {
                (a as usize) < sn
                    && (b as usize) < sn
                    && (c as usize) < sn
                    && a != b
                    && b != c
                    && a != c
                    && self.positions[a as usize].iter().all(|f| f.is_finite())
                    && self.positions[b as usize].iter().all(|f| f.is_finite())
                    && self.positions[c as usize].iter().all(|f| f.is_finite())
            })
            .collect();
        if raw_tris.is_empty() {
            return empty_out;
        }

        // Weld so the metric sees shared vertices (a per-face soup has no
        // edges to collapse). Build a clean indexed primitive over the
        // valid triangles, then weld.
        let clean = {
            let mut c = self.to_triangle_list();
            let mut flat: Vec<u32> = Vec::with_capacity(raw_tris.len() * 3);
            for t in &raw_tris {
                flat.extend_from_slice(t);
            }
            c.indices = Some(Indices::U32(flat));
            c
        };
        let welded = clean.weld_vertices();
        let faces0 = welded.triangle_indices();
        let nverts = welded.positions.len();
        if faces0.is_empty() || nverts == 0 {
            return empty_out;
        }

        let mut state = QemState::new(&welded, faces0);
        state.run(target_triangles, max_error);
        state.into_primitive(&welded, empty_out)
    }
}

/// Mutable working state for one simplification pass.
struct QemState {
    /// Live position per vertex (f64 for the metric; emitted as f32).
    pos: Vec<[f64; 3]>,
    /// Accumulated quadric per vertex.
    quad: Vec<Quadric>,
    /// Faces as vertex-index triples; a removed face is `None`.
    faces: Vec<Option<[u32; 3]>>,
    /// Incident face indices per vertex (may contain stale entries that
    /// are filtered against `faces` on read).
    vfaces: Vec<Vec<u32>>,
    /// Alive flag per vertex.
    alive: Vec<bool>,
    /// Version stamp per vertex, bumped on every quadric/position change.
    ver: Vec<u32>,
    /// Boundary flag per vertex.
    boundary: Vec<bool>,
    /// The collapse split fraction `t` chosen for the last applied
    /// collapse of edge `(u, v)` — used to blend attributes (`v̄ = u +
    /// t·(v − u)` projected). Recorded per merged vertex.
    /// (vertex → (other endpoint, t)) for attribute reconstruction.
    merge_log: Vec<(u32, u32, f64)>,
    live_faces: usize,
}

impl QemState {
    fn new(welded: &Primitive, faces: Vec<[u32; 3]>) -> Self {
        let nverts = welded.positions.len();
        let pos: Vec<[f64; 3]> = welded
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();

        let mut quad = vec![Quadric::default(); nverts];
        let mut vfaces: Vec<Vec<u32>> = vec![Vec::new(); nverts];

        // Face quadrics → vertex quadrics.
        for (fi, &[a, b, c]) in faces.iter().enumerate() {
            let (pa, pb, pc) = (pos[a as usize], pos[b as usize], pos[c as usize]);
            if let Some(n) = face_normal(pa, pb, pc) {
                let d = -dot(n, pa);
                let kp = Quadric::from_plane(n[0], n[1], n[2], d);
                for &v in &[a, b, c] {
                    quad[v as usize] = quad[v as usize].add(&kp);
                    vfaces[v as usize].push(fi as u32);
                }
            } else {
                // Degenerate face still records incidence (so its corners
                // are tracked) but contributes no plane.
                for &v in &[a, b, c] {
                    vfaces[v as usize].push(fi as u32);
                }
            }
        }

        // Boundary detection + boundary constraint quadrics: an edge used
        // by exactly one face is a boundary edge; add a perpendicular
        // plane through it (normal × edge direction) weighted by edge
        // length² so seams resist collapse.
        let mut edge_use: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        for f in &faces {
            let [a, b, c] = *f;
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let k = if u < v { (u, v) } else { (v, u) };
                *edge_use.entry(k).or_insert(0) += 1;
            }
        }
        let mut boundary = vec![false; nverts];
        // For the constraint plane we need a representative face normal of
        // each boundary edge; recompute per boundary edge from its single
        // owning face.
        // Map edge → owning face for boundary edges.
        let mut edge_face: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::new();
        for (fi, f) in faces.iter().enumerate() {
            let [a, b, c] = *f;
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let k = if u < v { (u, v) } else { (v, u) };
                edge_face.entry(k).or_insert(fi);
            }
        }
        for (&(u, v), &cnt) in &edge_use {
            if cnt == 1 {
                boundary[u as usize] = true;
                boundary[v as usize] = true;
                if let Some(&fi) = edge_face.get(&(u, v)) {
                    let [fa, fb, fc] = faces[fi];
                    let n = face_normal(pos[fa as usize], pos[fb as usize], pos[fc as usize]);
                    if let Some(n) = n {
                        let e = sub(pos[v as usize], pos[u as usize]);
                        // Plane through the edge, perpendicular to the
                        // face: normal = edge × face_normal, normalised.
                        let mut cn = cross(e, n);
                        let len = norm(cn);
                        if len.is_finite() && len > 1e-20 {
                            cn = [cn[0] / len, cn[1] / len, cn[2] / len];
                            let d = -dot(cn, pos[u as usize]);
                            // Weight by edge length² so longer seams are
                            // stiffer (matching the area-scaled interior
                            // quadrics).
                            let w = dot(e, e);
                            let kp = Quadric::from_plane(
                                cn[0] * w.sqrt(),
                                cn[1] * w.sqrt(),
                                cn[2] * w.sqrt(),
                                d * w.sqrt(),
                            );
                            quad[u as usize] = quad[u as usize].add(&kp);
                            quad[v as usize] = quad[v as usize].add(&kp);
                        }
                    }
                }
            }
        }

        let live_faces = faces.len();
        QemState {
            pos,
            quad,
            faces: faces.into_iter().map(Some).collect(),
            vfaces,
            alive: vec![true; nverts],
            ver: vec![0; nverts],
            boundary,
            merge_log: (0..nverts as u32).map(|i| (i, i, 0.0)).collect(),
            live_faces,
        }
    }

    /// Live incident faces of `v` (filters stale `vfaces` entries).
    fn incident(&self, v: u32) -> Vec<u32> {
        self.vfaces[v as usize]
            .iter()
            .copied()
            .filter(|&fi| {
                self.faces[fi as usize]
                    .map(|t| t.contains(&v))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Compute the optimal merge position + cost for collapsing `(u, v)`.
    /// Returns `(pos, cost, t)` where `t` is the split fraction along the
    /// edge (`pos ≈ u + t·(v − u)`), used to blend attributes.
    fn collapse_target(&self, u: u32, v: u32) -> ([f64; 3], f64, f64) {
        let q = self.quad[u as usize].add(&self.quad[v as usize]);
        let pu = self.pos[u as usize];
        let pv = self.pos[v as usize];

        // Boundary endpoints constrain the target to the edge so a seam
        // vertex never drifts into the interior.
        let bu = self.boundary[u as usize];
        let bv = self.boundary[v as usize];

        let pos = if bu && bv {
            // Both on boundary: keep on the edge, pick the cheaper end /
            // midpoint.
            self.best_on_edge(&q, pu, pv)
        } else if bu {
            pu
        } else if bv {
            pv
        } else if let Some(p) = q.optimal_position() {
            p
        } else {
            self.best_on_edge(&q, pu, pv)
        };

        let cost = q.eval(pos[0], pos[1], pos[2]).max(0.0);
        // Recover t = ((pos − pu)·(pv − pu)) / |pv − pu|² (clamped) for
        // attribute blending; degenerate edge → 0.
        let e = sub(pv, pu);
        let denom = dot(e, e);
        let t = if denom > 1e-30 {
            (dot(sub(pos, pu), e) / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (pos, cost, t)
    }

    /// Cheapest of `{u, midpoint, v}` under quadric `q`.
    fn best_on_edge(&self, q: &Quadric, pu: [f64; 3], pv: [f64; 3]) -> [f64; 3] {
        let mid = [
            0.5 * (pu[0] + pv[0]),
            0.5 * (pu[1] + pv[1]),
            0.5 * (pu[2] + pv[2]),
        ];
        let mut best = pu;
        let mut bc = q.eval(pu[0], pu[1], pu[2]);
        for cand in [mid, pv] {
            let c = q.eval(cand[0], cand[1], cand[2]);
            if c < bc {
                bc = c;
                best = cand;
            }
        }
        best
    }

    /// Would collapsing `(u, v) → np` flip or sliver any of `u`'s
    /// surviving incident triangles (those not also touching `v`)?
    fn would_flip(&self, u: u32, v: u32, np: [f64; 3]) -> bool {
        for fi in self.incident(u) {
            let [a, b, c] = self.faces[fi as usize].unwrap();
            // Skip the two faces that vanish (they touch both u and v).
            if [a, b, c].contains(&v) {
                continue;
            }
            // Old + new positions with u replaced by np.
            let old = [
                self.pos[a as usize],
                self.pos[b as usize],
                self.pos[c as usize],
            ];
            let mapped = |w: u32| if w == u { np } else { self.pos[w as usize] };
            let new = [mapped(a), mapped(b), mapped(c)];
            let no = face_normal(old[0], old[1], old[2]);
            let nn = face_normal(new[0], new[1], new[2]);
            match (no, nn) {
                // Reject a normal reversal / sliver collapse (dot small or
                // negative) — a fold-over.
                (Some(no), Some(nn)) if dot(no, nn) < 1e-3 => return true,
                // Was valid, became degenerate (sliver/zero area) → reject.
                (Some(_), None) => return true,
                _ => {}
            }
        }
        false
    }

    /// Build a fresh candidate for edge `(u, v)` if the collapse is
    /// currently legal; `None` if it should not be queued.
    fn make_candidate(&self, u: u32, v: u32) -> Option<Candidate> {
        if !self.alive[u as usize] || !self.alive[v as usize] || u == v {
            return None;
        }
        // Boundary→interior collapse is forbidden: a boundary vertex may
        // only merge along the boundary (into another boundary vertex).
        if self.boundary[u as usize] != self.boundary[v as usize] {
            return None;
        }
        // The edge must be shared by 1 or 2 live faces (manifold-ish).
        let mut shared = 0u32;
        for fi in self.incident(u) {
            let t = self.faces[fi as usize].unwrap();
            if t.contains(&v) {
                shared += 1;
            }
        }
        if shared == 0 || shared > 2 {
            return None;
        }
        // Link condition (cheap form): the two endpoints must not share a
        // neighbour outside the collapsing triangles, or the collapse
        // creates a non-manifold fold. Skip when both endpoints' shared
        // neighbours exceed the shared-face count.
        if shared == 2 && self.shared_neighbours(u, v) > 2 {
            return None;
        }
        if shared == 1 && self.shared_neighbours(u, v) > 1 {
            return None;
        }
        let (np, cost, _t) = self.collapse_target(u, v);
        if !cost.is_finite() {
            return None;
        }
        if self.would_flip(u, v, np) || self.would_flip(v, u, np) {
            return None;
        }
        Some(Candidate {
            cost,
            u,
            v,
            vu: self.ver[u as usize],
            vv: self.ver[v as usize],
        })
    }

    /// Number of vertices adjacent to both `u` and `v` via a live face.
    fn shared_neighbours(&self, u: u32, v: u32) -> usize {
        let nbr = |w: u32| -> std::collections::BTreeSet<u32> {
            let mut s = std::collections::BTreeSet::new();
            for fi in self.incident(w) {
                for &c in self.faces[fi as usize].unwrap().iter() {
                    if c != w {
                        s.insert(c);
                    }
                }
            }
            s
        };
        let nu = nbr(u);
        let nv = nbr(v);
        nu.intersection(&nv).filter(|&&w| w != u && w != v).count()
    }

    /// One-ring vertices of `v` via live faces (excluding `v`).
    fn one_ring(&self, v: u32) -> Vec<u32> {
        let mut s = std::collections::BTreeSet::new();
        for fi in self.incident(v) {
            for &c in self.faces[fi as usize].unwrap().iter() {
                if c != v {
                    s.insert(c);
                }
            }
        }
        s.into_iter().collect()
    }

    fn run(&mut self, target_triangles: usize, max_error: f64) {
        // Seed the heap with every undirected edge once.
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        let nf = self.faces.len();
        for fi in 0..nf {
            if let Some([a, b, c]) = self.faces[fi] {
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    let k = if u < v { (u, v) } else { (v, u) };
                    if seen.insert(k) {
                        if let Some(cand) = self.make_candidate(k.0, k.1) {
                            heap.push(cand);
                        }
                    }
                }
            }
        }

        while self.live_faces > target_triangles {
            let cand = match heap.pop() {
                Some(c) => c,
                None => break,
            };
            // Lazy-deletion: skip if either endpoint moved/changed since
            // this candidate was costed.
            if !self.alive[cand.u as usize]
                || !self.alive[cand.v as usize]
                || self.ver[cand.u as usize] != cand.vu
                || self.ver[cand.v as usize] != cand.vv
            {
                continue;
            }
            // Error-bounded stop: the heap key `cand.cost` is monotone
            // non-decreasing across pops, so once the cheapest live key
            // exceeds the budget every remaining one does too — we are
            // done. (Checking the heap key, not the re-validated cost,
            // keeps the bound sound even if a neighbourhood shift made
            // this particular edge more expensive: a still-cheaper edge
            // would have a smaller key and pop first.)
            if cand.cost > max_error {
                break;
            }
            // Re-validate legality at apply time (the neighbourhood may
            // have shifted) and recompute the exact target.
            let fresh = match self.make_candidate(cand.u, cand.v) {
                Some(c) => c,
                None => continue,
            };
            // If re-validation made this edge more expensive than its heap
            // key, it may no longer be the cheapest option — re-push with
            // the corrected cost and let the heap re-order instead of
            // applying an out-of-order collapse.
            if fresh.cost > cand.cost + 1e-12 {
                heap.push(fresh);
                continue;
            }
            // Apply the collapse: v collapses into u.
            let (u, v) = (fresh.u, fresh.v);
            let (np, _cost, t) = self.collapse_target(u, v);
            self.apply_collapse(u, v, np, t);

            // Re-queue every edge in u's refreshed one-ring.
            for w in self.one_ring(u) {
                if let Some(c) = self.make_candidate(u, w) {
                    heap.push(c);
                }
            }
        }
    }

    /// Collapse `v` into `u`, moving `u` to `np`. Updates faces,
    /// incidence, quadric, version, and the live-face count. The kept
    /// vertex is `u`; its blend record stores `(v, t)`.
    fn apply_collapse(&mut self, u: u32, v: u32, np: [f64; 3], t: f64) {
        // Remove the (≤2) faces that touch both u and v; retarget the
        // rest of v's faces onto u.
        let vfs = self.incident(v);
        for fi in vfs {
            let tri = self.faces[fi as usize].unwrap();
            if tri.contains(&u) {
                // Shared face vanishes.
                self.faces[fi as usize] = None;
                self.live_faces -= 1;
            } else {
                // Retarget v → u.
                let nt = tri.map(|c| if c == v { u } else { c });
                // Guard against a face becoming degenerate (two equal
                // corners) — drop it rather than emit a zero-area tri.
                if nt[0] == nt[1] || nt[1] == nt[2] || nt[0] == nt[2] {
                    self.faces[fi as usize] = None;
                    self.live_faces -= 1;
                } else {
                    self.faces[fi as usize] = Some(nt);
                    self.vfaces[u as usize].push(fi);
                }
            }
        }
        // u absorbs v's quadric and moves to np.
        self.quad[u as usize] = self.quad[u as usize].add(&self.quad[v as usize]);
        self.pos[u as usize] = np;
        // A boundary vertex stays a boundary vertex after merging into it.
        self.boundary[u as usize] = self.boundary[u as usize] || self.boundary[v as usize];
        self.alive[v as usize] = false;
        self.ver[u as usize] = self.ver[u as usize].wrapping_add(1);
        self.ver[v as usize] = self.ver[v as usize].wrapping_add(1);
        // Record the merge so attributes can be reconstructed: v blends
        // into u at fraction t.
        self.merge_log[v as usize] = (u, v, t);
        // Bump versions of u's one-ring so their stale candidates drop.
        for w in self.one_ring(u) {
            self.ver[w as usize] = self.ver[w as usize].wrapping_add(1);
        }
    }

    /// Emit the simplified primitive, blending attributes from `welded`
    /// along the recorded collapse chain.
    fn into_primitive(self, welded: &Primitive, empty_out: Primitive) -> Primitive {
        // Surviving faces, remapped to a compact vertex pool.
        let live: Vec<[u32; 3]> = self.faces.iter().filter_map(|f| *f).collect();
        if live.is_empty() {
            return empty_out;
        }

        // Compact: only vertices referenced by a live face survive.
        let nverts = self.alive.len();
        let mut remap = vec![u32::MAX; nverts];
        let mut order: Vec<u32> = Vec::new();
        for &[a, b, c] in &live {
            for v in [a, b, c] {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = order.len() as u32;
                    order.push(v);
                }
            }
        }

        // Build the blend weight per surviving vertex by following the
        // merge chain: each absorbed vertex contributed a fractional
        // position; for attributes we use the *final* split as a 0/1-ish
        // blend toward the absorbed endpoint. We approximate the merged
        // attribute as the average of the endpoint attributes weighted by
        // the recorded t at the last absorption — a stable, deterministic
        // rule that matches subdivide_loop's midpoint convention when
        // t = 0.5. Position comes straight from the metric solve.
        let mut out = welded.clone();
        out.topology = Topology::Triangles;

        // Positions from the metric (f64 → f32).
        let mut npos: Vec<[f32; 3]> = Vec::with_capacity(order.len());
        for &v in &order {
            let p = self.pos[v as usize];
            npos.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
        out.positions = npos;

        // Helper: blended attribute row for survivor v, lerping the kept
        // vertex's original row toward the last vertex it absorbed.
        // `merge_log[v] = (kept, absorbed, t)`; when nothing was absorbed
        // (kept == absorbed) the row is the original.
        let blend_t = |v: u32| -> (u32, u32, f32) {
            let (kept, absorbed, t) = self.merge_log[v as usize];
            // Only meaningful when v is the kept vertex.
            if kept == v && absorbed != v {
                (v, absorbed, t as f32)
            } else {
                (v, v, 0.0)
            }
        };

        macro_rules! lerp_set {
            ($src:expr, $k:expr) => {{
                let src = $src;
                let mut v: Vec<[f32; $k]> = Vec::with_capacity(order.len());
                for &o in &order {
                    let (a, b, t) = blend_t(o);
                    let ra = src.get(a as usize).copied().unwrap_or([0.0; $k]);
                    let rb = src.get(b as usize).copied().unwrap_or([0.0; $k]);
                    let mut row = [0.0f32; $k];
                    for k in 0..$k {
                        row[k] = ra[k] * (1.0 - t) + rb[k] * t;
                    }
                    v.push(row);
                }
                v
            }};
        }

        if let Some(ns) = &welded.normals {
            let mut v = lerp_set!(ns, 3);
            for nrm in v.iter_mut() {
                let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
                if len.is_finite() && len > 0.0 {
                    nrm[0] /= len;
                    nrm[1] /= len;
                    nrm[2] /= len;
                }
            }
            out.normals = Some(v);
        }
        if let Some(ts) = &welded.tangents {
            let mut v: Vec<[f32; 4]> = Vec::with_capacity(order.len());
            for &o in &order {
                let (a, b, t) = blend_t(o);
                let ta = ts.get(a as usize).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]);
                let tb = ts.get(b as usize).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]);
                let x = ta[0] * (1.0 - t) + tb[0] * t;
                let y = ta[1] * (1.0 - t) + tb[1] * t;
                let z = ta[2] * (1.0 - t) + tb[2] * t;
                let len = (x * x + y * y + z * z).sqrt();
                let (nx, ny, nz) = if len.is_finite() && len > 0.0 {
                    (x / len, y / len, z / len)
                } else {
                    (1.0, 0.0, 0.0)
                };
                // Handedness from the kept (lower-index) endpoint.
                v.push([nx, ny, nz, ta[3]]);
            }
            out.tangents = Some(v);
        }
        out.uvs = welded.uvs.iter().map(|set| lerp_set!(set, 2)).collect();
        out.colors = welded.colors.iter().map(|set| lerp_set!(set, 4)).collect();
        if let Some(ws) = &welded.weights {
            let mut v = lerp_set!(ws, 4);
            for w in v.iter_mut() {
                let s = w[0] + w[1] + w[2] + w[3];
                if s > 0.0 && s.is_finite() {
                    for c in w.iter_mut() {
                        *c /= s;
                    }
                }
            }
            out.weights = Some(v);
        }
        if let Some(js) = &welded.joints {
            // Categorical: take the kept endpoint's quad.
            let mut v: Vec<[u16; 4]> = Vec::with_capacity(order.len());
            for &o in &order {
                let (a, _b, _t) = blend_t(o);
                v.push(js.get(a as usize).copied().unwrap_or([0; 4]));
            }
            out.joints = Some(v);
        }
        // Morph deltas: same linear blend, per target — every delta
        // buffer (primary slots + in-between shapes) through one rule.
        out.targets = welded
            .targets
            .iter()
            .map(|tgt| tgt.map_buffers(|d| lerp_set!(d, 3)))
            .collect();

        // Remapped index buffer.
        let mut idx: Vec<u32> = Vec::with_capacity(live.len() * 3);
        for &[a, b, c] in &live {
            idx.push(remap[a as usize]);
            idx.push(remap[b as usize]);
            idx.push(remap[c as usize]);
        }
        let vcount = out.positions.len();
        out.indices = Some(if vcount <= 65_536 {
            Indices::U16(idx.iter().map(|&i| i as u16).collect())
        } else {
            Indices::U32(idx)
        });

        out
    }
}
