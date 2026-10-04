//! Ray + ray-triangle / ray-AABB intersection primitives.
//!
//! These are the low-level computational-geometry building blocks a
//! renderer / picker / collision-probe sits on top of:
//!
//! * [`Ray`] — value type with `origin` + (non-unit-required) `direction`,
//!   plus a `point_at(t)` helper.
//! * [`BoundingBox::intersect_ray`] (in `scene.rs`) — slab test returning
//!   the entry / exit parametric distances along the ray.
//! * [`Primitive::intersect_ray`] / [`Primitive::any_ray_intersection`]
//!   (in `mesh.rs`) — closest-hit / shadow-ray queries against the
//!   primitive's triangle tessellation.
//!
//! The ray-triangle math is the **Möller-Trumbore** closed form
//! (T. Möller & B. Trumbore, "Fast, Minimum Storage Ray-Triangle
//! Intersection", Journal of Graphics Tools 2(1), 1997 — open-access
//! published algorithm; no source-code dependency on any external
//! implementation). For a triangle with corners `P0, P1, P2` and edges
//! `E1 = P1 - P0`, `E2 = P2 - P0`, the system
//!
//! ```text
//!     O + t*D = P0 + u*E1 + v*E2
//! ```
//!
//! is solved with Cramer's rule on the 3x3 matrix `[-D | E1 | E2]`,
//! using the scalar-triple-product identity
//! `det([-D | E1 | E2]) = D · (E2 × E1) = -D · (E1 × E2)`. Defining
//! `P = D × E2`, the determinant is `det = E1 · P`. The barycentric
//! coordinates and ray parameter fall out as
//!
//! ```text
//!     u = ((O - P0) · P) / det
//!     Q = (O - P0) × E1
//!     v = (D · Q) / det
//!     t = (E2 · Q) / det
//! ```
//!
//! A hit requires `u ∈ [0, 1]`, `v ∈ [0, 1]`, `u + v ≤ 1` (inside the
//! triangle), `t ≥ 0` (in front of the ray origin), and `|det| > 0`
//! (the ray is not parallel to the triangle plane). The same
//! cross-product machinery already drives [`crate::Primitive::compute_normals`]
//! and [`crate::Primitive::surface_area`].
//!
//! The ray-AABB math is the **slab method** (Kay & Kajiya, "Ray Tracing
//! Complex Scenes", SIGGRAPH 1986). For each axis the ray's entry /
//! exit distance through the box's two parallel slabs is
//! `(min - O) / D` and `(max - O) / D`; the overall hit interval is the
//! intersection across all three axes. An axis-parallel ray
//! (`D[axis] == 0`) is handled by short-circuiting on the origin's
//! position relative to the slab (inside = pass-through, outside =
//! immediate miss).

/// A directed half-line in 3D space.
///
/// `origin` is the start point and `direction` is the displacement
/// vector — **not required to be unit length**. The ray parameter `t`
/// returned by intersection queries is measured along `direction`, so
/// `ray.point_at(1.0)` lands at `origin + direction`. Callers that need
/// `t` in world units should pass a unit `direction`.
///
/// A zero-length `direction` is a degenerate ray; intersection helpers
/// return `None` rather than panic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
}

impl Ray {
    /// Construct a ray from its two components.
    pub fn new(origin: [f32; 3], direction: [f32; 3]) -> Self {
        Self { origin, direction }
    }

    /// Point on the ray at parameter `t`: `origin + t * direction`.
    pub fn point_at(self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }
}

/// Closest-hit record produced by [`crate::Primitive::intersect_ray`].
///
/// `t` is the ray parameter at the hit point (so `ray.point_at(t)` is
/// the intersection point). `triangle_index` is the index into the
/// `Vec<[u32; 3]>` returned by [`crate::Primitive::triangle_indices`] —
/// callers needing the per-triangle vertex indices look it up there.
///
/// `barycentric` is the in-triangle coordinate `[w, u, v]` with
/// `w = 1 - u - v` so that the hit point reconstructs as
/// `w * P0 + u * P1 + v * P2` (the standard barycentric definition; the
/// Möller-Trumbore (u, v) are the latter two). `front_face` is `true`
/// when the ray approaches the triangle from its CCW-front-facing
/// side (right-handed, glTF-aligned) — i.e. when the ray direction
/// opposes the outward face normal `N = E1 × E2`, equivalently
/// `D · N < 0`. A back-side hit (ray parallel to the normal) reports
/// `false`. Both sides report a hit by default; `front_face` is the
/// signal a caller pairs with [`crate::Material::double_sided`] to
/// decide whether to act on it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub t: f32,
    pub triangle_index: usize,
    pub barycentric: [f32; 3],
    pub front_face: bool,
}

/// Möller-Trumbore ray-triangle intersection.
///
/// Returns `Some((t, u, v, front_face))` when the ray strikes the
/// triangle, `None` when it misses (parallel, behind the origin, or
/// outside the triangle's barycentric simplex). The barycentric pair
/// `(u, v)` corresponds to corners `(P1, P2)`; the third coordinate is
/// `w = 1 - u - v`.
///
/// `t_max` is the upper bound on the ray parameter — a hit beyond
/// `t_max` is reported as a miss. Pass `f32::INFINITY` for unbounded
/// queries.
///
/// A degenerate triangle (`|det| ≤ epsilon`, i.e. the ray is parallel
/// to the triangle plane or the triangle itself has zero area) is
/// silently treated as a miss, matching how the rest of the crate
/// handles zero-area faces (see [`crate::Primitive::compute_normals`] /
/// [`crate::Primitive::surface_area`]). NaN- or Inf-producing math
/// likewise returns `None`.
///
/// `front_face` is `true` when the ray approaches the triangle from
/// its CCW-front side — i.e. the ray direction opposes the outward
/// face normal `N = E1 × E2` (`D · N < 0`, equivalent to `det > 0`
/// since `det = -D · N`). A ray parallel to the normal hits the back
/// and reports `false`. Both sides hit by default; double-sided
/// behaviour is the caller's responsibility via the
/// [`crate::Material::double_sided`] flag.
pub fn intersect_triangle(
    ray: Ray,
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    t_max: f32,
) -> Option<(f32, f32, f32, bool)> {
    moller_trumbore(ray, p0, p1, p2, 0.0, t_max)
}

/// Slab-method ray-AABB intersection.
///
/// Returns `Some((t_enter, t_exit))` when the ray enters and exits the
/// box at finite parameters within `[0, t_max]`, `None` otherwise. The
/// returned interval is clamped to `t_enter ≥ 0` — a ray whose origin
/// is inside the box reports `t_enter = 0` and `t_exit` at the exit
/// face.
///
/// An axis-parallel ray (`direction[axis] == 0`) on which the origin
/// lies inside the corresponding slab passes through that axis test
/// untouched; an origin outside the slab is an immediate miss. NaN /
/// Inf components on either side return `None`.
///
/// This is the per-node early-out a BVH traverser calls before
/// recursing into the children that hold the actual triangles.
pub fn intersect_aabb(ray: Ray, min: [f32; 3], max: [f32; 3], t_max: f32) -> Option<(f32, f32)> {
    let mut t_enter = 0.0_f32;
    let mut t_exit = t_max;

    for axis in 0..3 {
        let o = ray.origin[axis];
        let d = ray.direction[axis];
        let a = min[axis];
        let b = max[axis];

        if !o.is_finite() || !d.is_finite() || !a.is_finite() || !b.is_finite() {
            return None;
        }

        if d.abs() < 1e-30 {
            // Axis-parallel: origin must already lie inside this slab,
            // otherwise the ray never enters the box.
            if o < a || o > b {
                return None;
            }
            continue;
        }

        let inv_d = 1.0 / d;
        let mut t0 = (a - o) * inv_d;
        let mut t1 = (b - o) * inv_d;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }

        if t0 > t_enter {
            t_enter = t0;
        }
        if t1 < t_exit {
            t_exit = t1;
        }
        if t_enter > t_exit {
            return None;
        }
    }

    Some((t_enter, t_exit))
}

/// Per-query parameters shared by every accelerated ray query
/// ([`crate::Bvh`] and [`crate::InstanceBvh`]).
///
/// * `t_min` / `t_max` — the accepted parametric interval. A hit
///   counts only when `t_min <= t <= t_max`. Shadow / occlusion rays
///   towards a light at distance `L` along a unit direction pass
///   `t_max = L` (minus a small epsilon); secondary rays leaving a
///   surface typically pass a small positive `t_min` (or offset the
///   origin) to avoid re-hitting the surface they start on.
/// * `triangle_test` — leaf intersection algorithm (see
///   [`TriangleTest`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayQuery {
    pub t_min: f32,
    pub t_max: f32,
    pub triangle_test: TriangleTest,
}

impl RayQuery {
    /// `[0, t_max]` with the default (Möller-Trumbore) test.
    pub fn new(t_max: f32) -> Self {
        Self {
            t_min: 0.0,
            t_max,
            triangle_test: TriangleTest::MollerTrumbore,
        }
    }

    /// Builder: set the lower bound of the accepted interval.
    pub fn with_t_min(mut self, t_min: f32) -> Self {
        self.t_min = t_min;
        self
    }

    /// Builder: select the leaf triangle algorithm.
    pub fn with_triangle_test(mut self, test: TriangleTest) -> Self {
        self.triangle_test = test;
        self
    }
}

impl Default for RayQuery {
    /// `[0, ∞)` with Möller-Trumbore.
    fn default() -> Self {
        Self::new(f32::INFINITY)
    }
}

/// Which ray-triangle algorithm a traversal uses at the leaves.
///
/// * [`TriangleTest::MollerTrumbore`] — the closed-form Cramer's-rule
///   solve described in the module docs (Möller & Trumbore 1997).
///   Fast, and bit-identical to [`intersect_triangle`] /
///   [`crate::Primitive::intersect_ray`], but not watertight: a ray
///   passing exactly through a shared edge or vertex can slip between
///   the two adjacent triangles because each evaluates its own edge
///   functions with different rounding, and an absolute `1e-8`
///   determinant epsilon rejects very small triangles.
/// * [`TriangleTest::Watertight`] — S. Woop, C. Benthin & I. Wald,
///   "Watertight Ray/Triangle Intersection", Journal of Computer
///   Graphics Techniques 2(1), 2013. The ray is turned into a
///   unit-`z` ray by an axis permutation + shear computed once per ray
///   ([`PreparedRay`]); the triangle's vertices are transformed into
///   that frame and the three 2-D edge functions are evaluated with
///   the exact same arithmetic for a shared edge on both adjacent
///   triangles, so no ray leaks through a closed mesh. Edge functions
///   that round to exactly zero are re-evaluated in `f64` as the
///   paper prescribes. No epsilon is involved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TriangleTest {
    /// Möller-Trumbore (default; matches the brute-force primitive
    /// queries bit-for-bit).
    #[default]
    MollerTrumbore,
    /// Woop-Benthin-Wald watertight test.
    Watertight,
}

/// `1 + 2·γ(3)` with `γ(n) = n·u / (1 - n·u)`, `u = 2⁻²⁴` — the
/// conservative far-distance inflation factor from T. Ize, "Robust
/// BVH Ray Traversal", Journal of Computer Graphics Techniques 2(2),
/// 2013. Scaling the slab-test exit distance by this factor
/// guarantees that floating-point rounding in `(bound - o) · (1/d)`
/// never culls a box the exact ray actually touches.
/// Numerically: `γ(3) ≈ 1.788e-7`, so the factor is `≈ 1.000000358`,
/// which rounds to the next-but-one `f32` above `1.0` (conservative).
const ROBUST_TFAR_SCALE: f32 = 1.000_000_4;

/// A [`Ray`] with every per-ray constant a BVH traversal needs
/// precomputed once: the reciprocal direction for the slab test, the
/// per-axis direction sign (which box corner is "near"), and the
/// Woop-Benthin-Wald permutation + shear for the watertight triangle
/// test.
///
/// # Robust slab test
///
/// [`PreparedRay::slab`] follows A. Williams, S. Barrus, R. K. Morley
/// & P. Shirley, "An Efficient and Robust Ray-Box Intersection
/// Algorithm", Journal of Graphics Tools 10(1), 2005: the reciprocal
/// direction is taken with IEEE semantics so a zero component becomes
/// `±∞` (signed by the zero's sign bit, so `-0.0` is handled), and the
/// near / far corner is selected by the reciprocal's sign instead of
/// a swap. The `0 · ∞ = NaN` case (ray origin exactly on a slab plane
/// of an axis it runs parallel to) is absorbed by `f32::max` /
/// `f32::min`, which return the non-NaN operand — the axis then simply
/// does not constrain the interval, which is the conservative answer.
/// The exit distance is inflated by `1 + 2γ(3)` per Ize 2013 (see
/// `ROBUST_TFAR_SCALE`) so rounding can never produce a false miss.
///
/// A ray whose origin or direction has a non-finite component, or
/// whose direction is the zero vector, is flagged invalid at
/// construction ([`PreparedRay::is_valid`]) and every traversal
/// returns "no hit" for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparedRay {
    /// The source ray.
    pub ray: Ray,
    /// Component-wise `1 / direction` (IEEE: `1/±0 = ±∞`).
    pub inv_dir: [f32; 3],
    /// `true` where `inv_dir` is negative (sign bit set) — the slab
    /// test then reads `max` as the near plane on that axis.
    pub neg: [bool; 3],
    /// Watertight-test axis permutation: `kz` is the dominant
    /// direction axis, `kx` / `ky` the remaining two (swapped when
    /// `direction[kz] < 0` to preserve winding).
    pub k: [usize; 3],
    /// Watertight-test shear constants
    /// `[d[kx]/d[kz], d[ky]/d[kz], 1/d[kz]]`.
    pub shear: [f32; 3],
    valid: bool,
}

impl PreparedRay {
    /// Precompute the per-ray traversal constants for `ray`.
    pub fn new(ray: Ray) -> Self {
        let o = ray.origin;
        let d = ray.direction;
        let finite = o.iter().chain(d.iter()).all(|v| v.is_finite());
        let nonzero = d[0] != 0.0 || d[1] != 0.0 || d[2] != 0.0;
        let inv_dir = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];
        let neg = [
            inv_dir[0].is_sign_negative(),
            inv_dir[1].is_sign_negative(),
            inv_dir[2].is_sign_negative(),
        ];
        // Woop et al. 2013 §3: dominant axis becomes z.
        let ad = [d[0].abs(), d[1].abs(), d[2].abs()];
        let kz = if ad[0] >= ad[1] && ad[0] >= ad[2] {
            0
        } else if ad[1] >= ad[2] {
            1
        } else {
            2
        };
        let mut kx = (kz + 1) % 3;
        let mut ky = (kx + 1) % 3;
        if d[kz] < 0.0 {
            std::mem::swap(&mut kx, &mut ky);
        }
        let shear = if d[kz] != 0.0 {
            [d[kx] / d[kz], d[ky] / d[kz], 1.0 / d[kz]]
        } else {
            [0.0, 0.0, 0.0]
        };
        Self {
            ray,
            inv_dir,
            neg,
            k: [kx, ky, kz],
            shear,
            valid: finite && nonzero,
        }
    }

    /// `false` for a ray with a non-finite component or a zero
    /// direction — such a ray intersects nothing.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Robust slab test against the box `[min, max]` restricted to the
    /// parametric interval `[t_min, t_max]`.
    ///
    /// Returns the entry distance (clamped to `t_min`) when the ray
    /// overlaps the box inside the interval, `None` otherwise. See the
    /// type-level docs for the NaN / ±∞ / rounding contract.
    #[inline(always)]
    pub fn slab(&self, min: &[f32; 3], max: &[f32; 3], t_min: f32, t_max: f32) -> Option<f32> {
        let o = &self.ray.origin;
        let inv = &self.inv_dir;
        let (nx, fx) = if self.neg[0] {
            (max[0], min[0])
        } else {
            (min[0], max[0])
        };
        let (ny, fy) = if self.neg[1] {
            (max[1], min[1])
        } else {
            (min[1], max[1])
        };
        let (nz, fz) = if self.neg[2] {
            (max[2], min[2])
        } else {
            (min[2], max[2])
        };
        let tx0 = (nx - o[0]) * inv[0];
        let tx1 = (fx - o[0]) * inv[0];
        let ty0 = (ny - o[1]) * inv[1];
        let ty1 = (fy - o[1]) * inv[1];
        let tz0 = (nz - o[2]) * inv[2];
        let tz1 = (fz - o[2]) * inv[2];
        // f32::max / f32::min return the non-NaN operand: a NaN
        // (0 · ∞, origin on a slab plane of a parallel axis) leaves
        // the interval unconstrained on that axis.
        let t0 = t_min.max(tx0).max(ty0).max(tz0);
        let t1 = t_max.min(tx1.min(ty1).min(tz1) * ROBUST_TFAR_SCALE);
        if t0 <= t1 {
            Some(t0)
        } else {
            None
        }
    }

    /// Ray-triangle test with the selected algorithm, accepting hits
    /// with `t ∈ [t_min, t_max]`.
    ///
    /// Returns `(t, u, v, front_face)` with the same conventions as
    /// [`intersect_triangle`]: `(u, v)` weight corners `(p1, p2)`,
    /// `w = 1 - u - v` weights `p0`, and `front_face` is `true` when
    /// the ray direction opposes `N = (p1 - p0) × (p2 - p0)`.
    #[inline]
    pub fn intersect_triangle(
        &self,
        test: TriangleTest,
        p0: [f32; 3],
        p1: [f32; 3],
        p2: [f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<(f32, f32, f32, bool)> {
        match test {
            TriangleTest::MollerTrumbore => moller_trumbore(self.ray, p0, p1, p2, t_min, t_max),
            TriangleTest::Watertight => self.watertight(p0, p1, p2, t_min, t_max),
        }
    }

    /// Woop-Benthin-Wald watertight ray-triangle intersection (JCGT
    /// 2(1), 2013, Fig. 1 / §3) with backface culling disabled.
    fn watertight(
        &self,
        p0: [f32; 3],
        p1: [f32; 3],
        p2: [f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<(f32, f32, f32, bool)> {
        let [kx, ky, kz] = self.k;
        let [sx, sy, sz] = self.shear;
        let o = self.ray.origin;
        // Vertices relative to the ray origin.
        let a = [p0[0] - o[0], p0[1] - o[1], p0[2] - o[2]];
        let b = [p1[0] - o[0], p1[1] - o[1], p1[2] - o[2]];
        let c = [p2[0] - o[0], p2[1] - o[1], p2[2] - o[2]];
        // Shear + scale into the ray's unit-z space.
        let ax = a[kx] - sx * a[kz];
        let ay = a[ky] - sy * a[kz];
        let bx = b[kx] - sx * b[kz];
        let by = b[ky] - sy * b[kz];
        let cx = c[kx] - sx * c[kz];
        let cy = c[ky] - sy * c[kz];
        // Scaled barycentric edge functions.
        let mut uu = cx * by - cy * bx;
        let mut vv = ax * cy - ay * cx;
        let mut ww = bx * ay - by * ax;
        // Exactly-zero edge function: fall back to double precision
        // (the paper's prescribed fix for edge / vertex hits).
        if uu == 0.0 || vv == 0.0 || ww == 0.0 {
            let (ax, ay, bx, by, cx, cy) = (
                ax as f64, ay as f64, bx as f64, by as f64, cx as f64, cy as f64,
            );
            uu = (cx * by - cy * bx) as f32;
            vv = (ax * cy - ay * cx) as f32;
            ww = (bx * ay - by * ax) as f32;
        }
        if (uu < 0.0 || vv < 0.0 || ww < 0.0) && (uu > 0.0 || vv > 0.0 || ww > 0.0) {
            return None;
        }
        let det = uu + vv + ww;
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let az = sz * a[kz];
        let bz = sz * b[kz];
        let cz = sz * c[kz];
        let t_scaled = uu * az + vv * bz + ww * cz;
        let rcp = 1.0 / det;
        let t = t_scaled * rcp;
        if !(t >= t_min && t <= t_max) {
            return None;
        }
        // uu weights p0, vv weights p1, ww weights p2.
        let u = vv * rcp;
        let v = ww * rcp;
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let front = dot(self.ray.direction, cross(e1, e2)) < 0.0;
        Some((t, u, v, front))
    }
}

/// Möller-Trumbore with an explicit `[t_min, t_max]` acceptance
/// interval; [`intersect_triangle`] is the `t_min = 0` case.
#[inline]
fn moller_trumbore(
    ray: Ray,
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    t_min: f32,
    t_max: f32,
) -> Option<(f32, f32, f32, bool)> {
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let p_vec = cross(ray.direction, e2);
    let det = dot(e1, p_vec);
    if !det.is_finite() || det.abs() < 1e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = [
        ray.origin[0] - p0[0],
        ray.origin[1] - p0[1],
        ray.origin[2] - p0[2],
    ];
    let u = dot(s, p_vec) * inv_det;
    if !u.is_finite() || !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(ray.direction, q) * inv_det;
    if !v.is_finite() || v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot(e2, q) * inv_det;
    if !t.is_finite() || t < t_min || t > t_max {
        return None;
    }
    Some((t, u, v, det > 0.0))
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_at_zero_is_origin() {
        let r = Ray::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert_eq!(r.point_at(0.0), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn point_at_one_is_origin_plus_direction() {
        let r = Ray::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert_eq!(r.point_at(1.0), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn triangle_centre_hit_from_back_side() {
        // Triangle in z=1 plane, CCW from +Z (outward normal +Z). A
        // ray shooting +Z from the origin approaches the triangle
        // from the -Z side (the back) and reports `front_face = false`.
        let r = Ray::new([0.3333, 0.3333, 0.0], [0.0, 0.0, 1.0]);
        let hit = intersect_triangle(
            r,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            f32::INFINITY,
        )
        .expect("centre hit");
        assert!((hit.0 - 1.0).abs() < 1e-5, "t = {}", hit.0);
        // u + v ≈ 2/3, both ≈ 1/3.
        assert!((hit.1 - 0.3333).abs() < 1e-3);
        assert!((hit.2 - 0.3333).abs() < 1e-3);
        assert!(!hit.3, "ray along the normal hits from the back side");
    }

    #[test]
    fn triangle_front_face_hit_from_above() {
        // Same triangle, ray from above shooting -Z hits the front face
        // (the ray opposes the outward normal +Z).
        let r = Ray::new([0.3333, 0.3333, 2.0], [0.0, 0.0, -1.0]);
        let hit = intersect_triangle(
            r,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            f32::INFINITY,
        )
        .expect("front hit");
        assert!(hit.3, "ray opposing the normal hits the front face");
    }

    #[test]
    fn ray_parallel_to_triangle_misses() {
        let r = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(intersect_triangle(
            r,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            f32::INFINITY,
        )
        .is_none());
    }

    #[test]
    fn triangle_outside_simplex_misses() {
        // Hit point would be at (2,2) on the z=1 plane — outside the
        // unit triangle.
        let r = Ray::new([2.0, 2.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(intersect_triangle(
            r,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            f32::INFINITY,
        )
        .is_none());
    }

    #[test]
    fn behind_origin_misses() {
        // Hit point would be at t = -1.
        let r = Ray::new([0.3333, 0.3333, 2.0], [0.0, 0.0, 1.0]);
        assert!(intersect_triangle(
            r,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            f32::INFINITY,
        )
        .is_none());
    }

    #[test]
    fn t_max_cull() {
        let r = Ray::new([0.3333, 0.3333, 0.0], [0.0, 0.0, 1.0]);
        // Hit is at t = 1.0; t_max = 0.5 should cull it.
        assert!(
            intersect_triangle(r, [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], 0.5,)
                .is_none()
        );
    }

    #[test]
    fn aabb_axis_aligned_through_centre() {
        let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let hit = intersect_aabb(r, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], f32::INFINITY).unwrap();
        assert!((hit.0 - 1.0).abs() < 1e-6, "enter t = {}", hit.0);
        assert!((hit.1 - 2.0).abs() < 1e-6, "exit t = {}", hit.1);
    }

    #[test]
    fn aabb_origin_inside_box() {
        let r = Ray::new([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let hit = intersect_aabb(r, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], f32::INFINITY).unwrap();
        assert_eq!(hit.0, 0.0, "origin inside → t_enter = 0");
        assert!((hit.1 - 0.5).abs() < 1e-6, "exit t = {}", hit.1);
    }

    #[test]
    fn aabb_misses_to_the_side() {
        let r = Ray::new([-1.0, 2.0, 0.5], [1.0, 0.0, 0.0]);
        assert!(
            intersect_aabb(r, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], f32::INFINITY).is_none(),
            "ray passes outside the box in y"
        );
    }

    #[test]
    fn aabb_axis_parallel_inside_slab_passes() {
        // Ray with d=(0,0,0)-in-x sense doesn't exist (we'd need a
        // non-degenerate direction). But d=(0,1,0) with origin inside
        // the x-slab: y-axis test does the work, x-axis short-circuits.
        let r = Ray::new([0.5, -1.0, 0.5], [0.0, 1.0, 0.0]);
        let hit = intersect_aabb(r, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], f32::INFINITY).unwrap();
        assert!((hit.0 - 1.0).abs() < 1e-6);
        assert!((hit.1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn aabb_axis_parallel_outside_slab_misses() {
        // Same as above but x-origin outside the x-slab.
        let r = Ray::new([2.0, -1.0, 0.5], [0.0, 1.0, 0.0]);
        assert!(intersect_aabb(r, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], f32::INFINITY).is_none());
    }
}
