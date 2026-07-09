//! Parametric extruded-solid construction — a planar profile (outer
//! boundary + optional hole boundaries) swept along a straight
//! direction into a closed, watertight triangle [`Primitive`].
//!
//! This is the tessellation kernel a swept-solid format producer
//! consumes. The semantics mirror the IFC 4.3 extruded-area-solid
//! representation (staged at
//! `docs/3d/ifc/ifc43-entity-IfcExtrudedAreaSolid.html`, §8.8.3.15):
//! the profile lies in the *xy* plane of its own coordinate system,
//! the extrusion direction may be any direction **not perpendicular
//! to the z axis** (i.e. its z component must be non-zero — oblique
//! extrusions are legal), the translation length is given by a
//! positive `depth` along the unit direction, and holes in the
//! planar area sweep into holes in the solid. The outer bound is
//! counter-clockwise seen from `+z` and inner bounds are clockwise
//! (input loops in the opposite winding are normalised
//! automatically). Placement into the parent scene is the caller's
//! job — apply a node [`Transform`](crate::Transform) for the
//! object-coordinate-system positioning the format carries.
//!
//! Cap triangulation is ear clipping. The correctness foundation is
//! the two-ears theorem (Meisters, "Polygons Have Ears", *American
//! Mathematical Monthly* 82(6), 1975: every simple polygon with more
//! than three vertices has at least two non-overlapping ears), which
//! guarantees the clipping loop always finds an ear on a simple
//! polygon and therefore terminates with exactly `n - 2` triangles.
//! Holes are reduced to the simple-polygon case by splicing each
//! hole into the outer ring through a zero-width bridge: the bridge
//! is traversed once in each direction, so the merged ring stays a
//! (weakly) simple counter-clockwise polygon and the standard ear
//! clip applies. The bridge end-point on the outer ring is found by
//! casting a `+x` ray from the hole's maximum-`x` vertex — the
//! nearest crossing edge yields a mutually visible vertex (refined
//! against reflex vertices inside the candidate triangle, picking
//! the one with the smallest angle to the ray, then the nearest).
//! All of this is built from the standard plane-geometry
//! definitions (signed area / cross product / barycentric sign
//! tests) this crate's other reductions already rest on, plus the
//! staged IFC entity page cited above for the sweep semantics.

use crate::mesh::{Indices, Primitive, Topology};

/// A planar profile in the *xy* plane: one outer boundary loop and
/// zero or more hole loops, each a list of 2D points.
///
/// The natural winding is outer counter-clockwise / holes clockwise
/// (seen from `+z`) — the convention the IFC 4.3 extruded-area-solid
/// texture-mapping rules state for arbitrary profiles
/// (`docs/3d/ifc/ifc43-entity-IfcExtrudedAreaSolid.html`) — but both
/// [`Profile2D::triangulate`] and [`Profile2D::extrude`] normalise
/// either input winding, so loops may be supplied either way round.
///
/// A closing duplicate point (`last == first`, common in wire
/// formats that store closed polylines explicitly closed) and exact
/// consecutive duplicate points are ignored. Holes must be strictly
/// inside the outer boundary and pairwise disjoint; loops must be
/// simple (non-self-intersecting). Violations are not detected — the
/// triangulation of a non-simple profile is unspecified (it still
/// terminates, but the covered area is meaningless).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Profile2D {
    /// Outer boundary loop.
    pub outer: Vec<[f32; 2]>,
    /// Hole loops, each strictly inside `outer` and pairwise
    /// disjoint.
    pub holes: Vec<Vec<[f32; 2]>>,
}

/// One vertex of the working ring: 2D position + index into the
/// flattened original vertex list (`outer ++ holes[0] ++ …`).
#[derive(Clone, Copy)]
struct RingVertex {
    p: [f64; 2],
    orig: u32,
}

/// `2 ×` signed area of triangle `(a, b, c)` — positive when the
/// corners run counter-clockwise.
fn cross2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// `2 ×` signed loop area (shoelace). Positive = counter-clockwise.
fn signed_area2(pts: &[RingVertex]) -> f64 {
    let n = pts.len();
    let mut s = 0.0;
    for i in 0..n {
        let a = pts[i].p;
        let b = pts[(i + 1) % n].p;
        s += a[0] * b[1] - b[0] * a[1];
    }
    s
}

/// Inclusive point-in-triangle test for a counter-clockwise triangle:
/// `true` when `p` is inside or on the boundary of `(a, b, c)`.
fn point_in_ccw_triangle(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    cross2(a, b, p) >= 0.0 && cross2(b, c, p) >= 0.0 && cross2(c, a, p) >= 0.0
}

/// Drop a closing duplicate (`last == first`) and exact consecutive
/// duplicates, tagging each survivor with its flattened original
/// index (starting at `base`).
fn clean_loop(pts: &[[f32; 2]], base: u32) -> Vec<RingVertex> {
    let mut out: Vec<RingVertex> = Vec::with_capacity(pts.len());
    for (i, p) in pts.iter().enumerate() {
        let q = [f64::from(p[0]), f64::from(p[1])];
        if let Some(last) = out.last() {
            if last.p == q {
                continue;
            }
        }
        out.push(RingVertex {
            p: q,
            orig: base + i as u32,
        });
    }
    // Closing duplicate: last point repeats the first.
    while out.len() > 1 && out[0].p == out[out.len() - 1].p {
        out.pop();
    }
    out
}

/// Splice `hole` (clockwise) into `ring` (counter-clockwise) through
/// a zero-width bridge so the result is one weakly-simple CCW ring.
/// Returns `None` when the `+x` ray from the hole's rightmost vertex
/// never crosses the ring (hole outside the boundary — malformed
/// profile).
fn bridge_hole(ring: &[RingVertex], hole: &[RingVertex]) -> Option<Vec<RingVertex>> {
    let n = ring.len();

    // M = hole vertex with lexicographically largest (x, y).
    let m_i = (0..hole.len())
        .max_by(|&a, &b| {
            let (pa, pb) = (hole[a].p, hole[b].p);
            pa[0]
                .partial_cmp(&pb[0])
                .unwrap()
                .then(pa[1].partial_cmp(&pb[1]).unwrap())
        })
        .expect("hole loops are non-empty by construction");
    let m = hole[m_i].p;

    // Cast a +x ray from M; find the nearest ring edge crossing.
    // The half-open rule ((a.y > m.y) != (b.y > m.y)) counts a vertex
    // lying exactly on the ray's line toward exactly one of its two
    // incident edges, so crossings are never double-counted.
    let mut best: Option<(f64, usize)> = None; // (intersection x, edge start)
    for i in 0..n {
        let a = ring[i].p;
        let b = ring[(i + 1) % n].p;
        if (a[1] > m[1]) != (b[1] > m[1]) {
            let t = a[0] + (m[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if t >= m[0] && t.is_finite() && best.map_or(true, |(bx, _)| t < bx) {
                best = Some((t, i));
            }
        }
    }
    let (ix, ei) = best?;
    let inter = [ix, m[1]];
    let (a_i, b_i) = (ei, (ei + 1) % n);
    let (a, b) = (ring[a_i].p, ring[b_i].p);

    // Candidate visible ring vertex P: the intersection point itself
    // when it lands on a ring vertex, otherwise the crossed edge's
    // endpoint with the larger x. Reflex ring vertices inside (or, in
    // collinear cases, on) the triangle (M, I, P) can occlude P; among
    // them the one with the smallest angle to the +x ray (ties:
    // nearest) is mutually visible with M, so it replaces the
    // candidate. When I lands on a vertex the triangle collapses to
    // the segment M–I and the containment test degenerates to an
    // on-segment test — exactly the occluders that matter then.
    let cand = if a == inter {
        a_i
    } else if b == inter {
        b_i
    } else if a[0] > b[0] {
        a_i
    } else {
        b_i
    };
    let pc = ring[cand].p;
    let (t0, t1, t2) = if cross2(m, inter, pc) >= 0.0 {
        (m, inter, pc)
    } else {
        (m, pc, inter)
    };
    let mut p_i = cand;
    let mut best_key: Option<(f64, f64)> = None; // (tan(angle), distance)
    for j in 0..n {
        let prev = ring[(j + n - 1) % n].p;
        let cur = ring[j].p;
        let next = ring[(j + 1) % n].p;
        if cross2(prev, cur, next) >= 0.0 {
            continue; // convex vertices cannot occlude
        }
        if cur == m || cur == inter || cur == pc {
            continue;
        }
        if !point_in_ccw_triangle(cur, t0, t1, t2) {
            continue;
        }
        let dx = cur[0] - m[0];
        let dy = (cur[1] - m[1]).abs();
        if dx <= 0.0 {
            continue; // behind the ray origin
        }
        let key = (dy / dx, dx * dx + dy * dy);
        if best_key.map_or(true, |bk| key < bk) {
            best_key = Some(key);
            p_i = j;
        }
    }
    // Several ring slots can share P's coordinates (bridge duplicates
    // from an earlier hole). Any of them works geometrically; take
    // the first occurrence for determinism.
    let p_pos = ring[p_i].p;
    p_i = (0..n).find(|&j| ring[j].p == p_pos).unwrap_or(p_i);

    // Splice: … P, M, (hole walked CW back round to M), M, P, …
    let mut out = Vec::with_capacity(n + hole.len() + 2);
    out.extend_from_slice(&ring[..=p_i]);
    out.extend_from_slice(&hole[m_i..]);
    out.extend_from_slice(&hole[..=m_i]);
    out.push(ring[p_i]);
    out.extend_from_slice(&ring[p_i + 1..]);
    Some(out)
}

/// Ear-clip a weakly-simple counter-clockwise ring into triangles
/// over the original (flattened) vertex indices. Triangles come out
/// counter-clockwise; zero-area clips are removed without emitting.
fn ear_clip(ring: &[RingVertex], area_eps: f64) -> Option<Vec<[u32; 3]>> {
    let n = ring.len();
    if n < 3 {
        return None;
    }

    // Doubly-linked ring over index slots.
    let mut prev: Vec<usize> = (0..n).map(|i| (i + n - 1) % n).collect();
    let mut next: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
    let mut alive = vec![true; n];
    let cross_at = |i: usize, prev: &[usize], next: &[usize]| -> f64 {
        cross2(ring[prev[i]].p, ring[i].p, ring[next[i]].p)
    };
    // Reflex bookkeeping: only reflex vertices can occlude an ear (a
    // convex vertex strictly inside an ear triangle would imply a
    // reflex vertex inside it too — follow its boundary chain), so
    // ear tests scan this list instead of the whole ring. Entries are
    // validated lazily against the live flags.
    let mut reflex = vec![false; n];
    let mut reflex_list: Vec<usize> = Vec::new();
    for (i, r) in reflex.iter_mut().enumerate() {
        if cross_at(i, &prev, &next) < 0.0 {
            *r = true;
            reflex_list.push(i);
        }
    }

    let is_ear = |i: usize,
                  prev: &[usize],
                  next: &[usize],
                  alive: &[bool],
                  reflex: &[bool],
                  reflex_list: &[usize]|
     -> bool {
        let (a, b, c) = (ring[prev[i]].p, ring[i].p, ring[next[i]].p);
        if cross2(a, b, c) <= 0.0 {
            return false; // reflex or degenerate corner
        }
        for &j in reflex_list {
            if !alive[j] || !reflex[j] || j == prev[i] || j == i || j == next[i] {
                continue;
            }
            let p = ring[j].p;
            if p == a || p == b || p == c {
                continue; // bridge duplicate of an ear corner
            }
            if point_in_ccw_triangle(p, a, b, c) {
                return false;
            }
        }
        true
    };

    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(n.saturating_sub(2));
    let mut remaining = n;
    let mut cursor = 0usize;

    // Unlink i; emit its ear triangle unless (near-)zero-area.
    macro_rules! clip {
        ($i:expr, $emit:expr) => {{
            let i = $i;
            if $emit {
                let c2 = cross_at(i, &prev, &next);
                if c2 > area_eps {
                    tris.push([ring[prev[i]].orig, ring[i].orig, ring[next[i]].orig]);
                }
            }
            let (p, nx) = (prev[i], next[i]);
            next[p] = nx;
            prev[nx] = p;
            alive[i] = false;
            remaining -= 1;
            // Re-derive the two neighbours' convexity; a vertex that
            // turns reflex joins the scan list.
            for &k in &[p, nx] {
                let was = reflex[k];
                let now = cross_at(k, &prev, &next) < 0.0;
                reflex[k] = now;
                if now && !was {
                    reflex_list.push(k);
                }
            }
            cursor = nx;
        }};
    }

    while remaining > 3 {
        // Scan for an ear starting at the cursor.
        let mut found = None;
        let mut i = cursor;
        for _ in 0..remaining {
            if is_ear(i, &prev, &next, &alive, &reflex, &reflex_list) {
                found = Some(i);
                break;
            }
            i = next[i];
        }
        match found {
            Some(i) => clip!(i, true),
            None => {
                // No ear: drop a collinear / zero-area corner if one
                // exists (safe — removing it changes no geometry)…
                let mut j = cursor;
                let mut degenerate = None;
                let mut convex = None;
                for _ in 0..remaining {
                    let c2 = cross_at(j, &prev, &next);
                    if c2.abs() <= area_eps {
                        degenerate = Some(j);
                        break;
                    }
                    if c2 > 0.0 && convex.is_none() {
                        convex = Some(j);
                    }
                    j = next[j];
                }
                if let Some(d) = degenerate {
                    clip!(d, false);
                } else {
                    // …otherwise force-clip a convex corner so the
                    // loop always terminates (only reachable on
                    // non-simple input, where the output is already
                    // documented as unspecified). Every corner reflex
                    // means this is not a CCW ring: bail (`?`).
                    let c = convex?;
                    clip!(c, true);
                }
            }
        }
    }
    // Final triangle.
    let i = (0..n).find(|&k| alive[k])?;
    let c2 = cross_at(i, &prev, &next);
    if c2 > area_eps {
        tris.push([ring[prev[i]].orig, ring[i].orig, ring[next[i]].orig]);
    }
    if tris.is_empty() {
        return None;
    }
    Some(tris)
}

impl Profile2D {
    /// Profile with the given outer boundary and no holes.
    pub fn new(outer: Vec<[f32; 2]>) -> Self {
        Self {
            outer,
            holes: Vec::new(),
        }
    }

    /// Builder: append one hole loop.
    pub fn with_hole(mut self, hole: Vec<[f32; 2]>) -> Self {
        self.holes.push(hole);
        self
    }

    /// Total stored vertex count (`outer` plus every hole), i.e. the
    /// length of the flattened vertex list that
    /// [`Profile2D::triangulate`] indices refer to.
    pub fn vertex_count(&self) -> usize {
        self.outer.len() + self.holes.iter().map(Vec::len).sum::<usize>()
    }

    /// Enclosed area: `|outer loop area| − Σ |hole loop area|`, by
    /// the shoelace formula, in the squared unit of the profile
    /// coordinates. Winding-agnostic (absolute values). For a valid
    /// profile (holes strictly inside the outer boundary, pairwise
    /// disjoint) this equals the area covered by
    /// [`Profile2D::triangulate`]'s triangles.
    pub fn area(&self) -> f64 {
        let loop_area = |pts: &[[f32; 2]]| -> f64 {
            let v = clean_loop(pts, 0);
            if v.len() < 3 {
                return 0.0;
            }
            signed_area2(&v).abs() / 2.0
        };
        let mut a = loop_area(&self.outer);
        for h in &self.holes {
            a -= loop_area(h);
        }
        a
    }

    /// Maximum absolute coordinate — the scale that calibrates the
    /// relative degenerate-area epsilon.
    fn coord_scale(&self) -> f64 {
        let mut s = 0.0f64;
        let mut visit = |pts: &[[f32; 2]]| {
            for p in pts {
                for c in p {
                    let a = f64::from(*c).abs();
                    if a.is_finite() && a > s {
                        s = a;
                    }
                }
            }
        };
        visit(&self.outer);
        for h in &self.holes {
            visit(h);
        }
        s
    }

    /// Triangulate the profile's enclosed area (outer boundary minus
    /// holes) into counter-clockwise triangles.
    ///
    /// Each returned `[u32; 3]` indexes the **flattened original
    /// vertex list** `outer ++ holes[0] ++ holes[1] ++ …` (so entry
    /// `outer.len()` is `holes[0][0]`), making the result directly
    /// usable as an index buffer over the profile's own points — a
    /// cap face, a hole-fill patch over a
    /// [`Primitive::boundary_loops`](crate::Primitive::boundary_loops)
    /// loop, or the input to [`Profile2D::extrude`]. Triangle count
    /// is `n + 2·h − 2` for `n` distinct vertices and `h` holes,
    /// minus any zero-area clips.
    ///
    /// Algorithm: hole loops are spliced into the outer ring through
    /// zero-width bridges (largest-`x` hole first, so later bridges
    /// can land on already-merged hole vertices), then the merged
    /// ring is ear-clipped; see the module docs for the derivation
    /// and termination argument. Input windings are normalised
    /// (outer → CCW, holes → CW); closing/consecutive duplicate
    /// points are ignored.
    ///
    /// Returns `None` when no area exists to triangulate: fewer than
    /// 3 distinct outer vertices, a (near-)zero-area or non-finite
    /// outer loop, a degenerate hole loop (fewer than 3 distinct
    /// vertices or zero area), or a hole that never sees the outer
    /// boundary along its `+x` ray (hole outside the boundary).
    /// Self-intersecting input is not detected; the output for such
    /// a profile is unspecified but the call still terminates.
    ///
    /// Cost: `O(n²)` worst case; `O(n · reflex_count)` in practice
    /// (a convex profile clips in linear time).
    pub fn triangulate(&self) -> Option<Vec<[u32; 3]>> {
        if self.vertex_count() > (u32::MAX / 2) as usize {
            return None;
        }
        let scale = self.coord_scale();
        if scale == 0.0 || !scale.is_finite() {
            return None;
        }
        // Relative epsilon under which a doubled triangle area counts
        // as degenerate (collinear within f32-input precision).
        let area_eps = (scale * 1e-9) * (scale * 1e-9);

        // Outer ring, normalised CCW.
        let mut ring = clean_loop(&self.outer, 0);
        if ring.len() < 3 {
            return None;
        }
        if ring
            .iter()
            .any(|v| !v.p[0].is_finite() || !v.p[1].is_finite())
        {
            return None;
        }
        if signed_area2(&ring).abs() <= area_eps {
            return None;
        }
        if signed_area2(&ring) < 0.0 {
            ring.reverse();
        }

        // Holes, normalised CW, merged largest-x first.
        let mut holes: Vec<Vec<RingVertex>> = Vec::with_capacity(self.holes.len());
        let mut base = self.outer.len() as u32;
        for h in &self.holes {
            let mut hv = clean_loop(h, base);
            base += h.len() as u32;
            if hv.len() < 3 || signed_area2(&hv).abs() <= area_eps {
                return None;
            }
            if hv
                .iter()
                .any(|v| !v.p[0].is_finite() || !v.p[1].is_finite())
            {
                return None;
            }
            if signed_area2(&hv) > 0.0 {
                hv.reverse();
            }
            holes.push(hv);
        }
        holes.sort_by(|a, b| {
            let mx = |h: &[RingVertex]| h.iter().map(|v| v.p[0]).fold(f64::NEG_INFINITY, f64::max);
            mx(b).partial_cmp(&mx(a)).unwrap()
        });
        for h in &holes {
            ring = bridge_hole(&ring, h)?;
        }

        ear_clip(&ring, area_eps)
    }

    /// Sweep the profile along a straight direction into a closed,
    /// watertight indexed triangle [`Primitive`] — the tessellation
    /// of the IFC 4.3 extruded-area-solid representation
    /// (`docs/3d/ifc/ifc43-entity-IfcExtrudedAreaSolid.html`): the
    /// profile lies in the *xy* plane (`z = 0`), the solid is the
    /// translation of that area by `depth · direction / |direction|`,
    /// holes sweep into holes of the solid.
    ///
    /// `direction` need not be unit length and need not be vertical —
    /// any direction whose **z component is non-zero** (i.e. not
    /// perpendicular to the z axis, the IFC validity rule) produces
    /// an oblique prism. `depth` is the translation **length** along
    /// the unit direction, so the prism's perpendicular height is
    /// `depth · |dz| / |direction|` and its volume is
    /// `area() · depth · |dz| / |direction|` — for the common
    /// straight-up `[0, 0, 1]` case simply `area() · depth`.
    ///
    /// # Output
    ///
    /// * `positions` is the flattened profile vertex list at `z = 0`
    ///   (the bottom ring, `vertex_count()` entries in stored order)
    ///   followed by the same list offset by the extrusion vector
    ///   (the top ring), so index `i + vertex_count()` is the swept
    ///   copy of profile vertex `i`.
    /// * The index buffer holds both caps (the
    ///   [`Profile2D::triangulate`] triangles, bottom facing `−z`-ish
    ///   / top facing `+z`-ish) plus two side-wall triangles per
    ///   boundary edge of every loop. All faces wind outward
    ///   (counter-clockwise seen from outside the solid), so
    ///   [`Primitive::signed_volume`] is positive and
    ///   [`Primitive::edge_manifold_report`] reports a closed
    ///   two-manifold
    ///   ([`EdgeManifoldReport::is_closed_manifold`](crate::EdgeManifoldReport::is_closed_manifold))
    ///   — extruding
    ///   downward (`dz < 0`) flips every winding to keep that true.
    ///   Width follows the crate's promotion rule: [`Indices::U16`]
    ///   while the vertex pool is `≤ 65 536` entries, else
    ///   [`Indices::U32`].
    /// * Vertices are shared between caps and walls (watertight
    ///   connectivity, no T-junctions). `normals` / `uvs` are left
    ///   unset — [`Primitive::compute_normals`] reconstructs smooth
    ///   normals, and a renderer wanting hard creases splits corners
    ///   first.
    /// * `material`, `extras`, and every other slot are default.
    ///
    /// Returns `None` when the profile does not triangulate (see
    /// [`Profile2D::triangulate`]), when `depth` is not a strictly
    /// positive finite number, or when `direction` is non-finite,
    /// zero length, or has `dz == 0` after normalisation. Pure
    /// (builds a fresh primitive); cost is the triangulation plus
    /// `O(vertex_count)`.
    pub fn extrude(&self, direction: [f32; 3], depth: f32) -> Option<Primitive> {
        if !depth.is_finite() || depth <= 0.0 {
            return None;
        }
        let d = [
            f64::from(direction[0]),
            f64::from(direction[1]),
            f64::from(direction[2]),
        ];
        if d.iter().any(|c| !c.is_finite()) {
            return None;
        }
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len == 0.0 || !len.is_finite() {
            return None;
        }
        let offset = [
            d[0] / len * f64::from(depth),
            d[1] / len * f64::from(depth),
            d[2] / len * f64::from(depth),
        ];
        if offset[2] == 0.0 {
            return None; // perpendicular to the z axis: no solid
        }

        let cap = self.triangulate()?;
        let v = self.vertex_count() as u32;

        // Bottom ring then top ring.
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(2 * v as usize);
        let mut flat: Vec<[f32; 2]> = Vec::with_capacity(v as usize);
        flat.extend_from_slice(&self.outer);
        for h in &self.holes {
            flat.extend_from_slice(h);
        }
        for p in &flat {
            positions.push([p[0], p[1], 0.0]);
        }
        for p in &flat {
            positions.push([
                (f64::from(p[0]) + offset[0]) as f32,
                (f64::from(p[1]) + offset[1]) as f32,
                offset[2] as f32,
            ]);
        }

        // Caps: bottom reversed (faces -z), top as-is (faces +z).
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(2 * cap.len() + 4 * v as usize);
        for t in &cap {
            tris.push([t[0], t[2], t[1]]);
            tris.push([t[0] + v, t[1] + v, t[2] + v]);
        }

        // Side walls along each loop in normalised orientation
        // (outer CCW, holes CW) so the quads face outward.
        let mut emit_walls = |pts: &[[f32; 2]], base: u32, want_ccw: bool| {
            let mut lp = clean_loop(pts, base);
            if lp.len() < 3 {
                return;
            }
            let ccw = signed_area2(&lp) > 0.0;
            if ccw != want_ccw {
                lp.reverse();
            }
            for k in 0..lp.len() {
                let u = lp[k].orig;
                let w = lp[(k + 1) % lp.len()].orig;
                tris.push([u, w, w + v]);
                tris.push([u, w + v, u + v]);
            }
        };
        emit_walls(&self.outer, 0, true);
        let mut base = self.outer.len() as u32;
        for h in &self.holes {
            emit_walls(h, base, false);
            base += h.len() as u32;
        }

        // Downward extrusion: flip every winding so faces still
        // point outward and the signed volume stays positive.
        if offset[2] < 0.0 {
            for t in &mut tris {
                t.swap(1, 2);
            }
        }

        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = positions;
        let flat_indices: Vec<u32> = tris.iter().flatten().copied().collect();
        prim.indices = Some(if prim.positions.len() <= 65_536 {
            Indices::U16(flat_indices.iter().map(|&i| i as u16).collect())
        } else {
            Indices::U32(flat_indices)
        });
        Some(prim)
    }
}
