//! Mesh, primitive, topology, and index buffer types.
//!
//! A [`Mesh`] is a named bag of [`Primitive`]s — each primitive is a
//! self-contained drawable: one vertex buffer (positions + optional
//! attributes), one optional index buffer, one optional material
//! reference, and one [`Topology`] (how the vertices are stitched).
//! This mirrors glTF 2.0 §3.7.2 mesh.primitive (which itself
//! generalises the OpenGL VAO).
//!
//! Morph targets — typed deltas applied on top of the base vertex
//! buffer to interpolate between named poses — live on
//! [`Primitive::targets`] (per the glTF 2.0 §3.7.2.2 schema), with the
//! per-target blend weights' default values on [`Mesh::weights`].

use std::collections::HashMap;

use crate::scene::{BoundingBox, MaterialId};

/// How the vertex buffer is interpreted as primitives.
///
/// Variants follow OpenGL/glTF naming so format crates can map the
/// wire encoding 1:1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    /// Disjoint triangles — every 3 vertices form one triangle.
    Triangles,
    /// `(0,1,2), (1,2,3), (2,3,4), …` (alternating winding).
    TriangleStrip,
    /// `(0,1,2), (0,2,3), (0,3,4), …` (shared anchor).
    TriangleFan,
    /// Disjoint line segments — every 2 vertices form one segment.
    Lines,
    /// `(0,1), (1,2), (2,3), …`.
    LineStrip,
    /// LineStrip closed back to vertex 0.
    LineLoop,
    /// One point per vertex.
    Points,
}

/// Index buffer payload. `U16` is glTF's default for compactness;
/// formats with > 65 535 vertices per primitive promote to `U32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Indices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl Indices {
    /// Number of indices, regardless of width.
    pub fn len(&self) -> usize {
        match self {
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
        }
    }

    /// `true` if no indices are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One named morph-target delta set applied on top of a [`Primitive`]'s
/// base vertex buffer.
///
/// Per glTF 2.0 §3.7.2.2, a morph target is an ordered map from
/// attribute name (`POSITION`, `NORMAL`, `TANGENT`) to a delta accessor
/// of the same length as the base attribute. The deltas are added to
/// the base values, scaled by the per-target weight (sourced from
/// either [`Mesh::weights`] or, at runtime, the
/// [`crate::AnimationProperty::MorphWeights`] channel).
///
/// We surface those three named slots as typed `Option`s so callers
/// don't have to round-trip through string keys. Other attribute names
/// allowed by future glTF extensions (e.g. `COLOR_0`) still travel via
/// [`Primitive::extras`].
///
/// All present buffers must have the same length as the corresponding
/// base attribute on the parent [`Primitive`]. Absent slots
/// (`None`/`tangent: None`) leave that attribute untouched at runtime
/// for this target.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MorphTarget {
    /// Per-vertex `POSITION` delta (added to the base `positions`).
    pub position: Option<Vec<[f32; 3]>>,
    /// Per-vertex `NORMAL` delta (added to the base `normals`).
    pub normal: Option<Vec<[f32; 3]>>,
    /// Per-vertex `TANGENT` delta (added to the base `tangents` xyz;
    /// the handedness `w` is *not* morphed per spec §3.7.2.2).
    pub tangent: Option<Vec<[f32; 3]>>,
}

impl MorphTarget {
    /// Empty target — no deltas in any slot. Useful as a starting
    /// builder before the format-crate decoder fills the slots that
    /// the wire actually carried.
    pub fn new() -> Self {
        Self::default()
    }
}

/// One drawable submesh.
///
/// `positions` is mandatory; every other attribute is optional and,
/// when present, must have the same `len()` as `positions`. UV and
/// vertex-colour buffers are vectors-of-vectors so multi-channel
/// content (lightmaps, second UV set) is representable without
/// flattening into the spec's TEXCOORD_0/_1 strings.
///
/// **`#[non_exhaustive]` (round 7):** new attribute fields land in
/// minor releases without breaking downstream callers. Construct via
/// [`Primitive::new`] + per-field assignment; struct-update syntax
/// (`Primitive { positions, ..Primitive::new(Topology::Triangles) }`)
/// works inside this crate but not from external crates — that's the
/// whole point of the attribute. Outside this crate, always go
/// through the constructor.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Primitive {
    pub topology: Topology,
    pub positions: Vec<[f32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    /// xyz + handedness in `w` (`±1.0`) per glTF.
    pub tangents: Option<Vec<[f32; 4]>>,
    /// `uvs[N]` is the Nth UV set. Empty outer vec means no UVs.
    pub uvs: Vec<Vec<[f32; 2]>>,
    /// `colors[N]` is the Nth vertex-colour set. Empty outer vec
    /// means no per-vertex colour.
    pub colors: Vec<Vec<[f32; 4]>>,
    /// 4 joint indices per vertex when skinning is active.
    pub joints: Option<Vec<[u16; 4]>>,
    /// 4 joint weights per vertex; should sum to 1.0 within tolerance.
    pub weights: Option<Vec<[f32; 4]>>,
    pub indices: Option<Indices>,
    pub material: Option<MaterialId>,
    /// Morph-target delta sets per glTF 2.0 §3.7.2.2. Empty vec means
    /// no morph targets on this primitive. Each entry is one named
    /// pose (e.g. "smile", "blink") whose blend weight is sourced
    /// from [`Mesh::weights`] (default) or an animation channel
    /// (runtime). The number of targets across every primitive in the
    /// parent [`Mesh`] should match — the spec mandates that the
    /// `i`th target on each primitive shares one weight slot.
    pub targets: Vec<MorphTarget>,
    pub extras: HashMap<String, serde_json::Value>,
}

impl Primitive {
    /// Empty primitive — no positions, no attributes, `Triangles`
    /// topology by default.
    pub fn new(topology: Topology) -> Self {
        Self {
            topology,
            positions: Vec::new(),
            normals: None,
            tangents: None,
            uvs: Vec::new(),
            colors: Vec::new(),
            joints: None,
            weights: None,
            indices: None,
            material: None,
            targets: Vec::new(),
            extras: HashMap::new(),
        }
    }

    /// Number of triangles produced by tessellating this primitive.
    ///
    /// The count uses the index buffer length when present, or
    /// `positions.len()` otherwise. Non-triangle topologies return 0.
    pub fn triangle_count(&self) -> usize {
        let n = self
            .indices
            .as_ref()
            .map(|i| i.len())
            .unwrap_or(self.positions.len());
        match self.topology {
            Topology::Triangles => n / 3,
            Topology::TriangleStrip | Topology::TriangleFan => n.saturating_sub(2),
            _ => 0,
        }
    }

    /// Axis-aligned bounding box over [`Primitive::positions`] in the
    /// primitive's local space (no transforms applied).
    ///
    /// Returns `None` for an empty primitive. Vertices not referenced
    /// by `indices` are still included — this is the "data extent",
    /// not the "drawn extent". For an index-aware extent, the caller
    /// can iterate `indices` themselves and feed positions through
    /// [`BoundingBox::from_points`].
    ///
    /// NaN coordinates are skipped (not propagated to the output).
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        BoundingBox::from_points(self.positions.iter().copied())
    }

    /// De-strip this primitive's topology into a flat triangle list of
    /// **vertex indices**, each triple being one triangle wound
    /// counter-clockwise (front-facing) in the same orientation the
    /// source topology specifies.
    ///
    /// This is the standard OpenGL/glTF strip→list expansion (see the
    /// [`Topology`] variant docs, which mirror the OpenGL primitive
    /// assembly rules):
    ///
    /// * [`Topology::Triangles`] — already a list; index triples are
    ///   returned verbatim (the trailing 0–2 leftover indices that
    ///   don't complete a triangle are dropped).
    /// * [`Topology::TriangleStrip`] — `v[0],v[1],v[2]` then
    ///   `v[1],v[2],v[3]`, … with **alternating winding**: every
    ///   odd-numbered triangle swaps its last two vertices so the
    ///   visible winding stays consistent (OpenGL §10.1 triangle-strip
    ///   rule; glTF inherits it).
    /// * [`Topology::TriangleFan`] — `v[0],v[1],v[2]` then
    ///   `v[0],v[2],v[3]`, … sharing the anchor `v[0]`; winding is
    ///   uniform (no alternation).
    /// * Non-triangle topologies ([`Topology::Lines`],
    ///   [`Topology::Points`], …) yield an empty list.
    ///
    /// The values returned are **vertex indices** into the attribute
    /// buffers (`positions`, `normals`, …): if an index buffer is
    /// present its entries are dereferenced (so the result indexes the
    /// vertex pool, not the index buffer); if absent, the implicit
    /// sequence `0,1,2,…` over `positions.len()` is used. Indices are
    /// widened to `u32` so a `U16` source and a `U32` source produce the
    /// same type.
    ///
    /// The output count equals [`Primitive::triangle_count`] for
    /// triangle topologies. Cost is `O(triangle_count)`; one
    /// `Vec<[u32; 3]>` is allocated.
    pub fn triangle_indices(&self) -> Vec<[u32; 3]> {
        // The logical vertex-index sequence: either the index buffer
        // widened to u32, or the implicit 0..positions.len() range.
        let seq: Vec<u32> = match &self.indices {
            Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
            Some(Indices::U32(v)) => v.clone(),
            None => (0..self.positions.len() as u32).collect(),
        };
        let n = seq.len();
        match self.topology {
            Topology::Triangles => {
                let tris = n / 3;
                let mut out = Vec::with_capacity(tris);
                for t in 0..tris {
                    out.push([seq[3 * t], seq[3 * t + 1], seq[3 * t + 2]]);
                }
                out
            }
            Topology::TriangleStrip => {
                if n < 3 {
                    return Vec::new();
                }
                let mut out = Vec::with_capacity(n - 2);
                for i in 0..(n - 2) {
                    // Even-indexed triangle keeps (i, i+1, i+2);
                    // odd-indexed swaps the last two to keep winding
                    // consistent (OpenGL triangle-strip rule).
                    if i % 2 == 0 {
                        out.push([seq[i], seq[i + 1], seq[i + 2]]);
                    } else {
                        out.push([seq[i], seq[i + 2], seq[i + 1]]);
                    }
                }
                out
            }
            Topology::TriangleFan => {
                if n < 3 {
                    return Vec::new();
                }
                let anchor = seq[0];
                let mut out = Vec::with_capacity(n - 2);
                for i in 1..(n - 1) {
                    out.push([anchor, seq[i], seq[i + 1]]);
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// De-strip this primitive into an equivalent
    /// [`Topology::Triangles`] primitive with a freshly built `U32`
    /// index buffer.
    ///
    /// The vertex attribute buffers (`positions`, `normals`,
    /// `tangents`, `uvs`, `colors`, `joints`, `weights`) and the
    /// `material` are carried over verbatim — only the connectivity is
    /// rewritten. The new index buffer is the flattening produced by
    /// [`Primitive::triangle_indices`] (so the alternating
    /// triangle-strip winding rule is honoured).
    ///
    /// `targets` (morph deltas) are carried over too: they are
    /// per-vertex-parallel to the attribute buffers, which are
    /// unchanged, so they stay valid. `extras` is cloned through.
    ///
    /// For a primitive that is already [`Topology::Triangles`] this is a
    /// normalising round-trip: the output is `Triangles` with an
    /// explicit index buffer even if the input was non-indexed. For a
    /// non-triangle topology (lines/points) the result is an empty-index
    /// `Triangles` primitive (the attribute buffers are still carried,
    /// but nothing is drawn) — callers that care about line/point
    /// topology should branch on [`Primitive::topology`] before calling.
    pub fn to_triangle_list(&self) -> Primitive {
        let tris = self.triangle_indices();
        let mut flat: Vec<u32> = Vec::with_capacity(tris.len() * 3);
        for t in &tris {
            flat.extend_from_slice(t);
        }
        let mut out = self.clone();
        out.topology = Topology::Triangles;
        out.indices = Some(Indices::U32(flat));
        out
    }

    /// Recompute smooth, area-weighted per-vertex normals from this
    /// primitive's triangle connectivity and return them as one
    /// `[f32; 3]` per vertex (length `positions.len()`).
    ///
    /// This is the standard smooth-shading normal-estimation scheme.
    /// For each triangle `(a, b, c)` the un-normalised face normal is
    /// the edge cross product
    ///
    /// ```text
    /// N_face = (P[b] - P[a]) × (P[c] - P[a])
    /// ```
    ///
    /// whose direction is the geometric normal and whose **magnitude
    /// equals twice the triangle's area** (`|u × v| = |u||v|sinθ`).
    /// Accumulating these un-normalised vectors into each of the
    /// triangle's three vertices, then normalising the per-vertex sum,
    /// therefore yields the **area-weighted** average of the incident
    /// face normals — larger faces pull the shared vertex normal more
    /// strongly, which is the textbook recomputation (the area weighting
    /// falls out of the cross-product magnitude; see the smooth-shading
    /// normal averaging of Gouraud, "Continuous Shading of Curved
    /// Surfaces", IEEE TC 1971, and the area-weighted face-normal
    /// accumulation in Foley, van Dam et al., *Computer Graphics:
    /// Principles and Practice*).
    ///
    /// Winding convention: vertices are taken counter-clockwise =
    /// front-facing (the crate's right-handed, glTF-aligned convention),
    /// so `N_face` points out of the front face. The connectivity is the
    /// de-stripped triangle list from [`Primitive::triangle_indices`], so
    /// `Triangles` / `TriangleStrip` (alternating winding honoured) /
    /// `TriangleFan` all feed in correctly; non-triangle topologies
    /// (lines/points) contribute no faces and every output normal stays
    /// at the `[0, 0, 1]` fallback.
    ///
    /// Contract:
    ///
    /// * **Output length is always `positions.len()`.** Vertices not
    ///   referenced by any triangle (or by an out-of-range index, which
    ///   is skipped) receive the fallback normal `[0, 0, 1]` rather than
    ///   a zero vector, so the result is always renderable.
    /// * **Degenerate faces contribute nothing.** A triangle whose edge
    ///   cross product is the zero vector (collinear or coincident
    ///   vertices) adds zero — it neither helps nor corrupts the
    ///   accumulation. A vertex touched only by degenerate faces falls
    ///   back to `[0, 0, 1]`.
    /// * **NaN-safe.** A face producing a non-finite normal is skipped;
    ///   a vertex whose accumulated sum is non-finite or zero-length
    ///   falls back to `[0, 0, 1]`.
    /// * **Does not mutate `self`.** Assign the result to
    ///   [`Primitive::normals`] (matching `positions` length) if you want
    ///   to store it. This is the recompute step a format decoder runs
    ///   when the wire stream omits normals (STL face normals aside, OBJ
    ///   without `vn`, glTF without `NORMAL`).
    ///
    /// Cost is `O(triangle_count + V)`; allocates one `Vec<[f32; 3]>`.
    pub fn compute_normals(&self) -> Vec<[f32; 3]> {
        const FALLBACK: [f32; 3] = [0.0, 0.0, 1.0];
        let n = self.positions.len();
        let mut acc = vec![[0.0f32; 3]; n];
        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            // Defensive: an index buffer can dereference out of range
            // for a malformed primitive — skip such a face rather than
            // panic. validate() catches it ahead of time.
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = self.positions[ia];
            let pb = self.positions[ib];
            let pc = self.positions[ic];
            let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            // Cross product u × v: magnitude is twice the triangle area,
            // so summing it area-weights the contribution automatically.
            let fn_ = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            if !fn_[0].is_finite() || !fn_[1].is_finite() || !fn_[2].is_finite() {
                continue;
            }
            for &i in &[ia, ib, ic] {
                acc[i][0] += fn_[0];
                acc[i][1] += fn_[1];
                acc[i][2] += fn_[2];
            }
        }
        for a in acc.iter_mut() {
            let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            if len.is_finite() && len > 0.0 {
                a[0] /= len;
                a[1] /= len;
                a[2] /= len;
            } else {
                *a = FALLBACK;
            }
        }
        acc
    }

    /// Evaluate the per-vertex morph-blend formula from glTF 2.0
    /// §3.7.2.2 against this primitive's [`Primitive::targets`] using
    /// the supplied per-target `weights`, and return the blended
    /// attribute buffers.
    ///
    /// Per spec §3.7.2.2:
    ///
    /// ```text
    /// morphed[k] = base[k]
    ///            + weights[0] * targets[0].ATTR[k]
    ///            + weights[1] * targets[1].ATTR[k]
    ///            + ...
    /// ```
    ///
    /// The contract:
    ///
    /// * **Per-attribute opt-in.** A target whose slot is `None`
    ///   contributes nothing for that attribute (spec line 3589:
    ///   *"Attributes present in the base mesh primitive but not
    ///   included in a given morph target MUST retain their original
    ///   values for the morph target."*). The output attribute is
    ///   `Some(_)` iff the base attribute was `Some(_)`/non-empty.
    ///   `POSITION` is always present in the output (it's required on
    ///   every primitive); the other two are mirrors of the base
    ///   presence.
    /// * **Tangent handedness preserved.** §3.7.2.2 (line 3616):
    ///   morph TANGENT deltas are VEC3 — the base TANGENT's `w`
    ///   handedness is **not** morphed and is copied through verbatim.
    /// * **Weight count is `weights.len()`.** Any target index `i`
    ///   beyond `weights.len()` is skipped (`weight = 0` per spec
    ///   line 3697: missing weights default to zero). Any `weights[i]`
    ///   for `i >= self.targets.len()` is also ignored (no target to
    ///   apply it to). Empty `weights` returns the base attributes
    ///   unmodified.
    /// * **Buffer-length mismatch is a soft error.** A target slot
    ///   whose length disagrees with the base attribute is skipped
    ///   for that vertex range (we still apply the prefix where lengths
    ///   line up). Callers should run [`crate::Scene3D::validate`]
    ///   first to catch this — the runtime path stays panic-free.
    ///
    /// Cost is `O(V * (1 + T))` where `V = positions.len()` and
    /// `T = min(weights.len(), targets.len())`. Allocates one
    /// `Vec<[f32; 3]>` (positions) plus one per present output
    /// attribute.
    pub fn apply_morph_weights(&self, weights: &[f32]) -> MorphedAttributes {
        let n = self.positions.len();
        let mut positions = self.positions.clone();
        let mut normals = self.normals.clone();
        let mut tangents = self.tangents.clone();

        let t_max = self.targets.len().min(weights.len());
        for (target, &w) in self.targets.iter().zip(weights.iter()).take(t_max) {
            if w == 0.0 {
                continue; // Skip no-op contributions; same observable result.
            }
            if let Some(d) = &target.position {
                let lim = n.min(d.len());
                for k in 0..lim {
                    positions[k][0] += w * d[k][0];
                    positions[k][1] += w * d[k][1];
                    positions[k][2] += w * d[k][2];
                }
            }
            if let (Some(base), Some(d)) = (normals.as_mut(), target.normal.as_ref()) {
                let lim = base.len().min(d.len());
                for k in 0..lim {
                    base[k][0] += w * d[k][0];
                    base[k][1] += w * d[k][1];
                    base[k][2] += w * d[k][2];
                }
            }
            if let (Some(base), Some(d)) = (tangents.as_mut(), target.tangent.as_ref()) {
                // TANGENT is [f32; 4] (xyz + handedness w). Morph
                // delta is [f32; 3] — handedness is NOT morphed
                // (spec §3.7.2.2 line 3616). Add xyz only; leave w
                // untouched.
                let lim = base.len().min(d.len());
                for k in 0..lim {
                    base[k][0] += w * d[k][0];
                    base[k][1] += w * d[k][1];
                    base[k][2] += w * d[k][2];
                }
            }
        }

        MorphedAttributes {
            positions,
            normals,
            tangents,
        }
    }
}

/// Evaluated output of [`Primitive::apply_morph_weights`].
///
/// One blended copy of each base attribute on a [`Primitive`]. The
/// `Option` shape mirrors the input primitive's attribute presence:
/// `normals` / `tangents` are `Some` iff the corresponding base
/// attribute was `Some`. `positions` is always present (every
/// primitive carries it).
///
/// The buffers live in mesh-local space — skin pose, parent
/// transforms, and the renderer's projection are not applied. Per
/// glTF 2.0 §3.7.2.2 line 3697, callers feeding into a draw call
/// should consume these as the input to skinning/projection rather
/// than re-blending each frame.
#[derive(Clone, Debug, PartialEq)]
pub struct MorphedAttributes {
    /// Blended `POSITION` buffer, length equal to the source
    /// primitive's `positions.len()`.
    pub positions: Vec<[f32; 3]>,
    /// Blended `NORMAL` buffer (only when the source primitive carried
    /// normals). Length matches `positions`.
    pub normals: Option<Vec<[f32; 3]>>,
    /// Blended `TANGENT` buffer, xyz blended and `w` handedness
    /// preserved verbatim (spec §3.7.2.2 forbids morphing handedness).
    /// Length matches `positions`.
    pub tangents: Option<Vec<[f32; 4]>>,
}

/// A named bag of [`Primitive`]s sharing nothing but a name.
///
/// Most authoring tools split a logical "object" into one primitive
/// per material so the renderer can issue one draw call per
/// primitive without rebinding state.
///
/// **`#[non_exhaustive]` (round 7):** new fields can be added in
/// minor releases without breaking downstream callers. Construct via
/// [`Mesh::new`] + the [`Mesh::with_primitive`] / [`Mesh::with_weights`]
/// builders. From outside this crate, struct literal syntax is
/// rejected by the compiler — go through the constructor.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Mesh {
    pub name: Option<String>,
    pub primitives: Vec<Primitive>,
    /// Default morph-target blend weights (glTF 2.0 §3.7.2.2
    /// `mesh.weights`). When non-empty, `weights[i]` is the static
    /// blend factor for the `i`th [`MorphTarget`] on every primitive
    /// in [`Mesh::primitives`]. An animation channel of property
    /// [`crate::AnimationProperty::MorphWeights`] overrides this
    /// vector at runtime. Empty vec means no static weights — the
    /// runtime falls back to zero (i.e. base mesh).
    pub weights: Vec<f32>,
}

impl Mesh {
    /// Empty mesh with the given name.
    pub fn new(name: impl Into<Option<String>>) -> Self {
        Self {
            name: name.into(),
            primitives: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Push a primitive and return `&mut self` for chaining.
    pub fn with_primitive(mut self, primitive: Primitive) -> Self {
        self.primitives.push(primitive);
        self
    }

    /// Set the static morph-blend `weights` and return `&mut self`
    /// for chaining. The vector length should match the number of
    /// [`MorphTarget`]s on each [`Primitive`] in this mesh.
    pub fn with_weights(mut self, weights: impl Into<Vec<f32>>) -> Self {
        self.weights = weights.into();
        self
    }

    /// Axis-aligned bounding box over every contained primitive in
    /// mesh-local space (no transforms applied; morph deltas + skin
    /// pose ignored).
    ///
    /// Returns `None` if every primitive is empty. Morph targets are
    /// not folded in — for a worst-case bound the caller would have
    /// to walk each [`MorphTarget`] and union the deltas; the typed
    /// model deliberately stays runtime-agnostic.
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        self.primitives
            .iter()
            .filter_map(|p| p.bounding_box())
            .reduce(BoundingBox::union)
    }
}
