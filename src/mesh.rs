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

    /// Merge bit-identical vertices into a shared pool and return an
    /// equivalent **indexed** primitive whose attribute buffers contain
    /// only the distinct vertices, with the index buffer rewritten to
    /// reference the deduplicated pool.
    ///
    /// This is the inverse of attribute "explosion": a decoder for a
    /// non-shared format (binary STL stores three fresh vertices per
    /// facet with no sharing; an OBJ `f` line that repeats a `v/vt/vn`
    /// triple still produces a distinct rendering vertex per face corner)
    /// produces a vertex *soup* where coincident corners are duplicated.
    /// Welding collapses those duplicates so a vertex shared by `k`
    /// faces is stored once and referenced `k` times, shrinking the
    /// vertex buffer and letting the GPU's post-transform vertex cache do
    /// its job. The reverse trip — `weld_vertices` then
    /// [`Primitive::to_triangle_list`] applied to an already-indexed
    /// primitive — is the explode step.
    ///
    /// # What counts as "the same vertex"
    ///
    /// Two source vertices merge **iff every attribute slot present on
    /// the primitive is bit-identical** between them: `positions`, each
    /// `NORMAL` / `TANGENT`, every UV set in `uvs`, every colour set in
    /// `colors`, the `joints` quad, the `weights` quad, **and** the
    /// per-vertex deltas of every [`MorphTarget`] in `targets`. A vertex
    /// that agrees in position but differs in (say) UV or a morph delta
    /// is a *distinct* rendering vertex and is kept separate — this is
    /// the only correct rule for an indexed draw call, where one index
    /// selects one tuple across *all* attribute streams simultaneously.
    /// Callers that want to merge by position alone (e.g. to fix a
    /// cracked surface before recomputing smooth normals) should strip
    /// the other attributes first.
    ///
    /// Float comparison is **exact** (bit pattern), which is the right
    /// choice for de-duplicating a decoder's vertex soup: identical
    /// source numbers decode to identical bits, so genuine duplicates
    /// collapse while authored-distinct values stay split. Two
    /// normalisations make the bit key well-behaved: `-0.0` is folded to
    /// `+0.0` (they are numerically equal and should merge), and every
    /// `NaN` is folded to one canonical bit pattern (so two `NaN`
    /// coordinates merge rather than the IEEE rule that `NaN != NaN`
    /// silently preventing dedup; geometry should not carry `NaN`, but
    /// the welder stays deterministic if it does). No epsilon tolerance
    /// is applied — proximity-based welding is a separate, lossy
    /// operation and is intentionally out of scope.
    ///
    /// # Output
    ///
    /// * `topology` is preserved verbatim — welding rewrites *which*
    ///   pool entry each draw step references, never the stitching rule,
    ///   so it is valid for every [`Topology`] (triangles, strips, fans,
    ///   lines, points), not just triangle lists.
    /// * The new index buffer walks the source's draw order: for an
    ///   already-indexed input the existing index sequence is remapped
    ///   through the dedup table; for a non-indexed input the implicit
    ///   `0,1,2,…` order is materialised into an explicit buffer. Index
    ///   width is [`Indices::U16`] when the deduplicated vertex count is
    ///   `≤ 65 536`, else [`Indices::U32`] (matching glTF's default
    ///   width-promotion).
    /// * Attribute buffers (`positions`, `normals`, `tangents`, every
    ///   `uvs` / `colors` set, `joints`, `weights`) and every
    ///   [`MorphTarget`] slot are gathered down to the distinct vertices
    ///   in first-seen order, so the pool is deterministic across runs.
    ///   `material`, `targets` roster shape, and `extras` are carried
    ///   over; only connectivity + the per-vertex buffers change.
    /// * An out-of-range entry in an existing index buffer (malformed
    ///   primitive) is dropped from the output index stream rather than
    ///   panicking — [`Scene3D::validate`](crate::Scene3D::validate)
    ///   catches such inputs ahead of time.
    /// * **Does not mutate `self`.** An empty primitive (no positions)
    ///   round-trips to an empty indexed primitive.
    ///
    /// Cost is `O(N · A)` where `N` is the source vertex count and `A`
    /// the per-vertex attribute byte width (the hash key length); one
    /// `HashMap` plus the gathered output buffers are allocated.
    pub fn weld_vertices(&self) -> Primitive {
        // Canonicalise an f32 to a stable hashable bit pattern: fold
        // -0.0 → +0.0 (numerically equal, must merge) and every NaN to
        // one pattern (so NaN coords merge instead of never matching).
        fn key(x: f32) -> u32 {
            if x == 0.0 {
                0 // covers both +0.0 and -0.0
            } else if x.is_nan() {
                0x7fc0_0000 // one canonical quiet-NaN pattern
            } else {
                x.to_bits()
            }
        }

        let n = self.positions.len();

        // Build a per-vertex bit key over every present attribute slot.
        // Order is fixed so the key is reproducible.
        let build_key = |i: usize| -> Vec<u32> {
            let mut k = Vec::new();
            let p = self.positions[i];
            k.extend([key(p[0]), key(p[1]), key(p[2])]);
            if let Some(ns) = &self.normals {
                if let Some(v) = ns.get(i) {
                    k.extend([key(v[0]), key(v[1]), key(v[2])]);
                }
            }
            if let Some(ts) = &self.tangents {
                if let Some(v) = ts.get(i) {
                    k.extend([key(v[0]), key(v[1]), key(v[2]), key(v[3])]);
                }
            }
            for set in &self.uvs {
                if let Some(v) = set.get(i) {
                    k.extend([key(v[0]), key(v[1])]);
                }
            }
            for set in &self.colors {
                if let Some(v) = set.get(i) {
                    k.extend([key(v[0]), key(v[1]), key(v[2]), key(v[3])]);
                }
            }
            if let Some(js) = &self.joints {
                if let Some(v) = js.get(i) {
                    k.extend([v[0] as u32, v[1] as u32, v[2] as u32, v[3] as u32]);
                }
            }
            if let Some(ws) = &self.weights {
                if let Some(v) = ws.get(i) {
                    k.extend([key(v[0]), key(v[1]), key(v[2]), key(v[3])]);
                }
            }
            // Morph deltas are per-vertex parallel — a corner that
            // differs only in a morph delta is a distinct vertex.
            for t in &self.targets {
                if let Some(d) = &t.position {
                    if let Some(v) = d.get(i) {
                        k.extend([key(v[0]), key(v[1]), key(v[2])]);
                    }
                }
                if let Some(d) = &t.normal {
                    if let Some(v) = d.get(i) {
                        k.extend([key(v[0]), key(v[1]), key(v[2])]);
                    }
                }
                if let Some(d) = &t.tangent {
                    if let Some(v) = d.get(i) {
                        k.extend([key(v[0]), key(v[1]), key(v[2])]);
                    }
                }
            }
            k
        };

        // Map each source vertex index → pool index; remember the
        // first source index that produced each pool slot.
        let mut dedup: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut remap: Vec<u32> = Vec::with_capacity(n);
        let mut sources: Vec<usize> = Vec::new();
        for i in 0..n {
            let k = build_key(i);
            let slot = *dedup.entry(k).or_insert_with(|| {
                let id = sources.len() as u32;
                sources.push(i);
                id
            });
            remap.push(slot);
        }

        // Gather the deduplicated attribute buffers in first-seen order.
        let gather3 =
            |src: &Vec<[f32; 3]>| -> Vec<[f32; 3]> { sources.iter().map(|&i| src[i]).collect() };
        let positions = gather3(&self.positions);
        let normals = self.normals.as_ref().map(|s| {
            sources
                .iter()
                .map(|&i| s.get(i).copied().unwrap_or([0.0; 3]))
                .collect()
        });
        let tangents = self.tangents.as_ref().map(|s| {
            sources
                .iter()
                .map(|&i| s.get(i).copied().unwrap_or([0.0; 4]))
                .collect()
        });
        let uvs = self
            .uvs
            .iter()
            .map(|set| {
                sources
                    .iter()
                    .map(|&i| set.get(i).copied().unwrap_or([0.0; 2]))
                    .collect()
            })
            .collect();
        let colors = self
            .colors
            .iter()
            .map(|set| {
                sources
                    .iter()
                    .map(|&i| set.get(i).copied().unwrap_or([0.0; 4]))
                    .collect()
            })
            .collect();
        let joints = self.joints.as_ref().map(|s| {
            sources
                .iter()
                .map(|&i| s.get(i).copied().unwrap_or([0; 4]))
                .collect()
        });
        let weights = self.weights.as_ref().map(|s| {
            sources
                .iter()
                .map(|&i| s.get(i).copied().unwrap_or([0.0; 4]))
                .collect()
        });
        let targets = self
            .targets
            .iter()
            .map(|t| MorphTarget {
                position: t.position.as_ref().map(|d| {
                    sources
                        .iter()
                        .map(|&i| d.get(i).copied().unwrap_or([0.0; 3]))
                        .collect()
                }),
                normal: t.normal.as_ref().map(|d| {
                    sources
                        .iter()
                        .map(|&i| d.get(i).copied().unwrap_or([0.0; 3]))
                        .collect()
                }),
                tangent: t.tangent.as_ref().map(|d| {
                    sources
                        .iter()
                        .map(|&i| d.get(i).copied().unwrap_or([0.0; 3]))
                        .collect()
                }),
            })
            .collect();

        // Rewrite the draw order: remap an existing index buffer through
        // the dedup table (dropping out-of-range entries), or
        // materialise the implicit 0..n order.
        let new_indices: Vec<u32> = match &self.indices {
            Some(Indices::U16(v)) => v
                .iter()
                .filter_map(|&i| remap.get(i as usize).copied())
                .collect(),
            Some(Indices::U32(v)) => v
                .iter()
                .filter_map(|&i| remap.get(i as usize).copied())
                .collect(),
            None => remap.clone(),
        };

        // glTF width-promotion: U16 while the pool fits, else U32.
        let indices = if sources.len() <= u16::MAX as usize + 1 {
            Indices::U16(new_indices.iter().map(|&i| i as u16).collect())
        } else {
            Indices::U32(new_indices)
        };

        Primitive {
            topology: self.topology,
            positions,
            normals,
            tangents,
            uvs,
            colors,
            joints,
            weights,
            indices: Some(indices),
            material: self.material,
            targets,
            extras: self.extras.clone(),
        }
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

    /// Total surface area of this primitive's triangle tessellation, in
    /// the unit-squared of [`Primitive::positions`] (matching the
    /// parent [`crate::Scene3D::unit`] — metres² by default).
    ///
    /// Topology handling matches [`Primitive::triangle_indices`]: the
    /// de-stripped triangle list is summed, so `Triangles` /
    /// `TriangleStrip` (alternating winding honoured) / `TriangleFan`
    /// all feed in correctly. Non-triangle topologies (lines/points)
    /// contribute 0.0 — they have no surface.
    ///
    /// # Derivation (clean-room, first-principles)
    ///
    /// For a triangle with corners `(P_a, P_b, P_c)` and edge vectors
    /// `E1 = P_b - P_a`, `E2 = P_c - P_a`, the parallelogram spanned
    /// by `E1` and `E2` has area `|E1 × E2|` (the cross-product
    /// magnitude is the definition of the parallelogram's signed area
    /// magnitude — any introductory vector calculus reference, e.g.
    /// Marsden & Tromba, *Vector Calculus*). A triangle occupies
    /// exactly half of that parallelogram, so
    ///
    /// ```text
    /// area = |E1 × E2| / 2
    /// ```
    ///
    /// Note the same `E1 × E2` cross product already drives
    /// [`Primitive::compute_normals`] (its magnitude is twice the
    /// triangle area, which is why summing the un-normalised face
    /// normal into each vertex automatically area-weights smooth
    /// shading). `surface_area` reuses the identical edge-cross
    /// machinery and divides by two; the two methods are sibling
    /// reductions of the same triangle walk.
    ///
    /// # Contract
    ///
    /// * Always returns a finite, non-negative `f64` for a primitive
    ///   whose positions are all finite. The accumulator is `f64` so
    ///   a million-triangle mesh doesn't drift under `f32` summation;
    ///   the per-triangle cross-product math is also done in `f64`.
    /// * **Degenerate triangles contribute zero.** A triangle whose
    ///   edge cross product is the zero vector (collinear/coincident
    ///   corners — the same set [`Primitive::degenerate_triangles`]
    ///   reports) adds 0.0 to the sum. They neither help nor corrupt
    ///   the total.
    /// * **NaN-safe.** A face whose edge differences or cross product
    ///   produces a non-finite component contributes 0.0 instead of
    ///   poisoning the sum with NaN/Inf. The whole result therefore
    ///   stays finite even on a partly-corrupt vertex buffer.
    /// * **Out-of-range index** entries (a malformed primitive whose
    ///   index buffer dereferences past `positions.len()`) are
    ///   skipped, not panicked.
    /// * Non-triangle topologies return 0.0.
    /// * **Does not mutate `self`.** Pure; cost is `O(triangle_count)`.
    ///
    /// # Use
    ///
    /// * STL validators: the Fabbers/Stratasys conformance recipe
    ///   asks for a total enclosed-volume check, of which surface
    ///   area is the cheap precursor.
    /// * Importers comparing two formats' tessellation densities for
    ///   LOD/decimation heuristics.
    /// * Texel-density readouts (texture pixels per square metre)
    ///   when combined with the UV-chart area returned by a future
    ///   `uv_area` helper.
    pub fn surface_area(&self) -> f64 {
        let n = self.positions.len();
        let mut total = 0.0_f64;
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
            // f64 from the edge differences onward so accumulation
            // stays stable across large meshes.
            let ux = pb[0] as f64 - pa[0] as f64;
            let uy = pb[1] as f64 - pa[1] as f64;
            let uz = pb[2] as f64 - pa[2] as f64;
            let vx = pc[0] as f64 - pa[0] as f64;
            let vy = pc[1] as f64 - pa[1] as f64;
            let vz = pc[2] as f64 - pa[2] as f64;
            // Cross product u × v; its magnitude is twice the
            // triangle area.
            let cx = uy * vz - uz * vy;
            let cy = uz * vx - ux * vz;
            let cz = ux * vy - uy * vx;
            if !cx.is_finite() || !cy.is_finite() || !cz.is_finite() {
                continue;
            }
            let m2 = cx * cx + cy * cy + cz * cz;
            if !m2.is_finite() {
                continue;
            }
            total += m2.sqrt() * 0.5;
        }
        total
    }

    /// Recompute per-vertex MikkTSpace-style tangent-space basis
    /// vectors from this primitive's positions, UVs (UV set `uv_set`),
    /// and per-vertex normals, returning one `[f32; 4]` per vertex
    /// (length `positions.len()`) — the xyz is the unit tangent T and
    /// the w is the handedness sign (`+1.0` or `-1.0`) such that the
    /// bitangent reconstructs as `B = w * (N × T)`. This is exactly the
    /// shape the existing [`Primitive::tangents`] field stores and
    /// what glTF 2.0 §3.7.2.1 specifies for the `TANGENT` accessor.
    ///
    /// # Derivation (clean-room, first-principles)
    ///
    /// Texture coordinates parameterise the mesh surface as
    /// `P(u, v)`. The tangent T and bitangent B are the partial
    /// derivatives `∂P/∂u` and `∂P/∂v` respectively. Over a single
    /// triangle the surface is linear, so for vertices `(P0, P1, P2)`
    /// with UVs `(Q0, Q1, Q2)` and edge vectors `E1 = P1 - P0`,
    /// `E2 = P2 - P0`, UV deltas
    /// `(Δu1, Δv1) = Q1 - Q0`, `(Δu2, Δv2) = Q2 - Q0`, the chain rule
    /// gives:
    ///
    /// ```text
    /// [E1]   [Δu1  Δv1] [T]
    /// [E2] = [Δu2  Δv2] [B]
    /// ```
    ///
    /// Inverting the 2×2 UV-delta matrix yields the closed-form
    /// per-triangle tangent and bitangent:
    ///
    /// ```text
    /// det = Δu1·Δv2 - Δu2·Δv1
    /// T   = ( Δv2·E1 - Δv1·E2) / det
    /// B   = (-Δu2·E1 + Δu1·E2) / det
    /// ```
    ///
    /// (This derivation appears in any partial-derivative treatment of
    /// surface parameterisation — see Lengyel, "Computing Tangent
    /// Space Basis Vectors for an Arbitrary Mesh" (2001), and the
    /// "Normal Mapping" chapter of Akenine-Möller, Haines & Hoffman,
    /// *Real-Time Rendering*. The math is just the inverse of a 2×2
    /// linear system.)
    ///
    /// We accumulate the un-normalised per-triangle `T` (divided by
    /// `det` only — so the sum is area-weighted, like
    /// [`Primitive::compute_normals`]: a degenerate UV triangle whose
    /// `det → 0` is skipped, not rescaled to infinity) into each of the
    /// three vertices. The same is done for `B` so the handedness sign
    /// can be tested per-vertex.
    ///
    /// After accumulation, at each vertex we project the accumulated
    /// `T_sum` against the per-vertex normal `N` and Gram-Schmidt
    /// orthonormalise:
    ///
    /// ```text
    /// T' = normalise(T_sum - (T_sum · N) * N)
    /// w  = sign((N × T') · B_sum)   // ±1
    /// ```
    ///
    /// This is the "MikkTSpace handedness rule" (per glTF 2.0
    /// §3.7.2.1): `B = w * (N × T)` — the renderer reconstructs the
    /// bitangent from `N`, `T`, `w` rather than storing a separate
    /// per-vertex `B`, halving the bandwidth.
    ///
    /// # Contract
    ///
    /// * Returns `None` if `normals` is absent, if UV set `uv_set` is
    ///   absent or empty, or if `positions` is empty. The caller can
    ///   then call [`Primitive::compute_normals`] + assignment and
    ///   retry — tangents are normal-dependent.
    /// * Output length always equals `positions.len()`. Vertices not
    ///   touched by any triangle, vertices whose UV chart is degenerate
    ///   (all triangles produce `det ≈ 0`), or vertices whose
    ///   accumulated `T_sum` is parallel to `N` (no UV gradient
    ///   information along the surface tangent plane) fall back to
    ///   `[1.0, 0.0, 0.0, 1.0]` — a unit vector and a positive
    ///   handedness, so the result is always renderable.
    /// * UV set `uv_set` selects which channel in
    ///   [`Primitive::uvs`] drives the tangent computation. Most
    ///   meshes have one UV set (`uv_set = 0`); a lightmap-uv-only
    ///   mesh would pass `uv_set = 1`.
    /// * NaN-safe. Any face whose computed `T_tri` or `B_tri` is
    ///   non-finite is skipped; any per-vertex sum that ends
    ///   non-finite or zero-length falls back as above.
    /// * The connectivity is the de-stripped triangle list from
    ///   [`Primitive::triangle_indices`], so `Triangles` /
    ///   `TriangleStrip` (alternating winding honoured) /
    ///   `TriangleFan` all feed in correctly; non-triangle topologies
    ///   produce an all-fallback buffer.
    /// * **Does not mutate `self`.** Assign the result to
    ///   [`Primitive::tangents`] if you want to store it — this is
    ///   the recompute step a format decoder runs when the wire stream
    ///   omits tangents (OBJ has no native tangent channel, glTF
    ///   without `TANGENT`).
    ///
    /// Cost is `O(triangle_count + V)`; allocates two scratch
    /// `Vec<[f32; 3]>` of length `V` (tangent and bitangent
    /// accumulators) plus the output `Vec<[f32; 4]>`.
    pub fn compute_tangents(&self, uv_set: usize) -> Option<Vec<[f32; 4]>> {
        const FALLBACK: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let n = self.positions.len();
        if n == 0 {
            return None;
        }
        let normals = self.normals.as_ref()?;
        if normals.len() != n {
            return None;
        }
        let uvs = self.uvs.get(uv_set)?;
        if uvs.len() != n {
            return None;
        }

        // Per-vertex tangent/bitangent accumulators (area-weighted by
        // construction: we skip the 1/det scaling that would otherwise
        // make a small UV triangle dominate).
        let mut t_acc = vec![[0.0f32; 3]; n];
        let mut b_acc = vec![[0.0f32; 3]; n];

        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = self.positions[ia];
            let pb = self.positions[ib];
            let pc = self.positions[ic];
            let qa = uvs[ia];
            let qb = uvs[ib];
            let qc = uvs[ic];

            // Edge vectors in object space.
            let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            // UV deltas.
            let du1 = qb[0] - qa[0];
            let dv1 = qb[1] - qa[1];
            let du2 = qc[0] - qa[0];
            let dv2 = qc[1] - qa[1];

            let det = du1 * dv2 - du2 * dv1;
            if !det.is_finite() || det == 0.0 {
                // Degenerate UV triangle — no surface tangent
                // information; contribute nothing.
                continue;
            }

            // The exact per-triangle tangent/bitangent (the actual
            // ∂P/∂u and ∂P/∂v on this triangle) are
            //   T = ( dv2·E1 - dv1·E2) / det
            //   B = (-du2·E1 + du1·E2) / det
            // We want to accumulate them area-weighted (so a small UV
            // triangle doesn't dominate). The unsigned UV triangle
            // area is |det|/2, so the area-weighted contribution is
            // numerator * sign(det) (= numerator/|det| * |det|).
            // Scaling by sign(det) keeps T pointing in the +U surface
            // direction even when the UV chart is mirrored (det<0):
            // we recover that mirror-vs-not signal separately at the
            // end via the cross-product handedness check, where it
            // belongs.
            let sgn = if det > 0.0 { 1.0 } else { -1.0 };
            let t_tri = [
                sgn * (dv2 * e1[0] - dv1 * e2[0]),
                sgn * (dv2 * e1[1] - dv1 * e2[1]),
                sgn * (dv2 * e1[2] - dv1 * e2[2]),
            ];
            let b_tri = [
                sgn * (-du2 * e1[0] + du1 * e2[0]),
                sgn * (-du2 * e1[1] + du1 * e2[1]),
                sgn * (-du2 * e1[2] + du1 * e2[2]),
            ];
            if !t_tri[0].is_finite()
                || !t_tri[1].is_finite()
                || !t_tri[2].is_finite()
                || !b_tri[0].is_finite()
                || !b_tri[1].is_finite()
                || !b_tri[2].is_finite()
            {
                continue;
            }
            for &i in &[ia, ib, ic] {
                t_acc[i][0] += t_tri[0];
                t_acc[i][1] += t_tri[1];
                t_acc[i][2] += t_tri[2];
                b_acc[i][0] += b_tri[0];
                b_acc[i][1] += b_tri[1];
                b_acc[i][2] += b_tri[2];
            }
        }

        // Per-vertex Gram-Schmidt + handedness recovery.
        let mut out = vec![FALLBACK; n];
        for i in 0..n {
            let n_v = normals[i];
            let t_sum = t_acc[i];
            let b_sum = b_acc[i];
            // Skip vertices that received no contribution.
            let tlen2 = t_sum[0] * t_sum[0] + t_sum[1] * t_sum[1] + t_sum[2] * t_sum[2];
            if !tlen2.is_finite() || tlen2 == 0.0 {
                continue;
            }
            // Skip vertices whose normal is degenerate (zero / NaN).
            let nlen2 = n_v[0] * n_v[0] + n_v[1] * n_v[1] + n_v[2] * n_v[2];
            if !nlen2.is_finite() || nlen2 == 0.0 {
                continue;
            }
            // Project T_sum onto the plane perpendicular to N:
            //   T' = T_sum - (T_sum · N) * N
            // We can assume N is already unit-length (compute_normals
            // returns unit normals); but to be safe against non-unit
            // user-supplied normals we don't rescale N here — the
            // Gram-Schmidt formula works for any N as long as we
            // normalise the result.
            let dot_tn = t_sum[0] * n_v[0] + t_sum[1] * n_v[1] + t_sum[2] * n_v[2];
            // If N happens to be non-unit, the projection coefficient
            // should be (T·N)/(N·N). Use the safe form.
            let coef = dot_tn / nlen2;
            let mut t = [
                t_sum[0] - coef * n_v[0],
                t_sum[1] - coef * n_v[1],
                t_sum[2] - coef * n_v[2],
            ];
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            if !len.is_finite() || len == 0.0 {
                // T_sum was parallel to N: no usable surface tangent.
                continue;
            }
            t[0] /= len;
            t[1] /= len;
            t[2] /= len;
            // Handedness: w = sign((N × T') · B_sum). +1.0 for
            // right-handed (N, T, B), -1.0 for mirrored / left-handed.
            let cross = [
                n_v[1] * t[2] - n_v[2] * t[1],
                n_v[2] * t[0] - n_v[0] * t[2],
                n_v[0] * t[1] - n_v[1] * t[0],
            ];
            let dot_cb = cross[0] * b_sum[0] + cross[1] * b_sum[1] + cross[2] * b_sum[2];
            let w = if dot_cb < 0.0 { -1.0 } else { 1.0 };
            out[i] = [t[0], t[1], t[2], w];
        }
        Some(out)
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

    /// Indices into [`Primitive::triangle_indices`] for **degenerate**
    /// triangles — triangles whose three vertices are collinear or
    /// coincident in 3D space.
    ///
    /// A triangle is degenerate iff its un-normalised face normal
    /// (the cross product of two edge vectors out of the same corner)
    /// is the zero vector. Equivalently, its signed area is zero —
    /// the three positions sit on a single line (or all three at one
    /// point). Such a triangle has no surface and contributes nothing
    /// to a shaded image; downstream code uniformly treats it as
    /// noise:
    ///
    /// * [`Primitive::compute_normals`] silently drops it (the
    ///   accumulator adds zero) — a vertex touched only by degenerate
    ///   faces ends up with the `[0, 0, 1]` fallback normal.
    /// * [`Primitive::compute_tangents`] silently drops it
    ///   (`det ≈ 0` in the UV-delta linear system).
    /// * STL spec (Fabbers / Stratasys 1989) explicitly forbids
    ///   degenerate facets — every facet must enclose three distinct
    ///   non-collinear vertices.
    ///
    /// This is the **detection-only** counterpart to those quiet
    /// drops: it surfaces *which* triangles are degenerate so a
    /// validator can warn, a repair pass can prune them, or a
    /// fixture-comparison test can pin them.
    ///
    /// # Contract
    ///
    /// * Returned indices reference [`Primitive::triangle_indices`] in
    ///   walk order. For [`Topology::Triangles`] index `t` is the
    ///   triangle whose corners are `triangle_indices()[t]`; for
    ///   `TriangleStrip` / `TriangleFan` the same. An empty `Vec` means
    ///   every (non-list-topology-implied-empty) triangle has non-zero
    ///   area in 3D.
    /// * **Collinear** is detected via the cross-product magnitude:
    ///   `|E1 × E2| == 0.0` exactly (no epsilon — a triangle that is
    ///   *almost* collinear within float precision but produces a
    ///   non-zero cross product is still considered valid; proximity
    ///   thresholding is a separate, lossy operation).
    /// * **Coincident** is the special case where two or three corners
    ///   share a position — the resulting edge vector is zero, the
    ///   cross product is zero, and the triangle is reported.
    /// * **Out-of-range index** entries are treated as degenerate
    ///   (the triangle can't be evaluated — same observable effect as
    ///   a zero-area triangle for downstream shaders).
    /// * **NaN-producing faces** are reported as degenerate (a face
    ///   whose cross product is non-finite has no well-defined area
    ///   and can't be safely shaded).
    /// * Non-triangle topologies (lines, points) return an empty `Vec`
    ///   — there are no triangles to test.
    /// * Pure (no `self` mutation). Cost is `O(triangle_count)`; one
    ///   small `Vec<usize>` allocation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bad = prim.degenerate_triangles();
    /// if !bad.is_empty() {
    ///     eprintln!("warning: {} degenerate triangles", bad.len());
    /// }
    /// ```
    pub fn degenerate_triangles(&self) -> Vec<usize> {
        let n = self.positions.len();
        let mut out = Vec::new();
        for (t, [ia, ib, ic]) in self.triangle_indices().into_iter().enumerate() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            // Out-of-range index: can't evaluate, treat as degenerate.
            if ia >= n || ib >= n || ic >= n {
                out.push(t);
                continue;
            }
            let pa = self.positions[ia];
            let pb = self.positions[ib];
            let pc = self.positions[ic];
            let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let cx = u[1] * v[2] - u[2] * v[1];
            let cy = u[2] * v[0] - u[0] * v[2];
            let cz = u[0] * v[1] - u[1] * v[0];
            // NaN-producing face: report as degenerate (can't shade
            // safely).
            if !cx.is_finite() || !cy.is_finite() || !cz.is_finite() {
                out.push(t);
                continue;
            }
            // Collinear / coincident: |E1 × E2| == 0.
            if cx == 0.0 && cy == 0.0 && cz == 0.0 {
                out.push(t);
            }
        }
        out
    }

    /// Classify every undirected triangle edge of this primitive by
    /// how many triangles use it, and return an [`EdgeManifoldReport`]
    /// summary.
    ///
    /// An **undirected edge** is the unordered pair of vertex pool
    /// indices `(min(a, b), max(a, b))`. The "use count" is the number
    /// of triangles in [`Primitive::triangle_indices`] that contain
    /// that pair as one of their three sides, regardless of corner
    /// winding direction. Each triangle contributes three undirected
    /// edges.
    ///
    /// # Classification (standard piecewise-linear topology)
    ///
    /// For each undirected edge:
    ///
    /// * **Boundary** — use count `= 1`. The edge sits on a hole, a
    ///   crack, or the outer rim of an open surface (a paper strip,
    ///   a half-cup). A *closed* manifold mesh — one a solid 3D
    ///   printer could fabricate — has **zero** boundary edges.
    /// * **Manifold-interior** — use count `= 2`. Exactly two
    ///   triangles meet at this edge, sharing the seam cleanly.
    ///   This is the standard "two-manifold" condition.
    /// * **Non-manifold** — use count `≥ 3`. Three or more triangles
    ///   meet at this edge (a "T-junction" / "book spine" /
    ///   "feather" defect). Most slicers and renderers can't
    ///   unambiguously compute a normal / interior side at such an
    ///   edge.
    ///
    /// The STL spec (Fabbers / Stratasys 1989) explicitly states the
    /// **vertex-to-vertex rule**: *"Each triangle must share two
    /// vertices with each of its adjacent triangles."* That rule
    /// implies every edge is used by exactly two facets — i.e. the
    /// mesh is closed and 2-manifold. This method gives a typed
    /// readout for that rule.
    ///
    /// # Contract
    ///
    /// * Only triangle topologies ([`Topology::Triangles`],
    ///   [`Topology::TriangleStrip`], [`Topology::TriangleFan`])
    ///   contribute edges. Lines/points/empty topologies yield an
    ///   all-zero report.
    /// * Triangle connectivity goes through
    ///   [`Primitive::triangle_indices`], so strip alternating winding
    ///   is honoured and out-of-range index entries are detected
    ///   ahead of edge counting.
    /// * A triangle whose three corner indices contain a duplicate
    ///   (so one or more of its edges has `a == b`, a *zero-length*
    ///   edge) is **excluded** from edge counting entirely — it's a
    ///   degenerate triangle by index, not a topology failure of its
    ///   neighbours. Use [`Primitive::degenerate_triangles`] to count
    ///   those.
    /// * An index entry that references a vertex slot beyond
    ///   `positions.len()` (out of range) is **excluded** along with
    ///   the whole triangle. The remaining triangles still feed in
    ///   normally — a single malformed corner doesn't poison the
    ///   neighbour count.
    /// * Topology comparison is by **vertex index**, not by 3D
    ///   position. Two corners with identical positions but different
    ///   indices are treated as distinct vertices on different edges
    ///   — run [`Primitive::weld_vertices`] first if you want
    ///   coincident corners to merge before counting.
    /// * Cost is `O(triangle_count)`; allocates one `HashMap` of
    ///   undirected edges. The `EdgeManifoldReport` itself does not
    ///   own the per-edge map — it stores only the counts.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let r = prim.edge_manifold_report();
    /// assert!(r.is_closed_manifold(),
    ///     "{} boundary, {} non-manifold edges",
    ///     r.boundary_edge_count, r.non_manifold_edge_count);
    /// ```
    pub fn edge_manifold_report(&self) -> EdgeManifoldReport {
        let n = self.positions.len();
        // Undirected edge → use count.
        let mut edge_uses: HashMap<(u32, u32), u32> = HashMap::new();
        for [ia, ib, ic] in self.triangle_indices() {
            // Out-of-range triangle: skip entirely.
            if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                continue;
            }
            // Triangle with a duplicate corner index has a zero-length
            // edge — degenerate-by-index. Skip the whole triangle so
            // its two "real" sides don't confuse neighbour counts.
            if ia == ib || ib == ic || ia == ic {
                continue;
            }
            for (a, b) in [(ia, ib), (ib, ic), (ic, ia)] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_uses.entry(key).or_insert(0) += 1;
            }
        }

        let mut boundary = 0usize;
        let mut interior = 0usize;
        let mut non_manifold = 0usize;
        let mut max_use = 0u32;
        for &count in edge_uses.values() {
            match count {
                0 => unreachable!("HashMap entry is always >= 1"),
                1 => boundary += 1,
                2 => interior += 1,
                _ => non_manifold += 1,
            }
            if count > max_use {
                max_use = count;
            }
        }

        EdgeManifoldReport {
            total_edge_count: edge_uses.len(),
            boundary_edge_count: boundary,
            manifold_interior_edge_count: interior,
            non_manifold_edge_count: non_manifold,
            max_edge_use: max_use,
        }
    }
}

/// Summary of the undirected-edge topology of a [`Primitive`], produced
/// by [`Primitive::edge_manifold_report`].
///
/// Every undirected edge of every (valid, non-degenerate-by-index)
/// triangle is bucketed by its **use count** — the number of
/// triangles that share it:
///
/// | Use count   | Bucket                     | Meaning                                                        |
/// | ----------- | -------------------------- | -------------------------------------------------------------- |
/// | `1`         | `boundary_edge_count`      | Edge on a hole / crack / open rim                              |
/// | `2`         | `manifold_interior_edge_count` | Standard two-manifold seam                                 |
/// | `≥ 3`       | `non_manifold_edge_count`  | Three or more faces meet here (T-junction / book-spine)        |
///
/// A **closed two-manifold** mesh has `boundary_edge_count == 0` and
/// `non_manifold_edge_count == 0` — every edge is shared by exactly
/// two faces, which is the STL spec's "vertex-to-vertex rule" and the
/// classical solid-printable condition. See
/// [`EdgeManifoldReport::is_closed_manifold`].
///
/// The report does **not** retain the per-edge map; the heavy
/// `HashMap` is freed as soon as it has been walked. Callers needing
/// the actual edge endpoints can re-derive them by walking
/// [`Primitive::triangle_indices`] themselves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgeManifoldReport {
    /// Total number of distinct undirected edges seen across all
    /// triangles. Sum of the three bucket counts.
    pub total_edge_count: usize,
    /// Edges used by exactly one triangle (open rim, crack, hole).
    pub boundary_edge_count: usize,
    /// Edges used by exactly two triangles (clean two-manifold seam).
    pub manifold_interior_edge_count: usize,
    /// Edges used by three or more triangles (non-manifold defect).
    pub non_manifold_edge_count: usize,
    /// Largest use count observed across all edges. `0` for an empty
    /// or all-degenerate primitive, `2` for a clean closed manifold,
    /// `≥ 3` when there is at least one non-manifold edge.
    pub max_edge_use: u32,
}

impl EdgeManifoldReport {
    /// `true` iff the primitive is a **closed two-manifold** — every
    /// edge is used by exactly two triangles. Equivalent to
    /// `boundary_edge_count == 0 && non_manifold_edge_count == 0 &&
    /// total_edge_count > 0`.
    ///
    /// An empty primitive (no triangles, `total_edge_count == 0`) is
    /// **not** considered closed — there is no surface to close.
    pub fn is_closed_manifold(&self) -> bool {
        self.total_edge_count > 0
            && self.boundary_edge_count == 0
            && self.non_manifold_edge_count == 0
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

    /// Sum of [`Primitive::surface_area`] across every contained
    /// primitive (mesh-local, no transforms / skin pose / morph deltas
    /// applied). Non-triangle primitives contribute 0.0.
    pub fn surface_area(&self) -> f64 {
        self.primitives.iter().map(|p| p.surface_area()).sum()
    }
}
