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

use std::collections::{HashMap, HashSet};

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

    /// Area-weighted surface centroid: the geometric centre of the
    /// primitive's triangle tessellation, treating the surface as a
    /// uniformly-dense flat shell. Returns `None` when no triangle
    /// contributes positive area (non-triangle topology, every face
    /// degenerate, all positions non-finite, every index out of range,
    /// or empty primitive).
    ///
    /// # Derivation (clean-room, first-principles)
    ///
    /// The continuous surface centroid of a body `S` under uniform
    /// surface density is
    ///
    /// ```text
    /// C = (∫∫_S x dS) / (∫∫_S dS).
    /// ```
    ///
    /// For a triangle tessellation, each triangle is a flat patch
    /// over which `x` varies linearly between the three corners; the
    /// well-known closed form for the integral of a linear function
    /// over a triangle is `area * value_at_centroid`, where the
    /// triangle centroid is the average of its three corners
    /// `(P_a + P_b + P_c) / 3` (Marsden & Tromba, *Vector Calculus*,
    /// chapter on triangle and parallelogram integrals — the
    /// barycentric weights average to 1/3 each). Summing over every
    /// triangle gives
    ///
    /// ```text
    /// C = (Σ area_i · centroid_i) / (Σ area_i)
    ///   = (Σ |E1 × E2|/2 · (P_a + P_b + P_c)/3) / surface_area.
    /// ```
    ///
    /// The same `|E1 × E2|/2` per-triangle area already drives
    /// [`Primitive::surface_area`]; `surface_centroid` reuses the
    /// identical edge-cross machinery and adds one three-corner sum +
    /// one scalar multiply per triangle, divided by the running area
    /// total at the end.
    ///
    /// # Contract
    ///
    /// * Returns `Some([f64; 3])` for any primitive with at least one
    ///   non-degenerate triangle whose corners are finite. Each
    ///   component is finite (NaN/Inf-producing triangles are skipped
    ///   the same way [`Primitive::surface_area`] skips them).
    /// * Returns `None` when the surface-area accumulator stays at
    ///   `0.0` — there is no positive-area surface to centre. This
    ///   matches the `None` policy on [`Primitive::bounding_box`] for
    ///   an empty positions buffer.
    /// * Coordinates are in the local frame of [`Primitive::positions`]
    ///   (matching the parent [`crate::Scene3D::unit`]). The centroid
    ///   is a position, not an offset; translation of the primitive
    ///   moves the centroid by the same vector.
    /// * **Degenerate triangles contribute nothing.** Same set as the
    ///   one [`Primitive::degenerate_triangles`] reports — a zero-area
    ///   triangle adds `0 * anything = 0` to both numerator and
    ///   denominator.
    /// * **Out-of-range index** entries are skipped, not panicked.
    /// * Non-triangle topologies return `None`.
    /// * Accumulators are `f64` so a million-triangle mesh doesn't
    ///   drift under `f32` summation; the per-triangle cross-product
    ///   math is also `f64`.
    /// * **Does not mutate `self`.** Pure; cost `O(triangle_count)`.
    ///
    /// # Use
    ///
    /// * Pivot-point heuristic for a "rotate around centre" gesture —
    ///   the area-weighted centroid sits inside the visible shell for
    ///   typical closed surfaces and is more stable than the AABB
    ///   centre under non-symmetric tessellations.
    /// * Importer round-trip checks — the surface centroid is invariant
    ///   under triangle subdivision (a single triangle and its
    ///   barycentric-subdivided version produce the same centroid),
    ///   making it a useful equivalence-class fingerprint.
    /// * Initial guess for a per-mesh local origin shift before a
    ///   `weld_vertices` / dedup pass; centring positions around the
    ///   centroid keeps the `f32` representable range balanced.
    ///
    /// See [`Mesh::surface_centroid`] for the per-mesh roll-up and
    /// [`crate::Scene3D::surface_centroid`] for the scene-level
    /// aggregate.
    pub fn surface_centroid(&self) -> Option<[f64; 3]> {
        let n = self.positions.len();
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_area = 0.0_f64;
        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = self.positions[ia];
            let pb = self.positions[ib];
            let pc = self.positions[ic];
            // f64 from the loaded position values onward so the
            // per-triangle area + centroid math stays stable at scale.
            let ax = pa[0] as f64;
            let ay = pa[1] as f64;
            let az = pa[2] as f64;
            let bx = pb[0] as f64;
            let by = pb[1] as f64;
            let bz = pb[2] as f64;
            let cx = pc[0] as f64;
            let cy = pc[1] as f64;
            let cz = pc[2] as f64;
            let ux = bx - ax;
            let uy = by - ay;
            let uz = bz - az;
            let vx = cx - ax;
            let vy = cy - ay;
            let vz = cz - az;
            // Cross product u × v; its magnitude is twice the
            // triangle area — same as `surface_area`.
            let crx = uy * vz - uz * vy;
            let cry = uz * vx - ux * vz;
            let crz = ux * vy - uy * vx;
            if !crx.is_finite() || !cry.is_finite() || !crz.is_finite() {
                continue;
            }
            let m2 = crx * crx + cry * cry + crz * crz;
            if !m2.is_finite() {
                continue;
            }
            let area = m2.sqrt() * 0.5;
            if !area.is_finite() || area == 0.0 {
                continue;
            }
            // Per-triangle centroid is the barycentre of its corners;
            // the contribution to the numerator is `area * centroid`,
            // i.e. `(area / 3) * (Pa + Pb + Pc)`.
            let w = area / 3.0;
            sum_x += w * (ax + bx + cx);
            sum_y += w * (ay + by + cy);
            sum_z += w * (az + bz + cz);
            sum_area += area;
        }
        if sum_area == 0.0 || !sum_area.is_finite() {
            return None;
        }
        let inv = 1.0 / sum_area;
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Transform-aware surface area: same triangle reduction as
    /// [`Primitive::surface_area`], but every corner is first mapped
    /// through the row-major column-vector affine 4x4 `world` matrix
    /// (same convention as [`crate::Transform::Matrix`] /
    /// [`crate::BoundingBox::transform`]) before the per-triangle area
    /// is accumulated. The translation column cancels in the edge
    /// differences, so only the upper-left 3x3 of `world` enters the
    /// per-triangle contribution.
    ///
    /// Used by [`crate::Scene3D::world_surface_area`] to fold each
    /// node's ancestor-chain transform into a per-instance area total
    /// without applying a single scalar scale to the local area (which
    /// would be wrong under non-uniform scale, since the
    /// post-transform area of a triangle depends on its orientation
    /// relative to the scale axes).
    ///
    /// Contract matches [`Primitive::surface_area`]:
    ///
    /// * `Triangles` / `TriangleStrip` / `TriangleFan` contribute their
    ///   transformed triangle area; other topologies contribute 0.0.
    /// * Degenerate triangles (after transform), out-of-range indices,
    ///   and NaN-/Inf-producing intermediates contribute 0.0.
    /// * Result is finite and non-negative; accumulator is `f64`.
    /// * Pure; cost `O(triangle_count)`.
    pub fn world_surface_area(&self, world: [[f32; 4]; 4]) -> f64 {
        let n = self.positions.len();
        let mut total = 0.0_f64;
        // Promote the 3x3 + translation to f64 once so the per-triangle
        // edge-mapping is a small fixed cost rather than a per-vertex
        // f32-cast cascade.
        let m00 = world[0][0] as f64;
        let m01 = world[0][1] as f64;
        let m02 = world[0][2] as f64;
        let m03 = world[0][3] as f64;
        let m10 = world[1][0] as f64;
        let m11 = world[1][1] as f64;
        let m12 = world[1][2] as f64;
        let m13 = world[1][3] as f64;
        let m20 = world[2][0] as f64;
        let m21 = world[2][1] as f64;
        let m22 = world[2][2] as f64;
        let m23 = world[2][3] as f64;
        let xform = |p: [f32; 3]| {
            let x = p[0] as f64;
            let y = p[1] as f64;
            let z = p[2] as f64;
            [
                m00 * x + m01 * y + m02 * z + m03,
                m10 * x + m11 * y + m12 * z + m13,
                m20 * x + m21 * y + m22 * z + m23,
            ]
        };
        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = xform(self.positions[ia]);
            let pb = xform(self.positions[ib]);
            let pc = xform(self.positions[ic]);
            let ux = pb[0] - pa[0];
            let uy = pb[1] - pa[1];
            let uz = pb[2] - pa[2];
            let vx = pc[0] - pa[0];
            let vy = pc[1] - pa[1];
            let vz = pc[2] - pa[2];
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

    /// Transform-aware area-weighted surface centroid: the same
    /// area-weighted recombination as [`Primitive::surface_centroid`],
    /// but every corner is first mapped through the row-major
    /// column-vector affine 4x4 `world` matrix (same convention as
    /// [`crate::Transform::Matrix`] / [`crate::BoundingBox::transform`])
    /// before each per-triangle area and centroid is accumulated. The
    /// translation column enters the per-corner position (and so the
    /// per-triangle centroid contribution), but cancels in the edge
    /// differences that drive the per-triangle area weight — the upper-
    /// left 3x3 alone fixes the area weighting; the full 4x4 fixes the
    /// position the weight multiplies.
    ///
    /// Used by [`crate::Scene3D::world_surface_centroid`] to fold each
    /// reachable node's ancestor-chain transform into a per-instance
    /// area-weighted centroid total. Returning the contribution
    /// numerator + denominator separately (instead of `world *
    /// local_centroid` scaled by a single area factor) is the only
    /// faithful answer under non-uniform scale, since both the per-
    /// triangle area weight *and* the per-triangle centroid bend with
    /// the transform in ways that don't factor through the local
    /// centroid alone.
    ///
    /// # Derivation
    ///
    /// For a triangle `(P_a, P_b, P_c)` mapped through the affine world
    /// matrix `M`, the post-transform corners are `M·P_*`, the post-
    /// transform centroid is `(M·P_a + M·P_b + M·P_c) / 3`, and the
    /// post-transform area is `|(M_3·E1) × (M_3·E2)| / 2` (the
    /// translation row cancels in the edge differences; `M_3` is the
    /// upper-left 3x3). Substituting in the continuous identity
    /// `C = (Σ area_i · centroid_i) / Σ area_i` and accumulating gives
    /// the closed form. Under a pure translation `t` (`M_3 = I`), every
    /// per-triangle area is unchanged and every per-triangle centroid
    /// gains `t`, so the area-weighted recombination gains `t` exactly
    /// — translation equivariance.
    ///
    /// # Contract
    ///
    /// * Topology handling, degenerate-triangle skipping, NaN guarding,
    ///   and out-of-range-index skipping all mirror
    ///   [`Primitive::surface_centroid`] / [`Primitive::world_surface_area`].
    ///   Non-triangle topologies return `None`. Result components are
    ///   finite for any finite input.
    /// * Returns `None` when the post-transform area accumulator stays
    ///   at `0.0` — either every triangle was degenerate, the
    ///   topology was non-triangular, or the transform collapsed every
    ///   triangle to zero area (e.g. a `[0, 1, 1]` scale flattens a
    ///   `Z=const` mesh's effective Y-Z extent only — but a `[0, 0, 1]`
    ///   scale on every axis is `None`).
    /// * Coordinates are in the **world** frame defined by the supplied
    ///   matrix — translation of `world` translates the result by the
    ///   same vector; rotation rotates it; uniform scale `s` around the
    ///   origin scales it from the origin by `s`.
    /// * Accumulators are `f64`; per-triangle math is `f64`.
    /// * **Does not mutate `self`.** Pure; cost `O(triangle_count)`.
    pub fn world_surface_centroid(&self, world: [[f32; 4]; 4]) -> Option<[f64; 3]> {
        let n = self.positions.len();
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_area = 0.0_f64;
        // Promote the 4x4 to f64 once so the per-corner mapping is a
        // small fixed cost rather than a per-vertex f32-cast cascade.
        let m00 = world[0][0] as f64;
        let m01 = world[0][1] as f64;
        let m02 = world[0][2] as f64;
        let m03 = world[0][3] as f64;
        let m10 = world[1][0] as f64;
        let m11 = world[1][1] as f64;
        let m12 = world[1][2] as f64;
        let m13 = world[1][3] as f64;
        let m20 = world[2][0] as f64;
        let m21 = world[2][1] as f64;
        let m22 = world[2][2] as f64;
        let m23 = world[2][3] as f64;
        let xform = |p: [f32; 3]| {
            let x = p[0] as f64;
            let y = p[1] as f64;
            let z = p[2] as f64;
            [
                m00 * x + m01 * y + m02 * z + m03,
                m10 * x + m11 * y + m12 * z + m13,
                m20 * x + m21 * y + m22 * z + m23,
            ]
        };
        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = xform(self.positions[ia]);
            let pb = xform(self.positions[ib]);
            let pc = xform(self.positions[ic]);
            if !pa[0].is_finite()
                || !pa[1].is_finite()
                || !pa[2].is_finite()
                || !pb[0].is_finite()
                || !pb[1].is_finite()
                || !pb[2].is_finite()
                || !pc[0].is_finite()
                || !pc[1].is_finite()
                || !pc[2].is_finite()
            {
                continue;
            }
            let ux = pb[0] - pa[0];
            let uy = pb[1] - pa[1];
            let uz = pb[2] - pa[2];
            let vx = pc[0] - pa[0];
            let vy = pc[1] - pa[1];
            let vz = pc[2] - pa[2];
            let crx = uy * vz - uz * vy;
            let cry = uz * vx - ux * vz;
            let crz = ux * vy - uy * vx;
            if !crx.is_finite() || !cry.is_finite() || !crz.is_finite() {
                continue;
            }
            let m2 = crx * crx + cry * cry + crz * crz;
            if !m2.is_finite() {
                continue;
            }
            let area = m2.sqrt() * 0.5;
            if !area.is_finite() || area == 0.0 {
                continue;
            }
            // Per-triangle contribution to the numerator is
            // `area * centroid` = `(area / 3) * (P_a + P_b + P_c)`,
            // same shape as `surface_centroid` in the local frame.
            let w = area / 3.0;
            sum_x += w * (pa[0] + pb[0] + pc[0]);
            sum_y += w * (pa[1] + pb[1] + pc[1]);
            sum_z += w * (pa[2] + pb[2] + pc[2]);
            sum_area += area;
        }
        if sum_area == 0.0 || !sum_area.is_finite() {
            return None;
        }
        let inv = 1.0 / sum_area;
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Signed volume enclosed by this primitive's triangle tessellation,
    /// in the unit-cubed of [`Primitive::positions`] (matching the parent
    /// [`crate::Scene3D::unit`] — metres³ by default). **The result is
    /// only physically meaningful for a closed two-manifold surface**
    /// (i.e. one for which [`Primitive::is_closed_manifold`] returns
    /// `true`); for an open/non-manifold mesh the sum is still well-
    /// defined arithmetically but no longer corresponds to a true
    /// enclosed volume.
    ///
    /// Sign follows the winding convention: CCW-viewed-from-outside
    /// (the crate's right-handed, glTF-aligned convention,
    /// `Triangles` / `TriangleStrip` / `TriangleFan` all matching
    /// [`Primitive::triangle_indices`]) produces a **positive** value
    /// for an outward-facing closed surface; a uniformly inside-out
    /// (clockwise-from-outside) mesh produces the same magnitude with
    /// the opposite sign. The unsigned [`Primitive::volume`] always
    /// returns the absolute value.
    ///
    /// Non-triangle topologies (lines/points) contribute 0.0.
    ///
    /// # Derivation (clean-room, first-principles)
    ///
    /// The divergence theorem (Gauss; Marsden & Tromba, *Vector
    /// Calculus*) states that for a vector field `F` on a closed
    /// region `V` bounded by `S`,
    ///
    /// ```text
    /// ∫∫∫_V (∇ · F) dV = ∫∫_S F · dS.
    /// ```
    ///
    /// Picking the radial field `F(x) = x / 3` gives `∇ · F = 1`, so
    /// the left side reduces to the enclosed volume `V`. The right side
    /// becomes the sum of `(x / 3) · n_face * area_face` over every
    /// triangle face. For a flat triangle with corners
    /// `(P_a, P_b, P_c)`, the centroid is `(P_a + P_b + P_c) / 3` and
    /// `n_face * area_face` is `(E1 × E2) / 2`. Substituting in and
    /// expanding the scalar triple product, the contribution of one
    /// triangle collapses to
    ///
    /// ```text
    /// V_tri = (1 / 6) · (P_a · (P_b × P_c)).
    /// ```
    ///
    /// Summing across every triangle gives the closed-form
    /// `(1 / 6) · Σ P_a · (P_b × P_c)`. (This is exactly the
    /// signed-tetrahedron-sum technique — each triangle plus the
    /// origin forms a tetrahedron of signed volume `P_a · (P_b × P_c)
    /// / 6`; the origin-coincident faces cancel pairwise for a closed
    /// mesh, leaving only the boundary contributions. The derivation
    /// matches the closed-form result in Cha Zhang & Tsuhan Chen,
    /// "Efficient feature extraction for 2D/3D objects in mesh
    /// representation", ICIP 2001.)
    ///
    /// The cross-product machinery is identical to the one
    /// [`Primitive::compute_normals`] and [`Primitive::surface_area`]
    /// already use; `signed_volume` adds one scalar dot per triangle
    /// (the third factor `P_a · (E1 × E2)`).
    ///
    /// # Contract
    ///
    /// * Always returns a finite `f64` for a primitive whose positions
    ///   are all finite. The accumulator is `f64` so a million-triangle
    ///   mesh doesn't drift under `f32` summation; per-triangle scalar
    ///   triple product is also `f64`.
    /// * **Degenerate triangles contribute zero.** A triangle whose
    ///   edge cross product is the zero vector adds 0.0 to the sum
    ///   (the dot with any `P_a` is also zero, but the early NaN guard
    ///   makes that explicit). They neither help nor corrupt the total.
    /// * **NaN-safe.** A face whose edge differences, cross product, or
    ///   triple product is non-finite contributes 0.0 instead of
    ///   poisoning the sum. The whole result stays finite even on a
    ///   partly-corrupt vertex buffer.
    /// * **Out-of-range index** entries are skipped, not panicked.
    /// * Non-triangle topologies return 0.0.
    /// * **Translation-invariant for a closed surface.** Because the
    ///   origin-coincident tetrahedron contributions cancel for a
    ///   closed mesh, the same closed mesh translated by any constant
    ///   offset gives the same signed volume (modulo float round-off
    ///   on the `O(n)` summation). An *open* mesh's signed_volume is
    ///   not translation-invariant (the open boundary leaks).
    /// * **Does not mutate `self`.** Pure; cost `O(triangle_count)`.
    ///
    /// # Use
    ///
    /// * STL conformance checks — the Fabbers/Stratasys
    ///   solid-printability recipe asks for a positive enclosed
    ///   volume; a closed manifold mesh with negative volume usually
    ///   means the file was authored inside-out (every facet wound
    ///   CW-from-outside).
    /// * 3D-print slicer pre-flight — compute total material volume
    ///   for cost / time estimates.
    /// * Importer sanity checks — comparing the volume reported by
    ///   format A's decoder vs format B's decoder against the same
    ///   mesh should round-trip to the same number.
    ///
    /// See [`Primitive::volume`] for the unsigned magnitude.
    pub fn signed_volume(&self) -> f64 {
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
            // f64 from the loaded position values onward so the scalar
            // triple product (a × b · c) stays stable at scale.
            let ax = pa[0] as f64;
            let ay = pa[1] as f64;
            let az = pa[2] as f64;
            let bx = pb[0] as f64;
            let by = pb[1] as f64;
            let bz = pb[2] as f64;
            let cx = pc[0] as f64;
            let cy = pc[1] as f64;
            let cz = pc[2] as f64;
            // P_b × P_c
            let crx = by * cz - bz * cy;
            let cry = bz * cx - bx * cz;
            let crz = bx * cy - by * cx;
            if !crx.is_finite() || !cry.is_finite() || !crz.is_finite() {
                continue;
            }
            // P_a · (P_b × P_c) — the signed volume of the tetrahedron
            // formed by the origin and the three corners.
            let tri = ax * crx + ay * cry + az * crz;
            if !tri.is_finite() {
                continue;
            }
            total += tri;
        }
        total / 6.0
    }

    /// Unsigned volume enclosed by this primitive's triangle
    /// tessellation — `|signed_volume()|` — in the unit-cubed of
    /// [`Primitive::positions`].
    ///
    /// Like [`Primitive::signed_volume`], the result is only
    /// physically meaningful for a closed two-manifold surface
    /// (`is_closed_manifold() == true`). The magnitude is robust to
    /// inside-out winding: a uniformly CW-from-outside mesh reports
    /// the same volume as the equivalent CCW-from-outside mesh.
    ///
    /// Non-triangle topologies (lines/points) and empty primitives
    /// return `0.0`.
    pub fn volume(&self) -> f64 {
        self.signed_volume().abs()
    }

    /// Volume-weighted centroid of the solid enclosed by this primitive's
    /// closed triangle tessellation — the **centre of mass** of a
    /// uniform-density body bounded by the surface, in the unit of
    /// [`Primitive::positions`].
    ///
    /// # Derivation
    ///
    /// The continuous identity is
    /// `C = ∫∫∫_V x dV / ∫∫∫_V dV`. The denominator is already the
    /// [`Primitive::signed_volume`] reduction. For the numerator, the
    /// same divergence-theorem trick `signed_volume` uses (fan the
    /// closed surface into origin-anchored tetrahedra, whose
    /// origin-coincident faces cancel pairwise for a closed mesh)
    /// applies directly: for each surface triangle `(P_a, P_b, P_c)`
    /// the corresponding tetrahedron `(0, P_a, P_b, P_c)` has signed
    /// volume `V_i = (P_a · (P_b × P_c)) / 6` and centroid
    /// `C_i = (0 + P_a + P_b + P_c) / 4 = (P_a + P_b + P_c) / 4`
    /// (the centroid of a tetrahedron is the average of its four
    /// vertices, a standard barycentric result). The
    /// volume-weighted centroid of the whole solid is then
    /// `C = (Σ V_i · C_i) / Σ V_i`. The same closed form is the
    /// "Volume Integration" reduction in any textbook treatment of
    /// rigid-body mass properties — e.g. Mirtich, "Fast and Accurate
    /// Computation of Polyhedral Mass Properties", *Journal of
    /// Graphics Tools* 1(2), 1996, equation (1.16); the closed form
    /// also appears in Cha & Chen, "Efficient feature extraction for
    /// 2D/3D objects in mesh representation", ICIP 2001, which we
    /// already cite for [`Primitive::signed_volume`].
    ///
    /// The cross-product machinery is exactly the same as
    /// [`Primitive::signed_volume`]; this helper adds three
    /// per-triangle corner sums plus one scalar multiply per axis,
    /// so the per-triangle cost stays a small constant factor.
    ///
    /// # Contract
    ///
    /// * Topology integration goes through
    ///   [`Primitive::triangle_indices`], so `Triangles` /
    ///   `TriangleStrip` (alternating winding) / `TriangleFan` all
    ///   feed in correctly; non-triangle topologies (lines/points)
    ///   return `None`.
    /// * Accumulators are `f64` so million-triangle meshes don't drift
    ///   under `f32` summation.
    /// * Degenerate (collinear/coincident corners), NaN- or
    ///   Inf-producing faces, and out-of-range index entries contribute
    ///   nothing — matching the silent-skip robustness contract of
    ///   every other reduction.
    /// * **Translation-equivariant for a closed surface** — under a
    ///   uniform translation `Δ` every corner shifts by `Δ`, every
    ///   per-tet centroid by `Δ`, and the volume-weighted average by
    ///   `Δ`. (Individual tetrahedra are *not* translation-invariant
    ///   because the origin is the implicit fourth vertex; the
    ///   surface-cancellation argument that makes
    ///   `signed_volume` translation-invariant also makes the volume
    ///   centroid translation-equivariant — the origin-anchored
    ///   contributions cancel pairwise for the closed boundary.)
    /// * **Sign-invariant** — flipping every triangle's winding flips
    ///   `signed_volume`'s sign *and* each `V_i`'s sign, so the
    ///   weighted-average ratio is unchanged. An inside-out cube
    ///   reports the same centre of mass as the equivalent
    ///   right-side-out cube.
    /// * Only physically meaningful for a closed two-manifold (see
    ///   [`Primitive::is_closed_manifold`]); arithmetically
    ///   well-defined regardless. An open surface (a hemisphere, a
    ///   plane) produces an answer that depends on where the origin
    ///   sits because the surface-cancellation argument no longer
    ///   applies. Callers wanting the centroid of an open patch
    ///   should use [`Primitive::surface_centroid`] instead.
    /// * Returns `None` when `Σ V_i` is `0.0` (degenerate mesh / flat
    ///   sheet / perfectly cancelling shells) or non-finite — there
    ///   is no centre of mass to report for a zero-volume body.
    /// * Pure; cost `O(triangle_count)`.
    ///
    /// # Use
    ///
    /// * Rigid-body physics setup — the body's centre of mass is the
    ///   torque-free axis of rotation around which the inertia tensor
    ///   is diagonalisable.
    /// * Camera framing / orbit-target picking — the volume centroid
    ///   of a closed mesh is closer to the perceptual centre of a
    ///   solid object than its [`Primitive::bounding_box`] centre
    ///   (which is pulled toward heavy / wide protrusions) or its
    ///   [`Primitive::surface_centroid`] (which is pulled toward
    ///   high-surface-area regions, e.g. spikes).
    /// * 3D-print balance pre-flight — the volume centroid relative to
    ///   the bed plane predicts tipping during print.
    ///
    /// See [`Mesh::volume_centroid`] for the per-mesh roll-up and
    /// [`crate::Scene3D::volume_centroid`] for the scene-level
    /// aggregate.
    pub fn volume_centroid(&self) -> Option<[f64; 3]> {
        let n = self.positions.len();
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_v = 0.0_f64;
        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = self.positions[ia];
            let pb = self.positions[ib];
            let pc = self.positions[ic];
            // f64 from the loaded position values onward so the scalar
            // triple product (a · (b × c)) and the per-tet centroid
            // stay stable at scale.
            let ax = pa[0] as f64;
            let ay = pa[1] as f64;
            let az = pa[2] as f64;
            let bx = pb[0] as f64;
            let by = pb[1] as f64;
            let bz = pb[2] as f64;
            let cx = pc[0] as f64;
            let cy = pc[1] as f64;
            let cz = pc[2] as f64;
            // P_b × P_c
            let crx = by * cz - bz * cy;
            let cry = bz * cx - bx * cz;
            let crz = bx * cy - by * cx;
            if !crx.is_finite() || !cry.is_finite() || !crz.is_finite() {
                continue;
            }
            // 6 · V_i — the signed volume of the tetrahedron formed by
            // the origin and the three corners times six. We divide by
            // six only at the end (it cancels between numerator and
            // denominator, but we keep the factor to make the final
            // signed-volume comparison faithful).
            let six_v = ax * crx + ay * cry + az * crz;
            if !six_v.is_finite() {
                continue;
            }
            // Per-tetrahedron centroid contribution `V_i · C_i` where
            // `C_i = (P_a + P_b + P_c) / 4`. We accumulate `6 · V_i ·
            // (P_a + P_b + P_c)` and divide by 24 at the end so the
            // per-step cost is one multiply per axis.
            let sx = ax + bx + cx;
            let sy = ay + by + cy;
            let sz = az + bz + cz;
            sum_x += six_v * sx;
            sum_y += six_v * sy;
            sum_z += six_v * sz;
            sum_v += six_v;
        }
        if sum_v == 0.0 || !sum_v.is_finite() {
            return None;
        }
        // sum_x is 24 · Σ (V_i · C_i_x); sum_v is 6 · Σ V_i.
        // Ratio is 4 · Σ (V_i · C_i_x) / Σ V_i → divide by 4 so the
        // final answer is the textbook `Σ V_i · C_i / Σ V_i`.
        let inv = 1.0 / (sum_v * 4.0);
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Transform-aware volume-weighted centroid (centre of mass) of the
    /// uniform-density solid enclosed by this primitive's triangle
    /// tessellation, after every corner is mapped through the row-major
    /// column-vector affine 4x4 `world` matrix (same convention as
    /// [`crate::Transform::Matrix`] / [`crate::BoundingBox::transform`]
    /// / [`Primitive::world_surface_area`] /
    /// [`Primitive::world_surface_centroid`]). Sibling of
    /// [`Primitive::volume_centroid`] for the per-instance world-frame
    /// case.
    ///
    /// Used by [`crate::Scene3D::world_volume_centroid`] to fold each
    /// reachable node's ancestor-chain transform into a per-instance
    /// volume-weighted centroid total. Returning the post-divide ratio
    /// rather than the raw numerator (`Σ V_i · C_i`) and denominator
    /// (`Σ V_i`) keeps the per-primitive shape natural for direct
    /// callers; the scene-level helper recovers each per-primitive
    /// signed volume from [`Primitive::world_signed_volume`] (one extra
    /// triangle pass) and multiplies — same pattern as
    /// [`Mesh::world_surface_centroid`].
    ///
    /// # Derivation
    ///
    /// The local helper [`Primitive::volume_centroid`] sums per-tet
    /// signed volumes `V_i = (P_a · (P_b × P_c)) / 6` weighted by the
    /// per-tet centroid `C_i = (P_a + P_b + P_c) / 4`. Under an affine
    /// map `M`, every corner `P_*` becomes `M·P_*`, so:
    ///
    /// ```text
    /// V_i_world  = (M·P_a · ((M·P_b) × (M·P_c))) / 6
    /// C_i_world  = (M·P_a + M·P_b + M·P_c) / 4.
    /// ```
    ///
    /// Both formulas mirror the local case with the transformed corners
    /// substituted in. Because the per-tet volume here is the signed
    /// volume of the **origin-anchored** tet `(0, M·P_a, M·P_b, M·P_c)`
    /// (not the local-then-transformed tet), the translation column of
    /// `M` *does* enter every term — translating `world` by `t` shifts
    /// every `V_i` by a boundary-dependent amount and every `C_i` by
    /// `t`. The boundary terms cancel pairwise for a **closed**
    /// two-manifold mesh (Σ V_i_world = det(M_3) · V_local; Σ V_i_world
    /// · C_i_world = det(M_3) · V_local · (C_local + 0) folded with the
    /// per-corner translation gives the post-transform centroid), so
    /// the post-divide ratio reduces to the textbook `C_world = M_3 ·
    /// C_local + t`. For an open patch the boundary terms remain; the
    /// returned ratio is still the closed-form per-tet volume integral
    /// and matches the arithmetic generalisation of
    /// [`Primitive::volume_centroid`].
    ///
    /// # Contract
    ///
    /// * Topology handling, degenerate / NaN guards, and out-of-range-
    ///   index skipping all mirror [`Primitive::volume_centroid`] /
    ///   [`Primitive::world_surface_centroid`]. Non-triangle topologies
    ///   return `None`. Result components are finite for any finite
    ///   input.
    /// * Returns `None` when the accumulated signed volume is `0.0` or
    ///   non-finite — every triangle degenerate, non-triangle topology,
    ///   transform collapsing every tet to zero signed volume (e.g. a
    ///   `[1, 1, 0]` scale), or perfectly cancelling shells.
    /// * Coordinates are in the **world** frame defined by `world`. For
    ///   a closed mesh under an invertible affine `M`, the result
    ///   equals `M · C_local` exactly (within `f64` round-off); for an
    ///   open patch the result depends on where the origin sits in the
    ///   transformed frame (same caveat as
    ///   [`Primitive::volume_centroid`]).
    /// * Accumulators are `f64`; per-triangle math is `f64`.
    /// * **Does not mutate `self`.** Pure; cost `O(triangle_count)`.
    pub fn world_volume_centroid(&self, world: [[f32; 4]; 4]) -> Option<[f64; 3]> {
        let n = self.positions.len();
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_v = 0.0_f64;
        // Promote the 4x4 to f64 once so the per-corner mapping is a
        // small fixed cost rather than a per-vertex f32-cast cascade —
        // mirrors `world_surface_centroid`.
        let m00 = world[0][0] as f64;
        let m01 = world[0][1] as f64;
        let m02 = world[0][2] as f64;
        let m03 = world[0][3] as f64;
        let m10 = world[1][0] as f64;
        let m11 = world[1][1] as f64;
        let m12 = world[1][2] as f64;
        let m13 = world[1][3] as f64;
        let m20 = world[2][0] as f64;
        let m21 = world[2][1] as f64;
        let m22 = world[2][2] as f64;
        let m23 = world[2][3] as f64;
        let xform = |p: [f32; 3]| {
            let x = p[0] as f64;
            let y = p[1] as f64;
            let z = p[2] as f64;
            [
                m00 * x + m01 * y + m02 * z + m03,
                m10 * x + m11 * y + m12 * z + m13,
                m20 * x + m21 * y + m22 * z + m23,
            ]
        };
        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = xform(self.positions[ia]);
            let pb = xform(self.positions[ib]);
            let pc = xform(self.positions[ic]);
            if !pa[0].is_finite()
                || !pa[1].is_finite()
                || !pa[2].is_finite()
                || !pb[0].is_finite()
                || !pb[1].is_finite()
                || !pb[2].is_finite()
                || !pc[0].is_finite()
                || !pc[1].is_finite()
                || !pc[2].is_finite()
            {
                continue;
            }
            // P_b × P_c (in world frame).
            let crx = pb[1] * pc[2] - pb[2] * pc[1];
            let cry = pb[2] * pc[0] - pb[0] * pc[2];
            let crz = pb[0] * pc[1] - pb[1] * pc[0];
            if !crx.is_finite() || !cry.is_finite() || !crz.is_finite() {
                continue;
            }
            // 6 · V_i — six times the signed volume of the origin-tet
            // formed in the post-transform frame. Same scaling trick as
            // `volume_centroid`: keep the factor of 6 in V and the
            // factor of 4 in C, divide both out at the end.
            let six_v = pa[0] * crx + pa[1] * cry + pa[2] * crz;
            if !six_v.is_finite() {
                continue;
            }
            let sx = pa[0] + pb[0] + pc[0];
            let sy = pa[1] + pb[1] + pc[1];
            let sz = pa[2] + pb[2] + pc[2];
            sum_x += six_v * sx;
            sum_y += six_v * sy;
            sum_z += six_v * sz;
            sum_v += six_v;
        }
        if sum_v == 0.0 || !sum_v.is_finite() {
            return None;
        }
        // sum_x = 24 · Σ (V_i · C_i_x), sum_v = 6 · Σ V_i → ratio is
        // `4 · Σ V_i · C_i / Σ V_i` → divide by 4 once at the end so
        // the final answer matches the textbook `Σ V_i · C_i / Σ V_i`.
        let inv = 1.0 / (sum_v * 4.0);
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Transform-aware signed volume of the origin-anchored tetrahedron
    /// sum after every corner is mapped through the row-major
    /// column-vector affine 4x4 `world` matrix. Helper used by
    /// [`Mesh::world_volume_centroid`] to recover each per-primitive
    /// signed-volume weight without re-walking the centroid path. The
    /// per-corner mapping is identical to
    /// [`Primitive::world_volume_centroid`]; only the centroid
    /// accumulators are dropped.
    ///
    /// # Contract
    ///
    /// * Mirrors [`Primitive::signed_volume`]'s `f64` accumulator and
    ///   non-triangle / out-of-range / NaN skipping policy. Always
    ///   returns a finite `f64` for finite input.
    /// * Coordinates are in the **world** frame. For a closed two-
    ///   manifold mesh under an affine `M` the result reduces to
    ///   `det(M_3) · signed_volume()` (closed-mesh translation
    ///   cancellation); for an open patch the translation column of
    ///   `M` enters the result.
    /// * Pure; cost `O(triangle_count)`.
    pub fn world_signed_volume(&self, world: [[f32; 4]; 4]) -> f64 {
        let n = self.positions.len();
        let mut sum_v = 0.0_f64;
        let m00 = world[0][0] as f64;
        let m01 = world[0][1] as f64;
        let m02 = world[0][2] as f64;
        let m03 = world[0][3] as f64;
        let m10 = world[1][0] as f64;
        let m11 = world[1][1] as f64;
        let m12 = world[1][2] as f64;
        let m13 = world[1][3] as f64;
        let m20 = world[2][0] as f64;
        let m21 = world[2][1] as f64;
        let m22 = world[2][2] as f64;
        let m23 = world[2][3] as f64;
        let xform = |p: [f32; 3]| {
            let x = p[0] as f64;
            let y = p[1] as f64;
            let z = p[2] as f64;
            [
                m00 * x + m01 * y + m02 * z + m03,
                m10 * x + m11 * y + m12 * z + m13,
                m20 * x + m21 * y + m22 * z + m23,
            ]
        };
        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = xform(self.positions[ia]);
            let pb = xform(self.positions[ib]);
            let pc = xform(self.positions[ic]);
            if !pa[0].is_finite()
                || !pa[1].is_finite()
                || !pa[2].is_finite()
                || !pb[0].is_finite()
                || !pb[1].is_finite()
                || !pb[2].is_finite()
                || !pc[0].is_finite()
                || !pc[1].is_finite()
                || !pc[2].is_finite()
            {
                continue;
            }
            let crx = pb[1] * pc[2] - pb[2] * pc[1];
            let cry = pb[2] * pc[0] - pb[0] * pc[2];
            let crz = pb[0] * pc[1] - pb[1] * pc[0];
            if !crx.is_finite() || !cry.is_finite() || !crz.is_finite() {
                continue;
            }
            let six_v = pa[0] * crx + pa[1] * cry + pa[2] * crz;
            if !six_v.is_finite() {
                continue;
            }
            sum_v += six_v;
        }
        // The accumulator has six times each per-tet signed volume; the
        // textbook formula is `(1/6) Σ P_a · (P_b × P_c)`. Divide once
        // at the end.
        sum_v / 6.0
    }

    /// Transform-aware unit-density inertia tensor of the solid enclosed
    /// by this primitive's closed triangle tessellation, taken about the
    /// **origin of the world frame** after every corner is mapped through
    /// the row-major column-vector affine 4x4 `world` matrix (same
    /// convention as [`crate::Transform::Matrix`] /
    /// [`crate::BoundingBox::transform`] /
    /// [`Primitive::world_volume_centroid`] /
    /// [`Primitive::world_surface_centroid`]). Sibling of
    /// [`Primitive::inertia_tensor`] for the per-instance world-frame
    /// case that round 259's prose flagged as the next-round candidate.
    ///
    /// Returned as a row-major symmetric `[[f64; 3]; 3]` matrix with the
    /// same rigid-body convention as [`Primitive::inertia_tensor`]:
    /// diagonal entries are the moments `I_αα = ∫_V (β² + γ²) dV` (the two
    /// world coordinates not equal to α), off-diagonals are the negated
    /// products of inertia `I_αβ = -∫_V x_α · x_β dV`.
    ///
    /// # Derivation
    ///
    /// The local helper [`Primitive::inertia_tensor`] fans the closed
    /// surface into origin-anchored tetrahedra `(0, P_a, P_b, P_c)` and
    /// evaluates the closed-form per-tet second-moment integrals. Under
    /// an affine map `M` every corner `P_*` becomes `M·P_*`, so the same
    /// closed form evaluated on the **world-frame** corners
    /// `(0, M·P_a, M·P_b, M·P_c)` gives `∫ x_α x_β dV` in the world frame
    /// directly. Mapping the corners first (rather than transforming the
    /// local tensor with `M_3 · I_local · M_3ᵀ` and a separate parallel-
    /// axis correction) folds rotation, non-uniform scale, **skew**, and
    /// the **translation** column of `M` into one pass — exactly the way
    /// [`Primitive::world_volume_centroid`] maps corners first so the
    /// translation column enters every origin-anchored tet term. For a
    /// closed two-manifold under an invertible affine `M` the result
    /// equals the analytic transform of the local tensor:
    /// `I_world(about world origin)` follows from `I_local(about local
    /// origin)` by the linear map `M_3 · (centred tensor) · M_3ᵀ`
    /// scaled by `det(M_3)` plus the parallel-axis shift induced by the
    /// translation `t` — but the direct corner-mapped integral computes
    /// it in one place without the bookkeeping. For an open patch the
    /// boundary terms remain and the result depends on where the world
    /// origin sits, the same caveat the local helper carries.
    ///
    /// # Contract
    ///
    /// * Topology integration, degenerate / NaN guards, and out-of-range-
    ///   index skipping all mirror [`Primitive::inertia_tensor`] /
    ///   [`Primitive::world_volume_centroid`]. Non-triangle topologies
    ///   return `None`.
    /// * Returns `None` only when every face has been silently skipped
    ///   (non-triangle, empty, or every per-tet integrand non-finite).
    /// * Coordinates are in the **world** frame defined by `world`. Scale
    ///   `s` along an axis scales the corresponding second moments by the
    ///   fifth power overall (`∫ x² dV` carries two position powers plus
    ///   three volume powers); a uniform scale `s` multiplies the whole
    ///   tensor by `s⁵`. A winding flip (or a mirror in `M`, `det(M_3) <
    ///   0`) negates the tensor, the same way it negates
    ///   [`Primitive::world_signed_volume`].
    /// * Accumulators are `f64`; per-triangle math is `f64`. Always
    ///   finite for finite input.
    /// * **Does not mutate `self`.** Pure; cost `O(triangle_count)`.
    ///
    /// Used by [`Mesh::world_inertia_tensor`] and
    /// [`crate::Scene3D::world_inertia_tensor`] to fold each reachable
    /// node's ancestor-chain transform into a per-instance world-frame
    /// inertia total.
    pub fn world_inertia_tensor(&self, world: [[f32; 4]; 4]) -> Option<[[f64; 3]; 3]> {
        if !matches!(
            self.topology,
            Topology::Triangles | Topology::TriangleStrip | Topology::TriangleFan
        ) {
            return None;
        }
        let n = self.positions.len();
        let mut acc_xx = 0.0_f64;
        let mut acc_yy = 0.0_f64;
        let mut acc_zz = 0.0_f64;
        let mut acc_xy = 0.0_f64;
        let mut acc_xz = 0.0_f64;
        let mut acc_yz = 0.0_f64;
        let mut any_finite = false;
        // Promote the 4x4 to f64 once so each corner mapping is a small
        // fixed cost — mirrors `world_volume_centroid` / `world_signed_volume`.
        let m00 = world[0][0] as f64;
        let m01 = world[0][1] as f64;
        let m02 = world[0][2] as f64;
        let m03 = world[0][3] as f64;
        let m10 = world[1][0] as f64;
        let m11 = world[1][1] as f64;
        let m12 = world[1][2] as f64;
        let m13 = world[1][3] as f64;
        let m20 = world[2][0] as f64;
        let m21 = world[2][1] as f64;
        let m22 = world[2][2] as f64;
        let m23 = world[2][3] as f64;
        let xform = |p: [f32; 3]| {
            let x = p[0] as f64;
            let y = p[1] as f64;
            let z = p[2] as f64;
            [
                m00 * x + m01 * y + m02 * z + m03,
                m10 * x + m11 * y + m12 * z + m13,
                m20 * x + m21 * y + m22 * z + m23,
            ]
        };
        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = xform(self.positions[ia]);
            let pb = xform(self.positions[ib]);
            let pc = xform(self.positions[ic]);
            let (ax, ay, az) = (pa[0], pa[1], pa[2]);
            let (bx, by, bz) = (pb[0], pb[1], pb[2]);
            let (cx, cy, cz) = (pc[0], pc[1], pc[2]);
            if !ax.is_finite()
                || !ay.is_finite()
                || !az.is_finite()
                || !bx.is_finite()
                || !by.is_finite()
                || !bz.is_finite()
                || !cx.is_finite()
                || !cy.is_finite()
                || !cz.is_finite()
            {
                continue;
            }
            // P_b × P_c in the world frame — same cross product as
            // `world_signed_volume`.
            let crx = by * cz - bz * cy;
            let cry = bz * cx - bx * cz;
            let crz = bx * cy - by * cx;
            if !crx.is_finite() || !cry.is_finite() || !crz.is_finite() {
                continue;
            }
            // six_v = 6 · V_i — the origin-anchored signed tetrahedron
            // volume (world frame) times six.
            let six_v = ax * crx + ay * cry + az * crz;
            if !six_v.is_finite() {
                continue;
            }
            // Same closed-form per-axis / cross-axis polynomials as
            // `inertia_tensor`, evaluated on the transformed corners.
            let mxx = ax * ax + bx * bx + cx * cx + ax * bx + ax * cx + bx * cx;
            let myy = ay * ay + by * by + cy * cy + ay * by + ay * cy + by * cy;
            let mzz = az * az + bz * bz + cz * cz + az * bz + az * cz + bz * cz;
            let mxy = 2.0 * (ax * ay + bx * by + cx * cy)
                + (ax * by + ay * bx)
                + (ax * cy + ay * cx)
                + (bx * cy + by * cx);
            let mxz = 2.0 * (ax * az + bx * bz + cx * cz)
                + (ax * bz + az * bx)
                + (ax * cz + az * cx)
                + (bx * cz + bz * cx);
            let myz = 2.0 * (ay * az + by * bz + cy * cz)
                + (ay * bz + az * by)
                + (ay * cz + az * cy)
                + (by * cz + bz * cy);
            if !mxx.is_finite()
                || !myy.is_finite()
                || !mzz.is_finite()
                || !mxy.is_finite()
                || !mxz.is_finite()
                || !myz.is_finite()
            {
                continue;
            }
            acc_xx += six_v * mxx;
            acc_yy += six_v * myy;
            acc_zz += six_v * mzz;
            acc_xy += six_v * mxy;
            acc_xz += six_v * mxz;
            acc_yz += six_v * myz;
            any_finite = true;
        }
        if !any_finite {
            return None;
        }
        // `∫_V x_α² dV = acc_αα / 60`; `∫_V x_α·x_β dV = acc_αβ / 120`.
        let int_xx = acc_xx / 60.0;
        let int_yy = acc_yy / 60.0;
        let int_zz = acc_zz / 60.0;
        let int_xy = acc_xy / 120.0;
        let int_xz = acc_xz / 120.0;
        let int_yz = acc_yz / 120.0;
        let i_xx = int_yy + int_zz;
        let i_yy = int_xx + int_zz;
        let i_zz = int_xx + int_yy;
        let i_xy = -int_xy;
        let i_xz = -int_xz;
        let i_yz = -int_yz;
        Some([[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]])
    }

    /// Unit-density inertia tensor of the solid enclosed by this
    /// primitive's closed triangle tessellation, taken about the
    /// **origin** of [`Primitive::positions`], in the unit-to-the-fifth
    /// of the position frame.
    ///
    /// Returned as a row-major `[[f64; 3]; 3]` symmetric matrix:
    ///
    /// ```text
    /// [ I_xx  I_xy  I_xz ]
    /// [ I_xy  I_yy  I_yz ]
    /// [ I_xz  I_yz  I_zz ]
    /// ```
    ///
    /// The convention is the standard rigid-body one: diagonal entries
    /// are the *moments* `I_αα = ∫_V (β² + γ²) dV` (the two coordinates
    /// not equal to α), and off-diagonal entries are the negated
    /// *products of inertia* `I_αβ = -∫_V x_α · x_β dV` (α ≠ β). The
    /// tensor maps an angular velocity `ω` to the angular momentum
    /// `L = I · ω` for a rigid body of unit density and the given shape.
    /// Multiply by the body's actual density `ρ` (or by the total mass
    /// `M = ρ · V` and divide by `V` if you prefer mass-normalised
    /// units) to get the physical inertia tensor.
    ///
    /// # Derivation
    ///
    /// The continuous identity `I_αβ = ∫∫∫_V f_αβ(x, y, z) dV` (with the
    /// integrand the moment or product-of-inertia kernel) is reduced by
    /// the same divergence-theorem decomposition
    /// [`Primitive::signed_volume`] and [`Primitive::volume_centroid`]
    /// already use: fan the closed surface into origin-anchored
    /// tetrahedra `(0, P_a, P_b, P_c)`, whose origin-coincident
    /// boundary terms cancel pairwise for a closed two-manifold. Each
    /// origin-anchored tetrahedron has signed volume
    /// `V_i = (P_a · (P_b × P_c)) / 6`, and the second-moment integrals
    /// over its interior have the closed form (with `P_0 = 0`,
    /// `P_1 = P_a`, `P_2 = P_b`, `P_3 = P_c`):
    ///
    /// ```text
    /// ∫_T x_α² dV  = (V_i / 10) · ( Σ_{k=1..3} p_α[k]²
    ///                              + Σ_{1 ≤ j < k ≤ 3} p_α[j]·p_α[k] )
    ///
    /// ∫_T x_α x_β dV = (V_i / 20) · ( 2·Σ_{k=1..3} p_α[k]·p_β[k]
    ///                                + Σ_{j ≠ k} p_α[j]·p_β[k] / 2 )
    /// ```
    ///
    /// (The `P_0 = 0` corner drops every term it appears in, collapsing
    /// the four-corner symmetric polynomials to the three-corner forms
    /// above.) These are the standard second-moment integrals of a
    /// tetrahedron, the same closed-form Mirtich, "Fast and Accurate
    /// Computation of Polyhedral Mass Properties", *Journal of Graphics
    /// Tools* 1(2), 1996 uses (we already cite Mirtich for
    /// [`Primitive::volume_centroid`]), specialised to the
    /// origin-anchored fourth vertex.
    ///
    /// The per-tetrahedron diagonal-moment contributions assemble
    /// directly:
    ///
    /// ```text
    /// I_xx_i = ∫_T (y² + z²) dV = ∫_T y² dV + ∫_T z² dV
    /// I_xy_i = - ∫_T x · y dV
    /// ```
    ///
    /// — and the same shape for the remaining tensor components. The
    /// per-tetrahedron sums add up across the closed surface to give the
    /// whole-body inertia tensor about the origin.
    ///
    /// To shift the tensor to a different reference point (e.g. the
    /// centre of mass returned by [`Primitive::volume_centroid`]), apply
    /// the parallel-axis theorem: `I_about_C = I_about_O - M · D` where
    /// `M = ρ · V` is the body's mass, and `D` is the
    /// rank-1-plus-trace-corrected displacement tensor
    /// `D_αβ = c_α · c_β - δ_αβ · |c|²` for the centroid offset
    /// `c = C - O`.
    ///
    /// # Contract
    ///
    /// * Topology integration goes through
    ///   [`Primitive::triangle_indices`], so `Triangles` /
    ///   `TriangleStrip` (alternating winding) / `TriangleFan` all feed
    ///   in correctly; non-triangle topologies (lines/points) return
    ///   `None`.
    /// * Accumulators are `f64`; per-triangle math is `f64`. Returns
    ///   `None` only for non-triangle topology, an empty primitive, or
    ///   when every face has been silently skipped because every per-tet
    ///   integrand was non-finite.
    /// * **Degenerate triangles contribute zero.** A face whose edge
    ///   cross product, signed-volume scalar, or per-axis polynomial
    ///   sum is non-finite is silently skipped — same robustness
    ///   contract as [`Primitive::signed_volume`] /
    ///   [`Primitive::volume_centroid`].
    /// * **Out-of-range index** entries are skipped, not panicked.
    /// * **Translation-equivariant for a closed surface.** Under a
    ///   uniform translation `Δ` the tensor about the *origin* shifts
    ///   by the parallel-axis correction (every per-tet integrand
    ///   changes), but the tensor about the *centre of mass* is
    ///   invariant (a coordinate-independent intrinsic property of the
    ///   shape). For an *open* mesh the closed-mesh boundary-term
    ///   cancellation argument does not apply and the returned tensor
    ///   depends on where the origin sits in the primitive's frame.
    /// * **Sign-invariant for the diagonal moments under winding flip.**
    ///   Flipping every triangle's winding flips every per-tet signed
    ///   volume, but the closed-form integrals scale linearly in that
    ///   signed volume, so an inside-out closed mesh produces the
    ///   *negated* tensor. Callers wanting the physical (positive-
    ///   diagonal) tensor of a closed body should either ensure the
    ///   winding is CCW-from-outside or take `[I_αα.abs()]`-stamped
    ///   diagonals — same caveat as [`Primitive::signed_volume`]'s
    ///   sign-vs-volume relationship.
    /// * Only physically meaningful for a closed two-manifold (see
    ///   [`Primitive::is_closed_manifold`]); arithmetically well-defined
    ///   regardless.
    /// * **Does not mutate `self`.** Pure; cost `O(triangle_count)`.
    ///
    /// # Use
    ///
    /// * Rigid-body dynamics — the body's inertia tensor is what an
    ///   engine multiplies into its angular-velocity update; the
    ///   centre-of-mass-shifted form is what most simulators want, and
    ///   the parallel-axis shift outlined in the derivation gets you
    ///   there from this origin-about result.
    /// * Principal-axis decomposition / oriented bounding boxes — the
    ///   eigenvectors of the centred inertia tensor are the body's
    ///   principal axes, useful as a tighter alternative to AABB for
    ///   picking / collision broad-phase.
    /// * Numerical 3D-print analysis — the principal moments tell you
    ///   the body's preferred resting orientation under gravity (the
    ///   axis with the largest moment is the most stable spin axis).
    ///
    /// See [`Mesh::inertia_tensor`] for the per-mesh roll-up.
    pub fn inertia_tensor(&self) -> Option<[[f64; 3]; 3]> {
        if !matches!(
            self.topology,
            Topology::Triangles | Topology::TriangleStrip | Topology::TriangleFan
        ) {
            return None;
        }
        let n = self.positions.len();
        // Accumulators scaled to keep the per-tet step division-free.
        // `acc_xx` holds `Σ six_v · (Ax² + Bx² + Cx² + Ax·Bx + Ax·Cx + Bx·Cx)`,
        // which is `60 · Σ V_i · ∫_T x² dV / V_i = 60 · Σ ∫_T x² dV`.
        // The whole-body `∫_V x² dV` is therefore `acc_xx / 60`.
        // Off-diagonals accumulate the doubled-symmetric polynomial; the
        // closed-form factor there is `V/20`, so per-tet contribution is
        // `six_v · (…) / 120` and the divisor at the end is 120.
        let mut acc_xx = 0.0_f64;
        let mut acc_yy = 0.0_f64;
        let mut acc_zz = 0.0_f64;
        let mut acc_xy = 0.0_f64;
        let mut acc_xz = 0.0_f64;
        let mut acc_yz = 0.0_f64;
        let mut any_finite = false;
        for [ia, ib, ic] in self.triangle_indices() {
            let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
            if ia >= n || ib >= n || ic >= n {
                continue;
            }
            let pa = self.positions[ia];
            let pb = self.positions[ib];
            let pc = self.positions[ic];
            // Promote to f64 once per corner; matches the
            // `signed_volume` / `volume_centroid` precision policy.
            let ax = pa[0] as f64;
            let ay = pa[1] as f64;
            let az = pa[2] as f64;
            let bx = pb[0] as f64;
            let by = pb[1] as f64;
            let bz = pb[2] as f64;
            let cx = pc[0] as f64;
            let cy = pc[1] as f64;
            let cz = pc[2] as f64;
            // P_b × P_c — same cross-product as `signed_volume`.
            let crx = by * cz - bz * cy;
            let cry = bz * cx - bx * cz;
            let crz = bx * cy - by * cx;
            if !crx.is_finite() || !cry.is_finite() || !crz.is_finite() {
                continue;
            }
            // six_v = 6 · V_i — the origin-anchored signed tetrahedron
            // volume times six.
            let six_v = ax * crx + ay * cry + az * crz;
            if !six_v.is_finite() {
                continue;
            }
            // Per-axis closed-form `Σ p[k]² + Σ_{j<k} p[j]·p[k]` for the
            // three non-origin corners — the polynomial that appears in
            // `∫_T x_α² dV = (V_i / 10) · (…)`.
            let mxx = ax * ax + bx * bx + cx * cx + ax * bx + ax * cx + bx * cx;
            let myy = ay * ay + by * by + cy * cy + ay * by + ay * cy + by * cy;
            let mzz = az * az + bz * bz + cz * cz + az * bz + az * cz + bz * cz;
            // Cross-axis closed-form
            // `2·Σ p_α[k]·p_β[k] + Σ_{j≠k} p_α[j]·p_β[k]` for the three
            // non-origin corners — the polynomial that appears in
            // `∫_T x_α·x_β dV = (V_i / 20) · (…)`.
            let mxy = 2.0 * (ax * ay + bx * by + cx * cy)
                + (ax * by + ay * bx)
                + (ax * cy + ay * cx)
                + (bx * cy + by * cx);
            let mxz = 2.0 * (ax * az + bx * bz + cx * cz)
                + (ax * bz + az * bx)
                + (ax * cz + az * cx)
                + (bx * cz + bz * cx);
            let myz = 2.0 * (ay * az + by * bz + cy * cz)
                + (ay * bz + az * by)
                + (ay * cz + az * cy)
                + (by * cz + bz * cy);
            if !mxx.is_finite()
                || !myy.is_finite()
                || !mzz.is_finite()
                || !mxy.is_finite()
                || !mxz.is_finite()
                || !myz.is_finite()
            {
                continue;
            }
            acc_xx += six_v * mxx;
            acc_yy += six_v * myy;
            acc_zz += six_v * mzz;
            acc_xy += six_v * mxy;
            acc_xz += six_v * mxz;
            acc_yz += six_v * myz;
            any_finite = true;
        }
        if !any_finite {
            return None;
        }
        // `∫_V x_α² dV = acc_αα / 60`; `∫_V x_α·x_β dV = acc_αβ / 120`.
        // Diagonals: `I_αα = ∫(β² + γ²) dV`.
        // Off-diagonals carry the minus sign by convention.
        let int_xx = acc_xx / 60.0;
        let int_yy = acc_yy / 60.0;
        let int_zz = acc_zz / 60.0;
        let int_xy = acc_xy / 120.0;
        let int_xz = acc_xz / 120.0;
        let int_yz = acc_yz / 120.0;
        let i_xx = int_yy + int_zz;
        let i_yy = int_xx + int_zz;
        let i_zz = int_xx + int_yy;
        let i_xy = -int_xy;
        let i_xz = -int_xz;
        let i_yz = -int_yz;
        Some([[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]])
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

    /// Return every **boundary edge** of this primitive — the undirected
    /// triangle edges used by exactly one triangle.
    ///
    /// This is the *detection-only* extractor counterpart to
    /// [`EdgeManifoldReport::boundary_edge_count`] (which only counts
    /// them), the same way [`Primitive::degenerate_triangles`] is the
    /// extractor counterpart to a degenerate-triangle count. A boundary
    /// edge sits on a hole, a crack, or the open rim of a non-closed
    /// surface; a closed two-manifold mesh
    /// ([`Primitive::is_closed_manifold`]) has **none**. Each returned
    /// `[u32; 2]` is the edge's two vertex-pool indices in ascending
    /// order (`[min(a, b), max(a, b)]`), so the pair is canonical
    /// regardless of which triangle's winding first introduced it.
    ///
    /// # Use cases
    ///
    /// * **Hole detection / hole-filling pre-pass** — the boundary edges
    ///   are the open seams; chaining them end-to-end recovers the
    ///   boundary loops a fill pass would triangulate.
    /// * **Open-rim outlining** — rendering just the boundary edges of an
    ///   open surface (a cloth patch, a terrain tile) as a wireframe
    ///   silhouette.
    /// * **Watertightness diagnostics** — a non-empty result on a mesh
    ///   that should be solid flags exactly where the surface is torn,
    ///   complementing the aggregate [`EdgeManifoldReport`] readout with
    ///   the concrete edge list.
    ///
    /// # Contract
    ///
    /// * Edge bucketing is identical to
    ///   [`Primitive::edge_manifold_report`]: undirected edges keyed by
    ///   `(min, max)` vertex index, counted over
    ///   [`Primitive::triangle_indices`] (so `Triangles` /
    ///   `TriangleStrip` / `TriangleFan` all feed in with the strip
    ///   alternating-winding rule honoured). Only edges whose use count
    ///   is exactly `1` are returned; manifold-interior (`2`) and
    ///   non-manifold (`≥ 3`) edges are not.
    /// * A triangle with an out-of-range corner index, or a duplicate
    ///   corner index (a zero-length edge), is **excluded whole** before
    ///   counting — its sides don't appear and don't perturb the
    ///   neighbour counts of valid triangles. Same exclusion rule as
    ///   [`Primitive::edge_manifold_report`].
    /// * Topology comparison is by **vertex index**, not 3D position.
    ///   Run [`Primitive::weld_vertices`] first if positionally
    ///   coincident corners on different indices should merge before the
    ///   boundary is computed (otherwise a welded-shut seam still reads
    ///   as two boundary edges).
    /// * Non-triangle topologies (lines/points) and empty primitives
    ///   return an empty `Vec`.
    /// * The result is sorted ascending by `(first, second)` index so
    ///   the output is deterministic across runs (the underlying
    ///   `HashMap` walk order is not). Pure (no `self` mutation); cost
    ///   `O(triangle_count + boundary_edge_count · log boundary_edge_count)`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A single open triangle has three boundary edges.
    /// let mut prim = Primitive::new(Topology::Triangles);
    /// prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    /// assert_eq!(prim.boundary_edges().len(), 3);
    /// assert!(!prim.edge_manifold_report().is_closed_manifold());
    /// ```
    pub fn boundary_edges(&self) -> Vec<[u32; 2]> {
        let n = self.positions.len();
        // Undirected edge → use count (same keying as
        // `edge_manifold_report`).
        let mut edge_uses: HashMap<(u32, u32), u32> = HashMap::new();
        for [ia, ib, ic] in self.triangle_indices() {
            // Out-of-range triangle: skip entirely.
            if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                continue;
            }
            // Duplicate corner index → zero-length edge; degenerate by
            // index. Skip the whole triangle so its real sides don't
            // confuse neighbour counts.
            if ia == ib || ib == ic || ia == ic {
                continue;
            }
            for (a, b) in [(ia, ib), (ib, ic), (ic, ia)] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_uses.entry(key).or_insert(0) += 1;
            }
        }

        let mut out: Vec<[u32; 2]> = edge_uses
            .into_iter()
            .filter_map(|((a, b), count)| (count == 1).then_some([a, b]))
            .collect();
        // Deterministic ordering — the HashMap walk order is not stable.
        out.sort_unstable();
        out
    }

    /// Chain this primitive's boundary edges end-to-end into ordered
    /// **boundary loops**.
    ///
    /// Where [`Primitive::boundary_edges`] returns the loose set of
    /// open-seam edges, this method stitches them into connected
    /// vertex-index sequences — the closed (or open) chains of vertices
    /// that bound each hole, crack, or open rim. It is the natural next
    /// step the [`Primitive::boundary_edges`] docs gesture toward: "a
    /// hole-detection / hole-filling pre-pass (chaining the boundary
    /// edges end-to-end recovers the boundary loops a fill pass
    /// triangulates)".
    ///
    /// Each returned `Vec<u32>` is one loop, listed as the ordered
    /// vertex-pool indices walked along the boundary in the surface's
    /// **winding-consistent direction** (a boundary half-edge keeps the
    /// orientation of the single triangle that owns it — for an
    /// outward-facing CCW mesh, a hole's loop therefore runs clockwise
    /// when viewed from outside, the standard "the surface is on your
    /// left" convention). The start vertex is **not** repeated at the
    /// end: a triangular hole returns three indices, not four. A
    /// well-formed loop is closed (its last vertex's outgoing boundary
    /// edge returns to the first); a chain that dead-ends (because a
    /// non-manifold defect consumed the continuation) is returned as the
    /// open path it is, so the caller still gets every boundary vertex.
    ///
    /// # Use cases
    ///
    /// * **Hole filling** — each closed loop is a polygon a fan / ear-clip
    ///   triangulator can cap to make the surface watertight.
    /// * **Open-rim outlining** — render each loop as a closed polyline
    ///   silhouette of an open patch.
    /// * **Genus / hole-count diagnostics** — the number of loops is the
    ///   number of distinct open seams on the surface.
    ///
    /// # Contract
    ///
    /// * The boundary-edge set is exactly [`Primitive::boundary_edges`]'s
    ///   (undirected edges used by exactly one triangle), but the chaining
    ///   uses each such edge's **directed** half-edge `a → b` taken from
    ///   the owning triangle's winding, so the loop direction is
    ///   well-defined. Edge bucketing, the out-of-range / duplicate-corner
    ///   whole-triangle exclusion, and the [`Primitive::triangle_indices`]
    ///   topology feed (`Triangles` / `TriangleStrip` / `TriangleFan`) all
    ///   match `boundary_edges`.
    /// * Walking starts from the boundary half-edge whose source vertex is
    ///   smallest and follows `b → next` through the outgoing boundary
    ///   half-edge at each vertex until the loop closes or no
    ///   continuation exists. At a pinch vertex with more than one
    ///   outgoing boundary half-edge (a figure-eight / non-manifold
    ///   vertex) the smallest-target continuation is chosen
    ///   deterministically; the remaining half-edges seed their own loops.
    ///   Every boundary half-edge is consumed exactly once, so the loops
    ///   partition the boundary-edge set.
    /// * The list of loops is sorted ascending by each loop's first
    ///   (smallest-rotation) vertex so the output is deterministic across
    ///   runs (the underlying `HashMap` walk order is not). Each loop is
    ///   additionally rotated to start at its own smallest vertex index,
    ///   so the same loop is reported identically regardless of which
    ///   half-edge seeded it.
    /// * Topology comparison is by **vertex index**, not 3D position — run
    ///   [`Primitive::weld_vertices`] first if positionally coincident
    ///   corners on different indices should merge before the boundary is
    ///   traced.
    /// * Non-triangle topologies (lines/points), empty primitives, and
    ///   closed two-manifolds
    ///   ([`EdgeManifoldReport::is_closed_manifold`]) return an empty
    ///   `Vec`. Pure (no `self` mutation); cost
    ///   `O(triangle_count + boundary_edge_count · log boundary_edge_count)`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A single open triangle's three boundary edges chain into one
    /// // three-vertex loop.
    /// let mut prim = Primitive::new(Topology::Triangles);
    /// prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    /// let loops = prim.boundary_loops();
    /// assert_eq!(loops.len(), 1);
    /// assert_eq!(loops[0].len(), 3);
    /// ```
    pub fn boundary_loops(&self) -> Vec<Vec<u32>> {
        let n = self.positions.len();
        // First pass: count undirected edge uses and record the directed
        // half-edge for each. Same exclusion rules as `boundary_edges`.
        let mut edge_uses: HashMap<(u32, u32), u32> = HashMap::new();
        let mut directed: Vec<(u32, u32)> = Vec::new();
        for [ia, ib, ic] in self.triangle_indices() {
            if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                continue;
            }
            if ia == ib || ib == ic || ia == ic {
                continue;
            }
            for (a, b) in [(ia, ib), (ib, ic), (ic, ia)] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_uses.entry(key).or_insert(0) += 1;
                directed.push((a, b));
            }
        }

        // Outgoing boundary half-edges per source vertex (use count == 1).
        // Sorted-target buckets give a deterministic continuation pick.
        let mut outgoing: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(a, b) in &directed {
            let key = if a < b { (a, b) } else { (b, a) };
            if edge_uses.get(&key) == Some(&1) {
                outgoing.entry(a).or_default().push(b);
            }
        }
        if outgoing.is_empty() {
            return Vec::new();
        }
        for targets in outgoing.values_mut() {
            // Largest-last so `pop()` consumes the smallest target first.
            targets.sort_unstable_by(|x, y| y.cmp(x));
        }

        // Deterministic seed order: ascending source vertex.
        let mut sources: Vec<u32> = outgoing.keys().copied().collect();
        sources.sort_unstable();

        let mut loops: Vec<Vec<u32>> = Vec::new();
        for &seed in &sources {
            // Drain every half-edge that still starts at `seed`.
            while outgoing.get(&seed).is_some_and(|t| !t.is_empty()) {
                let mut chain: Vec<u32> = Vec::new();
                let mut cur = seed;
                // Follow the boundary until it returns to the seed (closed
                // loop) or runs out of continuations (open chain).
                loop {
                    chain.push(cur);
                    let next = match outgoing.get_mut(&cur).and_then(|t| t.pop()) {
                        Some(v) => v,
                        None => break,
                    };
                    if next == seed {
                        // Loop closed — `next` is the already-recorded
                        // start vertex, so don't push it again.
                        break;
                    }
                    cur = next;
                }
                // Rotate so the loop starts at its smallest vertex index,
                // making the representation seed-independent.
                if let Some((min_pos, _)) = chain.iter().enumerate().min_by_key(|&(_, &v)| v) {
                    chain.rotate_left(min_pos);
                }
                loops.push(chain);
            }
        }

        // Deterministic loop ordering by first (smallest) vertex.
        loops.sort_unstable();
        loops
    }

    /// Cap every boundary loop of this surface with a triangle fan-free
    /// ear-clip patch, returning a new closed(-er) [`Topology::Triangles`]
    /// primitive — the hole-filling step the [`Primitive::boundary_edges`]
    /// / [`Primitive::boundary_loops`] docs name as their headline use.
    ///
    /// [`Primitive::boundary_loops`] traces each hole / crack / open rim as
    /// an ordered vertex-pool loop walked in the surface's
    /// winding-consistent direction; this method triangulates the polygon
    /// each loop bounds and appends the patch triangles to a de-stripped
    /// copy of the surface, so the seams those loops named become filled
    /// faces. **Every** boundary loop is capped — a free-floating surface
    /// patch has no intrinsic "outer" rim distinct from an interior hole,
    /// so a flat patch with a hole is closed into a (zero-thickness)
    /// double-sided shell with both its hole and its outer rim filled.
    /// Callers that need to keep a known outer rim open should splice only
    /// the caps they want from [`Primitive::boundary_loops`] instead. The
    /// intended pipeline is therefore
    /// `weld_vertices().fill_holes()` — weld positionally-coincident
    /// corners into a shared pool first (boundary detection is by vertex
    /// index, so an unwelded seam is not seen as a hole), then cap.
    ///
    /// # Output
    ///
    /// * The surface is first run through [`Primitive::to_triangle_list`],
    ///   so the result is always [`Topology::Triangles`] with an explicit
    ///   `U32` index buffer; the original triangles are preserved verbatim
    ///   and the cap triangles are appended after them.
    /// * Cap triangles reference **existing** vertex-pool indices only (a
    ///   boundary loop is made of pool vertices the surface already owns),
    ///   so no new vertices are introduced and every attribute buffer
    ///   (`normals`, `tangents`, `uvs`, `colors`, `joints`, `weights`,
    ///   morph `targets`) stays index-aligned and is carried over
    ///   unchanged. Re-run [`Primitive::compute_normals`] afterwards if a
    ///   smooth normal over the new faces is wanted — the caps inherit the
    ///   stored per-vertex normals, which were authored for the open rim.
    /// * Each cap triangle is wound so it **crosses every boundary edge in
    ///   the direction opposite** to the loop's traversal. Because a
    ///   boundary half-edge keeps the orientation of the single triangle
    ///   that owns it (glTF 2.0 §3.7.2.1: CCW = front-facing), crossing it
    ///   the other way makes each filled interior edge traversed once each
    ///   way — the same manifold-consistency condition
    ///   [`Primitive::orient_consistent`] enforces — so the patch's
    ///   front-face normal agrees with the surrounding surface and
    ///   `signed_volume`'s sign is preserved.
    ///
    /// # Triangulation
    ///
    /// A boundary loop is generally **not planar** (it bounds a hole in a
    /// curved surface), so it is first projected onto its best-fit plane.
    /// The plane normal is the area vector
    /// `N = ½ Σ (Pᵢ × Pᵢ₊₁)` (Newell's method — robust for a non-planar
    /// polygon, reducing to the exact face normal for a planar one); the
    /// loop is then expressed in an orthonormal in-plane basis `(u, v)`
    /// with `u × v = N̂` and ear-clipped in those 2D coordinates by the
    /// two-ears theorem (Meisters, "Polygons Have Ears", *American
    /// Mathematical Monthly* 82(6), 1975). The clip emits `k − 2`
    /// triangles for a `k`-vertex loop, indexing back through the loop's
    /// original pool indices, then each emitted triangle is reversed to
    /// satisfy the winding rule above. Reflex-vertex containment uses the
    /// standard barycentric-sign point-in-triangle test; a degenerate
    /// (zero-area) corner is dropped without emitting, and a fully
    /// non-simple projected loop still terminates (its patch is then
    /// best-effort, matching the [`extrude`](crate::extrude) cap's
    /// documented unspecified-on-non-simple contract).
    ///
    /// # Contract
    ///
    /// * Loops with fewer than three distinct projected vertices, a
    ///   non-finite Newell normal, or a zero-length area vector (every
    ///   loop vertex collinear) are skipped — they bound no fillable area.
    /// * Topology feed matches [`Primitive::boundary_loops`]: `Triangles`
    ///   / `TriangleStrip` / `TriangleFan` all feed in; a non-triangle
    ///   topology (lines/points) or a closed two-manifold (no boundary
    ///   loops) returns the de-stripped surface unchanged (no caps added).
    /// * Pure (does not mutate `self`). Cost is
    ///   `O(triangle_count + Σ kᵢ²)` over the loop lengths `kᵢ` (the
    ///   ear-clip's reflex scan), plus the [`Primitive::boundary_loops`]
    ///   walk it calls.
    pub fn fill_holes(&self) -> Primitive {
        let mut out = self.to_triangle_list();
        let loops = self.boundary_loops();
        if loops.is_empty() {
            return out;
        }
        let n = self.positions.len();

        // Accumulate cap triangles, then append to the (already U32) index
        // buffer carried by `out`.
        let mut caps: Vec<[u32; 3]> = Vec::new();
        for lp in &loops {
            // Gather the loop's 3D positions; bail on any out-of-range or
            // non-finite vertex (a malformed pool can't be projected).
            if lp.len() < 3 {
                continue;
            }
            let mut pts3: Vec<[f64; 3]> = Vec::with_capacity(lp.len());
            let mut ok = true;
            for &vi in lp {
                let i = vi as usize;
                if i >= n {
                    ok = false;
                    break;
                }
                let p = self.positions[i];
                if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                    ok = false;
                    break;
                }
                pts3.push([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
            }
            if !ok {
                continue;
            }
            if let Some(tris) = triangulate_loop_3d(&pts3, lp) {
                // Reverse each triangle's winding so it crosses every
                // boundary edge opposite to the loop traversal.
                for [a, b, c] in tris {
                    caps.push([a, c, b]);
                }
            }
        }

        if caps.is_empty() {
            return out;
        }
        if let Some(Indices::U32(idx)) = &mut out.indices {
            for t in caps {
                idx.extend_from_slice(&t);
            }
        }
        out
    }

    /// Summarise the combinatorial topology of this primitive's triangle
    /// tessellation: the vertex / edge / face counts, the
    /// **Euler characteristic** `χ = V − E + F`, the number of connected
    /// surface components, the number of boundary loops, and — for a
    /// closed orientable two-manifold — the **genus** (handle count).
    ///
    /// This is the aggregate topological-invariant readout the
    /// [`Primitive::boundary_loops`] docs gesture toward with "genus /
    /// hole-count diagnostics". Where [`Primitive::edge_manifold_report`]
    /// classifies edges and [`Primitive::boundary_loops`] traces open
    /// seams, this method rolls the whole connectivity graph up into the
    /// classical [`TopologySummary`] a mesh-repair or analysis pass keys
    /// off.
    ///
    /// # Counting conventions
    ///
    /// All three counts are over the *combinatorial* surface — by
    /// **vertex index**, not 3D position (run [`Primitive::weld_vertices`]
    /// first to merge positionally coincident corners):
    ///
    /// * **F** ([`TopologySummary::face_count`]) — the number of valid
    ///   triangles. A triangle with an out-of-range corner index or a
    ///   duplicate corner index (a zero-length edge) is **excluded whole**
    ///   — the same exclusion rule as
    ///   [`Primitive::edge_manifold_report`] / [`Primitive::boundary_edges`].
    /// * **E** ([`TopologySummary::edge_count`]) — the number of distinct
    ///   undirected edges `(min, max)` across the valid triangles, bucketed
    ///   identically to `edge_manifold_report` (this equals its
    ///   `total_edge_count`).
    /// * **V** ([`TopologySummary::vertex_count`]) — the number of distinct
    ///   vertex indices **referenced by a valid triangle**, *not*
    ///   `positions.len()`. Unreferenced pool slots (a decoder's slack, a
    ///   point cloud's stray points) do not inflate `V`, so an isolated
    ///   triangle reports `V = 3` regardless of pool size and the Euler
    ///   identity stays meaningful.
    ///
    /// # Euler characteristic and genus
    ///
    /// `χ = V − E + F` ([`TopologySummary::euler_characteristic`]) is the
    /// classical topological invariant (Euler's polyhedron formula,
    /// generalised to surfaces). For a disjoint union of `C` closed
    /// orientable surfaces of genera `g_1 … g_C` it equals
    /// `Σ (2 − 2·g_k)`; with `b` total boundary loops removed it is
    /// `2·C − 2·g_total − b`. When the primitive is a **single connected
    /// closed orientable two-manifold** (`component_count == 1`,
    /// `is_closed_manifold()`), this method inverts that to recover the
    /// genus `g = (2 − χ) / 2` ([`TopologySummary::genus`] = `Some(g)`):
    /// a sphere/cube/convex hull is `g = 0`, a torus/coffee-mug is
    /// `g = 1`, a double-torus is `g = 2`. The genus is **only** reported
    /// when that single-closed-component precondition holds *and*
    /// `2 − χ` is even and non-negative; otherwise it is `None` (an open
    /// patch, a non-manifold or self-touching surface, a multi-component
    /// mesh, or a non-orientable / defective surface whose `χ` does not
    /// admit an orientable-genus reading).
    ///
    /// # Connected components and boundary loops
    ///
    /// [`TopologySummary::component_count`] is the number of
    /// edge-connected triangle groups (two triangles are connected when
    /// they share an undirected edge — the standard "facet adjacency"
    /// relation, computed with a union-find over the edge buckets). A
    /// vertex touched by two otherwise-separate fans is **not** enough to
    /// join them (edge adjacency, not vertex adjacency), matching the
    /// two-manifold seam definition the rest of the crate uses. An empty
    /// or all-invalid primitive reports `0` components.
    /// [`TopologySummary::boundary_loop_count`] is
    /// `self.boundary_loops().len()` — the number of distinct open seams.
    ///
    /// # Contract
    ///
    /// * Topology feed and exclusion rules are identical to
    ///   [`Primitive::edge_manifold_report`] — `Triangles` /
    ///   `TriangleStrip` (alternating winding) / `TriangleFan` all feed in
    ///   through [`Primitive::triangle_indices`]. Non-triangle topologies
    ///   (lines/points) and empty primitives return an all-zero summary
    ///   with `genus == None`.
    /// * Pure (no `self` mutation). Cost
    ///   `O(triangle_count · α(V))` for the union-find pass plus the
    ///   `boundary_loops` walk it calls; `α` is the inverse-Ackermann
    ///   function (effectively constant).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A single open triangle: V=3, E=3, F=1, χ=1, one component,
    /// // one boundary loop, no closed genus.
    /// let mut prim = Primitive::new(Topology::Triangles);
    /// prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    /// let t = prim.topology_summary();
    /// assert_eq!(
    ///     (t.vertex_count, t.edge_count, t.face_count), (3, 3, 1));
    /// assert_eq!(t.euler_characteristic, 1);
    /// assert_eq!(t.component_count, 1);
    /// assert_eq!(t.boundary_loop_count, 1);
    /// assert_eq!(t.genus, None);
    /// ```
    pub fn topology_summary(&self) -> TopologySummary {
        let n = self.positions.len();
        // Gather the valid triangles once (same exclusion rules as
        // `edge_manifold_report`): in-range corners, no duplicate corner.
        let mut faces: Vec<[u32; 3]> = Vec::new();
        for [ia, ib, ic] in self.triangle_indices() {
            if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                continue;
            }
            if ia == ib || ib == ic || ia == ic {
                continue;
            }
            faces.push([ia, ib, ic]);
        }

        if faces.is_empty() {
            return TopologySummary::default();
        }

        // Distinct referenced vertices (V) — by index, referenced-only.
        // Distinct undirected edges (E), with the owning face indices so
        // facet adjacency can be derived without a second pass.
        let mut verts: HashSet<u32> = HashSet::new();
        // Undirected edge → the (up to two) face indices we union across.
        // A third+ sharer (non-manifold edge) still unions transitively;
        // we only need one representative per edge to chain the component.
        let mut edge_first_face: HashMap<(u32, u32), usize> = HashMap::new();
        let mut edge_set: HashSet<(u32, u32)> = HashSet::new();

        // Union-find over face indices for connected components.
        let mut parent: Vec<usize> = (0..faces.len()).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path halving
                x = parent[x];
            }
            x
        }

        for (fi, &[ia, ib, ic]) in faces.iter().enumerate() {
            verts.insert(ia);
            verts.insert(ib);
            verts.insert(ic);
            for (a, b) in [(ia, ib), (ib, ic), (ic, ia)] {
                let key = if a < b { (a, b) } else { (b, a) };
                edge_set.insert(key);
                match edge_first_face.get(&key) {
                    Some(&other) => {
                        // Share an edge → union the two facets.
                        let ra = find(&mut parent, fi);
                        let rb = find(&mut parent, other);
                        if ra != rb {
                            parent[ra] = rb;
                        }
                    }
                    None => {
                        edge_first_face.insert(key, fi);
                    }
                }
            }
        }

        // Count distinct roots = connected components.
        let mut roots: HashSet<usize> = HashSet::new();
        for fi in 0..faces.len() {
            let r = find(&mut parent, fi);
            roots.insert(r);
        }

        let vertex_count = verts.len();
        let edge_count = edge_set.len();
        let face_count = faces.len();
        let component_count = roots.len();
        let euler_characteristic = vertex_count as i64 - edge_count as i64 + face_count as i64;

        let boundary_loop_count = self.boundary_loops().len();

        // Genus only for a single connected closed orientable manifold.
        // χ = 2 − 2g ⇒ g = (2 − χ)/2, requiring (2 − χ) even and ≥ 0.
        let genus = if component_count == 1 && self.edge_manifold_report().is_closed_manifold() {
            let two_minus_chi = 2 - euler_characteristic;
            if two_minus_chi >= 0 && two_minus_chi % 2 == 0 {
                Some((two_minus_chi / 2) as u32)
            } else {
                None
            }
        } else {
            None
        };

        TopologySummary {
            vertex_count,
            edge_count,
            face_count,
            euler_characteristic,
            component_count,
            boundary_loop_count,
            genus,
        }
    }

    /// Build the **face-dual adjacency graph** of this primitive's
    /// triangle tessellation: for every triangle, the (up to three)
    /// triangles that share one of its edges.
    ///
    /// The result is indexed by the same enumeration
    /// [`Primitive::triangle_indices`] produces — entry `i` describes the
    /// neighbours of triangle `i`. Each entry is `[n01, n12, n20]`, where
    /// `n01` is the index of the triangle sharing this triangle's first
    /// edge (corners `0→1`), `n12` the one across edge `1→2`, and `n20`
    /// the one across edge `2→0`. A slot is `None` when that edge has no
    /// single well-defined neighbour:
    ///
    /// * a **boundary edge** (used by exactly one triangle — this one)
    ///   has no neighbour, the same edge
    ///   [`Primitive::boundary_edges`] reports;
    /// * a **non-manifold edge** (used by three or more triangles —
    ///   the `≥ 3` bucket of [`EdgeManifoldReport`]) has no *single*
    ///   neighbour, so every triangle on it gets `None` on that side
    ///   rather than an arbitrary pick.
    ///
    /// Only a clean **manifold-interior edge** (used by exactly two
    /// triangles) yields a `Some(other)` link, and the relation is then
    /// symmetric: if triangle `i`'s edge points to `j`, one of `j`'s
    /// slots points back to `i`. The number of `Some` links across the
    /// whole result is therefore `2 ·
    /// EdgeManifoldReport::manifold_interior_edge_count`.
    ///
    /// This is the explicit form of the facet-adjacency graph that
    /// [`Primitive::topology_summary`] walks implicitly with its
    /// union-find component pass — exposed here so a caller can traverse
    /// the dual graph directly. The edge two triangles share is the same
    /// shared-edge relation the STL "vertex-to-vertex rule" rests on
    /// (each triangle of a closed solid shares an edge — two vertices —
    /// with each adjacent triangle).
    ///
    /// # Use cases
    ///
    /// * **Winding / normal-consistency repair** — flood-fill across the
    ///   dual graph, flipping any neighbour whose shared edge is
    ///   traversed in the same direction by both triangles (a winding
    ///   disagreement), so a soup of inconsistently-wound facets becomes
    ///   coherently oriented.
    /// * **Region growing / mesh segmentation** — grow connected patches
    ///   by hopping `Some` links while a per-edge predicate (dihedral
    ///   angle below a crease threshold, same material) holds.
    /// * **Triangle-strip generation** — walk a chain of edge-adjacent
    ///   triangles to emit a long strip from a list.
    /// * **Connected-component labelling** — a breadth-first walk over
    ///   the `Some` links partitions the faces into the same components
    ///   [`TopologySummary::component_count`] reports.
    ///
    /// # Contract
    ///
    /// * Adjacency is by **vertex index**, not 3D position — run
    ///   [`Primitive::weld_vertices`] first if positionally coincident
    ///   corners on different indices should be treated as the same
    ///   vertex (otherwise a welded-shut seam reads as two boundary
    ///   edges with no link across it).
    /// * Edge bucketing and the out-of-range / duplicate-corner
    ///   whole-triangle exclusion are identical to
    ///   [`Primitive::edge_manifold_report`] /
    ///   [`Primitive::topology_summary`]. A triangle that is excluded
    ///   keeps its slot in the output (so indices line up with
    ///   `triangle_indices`) but every slot is `None`, and its edges
    ///   never count toward any other triangle's neighbour total.
    /// * Topology integration goes through
    ///   [`Primitive::triangle_indices`], so `Triangles` /
    ///   `TriangleStrip` (alternating winding) / `TriangleFan` all feed
    ///   in. Non-triangle topologies (lines/points) and empty primitives
    ///   return an empty `Vec`.
    /// * The output length always equals
    ///   `self.triangle_indices().len()` (= [`Primitive::triangle_count`]
    ///   for triangle topologies). Deterministic: the result depends only
    ///   on the index data, not on `HashMap` walk order. Pure (no `self`
    ///   mutation); cost `O(triangle_count)`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Two triangles sharing the diagonal edge 1→2 of a quad.
    /// let mut prim = Primitive::new(Topology::Triangles);
    /// prim.positions =
    ///     vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
    /// prim.indices = Some(Indices::U32(vec![0, 1, 2, 2, 1, 3]));
    /// let adj = prim.triangle_adjacency();
    /// // Triangle 0's edge 1→2 (slot index 1) is shared with triangle 1.
    /// assert_eq!(adj[0][1], Some(1));
    /// // The other two edges of triangle 0 are boundary.
    /// assert_eq!(adj[0][0], None);
    /// assert_eq!(adj[0][2], None);
    /// ```
    pub fn triangle_adjacency(&self) -> Vec<[Option<u32>; 3]> {
        let n = self.positions.len();
        let tris = self.triangle_indices();

        // First pass: bucket every valid triangle's undirected edges to
        // the faces that own them. The same exclusion rules as
        // `edge_manifold_report` / `topology_summary`: out-of-range
        // corners and duplicate-corner (zero-length-edge) triangles are
        // dropped whole so their bogus edges don't pollute neighbour
        // counts. A `Vec` per edge captures non-manifold (≥ 3) sharers,
        // which resolve to `None` rather than an arbitrary pick.
        let mut edge_faces: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        let mut valid: Vec<bool> = vec![false; tris.len()];
        for (fi, &[ia, ib, ic]) in tris.iter().enumerate() {
            if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                continue;
            }
            if ia == ib || ib == ic || ia == ic {
                continue;
            }
            valid[fi] = true;
            for (a, b) in [(ia, ib), (ib, ic), (ic, ia)] {
                let key = if a < b { (a, b) } else { (b, a) };
                edge_faces.entry(key).or_default().push(fi as u32);
            }
        }

        // Second pass: for each valid triangle, look up each of its three
        // edges. A `Some(other)` link exists only when the edge is shared
        // by exactly two faces (this one + one neighbour).
        let mut out: Vec<[Option<u32>; 3]> = vec![[None; 3]; tris.len()];
        for (fi, &[ia, ib, ic]) in tris.iter().enumerate() {
            if !valid[fi] {
                continue;
            }
            for (slot, (a, b)) in [(ia, ib), (ib, ic), (ic, ia)].into_iter().enumerate() {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(faces) = edge_faces.get(&key) {
                    if faces.len() == 2 {
                        // Exactly two sharers — the neighbour is the one
                        // that is not this face.
                        let neighbour = if faces[0] == fi as u32 {
                            faces[1]
                        } else {
                            faces[0]
                        };
                        out[fi][slot] = Some(neighbour);
                    }
                    // len == 1 (boundary) or len ≥ 3 (non-manifold):
                    // no single well-defined neighbour → leave None.
                }
            }
        }

        out
    }

    /// Propagate a **consistent triangle winding** across each
    /// edge-connected component and return the re-oriented triangle list
    /// together with an [`OrientationReport`].
    ///
    /// A vertex soup assembled from independently-authored facets (binary
    /// STL stores three loose vertices per facet; an OBJ stitched from
    /// several `g`-groups; a boolean/CSG result) frequently carries
    /// **mixed winding**: some triangles list their corners
    /// counter-clockwise (front-facing, per glTF 2.0 §3.7.2.1 — a
    /// positive-determinant transform makes the CCW triangle the front
    /// face) and some clockwise, so per-face normals point inconsistently
    /// in and out and back-face culling / two-sided lighting break. This
    /// flood-fills the face-dual adjacency graph (see
    /// [`Primitive::triangle_adjacency`]) and flips whichever neighbour
    /// disagrees, so within one edge-connected component every triangle
    /// ends up wound the same way **relative to the component's seed**.
    ///
    /// # The consistency rule
    ///
    /// Two triangles that share an undirected edge are *consistently*
    /// wound iff they traverse that shared edge in **opposite**
    /// directions — triangle A walking `u → v` and triangle B walking
    /// `v → u`. (Each interior edge of a coherently-oriented manifold is
    /// crossed once in each direction by its two faces.) When both
    /// traverse it the **same** way (`u → v` and `u → v`) the neighbour's
    /// winding disagrees and is flipped by swapping its last two corners
    /// (`[a, b, c] → [a, c, b]`), which reverses the per-face normal.
    ///
    /// # Seeding and the global flip ambiguity
    ///
    /// Winding consistency is only defined **relative to a reference**:
    /// flipping *every* triangle of a closed surface inside-out is still
    /// internally consistent. This routine fixes the reference per
    /// component to the **lowest-indexed valid triangle**, whose winding
    /// is kept verbatim; the rest of that component is brought into
    /// agreement with it. It does **not** attempt to decide which global
    /// orientation is "outward" (that needs the signed volume — see
    /// [`Primitive::signed_volume`], whose sign flips with the winding —
    /// or a known camera/seed face). Each connected component is seeded
    /// independently, so a multi-shell mesh's shells are each
    /// self-consistent but not necessarily co-oriented with one another.
    ///
    /// # Output
    ///
    /// `(faces, report)` where `faces` is parallel to
    /// [`Primitive::triangle_indices`] — same length, same order — with
    /// each disagreeing triangle's corners reordered. The
    /// [`OrientationReport`] carries the flip count, the component count,
    /// and a `non_orientable` flag set when a component contains a
    /// contradiction the flood-fill cannot satisfy (a Möbius-style
    /// closed loop of faces that forces a triangle into both windings).
    /// In that case the first assignment along the walk is kept and the
    /// flag warns the caller the result is a best effort.
    ///
    /// # Contract
    ///
    /// * Adjacency is by **vertex index** — run
    ///   [`Primitive::weld_vertices`] first so a positionally-coincident
    ///   seam links across (an unwelded crack reads as two boundary edges
    ///   and the two sides orient independently).
    /// * Edge bucketing and the out-of-range / duplicate-corner
    ///   whole-triangle exclusion match
    ///   [`Primitive::triangle_adjacency`]: an excluded triangle keeps its
    ///   slot in `faces` **verbatim** (never flipped, never linked) so the
    ///   output lines up with `triangle_indices`.
    /// * Only clean **manifold-interior** edges (exactly two sharers)
    ///   carry an orientation constraint. A boundary edge (one face) has
    ///   no neighbour to agree with; a non-manifold edge (≥ 3 faces) has
    ///   no single well-defined neighbour, so it is left unconstrained —
    ///   the components it would have joined orient independently, exactly
    ///   as [`Primitive::triangle_adjacency`] reports `None` for it.
    /// * `Triangles` / `TriangleStrip` (alternating winding already
    ///   resolved) / `TriangleFan` all feed in through
    ///   `triangle_indices`. Non-triangle topologies and empty primitives
    ///   return `(vec![], OrientationReport::default())`.
    /// * Deterministic (lowest-index seeds, sorted neighbour walk; does
    ///   not depend on `HashMap` order) and pure (no `self` mutation).
    ///   Cost `O(triangle_count · α)` over the union of edges.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Two triangles of a quad; the second is wound the wrong way so
    /// // it shares edge 1→2 in the SAME direction as the first.
    /// let mut prim = Primitive::new(Topology::Triangles);
    /// prim.positions =
    ///     vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
    /// prim.indices = Some(Indices::U32(vec![0, 1, 2, /*bad:*/ 1, 2, 3]));
    /// let (faces, report) = prim.orient_consistent();
    /// assert_eq!(report.flipped_count, 1);
    /// assert_eq!(report.component_count, 1);
    /// assert!(!report.non_orientable);
    /// // The neighbour was flipped to share the edge the opposite way.
    /// assert_eq!(faces[1], [1, 3, 2]);
    /// ```
    pub fn orient_consistent(&self) -> (Vec<[u32; 3]>, OrientationReport) {
        let n = self.positions.len();
        let tris = self.triangle_indices();

        // Identify the valid triangles (same exclusion rules as
        // `triangle_adjacency`). Invalid triangles keep their slot in the
        // output verbatim and never participate in orientation.
        let mut valid: Vec<bool> = vec![false; tris.len()];
        for (fi, &[ia, ib, ic]) in tris.iter().enumerate() {
            if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                continue;
            }
            if ia == ib || ib == ic || ia == ic {
                continue;
            }
            valid[fi] = true;
        }

        // Bucket every valid triangle's undirected edges to its sharers.
        let mut edge_faces: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (fi, &[ia, ib, ic]) in tris.iter().enumerate() {
            if !valid[fi] {
                continue;
            }
            for (a, b) in [(ia, ib), (ib, ic), (ic, ia)] {
                let key = if a < b { (a, b) } else { (b, a) };
                edge_faces.entry(key).or_default().push(fi as u32);
            }
        }

        // Output starts as the verbatim triangle list; flips rewrite
        // individual entries below. `flip` tracks the current decision
        // per triangle so neighbour comparisons use the *oriented* edge
        // direction, not the raw input.
        let mut faces = tris.clone();
        let mut flip: Vec<bool> = vec![false; tris.len()];
        let mut visited: Vec<bool> = vec![false; tris.len()];
        let mut flipped_count = 0usize;
        let mut component_count = 0usize;
        let mut non_orientable = false;

        // Directed corner pair `(a, b)` for slot index, honouring the
        // triangle's current flip state. A flipped `[a, b, c]` reads as
        // `[a, c, b]`, so its directed edges become a→c, c→b, b→a.
        let oriented = |tri: [u32; 3], flipped: bool| -> [(u32, u32); 3] {
            let [a, b, c] = tri;
            if flipped {
                [(a, c), (c, b), (b, a)]
            } else {
                [(a, b), (b, c), (c, a)]
            }
        };

        // Flood-fill each edge-connected component from its lowest-index
        // seed (deterministic). The seed keeps its input winding.
        for seed in 0..tris.len() {
            if !valid[seed] || visited[seed] {
                continue;
            }
            component_count += 1;
            visited[seed] = true;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(seed);
            while let Some(fi) = queue.pop_front() {
                let cur_edges = oriented(tris[fi], flip[fi]);
                for &(a, b) in cur_edges.iter() {
                    let key = if a < b { (a, b) } else { (b, a) };
                    let Some(sharers) = edge_faces.get(&key) else {
                        continue;
                    };
                    // Only a clean manifold-interior edge constrains
                    // orientation; boundary / non-manifold edges do not.
                    if sharers.len() != 2 {
                        continue;
                    }
                    let other = if sharers[0] == fi as u32 {
                        sharers[1]
                    } else {
                        sharers[0]
                    } as usize;
                    // Does the neighbour, at its current flip state,
                    // already traverse this edge the opposite way?
                    let nb_edges = oriented(tris[other], flip[other]);
                    let nb_same_dir = nb_edges.iter().any(|&(na, nb)| na == a && nb == b);
                    let nb_opp_dir = nb_edges.iter().any(|&(na, nb)| na == b && nb == a);
                    // Consistent ⇔ opposite traversal. If the neighbour
                    // shares the edge in the SAME direction it must flip.
                    let want_flip = nb_same_dir && !nb_opp_dir;
                    if !visited[other] {
                        visited[other] = true;
                        if want_flip {
                            flip[other] = true;
                            faces[other] = {
                                let [oa, ob, oc] = tris[other];
                                [oa, oc, ob]
                            };
                            flipped_count += 1;
                        }
                        queue.push_back(other);
                    } else {
                        // Already decided. If it now disagrees with this
                        // face, the component cannot be coherently
                        // oriented (a non-orientable loop): keep the
                        // earlier decision and flag it.
                        if want_flip {
                            non_orientable = true;
                        }
                    }
                }
            }
        }

        (
            faces,
            OrientationReport {
                flipped_count,
                component_count,
                non_orientable,
            },
        )
    }

    /// Closest-hit ray query against this primitive's triangle
    /// tessellation.
    ///
    /// Walks every triangle returned by [`Primitive::triangle_indices`]
    /// (so `Triangles` / `TriangleStrip` / `TriangleFan` all feed in,
    /// with the strip alternating-winding rule honoured) and runs the
    /// Möller-Trumbore ray-triangle intersection (see
    /// [`crate::ray::intersect_triangle`]). Returns the [`RayHit`]
    /// with the smallest `t ≥ 0` (closest along the ray), or `None`
    /// when nothing within `t_max` is struck.
    ///
    /// The returned `triangle_index` indexes the
    /// `Vec<[u32; 3]>` from `triangle_indices()` — callers needing
    /// the per-vertex indices look them up there. `barycentric` is
    /// `[w, u, v]` with `w = 1 - u - v` so the hit point reconstructs
    /// as `w * P0 + u * P1 + v * P2`; `front_face` follows the
    /// CCW-from-outside convention used everywhere else in the crate.
    ///
    /// Out-of-range index entries, NaN-producing math, and degenerate
    /// (zero-area / ray-parallel-to-plane) faces are silently skipped
    /// — same robustness contract as
    /// [`Primitive::compute_normals`] / [`Primitive::surface_area`].
    /// A degenerate ray (`direction == [0, 0, 0]`) misses everything.
    /// Non-triangle topologies (lines/points) return `None`.
    ///
    /// This is the brute-force O(triangle_count) query — adequate for
    /// small primitives and as the inner loop of a BVH leaf. Spatial
    /// acceleration (BVH/kd-tree) is a separate higher-level concern
    /// the caller layers on top by calling
    /// [`crate::BoundingBox::intersect_ray`] for early-out and
    /// recursing into per-primitive `intersect_ray` only on the leaves
    /// whose AABB the ray actually enters.
    pub fn intersect_ray(&self, ray: crate::ray::Ray, t_max: f32) -> Option<crate::ray::RayHit> {
        let n = self.positions.len();
        let mut closest: Option<crate::ray::RayHit> = None;
        let mut best_t = t_max;
        for (tri_idx, [ia, ib, ic]) in self.triangle_indices().into_iter().enumerate() {
            if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                continue;
            }
            let p0 = self.positions[ia as usize];
            let p1 = self.positions[ib as usize];
            let p2 = self.positions[ic as usize];
            if let Some((t, u, v, front)) = crate::ray::intersect_triangle(ray, p0, p1, p2, best_t)
            {
                let w = 1.0 - u - v;
                closest = Some(crate::ray::RayHit {
                    t,
                    triangle_index: tri_idx,
                    barycentric: [w, u, v],
                    front_face: front,
                });
                best_t = t;
            }
        }
        closest
    }

    /// Build a [`crate::Bvh`] over this primitive's triangle
    /// tessellation.
    ///
    /// Convenience wrapper around [`crate::Bvh::build`]. Returns
    /// `None` when the primitive has no usable triangles
    /// (non-triangle topology, all-NaN positions, or all out-of-range
    /// index entries — see [`crate::Bvh::build`] for the robustness
    /// contract).
    ///
    /// Many-ray workloads against the same primitive should build the
    /// BVH once and call [`crate::Bvh::intersect_ray`] for every ray,
    /// turning the per-query cost from `O(triangle_count)` (the
    /// brute-force [`Primitive::intersect_ray`] path) to roughly
    /// `O(log triangle_count)`. The returned `Bvh` carries a copy of
    /// the permuted triangle indices; the source primitive is not
    /// mutated.
    pub fn build_bvh(&self) -> Option<crate::Bvh> {
        crate::Bvh::build(self)
    }

    /// Shadow-ray early-exit query: `true` if **any** triangle in this
    /// primitive is hit at a parameter `t ∈ (epsilon, t_max]`.
    ///
    /// A small `epsilon` (`1e-4`) is subtracted from the starting
    /// parameter to avoid self-shadowing artefacts when the ray origin
    /// is itself a surface hit. For a ray whose origin is genuinely
    /// inside or behind the geometry, prefer
    /// [`Primitive::intersect_ray`] and inspect the returned `t`.
    ///
    /// Stops on the first hit found — does **not** return the closest
    /// hit. Same out-of-range / degenerate-face skipping as
    /// [`Primitive::intersect_ray`]; non-triangle topologies return
    /// `false`.
    pub fn any_ray_intersection(&self, ray: crate::ray::Ray, t_max: f32) -> bool {
        let n = self.positions.len();
        for [ia, ib, ic] in self.triangle_indices() {
            if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                continue;
            }
            let p0 = self.positions[ia as usize];
            let p1 = self.positions[ib as usize];
            let p2 = self.positions[ic as usize];
            if crate::ray::intersect_triangle(ray, p0, p1, p2, t_max).is_some() {
                return true;
            }
        }
        false
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

/// Combinatorial-topology summary of a [`Primitive`]'s triangle
/// tessellation, produced by [`Primitive::topology_summary`].
///
/// Rolls the whole connectivity graph up into the classical topological
/// invariants: the vertex / edge / face counts, the **Euler
/// characteristic** `χ = V − E + F`, the connected-component count, the
/// boundary-loop count, and — for a single closed orientable
/// two-manifold — the [`genus`](TopologySummary::genus).
///
/// All counts are by **vertex index**, not 3D position, and exclude
/// triangles with an out-of-range or duplicate corner index (the same
/// exclusion rule as [`EdgeManifoldReport`]). `V` counts only the
/// vertices a valid triangle actually references — unreferenced pool
/// slots do not inflate it — so the Euler identity stays meaningful for
/// a primitive whose `positions` pool carries slack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TopologySummary {
    /// **V** — distinct vertex indices referenced by a valid triangle
    /// (not `positions.len()`).
    pub vertex_count: usize,
    /// **E** — distinct undirected edges across the valid triangles
    /// (equals [`EdgeManifoldReport::total_edge_count`]).
    pub edge_count: usize,
    /// **F** — number of valid triangles (out-of-range / duplicate-corner
    /// triangles excluded).
    pub face_count: usize,
    /// **χ = V − E + F** — the Euler characteristic. Signed because an
    /// open surface with many holes can drive it negative.
    pub euler_characteristic: i64,
    /// Number of edge-connected triangle groups (facet adjacency: two
    /// triangles connect when they share an undirected edge). `0` for an
    /// empty / all-invalid primitive.
    pub component_count: usize,
    /// Number of distinct boundary loops — equals
    /// `Primitive::boundary_loops().len()`. `0` for a closed surface.
    pub boundary_loop_count: usize,
    /// Handle count of the surface, **only** populated for a single
    /// connected closed orientable two-manifold (`component_count == 1`
    /// and [`EdgeManifoldReport::is_closed_manifold`]) whose
    /// `g = (2 − χ) / 2` is a non-negative integer: `Some(0)` for a
    /// sphere/cube, `Some(1)` for a torus, `Some(2)` for a double-torus.
    /// `None` for an open patch, a multi-component mesh, a
    /// non-manifold / self-touching surface, or any surface whose `χ`
    /// does not admit an orientable-genus reading.
    pub genus: Option<u32>,
}

/// Outcome of [`Primitive::orient_consistent`].
///
/// Summarises a winding-consistency flood-fill: how many triangles were
/// flipped to agree with their component's seed, how many edge-connected
/// components the walk discovered (each seeded and oriented
/// independently), and whether any component was found to be
/// **non-orientable** — a face loop that forces a triangle into both
/// windings, which the flood-fill cannot satisfy.
///
/// The companion `Vec<[u32; 3]>` returned alongside this carries the
/// actual re-oriented triangle indices. This struct is only the
/// scalar tally; it never retains the per-edge adjacency map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OrientationReport {
    /// Number of triangles whose corners were reordered (`[a, b, c] →
    /// [a, c, b]`) to bring their winding into agreement with the
    /// lowest-indexed seed of their edge-connected component. `0` when
    /// the input was already coherently wound (or had no constrained
    /// edges).
    pub flipped_count: usize,
    /// Number of edge-connected triangle components the flood-fill
    /// visited — matches [`TopologySummary::component_count`] for the
    /// same primitive (boundary / non-manifold edges split components
    /// identically). `0` for an empty / all-invalid primitive.
    pub component_count: usize,
    /// `true` iff at least one component could **not** be coherently
    /// oriented: walking the face-dual graph reached an
    /// already-decided triangle that the current face's edge contradicts
    /// (a Möbius-style loop). The earlier decision along the walk is
    /// kept, so the returned faces are a best effort rather than a
    /// guaranteed-consistent orientation.
    pub non_orientable: bool,
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

    /// Area-weighted surface centroid across every contained primitive
    /// — the area-weighted combination of every primitive's own
    /// [`Primitive::surface_centroid`].
    ///
    /// # Derivation
    ///
    /// The continuous identity
    /// `C = (Σ area_i · centroid_i) / Σ area_i` from
    /// [`Primitive::surface_centroid`] generalises to a union of
    /// patches by additivity of the surface integral: integrating `x`
    /// over the union is the sum of the per-patch integrals
    /// (`area_i · centroid_i`), and integrating `1` is the sum of the
    /// per-patch areas. So the mesh-level centroid is the per-primitive
    /// centroid recombined with the per-primitive areas as weights —
    /// `Σ area_i · primitive_centroid_i / Σ area_i`. This matches
    /// [`Mesh::surface_area`]'s additive roll-up.
    ///
    /// Mesh-local — no transforms, skin pose, or morph deltas are
    /// applied. Primitives with `None` from
    /// [`Primitive::surface_centroid`] (no positive-area triangles)
    /// contribute nothing and don't pull the result toward
    /// `[0, 0, 0]`. Returns `None` when every contained primitive
    /// returns `None` (or the mesh holds zero primitives).
    pub fn surface_centroid(&self) -> Option<[f64; 3]> {
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_area = 0.0_f64;
        for p in &self.primitives {
            let area = p.surface_area();
            if area == 0.0 || !area.is_finite() {
                continue;
            }
            if let Some(c) = p.surface_centroid() {
                sum_x += c[0] * area;
                sum_y += c[1] * area;
                sum_z += c[2] * area;
                sum_area += area;
            }
        }
        if sum_area == 0.0 || !sum_area.is_finite() {
            return None;
        }
        let inv = 1.0 / sum_area;
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Transform-aware area-weighted surface centroid across every
    /// contained primitive: every primitive's
    /// [`Primitive::world_surface_centroid`] (numerator + denominator)
    /// is combined into a single mesh-level numerator + denominator
    /// before the final division.
    ///
    /// # Derivation
    ///
    /// The per-primitive helper supplies a closed-form ratio
    /// `(Σ area_world · centroid_world) / Σ area_world`. To recombine
    /// across primitives correctly, recover each primitive's numerator
    /// (centroid scaled by world surface area) and add it to a running
    /// total, then divide once at the end. Because
    /// [`Primitive::world_surface_centroid`] returns the post-divide
    /// ratio rather than the raw numerator, the mesh helper recovers
    /// each per-primitive area from [`Primitive::world_surface_area`]
    /// (a single fixed-cost extra triangle pass) and multiplies. The
    /// extra pass is intentional: keeping the per-primitive helper at
    /// its natural ratio shape gives callers a direct answer without
    /// teaching them about the recombination contract.
    ///
    /// # Contract
    ///
    /// * Skips primitives whose [`Primitive::world_surface_centroid`]
    ///   returns `None` or whose [`Primitive::world_surface_area`] is
    ///   `0.0` / non-finite. Returns `None` when every primitive
    ///   contributes nothing under `world`.
    /// * Mirrors [`Mesh::surface_centroid`]'s `f64` accumulator and
    ///   silent-skip policy for partly-corrupt buffers.
    /// * Pure; cost `O(Σ triangle_count_per_primitive)`.
    pub fn world_surface_centroid(&self, world: [[f32; 4]; 4]) -> Option<[f64; 3]> {
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_area = 0.0_f64;
        for p in &self.primitives {
            let area = p.world_surface_area(world);
            if area == 0.0 || !area.is_finite() {
                continue;
            }
            if let Some(c) = p.world_surface_centroid(world) {
                sum_x += c[0] * area;
                sum_y += c[1] * area;
                sum_z += c[2] * area;
                sum_area += area;
            }
        }
        if sum_area == 0.0 || !sum_area.is_finite() {
            return None;
        }
        let inv = 1.0 / sum_area;
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Sum of [`Primitive::signed_volume`] across every contained
    /// primitive (mesh-local, no transforms / skin pose / morph deltas
    /// applied). Non-triangle primitives contribute 0.0.
    ///
    /// **Only physically meaningful when every contained primitive is
    /// a closed two-manifold surface** (see
    /// [`Primitive::is_closed_manifold`]). A mesh that bundles, say, a
    /// closed cube with a non-closed UV strip will report the cube's
    /// signed volume plus an open-mesh leak from the strip; the leak
    /// is well-defined arithmetically but doesn't correspond to a
    /// physical volume. Sign follows the per-primitive convention
    /// (CCW-from-outside = positive).
    pub fn signed_volume(&self) -> f64 {
        self.primitives.iter().map(|p| p.signed_volume()).sum()
    }

    /// Unsigned `|signed_volume()|` aggregated over every contained
    /// primitive. **Note: this is `|Σ signed|`, not `Σ |signed|`** — two
    /// primitives whose signed volumes cancel will report a smaller
    /// magnitude than either one alone. For a typical single-shell mesh
    /// (every primitive part of one closed surface) the distinction
    /// doesn't matter; for a multi-shell mesh, prefer summing each
    /// primitive's [`Primitive::volume`] separately.
    pub fn volume(&self) -> f64 {
        self.signed_volume().abs()
    }

    /// Volume-weighted centroid (centre of mass) across every contained
    /// primitive — the signed-volume-weighted combination of every
    /// primitive's [`Primitive::volume_centroid`].
    ///
    /// # Derivation
    ///
    /// The continuous identity `C = ∫∫∫_V x dV / ∫∫∫_V dV` from
    /// [`Primitive::volume_centroid`] generalises to a union of solid
    /// bodies by additivity of the volume integral: integrating `x`
    /// over the union is the sum of the per-body integrals
    /// (`V_i · C_i`), and integrating `1` is the sum of the per-body
    /// signed volumes (`V_i`). So the mesh-level volume centroid is
    /// the per-primitive centroid recombined with the per-primitive
    /// **signed** volumes as weights — `Σ V_i · C_i / Σ V_i`. This
    /// matches [`Mesh::signed_volume`]'s additive roll-up; the signed
    /// weights cause an inside-out subshell to subtract correctly.
    ///
    /// # Contract
    ///
    /// * Mesh-local — no transforms, skin pose, or morph deltas are
    ///   applied.
    /// * Skips primitives whose [`Primitive::volume_centroid`] returns
    ///   `None` (non-triangle / zero-signed-volume) or whose
    ///   [`Primitive::signed_volume`] is non-finite. Returns `None`
    ///   when the accumulated signed volume is `0.0` or non-finite.
    /// * Mirrors [`Mesh::surface_centroid`]'s `f64` accumulator and
    ///   silent-skip policy for partly-corrupt buffers.
    /// * Pure; cost `O(Σ triangle_count_per_primitive)`.
    pub fn volume_centroid(&self) -> Option<[f64; 3]> {
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_v = 0.0_f64;
        for p in &self.primitives {
            let v = p.signed_volume();
            if v == 0.0 || !v.is_finite() {
                continue;
            }
            if let Some(c) = p.volume_centroid() {
                sum_x += c[0] * v;
                sum_y += c[1] * v;
                sum_z += c[2] * v;
                sum_v += v;
            }
        }
        if sum_v == 0.0 || !sum_v.is_finite() {
            return None;
        }
        let inv = 1.0 / sum_v;
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Transform-aware volume-weighted centroid (centre of mass) across
    /// every contained primitive: every primitive's
    /// [`Primitive::world_volume_centroid`] (post-divide centroid) is
    /// recombined with [`Primitive::world_signed_volume`] as the weight,
    /// then divided once at the end.
    ///
    /// # Derivation
    ///
    /// Same additivity argument as [`Mesh::volume_centroid`]: the
    /// continuous identity `C = ∫∫∫_V x dV / ∫∫∫_V dV` over a union of
    /// solid bodies splits into per-body integrals, so the mesh-level
    /// centroid is the per-primitive centroid recombined with the per-
    /// primitive **signed** volumes as weights. Because
    /// [`Primitive::world_volume_centroid`] returns the post-divide
    /// ratio rather than the raw numerator, the mesh helper recovers
    /// each per-primitive signed volume from
    /// [`Primitive::world_signed_volume`] (one fixed-cost extra
    /// triangle pass) and multiplies — same recombination shape as
    /// [`Mesh::world_surface_centroid`].
    ///
    /// # Contract
    ///
    /// * Skips primitives whose [`Primitive::world_volume_centroid`]
    ///   returns `None` or whose [`Primitive::world_signed_volume`] is
    ///   `0.0` / non-finite. Returns `None` when every primitive
    ///   contributes nothing under `world`.
    /// * Mirrors [`Mesh::volume_centroid`]'s `f64` accumulator and
    ///   silent-skip policy for partly-corrupt buffers.
    /// * Pure; cost `O(Σ triangle_count_per_primitive)`.
    pub fn world_volume_centroid(&self, world: [[f32; 4]; 4]) -> Option<[f64; 3]> {
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_v = 0.0_f64;
        for p in &self.primitives {
            let v = p.world_signed_volume(world);
            if v == 0.0 || !v.is_finite() {
                continue;
            }
            if let Some(c) = p.world_volume_centroid(world) {
                sum_x += c[0] * v;
                sum_y += c[1] * v;
                sum_z += c[2] * v;
                sum_v += v;
            }
        }
        if sum_v == 0.0 || !sum_v.is_finite() {
            return None;
        }
        let inv = 1.0 / sum_v;
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Unit-density inertia tensor of the solid enclosed by this mesh's
    /// closed triangle tessellation, taken about the **origin** of
    /// [`Primitive::positions`], summed across every contained
    /// primitive.
    ///
    /// # Derivation
    ///
    /// The continuous identity `I_αβ = ∫∫∫_V f_αβ(x, y, z) dV` (with the
    /// integrand the moment or product-of-inertia kernel from
    /// [`Primitive::inertia_tensor`]) is additive over a union of
    /// disjoint volumes: integrating any of the second-moment kernels
    /// over a union is the sum of the per-body integrals. So the
    /// mesh-level tensor is the element-wise sum of every primitive's
    /// own [`Primitive::inertia_tensor`] — the same additivity argument
    /// [`Mesh::signed_volume`] / [`Mesh::volume_centroid`] use to
    /// recombine per-primitive reductions. **Sign-aware**, just like
    /// [`Mesh::signed_volume`]: an inside-out subshell contributes a
    /// negated tensor, the same way it contributes a negative signed
    /// volume.
    ///
    /// # Contract
    ///
    /// * Mesh-local — no transforms, skin pose, or morph deltas are
    ///   applied.
    /// * Skips primitives whose [`Primitive::inertia_tensor`] returns
    ///   `None` (non-triangle topology, empty primitive, every-face-
    ///   degenerate). Returns `None` when **every** primitive returned
    ///   `None` (or the mesh holds zero primitives).
    /// * Mirrors [`Mesh::volume_centroid`]'s `f64` accumulator and
    ///   silent-skip policy for partly-corrupt buffers.
    /// * Symmetric matrix; the off-diagonal entries are populated
    ///   symmetrically (`I[0][1] == I[1][0]`, etc.). The diagonals are
    ///   `I_xx = ∫(y² + z²) dV`, `I_yy = ∫(x² + z²) dV`,
    ///   `I_zz = ∫(x² + y²) dV`. The off-diagonals carry the standard
    ///   minus sign (`I_xy = -∫ x·y dV`).
    /// * Pure; cost `O(Σ triangle_count_per_primitive)`.
    pub fn inertia_tensor(&self) -> Option<[[f64; 3]; 3]> {
        let mut total = [[0.0_f64; 3]; 3];
        let mut any = false;
        for p in &self.primitives {
            if let Some(t) = p.inertia_tensor() {
                for r in 0..3 {
                    for c in 0..3 {
                        total[r][c] += t[r][c];
                    }
                }
                any = true;
            }
        }
        if !any {
            None
        } else {
            Some(total)
        }
    }

    /// Transform-aware unit-density inertia tensor across every contained
    /// primitive: every primitive's [`Primitive::world_inertia_tensor`] is
    /// summed element-wise after each corner is mapped through the
    /// row-major column-vector affine 4x4 `world` matrix. Sibling of
    /// [`Mesh::inertia_tensor`] for the per-instance world-frame case.
    ///
    /// # Derivation
    ///
    /// Same additivity argument as [`Mesh::inertia_tensor`]: the
    /// second-moment integral `I_αβ = ∫∫∫_V f_αβ dV` is additive over a
    /// union of disjoint volumes, so the mesh-level world tensor is the
    /// element-wise sum of every primitive's own
    /// [`Primitive::world_inertia_tensor`]. Because the transform is
    /// folded in per primitive (every corner mapped through `world`
    /// before the integral), the sum is taken in the **world** frame and
    /// is sign-aware — an inside-out subshell contributes a negated
    /// tensor, the same way it contributes a negative
    /// [`Primitive::world_signed_volume`].
    ///
    /// # Contract
    ///
    /// * Skips primitives whose [`Primitive::world_inertia_tensor`]
    ///   returns `None` (non-triangle topology, empty primitive, every-
    ///   face-degenerate under `world`). Returns `None` when **every**
    ///   primitive returned `None` (or the mesh holds zero primitives).
    /// * Mirrors [`Mesh::world_volume_centroid`]'s `f64` accumulator and
    ///   silent-skip policy for partly-corrupt buffers.
    /// * Symmetric matrix; same diagonal-moment / negated-product-of-
    ///   inertia convention as [`Mesh::inertia_tensor`], in the world
    ///   frame.
    /// * Pure; cost `O(Σ triangle_count_per_primitive)`.
    pub fn world_inertia_tensor(&self, world: [[f32; 4]; 4]) -> Option<[[f64; 3]; 3]> {
        let mut total = [[0.0_f64; 3]; 3];
        let mut any = false;
        for p in &self.primitives {
            if let Some(t) = p.world_inertia_tensor(world) {
                for r in 0..3 {
                    for c in 0..3 {
                        total[r][c] += t[r][c];
                    }
                }
                any = true;
            }
        }
        if !any {
            None
        } else {
            Some(total)
        }
    }

    /// Closest-hit ray query across every contained primitive.
    ///
    /// Calls [`Primitive::intersect_ray`] on each primitive in turn,
    /// shrinking the search bound as hits land so each call only
    /// considers triangles in front of the current best `t`. Returns
    /// `(primitive_index, RayHit)` of the closest hit, or `None` when
    /// nothing within `t_max` is struck.
    ///
    /// Mesh-local space — parent node transforms, skin pose, and morph
    /// deltas are **not** applied. Transform the ray into mesh-local
    /// space by multiplying its origin + direction by the inverse of
    /// the node's world matrix before calling, or use this as the
    /// inner loop of a per-instance walk over
    /// [`crate::Scene3D::world_node_transforms`].
    pub fn intersect_ray(
        &self,
        ray: crate::ray::Ray,
        t_max: f32,
    ) -> Option<(usize, crate::ray::RayHit)> {
        let mut best: Option<(usize, crate::ray::RayHit)> = None;
        let mut best_t = t_max;
        for (idx, prim) in self.primitives.iter().enumerate() {
            if let Some(hit) = prim.intersect_ray(ray, best_t) {
                best_t = hit.t;
                best = Some((idx, hit));
            }
        }
        best
    }
}

/// Triangulate a single (possibly non-planar) 3D boundary loop into a fan
/// of triangles indexing back through `orig` (the loop's vertex-pool
/// indices, parallel to `pts3`).
///
/// The loop is projected onto its best-fit plane via Newell's area-vector
/// normal, expressed in an orthonormal in-plane basis, and ear-clipped by
/// the two-ears theorem. Returns `None` for fewer than three distinct
/// projected vertices, a non-finite / zero-length Newell normal, or a loop
/// whose projection collapses (every vertex collinear). The emitted
/// triangles wind counter-clockwise about the Newell normal in the
/// projected plane; the caller reverses them to satisfy its boundary-edge
/// crossing rule.
fn triangulate_loop_3d(pts3: &[[f64; 3]], orig: &[u32]) -> Option<Vec<[u32; 3]>> {
    let k = pts3.len();
    if k < 3 || orig.len() != k {
        return None;
    }

    // Newell's method: N = ½ Σ (Pᵢ × Pᵢ₊₁). Robust for a non-planar loop
    // and exact for a planar one.
    let mut nrm = [0.0f64; 3];
    for i in 0..k {
        let a = pts3[i];
        let b = pts3[(i + 1) % k];
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
    if !len.is_finite() || len <= 0.0 {
        return None;
    }
    let n_hat = [nrm[0] / len, nrm[1] / len, nrm[2] / len];

    // Build an orthonormal in-plane basis (u, v) with u × v = n_hat.
    // Pick the world axis least aligned with n_hat as the seed so the
    // cross product is well-conditioned.
    let seed = {
        let ax = n_hat[0].abs();
        let ay = n_hat[1].abs();
        let az = n_hat[2].abs();
        if ax <= ay && ax <= az {
            [1.0, 0.0, 0.0]
        } else if ay <= az {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        }
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let normalize = |a: [f64; 3]| -> Option<[f64; 3]> {
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        if l > 0.0 && l.is_finite() {
            Some([a[0] / l, a[1] / l, a[2] / l])
        } else {
            None
        }
    };
    let u = normalize(cross(seed, n_hat))?;
    // v = n_hat × u completes a right-handed (u, v, n_hat) frame, so a
    // CCW loop about n_hat reads CCW (positive shoelace) in (u, v).
    let v = cross(n_hat, u);

    // Project to 2D, dropping closing / consecutive duplicate points so the
    // ear clip sees a clean simple loop.
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut p2: Vec<[f64; 2]> = Vec::with_capacity(k);
    let mut keep: Vec<u32> = Vec::with_capacity(k);
    for i in 0..k {
        let q = [dot(pts3[i], u), dot(pts3[i], v)];
        if let Some(last) = p2.last() {
            if *last == q {
                continue;
            }
        }
        p2.push(q);
        keep.push(orig[i]);
    }
    while p2.len() > 1 && p2[0] == p2[p2.len() - 1] {
        p2.pop();
        keep.pop();
    }
    let m = p2.len();
    if m < 3 {
        return None;
    }

    // Orient the working loop counter-clockwise (positive shoelace) so the
    // ear test's convex/reflex sign convention holds.
    let area2 = {
        let mut s = 0.0;
        for i in 0..m {
            let a = p2[i];
            let b = p2[(i + 1) % m];
            s += a[0] * b[1] - b[0] * a[1];
        }
        s
    };
    if !area2.is_finite() || area2 == 0.0 {
        return None;
    }
    if area2 < 0.0 {
        p2.reverse();
        keep.reverse();
    }

    // Standard ear clip over a doubly-linked ring.
    let cross2 = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| -> f64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let in_tri = |p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]| -> bool {
        cross2(a, b, p) >= 0.0 && cross2(b, c, p) >= 0.0 && cross2(c, a, p) >= 0.0
    };
    let mut prev: Vec<usize> = (0..m).map(|i| (i + m - 1) % m).collect();
    let mut next: Vec<usize> = (0..m).map(|i| (i + 1) % m).collect();
    let mut alive = vec![true; m];
    let mut remaining = m;
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(m - 2);

    let is_ear = |i: usize, prev: &[usize], next: &[usize], alive: &[bool]| -> bool {
        let (a, b, c) = (p2[prev[i]], p2[i], p2[next[i]]);
        if cross2(a, b, c) <= 0.0 {
            return false; // reflex or degenerate
        }
        let mut j = next[next[i]];
        while j != prev[i] {
            if alive[j] && p2[j] != a && p2[j] != b && p2[j] != c && in_tri(p2[j], a, b, c) {
                return false;
            }
            j = next[j];
        }
        true
    };

    let mut cursor = 0usize;
    let mut guard = 0usize;
    let guard_max = m * m + 4;
    while remaining > 3 {
        guard += 1;
        if guard > guard_max {
            return None; // non-simple projection: bail rather than spin
        }
        // Find an ear, or fall back to a degenerate / forced clip.
        let mut found = None;
        let mut degenerate = None;
        let mut convex = None;
        let mut i = cursor;
        for _ in 0..remaining {
            let c2 = cross2(p2[prev[i]], p2[i], p2[next[i]]);
            if c2.abs() == 0.0 && degenerate.is_none() {
                degenerate = Some(i);
            }
            if c2 > 0.0 && convex.is_none() {
                convex = Some(i);
            }
            if is_ear(i, &prev, &next, &alive) {
                found = Some(i);
                break;
            }
            i = next[i];
        }
        let (clip_i, emit) = match (found, degenerate, convex) {
            (Some(i), _, _) => (i, true),
            (None, Some(d), _) => (d, false),
            (None, None, Some(c)) => (c, true),
            (None, None, None) => return None,
        };
        if emit {
            tris.push([keep[prev[clip_i]], keep[clip_i], keep[next[clip_i]]]);
        }
        let (p, nx) = (prev[clip_i], next[clip_i]);
        next[p] = nx;
        prev[nx] = p;
        alive[clip_i] = false;
        remaining -= 1;
        cursor = nx;
    }
    // Final triangle.
    let last = (0..m).find(|&k| alive[k])?;
    if cross2(p2[prev[last]], p2[last], p2[next[last]]) > 0.0 {
        tris.push([keep[prev[last]], keep[last], keep[next[last]]]);
    }
    if tris.is_empty() {
        None
    } else {
        Some(tris)
    }
}
