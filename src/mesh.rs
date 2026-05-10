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

use crate::scene::MaterialId;

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
/// **Round-7 candidate:** mark `#[non_exhaustive]` once every
/// downstream caller (the format crates in particular) has migrated
/// off literal `Primitive { … }` construction onto
/// [`Primitive::new`] + per-field assignment. Tracked in MEMORY
/// alongside the same deferral on `oxideav_core::Group`.
#[derive(Clone, Debug)]
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
}

/// A named bag of [`Primitive`]s sharing nothing but a name.
///
/// Most authoring tools split a logical "object" into one primitive
/// per material so the renderer can issue one draw call per
/// primitive without rebinding state.
///
/// **Round-7 candidate:** mark `#[non_exhaustive]` once every
/// downstream caller has migrated off literal `Mesh { … }`
/// construction onto [`Mesh::new`] + builders + struct-update
/// `..Mesh::default()` syntax. See the [`Primitive`] doc-note for
/// the rationale.
#[derive(Clone, Debug, Default)]
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
}
