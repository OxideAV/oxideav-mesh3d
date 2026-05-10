//! Scene-graph root, nodes, transforms, ID newtypes, and coordinate
//! metadata.
//!
//! The container is [`Scene3D`]. Every collection it owns
//! ([`Node`], [`Mesh`](crate::Mesh), [`Material`](crate::Material),
//! ...) is addressed by an `IdT(u32)` newtype that indexes into the
//! corresponding `Vec`. This keeps the model arena-friendly — clones
//! are cheap, identity is comparable, and serde round-tripping works
//! without back-references — while still letting decoders bulk-load
//! every mesh first and then point nodes at them.
//!
//! Coordinate convention defaults to **glTF 2.0**: right-handed,
//! Y-up, -Z forward, metres. Format crates that consume Z-up content
//! (STL, OBJ Wavefront) set [`Scene3D::up_axis`] to [`Axis::PosZ`]
//! and leave geometry untouched — the orientation metadata is
//! authoritative, no implicit rotation is applied.

use std::collections::HashMap;

use crate::{
    animation::Animation,
    audio::{AudioEmitter, AudioEmitterId, AudioSource, AudioSourceId},
    camera::Camera,
    light::Light,
    material::Material,
    mesh::Mesh,
    skin::Skeleton,
    skin::Skin,
    texture::Texture,
};

macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
    };
}

id_newtype!(
    /// Index into [`Scene3D::nodes`].
    NodeId
);
id_newtype!(
    /// Index into [`Scene3D::meshes`].
    MeshId
);
id_newtype!(
    /// Index into [`Scene3D::materials`].
    MaterialId
);
id_newtype!(
    /// Index into [`Scene3D::textures`].
    TextureId
);
id_newtype!(
    /// Index into [`Scene3D::skeletons`].
    SkeletonId
);
id_newtype!(
    /// Index into [`Scene3D::skins`].
    SkinId
);
id_newtype!(
    /// Index into [`Scene3D::cameras`].
    CameraId
);
id_newtype!(
    /// Index into [`Scene3D::lights`].
    LightId
);

/// Coordinate-system principal axis. Stored on [`Scene3D`] so a
/// renderer can apply (or skip) a global rotation when the file
/// convention disagrees with its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

/// Linear unit a single coordinate-space-1.0 represents in the file.
/// glTF defaults to metres; CAD/STL files often ship in millimetres
/// or inches. Renderers that mix scenes from different unit systems
/// scale by the ratio of [`Unit::to_metres`] values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Metres,
    Centimetres,
    Millimetres,
    Inches,
    Feet,
    Yards,
}

impl Unit {
    /// Multiplier from this unit to metres, e.g. `Inches.to_metres() == 0.0254`.
    pub fn to_metres(self) -> f32 {
        match self {
            Self::Metres => 1.0,
            Self::Centimetres => 0.01,
            Self::Millimetres => 0.001,
            Self::Inches => 0.0254,
            Self::Feet => 0.3048,
            Self::Yards => 0.9144,
        }
    }
}

/// Per-node local-to-parent transform. Decoders can store the raw
/// matrix as-is or decompose into translation/rotation/scale; the
/// [`Transform::to_matrix`] / [`Transform::from_matrix`] helpers
/// convert in either direction within float tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform {
    /// Row-major column-vector 4x4 transform — pre-multiplied
    /// (`out = M * v`). Layout matches glTF's `node.matrix` field.
    Matrix([[f32; 4]; 4]),
    /// Decomposed translation + rotation (xyzw quaternion) + scale.
    /// glTF's TRS form; preferred for animation since each channel is
    /// independent.
    Trs {
        translation: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
    },
}

impl Transform {
    /// Identity TRS — `(0,0,0)` translation, identity quaternion,
    /// `(1,1,1)` scale.
    pub fn identity() -> Self {
        Self::Trs {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }

    /// Compose this transform into a single 4x4 matrix.
    ///
    /// For `Matrix(m)` this is the identity passthrough; for
    /// `Trs { t, r, s }` the build order is `T * R * S`.
    pub fn to_matrix(&self) -> [[f32; 4]; 4] {
        match *self {
            Self::Matrix(m) => m,
            Self::Trs {
                translation,
                rotation,
                scale,
            } => trs_to_matrix(translation, rotation, scale),
        }
    }

    /// Best-effort decomposition of a 4x4 affine transform into TRS.
    ///
    /// Assumes the input is `T * R * S` with no shear and no negative
    /// scale; under that assumption the recovery is exact within
    /// float epsilon. For matrices with shear the output is the
    /// closest pure TRS (scales are column lengths, rotation is the
    /// orthonormalised basis).
    pub fn from_matrix(m: [[f32; 4]; 4]) -> Self {
        let translation = [m[0][3], m[1][3], m[2][3]];
        let cx = [m[0][0], m[1][0], m[2][0]];
        let cy = [m[0][1], m[1][1], m[2][1]];
        let cz = [m[0][2], m[1][2], m[2][2]];
        let sx = vec3_len(cx);
        let sy = vec3_len(cy);
        let sz = vec3_len(cz);
        // Avoid div-by-zero if a column was zero — fall back to a sentinel
        // axis; this lets the from_matrix(to_matrix(t)) round-trip remain
        // total even for pathological inputs.
        let inv_sx = if sx > f32::EPSILON { 1.0 / sx } else { 1.0 };
        let inv_sy = if sy > f32::EPSILON { 1.0 / sy } else { 1.0 };
        let inv_sz = if sz > f32::EPSILON { 1.0 / sz } else { 1.0 };
        let r00 = cx[0] * inv_sx;
        let r10 = cx[1] * inv_sx;
        let r20 = cx[2] * inv_sx;
        let r01 = cy[0] * inv_sy;
        let r11 = cy[1] * inv_sy;
        let r21 = cy[2] * inv_sy;
        let r02 = cz[0] * inv_sz;
        let r12 = cz[1] * inv_sz;
        let r22 = cz[2] * inv_sz;
        let rotation = rot_matrix_to_quat([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]);
        Self::Trs {
            translation,
            rotation,
            scale: [sx, sy, sz],
        }
    }
}

fn vec3_len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn trs_to_matrix(t: [f32; 3], r: [f32; 4], s: [f32; 3]) -> [[f32; 4]; 4] {
    // Quaternion (x, y, z, w) → 3x3 rotation matrix (Shoemake).
    let (x, y, z, w) = (r[0], r[1], r[2], r[3]);
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    let r00 = 1.0 - 2.0 * (yy + zz);
    let r01 = 2.0 * (xy - wz);
    let r02 = 2.0 * (xz + wy);
    let r10 = 2.0 * (xy + wz);
    let r11 = 1.0 - 2.0 * (xx + zz);
    let r12 = 2.0 * (yz - wx);
    let r20 = 2.0 * (xz - wy);
    let r21 = 2.0 * (yz + wx);
    let r22 = 1.0 - 2.0 * (xx + yy);
    [
        [r00 * s[0], r01 * s[1], r02 * s[2], t[0]],
        [r10 * s[0], r11 * s[1], r12 * s[2], t[1]],
        [r20 * s[0], r21 * s[1], r22 * s[2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rot_matrix_to_quat(m: [[f32; 3]; 3]) -> [f32; 4] {
    // Shepperd's branchless variant — picks the column with the
    // largest diagonal to avoid catastrophic cancellation near
    // 180-degree rotations. Returns (x, y, z, w).
    let trace = m[0][0] + m[1][1] + m[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (m[2][1] - m[1][2]) / s;
        let y = (m[0][2] - m[2][0]) / s;
        let z = (m[1][0] - m[0][1]) / s;
        [x, y, z, w]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        let w = (m[2][1] - m[1][2]) / s;
        let x = 0.25 * s;
        let y = (m[0][1] + m[1][0]) / s;
        let z = (m[0][2] + m[2][0]) / s;
        [x, y, z, w]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        let w = (m[0][2] - m[2][0]) / s;
        let x = (m[0][1] + m[1][0]) / s;
        let y = 0.25 * s;
        let z = (m[1][2] + m[2][1]) / s;
        [x, y, z, w]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        let w = (m[1][0] - m[0][1]) / s;
        let x = (m[0][2] + m[2][0]) / s;
        let y = (m[1][2] + m[2][1]) / s;
        let z = 0.25 * s;
        [x, y, z, w]
    }
}

/// A single scene-graph node.
///
/// Nodes form a forest rooted at [`Scene3D::roots`]. Each node has at
/// most one parent (enforced by walking children top-down only — the
/// `parent` back-pointer isn't stored; decoders that need it should
/// build a side-table).
#[derive(Clone, Debug)]
pub struct Node {
    pub name: Option<String>,
    pub transform: Transform,
    pub children: Vec<NodeId>,
    pub mesh: Option<MeshId>,
    pub camera: Option<CameraId>,
    pub light: Option<LightId>,
    pub skin: Option<SkinId>,
    /// Optional audio emitter attached to this node. The emitter's
    /// position + orientation come from this node's world transform
    /// when [`AudioEmitter::spatial`](crate::AudioEmitter::spatial)
    /// is `Some`; non-spatial emitters ignore the transform and play
    /// globally.
    pub audio_emitter: Option<AudioEmitterId>,
    pub extras: HashMap<String, serde_json::Value>,
}

impl Node {
    /// Construct an empty node with identity transform.
    pub fn new() -> Self {
        Self {
            name: None,
            transform: Transform::identity(),
            children: Vec::new(),
            mesh: None,
            camera: None,
            light: None,
            skin: None,
            audio_emitter: None,
            extras: HashMap::new(),
        }
    }

    /// Builder-style name setter.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Builder-style transform setter.
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    /// Builder-style mesh attachment.
    pub fn with_mesh(mut self, mesh: MeshId) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Builder-style audio-emitter attachment.
    pub fn with_audio_emitter(mut self, emitter: AudioEmitterId) -> Self {
        self.audio_emitter = Some(emitter);
        self
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-level container for a 3D scene.
///
/// Owns every resource referenced by the scene graph. Add resources
/// with the `add_*` helpers — they push into the corresponding `Vec`
/// and return the freshly-issued ID. Roots are explicit: a node added
/// with [`Scene3D::add_node`] is not automatically a root, so that
/// child nodes added later can be re-parented without re-shuffling.
#[derive(Clone, Debug)]
pub struct Scene3D {
    pub nodes: Vec<Node>,
    pub roots: Vec<NodeId>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub skeletons: Vec<Skeleton>,
    pub skins: Vec<Skin>,
    pub animations: Vec<Animation>,
    pub cameras: Vec<Camera>,
    pub lights: Vec<Light>,
    /// Audio assets owned by the scene; addressed by [`AudioSourceId`].
    pub audio_sources: Vec<AudioSource>,
    /// In-scene audio-emitter instances; addressed by [`AudioEmitterId`].
    pub audio_emitters: Vec<AudioEmitter>,
    pub up_axis: Axis,
    pub front_axis: Axis,
    pub unit: Unit,
    pub extras: HashMap<String, serde_json::Value>,
}

impl Scene3D {
    /// Empty scene with glTF-default orientation (Y-up, -Z forward,
    /// metres) and no resources.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
            meshes: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            skeletons: Vec::new(),
            skins: Vec::new(),
            animations: Vec::new(),
            cameras: Vec::new(),
            lights: Vec::new(),
            audio_sources: Vec::new(),
            audio_emitters: Vec::new(),
            up_axis: Axis::PosY,
            front_axis: Axis::NegZ,
            unit: Unit::Metres,
            extras: HashMap::new(),
        }
    }

    /// Push a node and return its id.
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Push a mesh and return its id.
    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshId {
        let id = MeshId(self.meshes.len() as u32);
        self.meshes.push(mesh);
        id
    }

    /// Push a material and return its id.
    pub fn add_material(&mut self, material: Material) -> MaterialId {
        let id = MaterialId(self.materials.len() as u32);
        self.materials.push(material);
        id
    }

    /// Push a texture and return its id.
    pub fn add_texture(&mut self, texture: Texture) -> TextureId {
        let id = TextureId(self.textures.len() as u32);
        self.textures.push(texture);
        id
    }

    /// Push a skeleton and return its id.
    pub fn add_skeleton(&mut self, skeleton: Skeleton) -> SkeletonId {
        let id = SkeletonId(self.skeletons.len() as u32);
        self.skeletons.push(skeleton);
        id
    }

    /// Push a skin and return its id.
    pub fn add_skin(&mut self, skin: Skin) -> SkinId {
        let id = SkinId(self.skins.len() as u32);
        self.skins.push(skin);
        id
    }

    /// Push an animation and return its id (animations are
    /// list-ordered, no separate id type — reference by index).
    pub fn add_animation(&mut self, animation: Animation) -> usize {
        let idx = self.animations.len();
        self.animations.push(animation);
        idx
    }

    /// Push a camera and return its id.
    pub fn add_camera(&mut self, camera: Camera) -> CameraId {
        let id = CameraId(self.cameras.len() as u32);
        self.cameras.push(camera);
        id
    }

    /// Push a light and return its id.
    pub fn add_light(&mut self, light: Light) -> LightId {
        let id = LightId(self.lights.len() as u32);
        self.lights.push(light);
        id
    }

    /// Push an [`AudioSource`] and return its id.
    pub fn add_audio_source(&mut self, source: AudioSource) -> AudioSourceId {
        let id = AudioSourceId(self.audio_sources.len() as u32);
        self.audio_sources.push(source);
        id
    }

    /// Push an [`AudioEmitter`] and return its id.
    pub fn add_audio_emitter(&mut self, emitter: AudioEmitter) -> AudioEmitterId {
        let id = AudioEmitterId(self.audio_emitters.len() as u32);
        self.audio_emitters.push(emitter);
        id
    }

    /// Borrow an audio source by id, if it exists.
    pub fn audio_source(&self, id: AudioSourceId) -> Option<&AudioSource> {
        self.audio_sources.get(id.0 as usize)
    }

    /// Borrow an audio emitter by id, if it exists.
    pub fn audio_emitter(&self, id: AudioEmitterId) -> Option<&AudioEmitter> {
        self.audio_emitters.get(id.0 as usize)
    }

    /// Promote a node to a root of the scene-graph forest.
    pub fn add_root(&mut self, node: NodeId) {
        self.roots.push(node);
    }

    /// Borrow a node by id, if it exists.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0 as usize)
    }

    /// Mutably borrow a node by id, if it exists.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id.0 as usize)
    }

    /// Borrow a mesh by id, if it exists.
    pub fn mesh(&self, id: MeshId) -> Option<&Mesh> {
        self.meshes.get(id.0 as usize)
    }

    /// Sum of triangles across every mesh primitive.
    ///
    /// Lists / strips / fans contribute as if tessellated:
    /// - `Triangles` → `vertex_count / 3` (or `index_count / 3`)
    /// - `TriangleStrip` / `TriangleFan` → `max(0, n - 2)` triangles
    /// - non-triangle topologies contribute 0.
    pub fn triangle_count(&self) -> usize {
        self.meshes
            .iter()
            .flat_map(|m| m.primitives.iter())
            .map(|p| p.triangle_count())
            .sum()
    }

    /// Sum of `positions.len()` across every mesh primitive.
    pub fn vertex_count(&self) -> usize {
        self.meshes
            .iter()
            .flat_map(|m| m.primitives.iter())
            .map(|p| p.positions.len())
            .sum()
    }

    /// Walk every cross-collection reference and report dangling
    /// indices + inconsistent buffer lengths. Returns `Ok(())` when
    /// the scene is internally consistent, or `Err` carrying every
    /// problem found (the walk does not short-circuit, so callers see
    /// the full set in one pass).
    ///
    /// Currently checks:
    ///
    /// * `roots` reference live `nodes`.
    /// * Every `Node::children`, `Node::mesh`, `Node::camera`,
    ///   `Node::light`, `Node::skin`, `Node::audio_emitter` references
    ///   a live entry in the corresponding arena.
    /// * Every primitive's optional attribute buffer (`normals`,
    ///   `tangents`, `uvs[i]`, `colors[i]`, `joints`, `weights`)
    ///   matches `positions.len()`.
    /// * `Primitive::indices` values stay within `positions.len()`.
    /// * `Primitive::material` indices are live.
    /// * Each `MorphTarget` slot length matches the corresponding
    ///   base attribute on the parent `Primitive`.
    /// * `Mesh::weights.len()` matches the morph-target count of
    ///   every contained primitive (or every primitive has zero
    ///   targets and `weights` is empty).
    ///
    /// This is a defensive check for fuzzers and codec authors —
    /// production decoders are expected to produce valid scenes
    /// already; the runtime cost is `O(N)` over every typed buffer.
    pub fn validate(&self) -> std::result::Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        let n_nodes = self.nodes.len();
        let n_meshes = self.meshes.len();
        let n_materials = self.materials.len();
        let n_cameras = self.cameras.len();
        let n_lights = self.lights.len();
        let n_skins = self.skins.len();
        let n_emitters = self.audio_emitters.len();

        for (i, root) in self.roots.iter().enumerate() {
            if (root.0 as usize) >= n_nodes {
                errors.push(ValidationError::DanglingId {
                    location: format!("roots[{i}]"),
                    id: root.0,
                    arena: "nodes",
                });
            }
        }
        for (i, node) in self.nodes.iter().enumerate() {
            for (j, child) in node.children.iter().enumerate() {
                if (child.0 as usize) >= n_nodes {
                    errors.push(ValidationError::DanglingId {
                        location: format!("nodes[{i}].children[{j}]"),
                        id: child.0,
                        arena: "nodes",
                    });
                }
            }
            if let Some(m) = node.mesh {
                if (m.0 as usize) >= n_meshes {
                    errors.push(ValidationError::DanglingId {
                        location: format!("nodes[{i}].mesh"),
                        id: m.0,
                        arena: "meshes",
                    });
                }
            }
            if let Some(c) = node.camera {
                if (c.0 as usize) >= n_cameras {
                    errors.push(ValidationError::DanglingId {
                        location: format!("nodes[{i}].camera"),
                        id: c.0,
                        arena: "cameras",
                    });
                }
            }
            if let Some(l) = node.light {
                if (l.0 as usize) >= n_lights {
                    errors.push(ValidationError::DanglingId {
                        location: format!("nodes[{i}].light"),
                        id: l.0,
                        arena: "lights",
                    });
                }
            }
            if let Some(s) = node.skin {
                if (s.0 as usize) >= n_skins {
                    errors.push(ValidationError::DanglingId {
                        location: format!("nodes[{i}].skin"),
                        id: s.0,
                        arena: "skins",
                    });
                }
            }
            if let Some(e) = node.audio_emitter {
                if (e.0 as usize) >= n_emitters {
                    errors.push(ValidationError::DanglingId {
                        location: format!("nodes[{i}].audio_emitter"),
                        id: e.0,
                        arena: "audio_emitters",
                    });
                }
            }
        }

        for (mi, mesh) in self.meshes.iter().enumerate() {
            let mesh_weights = mesh.weights.len();
            for (pi, prim) in mesh.primitives.iter().enumerate() {
                let n_pos = prim.positions.len();
                let here = |field: &str| format!("meshes[{mi}].primitives[{pi}].{field}");
                if let Some(v) = &prim.normals {
                    if v.len() != n_pos {
                        errors.push(ValidationError::AttributeLengthMismatch {
                            location: here("normals"),
                            expected: n_pos,
                            actual: v.len(),
                        });
                    }
                }
                if let Some(v) = &prim.tangents {
                    if v.len() != n_pos {
                        errors.push(ValidationError::AttributeLengthMismatch {
                            location: here("tangents"),
                            expected: n_pos,
                            actual: v.len(),
                        });
                    }
                }
                for (k, set) in prim.uvs.iter().enumerate() {
                    if set.len() != n_pos {
                        errors.push(ValidationError::AttributeLengthMismatch {
                            location: here(&format!("uvs[{k}]")),
                            expected: n_pos,
                            actual: set.len(),
                        });
                    }
                }
                for (k, set) in prim.colors.iter().enumerate() {
                    if set.len() != n_pos {
                        errors.push(ValidationError::AttributeLengthMismatch {
                            location: here(&format!("colors[{k}]")),
                            expected: n_pos,
                            actual: set.len(),
                        });
                    }
                }
                if let Some(v) = &prim.joints {
                    if v.len() != n_pos {
                        errors.push(ValidationError::AttributeLengthMismatch {
                            location: here("joints"),
                            expected: n_pos,
                            actual: v.len(),
                        });
                    }
                }
                if let Some(v) = &prim.weights {
                    if v.len() != n_pos {
                        errors.push(ValidationError::AttributeLengthMismatch {
                            location: here("weights"),
                            expected: n_pos,
                            actual: v.len(),
                        });
                    }
                }
                if let Some(idx) = &prim.indices {
                    let max_ok = n_pos as u32;
                    let bad = match idx {
                        crate::mesh::Indices::U16(v) => v.iter().any(|i| (*i as u32) >= max_ok),
                        crate::mesh::Indices::U32(v) => v.iter().any(|i| *i >= max_ok),
                    };
                    if bad {
                        errors.push(ValidationError::IndexOutOfRange {
                            location: here("indices"),
                            vertex_count: n_pos,
                        });
                    }
                }
                if let Some(m) = prim.material {
                    if (m.0 as usize) >= n_materials {
                        errors.push(ValidationError::DanglingId {
                            location: here("material"),
                            id: m.0,
                            arena: "materials",
                        });
                    }
                }
                for (ti, tgt) in prim.targets.iter().enumerate() {
                    let tgt_loc = |field: &str| here(&format!("targets[{ti}].{field}"));
                    if let Some(v) = &tgt.position {
                        if v.len() != n_pos {
                            errors.push(ValidationError::AttributeLengthMismatch {
                                location: tgt_loc("position"),
                                expected: n_pos,
                                actual: v.len(),
                            });
                        }
                    }
                    if let Some(v) = &tgt.normal {
                        if v.len() != n_pos {
                            errors.push(ValidationError::AttributeLengthMismatch {
                                location: tgt_loc("normal"),
                                expected: n_pos,
                                actual: v.len(),
                            });
                        }
                    }
                    if let Some(v) = &tgt.tangent {
                        if v.len() != n_pos {
                            errors.push(ValidationError::AttributeLengthMismatch {
                                location: tgt_loc("tangent"),
                                expected: n_pos,
                                actual: v.len(),
                            });
                        }
                    }
                }
                if mesh_weights != 0 && prim.targets.len() != mesh_weights {
                    errors.push(ValidationError::MorphWeightCountMismatch {
                        location: format!("meshes[{mi}].primitives[{pi}].targets"),
                        mesh_weights,
                        primitive_targets: prim.targets.len(),
                    });
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// One issue surfaced by [`Scene3D::validate`]. The variants intentionally
/// carry breadcrumb strings (`"meshes[3].primitives[0].normals"`) so a
/// caller can render a usable diagnostic without re-walking the scene.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationError {
    /// A typed `IdT(u32)` field points outside its arena.
    DanglingId {
        location: String,
        id: u32,
        arena: &'static str,
    },
    /// An optional attribute buffer is present but its length disagrees
    /// with the parent primitive's `positions.len()`.
    AttributeLengthMismatch {
        location: String,
        expected: usize,
        actual: usize,
    },
    /// A primitive's index buffer references a vertex past
    /// `positions.len()`.
    IndexOutOfRange {
        location: String,
        vertex_count: usize,
    },
    /// `Mesh::weights` is non-empty and disagrees with one of the
    /// child primitives' morph-target count.
    MorphWeightCountMismatch {
        location: String,
        mesh_weights: usize,
        primitive_targets: usize,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DanglingId {
                location,
                id,
                arena,
            } => write!(f, "{location}: id {id} is out of bounds for {arena}"),
            Self::AttributeLengthMismatch {
                location,
                expected,
                actual,
            } => write!(
                f,
                "{location}: length {actual} disagrees with positions length {expected}"
            ),
            Self::IndexOutOfRange {
                location,
                vertex_count,
            } => write!(
                f,
                "{location}: index buffer references vertex >= {vertex_count}"
            ),
            Self::MorphWeightCountMismatch {
                location,
                mesh_weights,
                primitive_targets,
            } => write!(
                f,
                "{location}: mesh has {mesh_weights} weights but primitive carries {primitive_targets} morph targets"
            ),
        }
    }
}

impl std::error::Error for ValidationError {}

impl Default for Scene3D {
    fn default() -> Self {
        Self::new()
    }
}
