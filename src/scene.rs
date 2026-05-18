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

/// Axis-aligned bounding box over a set of 3D points.
///
/// `min` is the componentwise minimum corner, `max` the componentwise
/// maximum corner. Both are inclusive; for an empty point set this
/// type returns [`None`] from its constructors rather than carrying a
/// degenerate `[inf; 3]` / `[-inf; 3]` sentinel.
///
/// Use [`BoundingBox::from_points`] to build one from an iterator of
/// `[f32; 3]`, [`BoundingBox::union`] to merge two boxes, and
/// [`BoundingBox::transform`] to rotate / translate / scale the box
/// by a 4x4 row-major-column-vector matrix (the eight corners are
/// transformed and a new AABB is fitted around them — the rotated
/// box's tight bound).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl BoundingBox {
    /// Bounding box of exactly one point. Both corners coincide.
    pub fn from_point(p: [f32; 3]) -> Self {
        Self { min: p, max: p }
    }

    /// Bounding box over a stream of points. Returns `None` if the
    /// iterator yields zero finite points (NaN coordinates are
    /// skipped on a per-component basis).
    pub fn from_points<I: IntoIterator<Item = [f32; 3]>>(points: I) -> Option<Self> {
        let mut acc: Option<Self> = None;
        for p in points {
            if p[0].is_nan() || p[1].is_nan() || p[2].is_nan() {
                continue;
            }
            acc = Some(match acc {
                None => Self::from_point(p),
                Some(b) => b.expand(p),
            });
        }
        acc
    }

    /// Grow the box to include `p`. Returns a new box; the input is
    /// left unchanged. NaN components are kept as-is on the
    /// existing box (they are not propagated by [`from_points`] either).
    pub fn expand(self, p: [f32; 3]) -> Self {
        Self {
            min: [
                self.min[0].min(p[0]),
                self.min[1].min(p[1]),
                self.min[2].min(p[2]),
            ],
            max: [
                self.max[0].max(p[0]),
                self.max[1].max(p[1]),
                self.max[2].max(p[2]),
            ],
        }
    }

    /// Componentwise union of two boxes — the smallest AABB
    /// containing both.
    pub fn union(self, other: Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Centre of the box (average of `min` and `max`).
    pub fn center(self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Componentwise size of the box (`max - min`).
    pub fn size(self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// `true` if every component of `min` is less than or equal to the
    /// corresponding component of `max` (i.e. the box is non-empty
    /// and well-formed).
    pub fn is_valid(self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1] && self.min[2] <= self.max[2]
    }

    /// Tight AABB around the box transformed by a row-major
    /// column-vector 4x4 matrix (`out = M * v`, same convention as
    /// [`Transform::Matrix`]).
    ///
    /// Returns the AABB of the eight transformed corners. For
    /// non-affine matrices (perspective `w != 1`) the result may not
    /// be physically meaningful — this method is intended for the
    /// scene-graph TRS / matrix chain composing every ancestor node's
    /// local transform.
    pub fn transform(self, m: [[f32; 4]; 4]) -> Self {
        let corners = [
            [self.min[0], self.min[1], self.min[2]],
            [self.max[0], self.min[1], self.min[2]],
            [self.min[0], self.max[1], self.min[2]],
            [self.max[0], self.max[1], self.min[2]],
            [self.min[0], self.min[1], self.max[2]],
            [self.max[0], self.min[1], self.max[2]],
            [self.min[0], self.max[1], self.max[2]],
            [self.max[0], self.max[1], self.max[2]],
        ];
        let xf = corners.map(|c| {
            [
                m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2] + m[0][3],
                m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2] + m[1][3],
                m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2] + m[2][3],
            ]
        });
        Self::from_points(xf).expect("eight corners always yield a finite AABB")
    }
}

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

/// Row-major column-vector 4x4 matrix multiply `a * b`.
fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
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

    /// Axis-aligned bounding box over every mesh referenced by a node
    /// reachable from [`Scene3D::roots`], with each mesh's vertices
    /// projected through its node's full ancestor transform chain.
    ///
    /// Returns `None` when no reachable node carries a mesh, or every
    /// reachable mesh is empty.
    ///
    /// **What this does NOT include:**
    ///
    /// * Skin pose deformation — the rest-pose vertices are used
    ///   verbatim. A bound mesh whose vertices are rigged to a
    ///   skeleton will report the *rest-pose* extent, not the
    ///   skinned-pose extent at any particular animation time.
    /// * Morph targets — only base [`Primitive::positions`](crate::Primitive::positions)
    ///   are folded in.
    /// * Meshes referenced by `nodes` not reachable from any root —
    ///   detached resources are ignored. Use [`Scene3D::meshes`] +
    ///   [`Mesh::bounding_box`](crate::Mesh::bounding_box) directly if
    ///   you need every resource regardless of scene-graph reachability.
    ///
    /// Re-entry through a cycle (a node listed as its own descendant)
    /// is guarded against — each node is visited at most once.
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        let n_nodes = self.nodes.len();
        let n_meshes = self.meshes.len();
        if n_nodes == 0 || n_meshes == 0 {
            return None;
        }
        let mut visited = vec![false; n_nodes];
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut acc: Option<BoundingBox> = None;
        // Iterative depth-first walk; stack carries (node_id, ancestor_matrix).
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().map(|r| (*r, identity)).collect();
        while let Some((nid, parent)) = stack.pop() {
            let idx = nid.0 as usize;
            if idx >= n_nodes || visited[idx] {
                continue;
            }
            visited[idx] = true;
            let node = &self.nodes[idx];
            let world = mat4_mul(parent, node.transform.to_matrix());
            if let Some(m) = node.mesh {
                if let Some(mesh) = self.meshes.get(m.0 as usize) {
                    if let Some(b) = mesh.bounding_box() {
                        let xf = b.transform(world);
                        acc = Some(match acc {
                            None => xf,
                            Some(a) => a.union(xf),
                        });
                    }
                }
            }
            // Walk children in reverse so leftmost child is popped first
            // (deterministic for the deterministic-debug-output use case).
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        acc
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
    /// * Every `Skeleton::inverse_bind_matrices` entry has its fourth
    ///   row set to `[0, 0, 0, 1]` (glTF 2.0 §5.28.1 affine-IBM
    ///   constraint).
    ///
    /// This is a defensive check for fuzzers and codec authors —
    /// production decoders are expected to produce valid scenes
    /// already; the runtime cost is `O(N)` over every typed buffer.
    pub fn validate(&self) -> std::result::Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        let n_nodes = self.nodes.len();
        let n_meshes = self.meshes.len();
        let n_materials = self.materials.len();
        let n_textures = self.textures.len();
        let n_cameras = self.cameras.len();
        let n_lights = self.lights.len();
        let n_skeletons = self.skeletons.len();
        let n_skins = self.skins.len();
        let n_emitters = self.audio_emitters.len();
        let n_audio_sources = self.audio_sources.len();

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

        // Materials → textures.
        for (mi, mat) in self.materials.iter().enumerate() {
            let slot = |field: &str| format!("materials[{mi}].{field}");
            let mut check = |field: &str, t: Option<crate::material::TextureRef>| {
                if let Some(r) = t {
                    if (r.texture.0 as usize) >= n_textures {
                        errors.push(ValidationError::DanglingId {
                            location: slot(field),
                            id: r.texture.0,
                            arena: "textures",
                        });
                    }
                }
            };
            check("base_color_texture", mat.base_color_texture);
            check("metallic_roughness_texture", mat.metallic_roughness_texture);
            check("normal_texture", mat.normal_texture);
            check("occlusion_texture", mat.occlusion_texture);
            check("emissive_texture", mat.emissive_texture);
        }

        // Skeletons → nodes + inverse-bind-matrix parity.
        for (si, skel) in self.skeletons.iter().enumerate() {
            for (ji, joint) in skel.joints.iter().enumerate() {
                if (joint.0 as usize) >= n_nodes {
                    errors.push(ValidationError::DanglingId {
                        location: format!("skeletons[{si}].joints[{ji}]"),
                        id: joint.0,
                        arena: "nodes",
                    });
                }
            }
            if !skel.inverse_bind_matrices.is_empty()
                && skel.inverse_bind_matrices.len() != skel.joints.len()
            {
                errors.push(ValidationError::SkeletonBindMatrixCountMismatch {
                    location: format!("skeletons[{si}]"),
                    joints: skel.joints.len(),
                    inverse_bind_matrices: skel.inverse_bind_matrices.len(),
                });
            }
            // glTF 2.0 §5.28.1: an accessor referenced by
            // `inverseBindMatrices` MUST have its fourth row set to
            // `[0.0, 0.0, 0.0, 1.0]` (the matrix is affine — a pure
            // composition of rotations/translations/scales/shears,
            // never projective). Our matrix is row-major
            // column-vector, so the "fourth row" of the math matrix
            // is the row at index 3.
            for (ji, ibm) in skel.inverse_bind_matrices.iter().enumerate() {
                let last = ibm[3];
                if last[0] != 0.0 || last[1] != 0.0 || last[2] != 0.0 || last[3] != 1.0 {
                    errors.push(ValidationError::SkeletonBindMatrixNotAffine {
                        location: format!("skeletons[{si}].inverse_bind_matrices[{ji}]"),
                        last_row: last,
                    });
                }
            }
        }

        // Skins → skeletons + optional root node.
        for (si, skin) in self.skins.iter().enumerate() {
            if (skin.skeleton.0 as usize) >= n_skeletons {
                errors.push(ValidationError::DanglingId {
                    location: format!("skins[{si}].skeleton"),
                    id: skin.skeleton.0,
                    arena: "skeletons",
                });
            }
            if let Some(r) = skin.root_node {
                if (r.0 as usize) >= n_nodes {
                    errors.push(ValidationError::DanglingId {
                        location: format!("skins[{si}].root_node"),
                        id: r.0,
                        arena: "nodes",
                    });
                }
            }
        }

        // Audio emitters → audio sources.
        for (ei, em) in self.audio_emitters.iter().enumerate() {
            if (em.source.0 as usize) >= n_audio_sources {
                errors.push(ValidationError::DanglingId {
                    location: format!("audio_emitters[{ei}].source"),
                    id: em.source.0,
                    arena: "audio_sources",
                });
            }
        }

        // Animations: channel target nodes + sampler parity.
        for (ai, anim) in self.animations.iter().enumerate() {
            for (ci, ch) in anim.channels.iter().enumerate() {
                let loc = |suffix: &str| format!("animations[{ai}].channels[{ci}]{suffix}");
                if (ch.target.node.0 as usize) >= n_nodes {
                    errors.push(ValidationError::DanglingId {
                        location: loc(".target.node"),
                        id: ch.target.node.0,
                        arena: "nodes",
                    });
                }
                let k = ch.sampler.keyframes.len();
                if k == 0 {
                    errors.push(ValidationError::AnimationSamplerEmpty {
                        location: loc(".sampler"),
                    });
                } else {
                    let mut prev = f32::NEG_INFINITY;
                    for (ki, t) in ch.sampler.keyframes.iter().enumerate() {
                        if t.partial_cmp(&prev) != Some(std::cmp::Ordering::Greater) {
                            errors.push(ValidationError::AnimationKeyframesNotStrictlyIncreasing {
                                location: loc(&format!(".sampler.keyframes[{ki}]")),
                                at: *t,
                                previous: prev,
                            });
                            break;
                        }
                        prev = *t;
                    }
                }

                use crate::animation::{AnimationProperty as P, AnimationValues as V};
                let variant_ok = matches!(
                    (ch.target.property, &ch.sampler.values),
                    (P::Translation | P::Scale, V::Vec3(_))
                        | (P::Rotation, V::Quat(_))
                        | (P::MorphWeights, V::Scalar(_))
                );
                if !variant_ok {
                    let expected: &'static str = match ch.target.property {
                        P::Translation | P::Scale => "Vec3",
                        P::Rotation => "Quat",
                        P::MorphWeights => "Scalar",
                    };
                    let actual: &'static str = match ch.sampler.values {
                        V::Vec3(_) => "Vec3",
                        V::Quat(_) => "Quat",
                        V::Scalar(_) => "Scalar",
                    };
                    errors.push(ValidationError::AnimationValueVariantMismatch {
                        location: loc(""),
                        property: match ch.target.property {
                            P::Translation => "Translation",
                            P::Rotation => "Rotation",
                            P::Scale => "Scale",
                            P::MorphWeights => "MorphWeights",
                        },
                        expected_variant: expected,
                        actual_variant: actual,
                    });
                }

                if k != 0 {
                    let v = ch.sampler.values.len();
                    let expected_factor = match ch.sampler.interpolation {
                        crate::animation::Interpolation::CubicSpline => 3,
                        _ => 1,
                    };
                    let ok = match (ch.target.property, &ch.sampler.values) {
                        (P::MorphWeights, V::Scalar(_)) => {
                            let denom = k * expected_factor;
                            denom != 0 && v % denom == 0 && v >= denom
                        }
                        _ => v == k * expected_factor,
                    };
                    if !ok {
                        errors.push(ValidationError::AnimationSamplerLengthMismatch {
                            location: loc(".sampler"),
                            keyframes: k,
                            values: v,
                            interpolation: match ch.sampler.interpolation {
                                crate::animation::Interpolation::Step => "Step",
                                crate::animation::Interpolation::Linear => "Linear",
                                crate::animation::Interpolation::CubicSpline => "CubicSpline",
                            },
                        });
                    }
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
///
/// `Eq` is not implemented because
/// [`AnimationKeyframesNotStrictlyIncreasing`](Self::AnimationKeyframesNotStrictlyIncreasing)
/// carries `f32` keyframe values; use `PartialEq` or pattern-match on
/// the variant fields when asserting in tests.
#[derive(Clone, Debug, PartialEq)]
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
    /// [`Skeleton::inverse_bind_matrices`](crate::Skeleton::inverse_bind_matrices)
    /// is non-empty and its length disagrees with
    /// [`Skeleton::joints`](crate::Skeleton::joints).
    SkeletonBindMatrixCountMismatch {
        location: String,
        joints: usize,
        inverse_bind_matrices: usize,
    },
    /// One of [`Skeleton::inverse_bind_matrices`](crate::Skeleton::inverse_bind_matrices)
    /// has a non-affine fourth row. The glTF 2.0 spec §5.28.1
    /// requires every IBM's last row to be `[0.0, 0.0, 0.0, 1.0]`;
    /// any other value implies a projective component that the
    /// skinning math `(weight_i * joint_world_i * IBM_i * pos)` would
    /// silently corrupt.
    SkeletonBindMatrixNotAffine {
        location: String,
        last_row: [f32; 4],
    },
    /// An animation channel's sampler has zero keyframes; no
    /// keyframe-time table to interpolate against.
    AnimationSamplerEmpty { location: String },
    /// An animation sampler's keyframe times are not strictly
    /// increasing — the renderer would search ambiguously.
    AnimationKeyframesNotStrictlyIncreasing {
        location: String,
        at: f32,
        previous: f32,
    },
    /// An animation sampler's value variant disagrees with the
    /// channel target's property kind (e.g. `Rotation` channel
    /// fed `Vec3` values).
    AnimationValueVariantMismatch {
        location: String,
        property: &'static str,
        expected_variant: &'static str,
        actual_variant: &'static str,
    },
    /// An animation sampler's value count doesn't match the expected
    /// `keyframes.len() * factor` (`factor = 1` for Step/Linear,
    /// `factor = 3` for CubicSpline; MorphWeights additionally
    /// multiplies by per-mesh morph-target count, so we only check
    /// divisibility there).
    AnimationSamplerLengthMismatch {
        location: String,
        keyframes: usize,
        values: usize,
        interpolation: &'static str,
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
            Self::SkeletonBindMatrixCountMismatch {
                location,
                joints,
                inverse_bind_matrices,
            } => write!(
                f,
                "{location}: skeleton has {joints} joints but {inverse_bind_matrices} inverse-bind matrices"
            ),
            Self::SkeletonBindMatrixNotAffine { location, last_row } => write!(
                f,
                "{location}: inverse-bind matrix last row {last_row:?} is not [0, 0, 0, 1]"
            ),
            Self::AnimationSamplerEmpty { location } => {
                write!(f, "{location}: sampler has no keyframes")
            }
            Self::AnimationKeyframesNotStrictlyIncreasing {
                location,
                at,
                previous,
            } => write!(
                f,
                "{location}: keyframe time {at} is not greater than previous {previous}"
            ),
            Self::AnimationValueVariantMismatch {
                location,
                property,
                expected_variant,
                actual_variant,
            } => write!(
                f,
                "{location}: property {property} expects {expected_variant} values but sampler carries {actual_variant}"
            ),
            Self::AnimationSamplerLengthMismatch {
                location,
                keyframes,
                values,
                interpolation,
            } => write!(
                f,
                "{location}: interpolation {interpolation} with {keyframes} keyframes expects matching values, got {values}"
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
