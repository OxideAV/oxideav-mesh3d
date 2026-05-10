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
    animation::Animation, camera::Camera, light::Light, material::Material, mesh::Mesh,
    skin::Skeleton, skin::Skin, texture::Texture,
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
}

impl Default for Scene3D {
    fn default() -> Self {
        Self::new()
    }
}
