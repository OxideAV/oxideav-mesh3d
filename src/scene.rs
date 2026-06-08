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

    /// Slab-method ray-AABB intersection — returns the entry / exit
    /// parametric distances along the ray clamped to `[0, t_max]`, or
    /// `None` if the ray misses.
    ///
    /// `t_enter == 0.0` indicates the ray's origin lies inside the
    /// box; `t_exit` is the parameter at which the ray leaves through
    /// the far face. Both values are along the (not-necessarily-unit)
    /// `ray.direction`, so the actual world-space point at the
    /// intersection is `ray.point_at(t)`.
    ///
    /// Delegates to [`crate::ray::intersect_aabb`]; see its docs for
    /// the axis-parallel-ray + NaN / Inf handling.
    pub fn intersect_ray(self, ray: crate::ray::Ray, t_max: f32) -> Option<(f32, f32)> {
        crate::ray::intersect_aabb(ray, self.min, self.max, t_max)
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
pub(crate) fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

/// Signed determinant of the upper-left 3x3 of a row-major
/// column-vector 4x4 matrix, returned as `f64` for accumulator-safe
/// volume scaling. The translation column does not enter the result.
fn mat3_det_of_world(m: [[f32; 4]; 4]) -> f64 {
    let a = m[0][0] as f64;
    let b = m[0][1] as f64;
    let c = m[0][2] as f64;
    let d = m[1][0] as f64;
    let e = m[1][1] as f64;
    let f = m[1][2] as f64;
    let g = m[2][0] as f64;
    let h = m[2][1] as f64;
    let i = m[2][2] as f64;
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Inverse of an affine row-major column-vector 4x4 whose bottom row
/// is `[0, 0, 0, 1]`. Returns `None` when the upper-left 3x3 is
/// singular (zero or non-finite determinant), when any input entry
/// is non-finite, or when the bottom row deviates from
/// `[0, 0, 0, 1]` (the matrix is non-affine and not handled here —
/// `world_node_transforms` only produces affines from TRS, so a
/// non-affine input is a malformed user transform).
///
/// The inverse uses the classical adjugate / determinant of the
/// 3x3 linear part, then applies the inverse linear to the negated
/// translation column. Computed in `f64` so very-different-scale
/// matrices (e.g. `1e-3` cm-scale child of a `1e3` km-scale parent)
/// round-trip without precision collapse, then cast back to `f32`.
pub(crate) fn mat4_affine_inverse(m: [[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    for row in &m {
        for v in row {
            if !v.is_finite() {
                return None;
            }
        }
    }
    // Affinity guard: bottom row must be [0, 0, 0, 1] within a tight
    // tolerance (TRS-derived matrices satisfy this exactly).
    let bot_eps = 1e-6_f32;
    if m[3][0].abs() > bot_eps
        || m[3][1].abs() > bot_eps
        || m[3][2].abs() > bot_eps
        || (m[3][3] - 1.0).abs() > bot_eps
    {
        return None;
    }
    let a = m[0][0] as f64;
    let b = m[0][1] as f64;
    let c = m[0][2] as f64;
    let d = m[1][0] as f64;
    let e = m[1][1] as f64;
    let f = m[1][2] as f64;
    let g = m[2][0] as f64;
    let h = m[2][1] as f64;
    let i = m[2][2] as f64;
    // Cofactors of the 3x3 linear part.
    let c00 = e * i - f * h;
    let c01 = -(d * i - f * g);
    let c02 = d * h - e * g;
    let c10 = -(b * i - c * h);
    let c11 = a * i - c * g;
    let c12 = -(a * h - b * g);
    let c20 = b * f - c * e;
    let c21 = -(a * f - c * d);
    let c22 = a * e - b * d;
    let det = a * c00 + b * c01 + c * c02;
    if !det.is_finite() || det == 0.0 {
        return None;
    }
    let inv = 1.0 / det;
    // Adjugate-transpose: inv_linear[row][col] = cofactor[col][row] / det.
    let l00 = c00 * inv;
    let l01 = c10 * inv;
    let l02 = c20 * inv;
    let l10 = c01 * inv;
    let l11 = c11 * inv;
    let l12 = c21 * inv;
    let l20 = c02 * inv;
    let l21 = c12 * inv;
    let l22 = c22 * inv;
    // Translation: t_inv = -L^-1 * t.
    let tx = m[0][3] as f64;
    let ty = m[1][3] as f64;
    let tz = m[2][3] as f64;
    let ix = -(l00 * tx + l01 * ty + l02 * tz);
    let iy = -(l10 * tx + l11 * ty + l12 * tz);
    let iz = -(l20 * tx + l21 * ty + l22 * tz);
    // Finite-check on the assembled inverse — a near-singular det can
    // produce inf / NaN entries; reject so callers fall through to the
    // "skip this instance" branch.
    let out = [
        [l00 as f32, l01 as f32, l02 as f32, ix as f32],
        [l10 as f32, l11 as f32, l12 as f32, iy as f32],
        [l20 as f32, l21 as f32, l22 as f32, iz as f32],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for row in &out {
        for v in row {
            if !v.is_finite() {
                return None;
            }
        }
    }
    Some(out)
}

/// Transform an [`crate::ray::Ray`] into mesh-local space by an affine
/// 4x4 inverse. The translation column moves the origin; the 3x3
/// linear part rotates / scales the direction (but is not normalised —
/// the same ray parameter `t` resolves to the same world-space point
/// before and after the change of frame).
pub(crate) fn ray_into_local(world_inv: [[f32; 4]; 4], ray: crate::ray::Ray) -> crate::ray::Ray {
    let o = ray.origin;
    let d = ray.direction;
    let lo = [
        world_inv[0][0] * o[0] + world_inv[0][1] * o[1] + world_inv[0][2] * o[2] + world_inv[0][3],
        world_inv[1][0] * o[0] + world_inv[1][1] * o[1] + world_inv[1][2] * o[2] + world_inv[1][3],
        world_inv[2][0] * o[0] + world_inv[2][1] * o[1] + world_inv[2][2] * o[2] + world_inv[2][3],
    ];
    let ld = [
        world_inv[0][0] * d[0] + world_inv[0][1] * d[1] + world_inv[0][2] * d[2],
        world_inv[1][0] * d[0] + world_inv[1][1] * d[1] + world_inv[1][2] * d[2],
        world_inv[2][0] * d[0] + world_inv[2][1] * d[1] + world_inv[2][2] * d[2],
    ];
    crate::ray::Ray::new(lo, ld)
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

/// Closest-hit record produced by [`Scene3D::intersect_ray`].
///
/// Pairs a world-space ray query with the scene-graph location of
/// the hit:
///
/// * `node` — the [`NodeId`] of the reachable node whose attached
///   mesh produced the hit. Look up `nodes[node]` for the node's
///   transform / parenting; pass the same index into
///   [`Scene3D::world_node_transforms`]`[node.0 as usize]` for the
///   world matrix that maps mesh-local coordinates back to world
///   space.
/// * `primitive_index` — the index into `nodes[node].mesh`'s
///   `Mesh::primitives` array identifying which primitive within the
///   mesh was struck. The inner `hit.triangle_index` then names the
///   triangle inside *that* primitive's
///   [`crate::Primitive::triangle_indices`] enumeration.
/// * `hit` — the underlying [`crate::ray::RayHit`] in mesh-local
///   coordinates (barycentric, triangle index, front-face flag) but
///   with `t` already in world-space units: affine change-of-frame
///   leaves the ray-parameter scalar invariant, so the same `t`
///   reconstructs the world hit point via
///   `world_ray.point_at(scene_hit.hit.t)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneRayHit {
    pub node: NodeId,
    pub primitive_index: usize,
    pub hit: crate::ray::RayHit,
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

    /// World-space 4x4 transform per scene node, indexed by `NodeId.0`.
    ///
    /// Walks every root in [`Scene3D::roots`] in order, composing each
    /// node's local [`Transform`] (via [`Transform::to_matrix`]) onto
    /// its parent's already-composed world transform. The returned
    /// vector has length `nodes.len()`; each slot holds:
    ///
    /// * `Some([[f32; 4]; 4])` — the row-major column-vector world
    ///   transform of that node, i.e. the matrix that takes a position
    ///   in the node's local frame to world space (`p_world = M *
    ///   p_local`, treating `p_local` as `[x, y, z, 1]ᵀ`).
    /// * `None` — the node is not reachable from any root in
    ///   [`Scene3D::roots`] (detached). Detached nodes are common
    ///   during incremental scene construction; the caller can detect
    ///   them without a separate reachability pass.
    ///
    /// The walk is depth-first iterative on an explicit stack, matching
    /// [`Scene3D::bounding_box`]'s traversal. Re-entry through a cycle
    /// (a node listed as its own descendant) is guarded against — each
    /// node receives **exactly one** world transform, the first one
    /// encountered on the depth-first walk. Out-of-range `NodeId`
    /// entries in `roots` / `children` are silently skipped.
    ///
    /// A node referenced by two parents (shared-instance pattern) is
    /// visited only once, so `world_node_transforms()[id.0 as usize]`
    /// resolves to a single matrix — the one obtained via the first
    /// parent on the DFS path. Decoders that need per-instance world
    /// transforms (mesh-instancing) should keep an explicit
    /// instance-list side-channel rather than relying on this helper.
    ///
    /// **What this does NOT include:**
    ///
    /// * Skin pose deformation — the static scene-graph transform is
    ///   reported, not the skinned-pose transform at any particular
    ///   animation time. Apply animation channels separately to obtain
    ///   pose-time transforms.
    /// * Camera / projection transforms.
    /// * Up-axis or unit conversion. [`Scene3D::up_axis`] and
    ///   [`Scene3D::unit`] are metadata; the returned matrices live in
    ///   whatever coordinate system the scene stored.
    ///
    /// ## Use cases
    ///
    /// * Transform-aware aggregate metrics (multiply each primitive's
    ///   `surface_area` by `|det(scale_part)|` or its `signed_volume`
    ///   by `sign(det) * |det|` to obtain a transform-folded total —
    ///   the per-component scales fall out of the upper-left 3x3 of
    ///   the world matrix).
    /// * Renderer-side world-matrix prep (one DFS pass at scene load,
    ///   then constant-time lookup per node when issuing draw calls).
    /// * Authoring-tool node inspection ("show me the world position
    ///   of `nodes[7]`" without re-walking the ancestor chain).
    ///
    /// Cost: `O(nodes.len() + total_children)`; allocates one
    /// `Vec<Option<...>>` of length `nodes.len()` plus the DFS stack.
    pub fn world_node_transforms(&self) -> Vec<Option<[[f32; 4]; 4]>> {
        let n_nodes = self.nodes.len();
        let mut out: Vec<Option<[[f32; 4]; 4]>> = vec![None; n_nodes];
        if n_nodes == 0 {
            return out;
        }
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // Iterative depth-first walk; stack carries (node_id, ancestor_matrix).
        // Roots are pushed in order so that, after the LIFO pop order,
        // the leftmost root is visited first — matching `bounding_box`'s
        // determinism contract.
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().rev().map(|r| (*r, identity)).collect();
        while let Some((nid, parent)) = stack.pop() {
            let idx = nid.0 as usize;
            if idx >= n_nodes || out[idx].is_some() {
                continue;
            }
            let node = &self.nodes[idx];
            let world = mat4_mul(parent, node.transform.to_matrix());
            out[idx] = Some(world);
            // Walk children in reverse so leftmost child is popped first
            // (deterministic ordering for snapshot consumers).
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        out
    }

    /// World-space axis-aligned bounding box per scene node, indexed by
    /// `NodeId.0`.
    ///
    /// Walks the [`Scene3D::roots`] forest with the same depth-first
    /// shape as [`Scene3D::world_node_transforms`] / [`Scene3D::bounding_box`].
    /// For every reachable node that carries a mesh (`Node::mesh ==
    /// Some(id)`), the contained mesh's local AABB
    /// ([`crate::Mesh::bounding_box`]) is transformed through the
    /// node's full ancestor-chain world matrix via
    /// [`BoundingBox::transform`] (eight-corner refit). The returned
    /// vector has length `nodes.len()`; each slot holds:
    ///
    /// * `Some(BoundingBox)` — the world-space tight AABB of the
    ///   node's attached mesh, transformed by the node's world matrix.
    /// * `None` — the node is not reachable from any root, the node
    ///   carries no mesh, the referenced mesh is empty, or the mesh
    ///   reference is out of range. The four `None` reasons are not
    ///   distinguished; callers needing the distinction can pair this
    ///   with [`Scene3D::world_node_transforms`] (whose `None` slots
    ///   are reachability-only).
    ///
    /// The output is the per-instance complement to
    /// [`Scene3D::bounding_box`], which collapses every reachable
    /// instance into a single scene-wide union. Each slot here is the
    /// tight bound of one instance, fit around the eight transformed
    /// corners of its mesh's local AABB — orientation-aware (an
    /// arbitrary rotation widens the AABB to wrap the rotated content)
    /// the same way [`BoundingBox::transform`] documents.
    ///
    /// ## Use cases
    ///
    /// * **Per-instance frustum / view-volume culling.** A renderer
    ///   tests each slot's AABB against the view frustum before
    ///   issuing the instance's draw call.
    /// * **Scene-level ray AABB pre-pass.** A ray query walks the
    ///   slots once and only descends into
    ///   [`crate::Mesh::intersect_ray`] for instances whose AABB the
    ///   ray actually pierces (via [`BoundingBox::intersect_ray`]).
    ///   For triangle-budget-dominated scenes this reduces the ray
    ///   walk from `Σ triangle_count` to `Σ triangle_count over hit instances` —
    ///   the same kind of pruning [`crate::Bvh::intersect_ray`] applies
    ///   at the leaf level, lifted to the per-instance level.
    /// * **BVH-of-instances seed.** A future scene-level BVH builder
    ///   feeds each slot's AABB + the slot index (a `NodeId`) into
    ///   the same median-split AABB-tree construction
    ///   [`crate::Bvh::build`] already runs per primitive — see
    ///   the round-210 docs gesture toward this layered acceleration.
    ///
    /// ## What this does NOT include
    ///
    /// * Skin pose deformation — the rest-pose vertices are used
    ///   verbatim. A rigged mesh reports the rest-pose extent, not
    ///   the skinned-pose extent at any particular animation time.
    /// * Morph targets — only base
    ///   [`crate::Primitive::positions`] are folded into each mesh's
    ///   local AABB.
    /// * Animation channels — the static scene-graph transform is
    ///   reported, not the post-animation transform.
    /// * Up-axis or unit conversion. [`Scene3D::up_axis`] and
    ///   [`Scene3D::unit`] are metadata; the returned boxes live in
    ///   whatever coordinate system the scene stored.
    ///
    /// ## Determinism + cycle contract
    ///
    /// The walk is depth-first iterative on an explicit stack, with
    /// roots visited in `roots`-order and children in source order —
    /// identical to [`Scene3D::world_node_transforms`]. A node listed
    /// as its own descendant (cycle) is visited once via the
    /// first-arrival DFS path; out-of-range `NodeId` entries in
    /// `roots` / `children` are silently skipped. A node referenced
    /// by two parents (shared-instance) resolves to the first
    /// parent's chain — per-instance world AABBs for the
    /// shared-instance pattern need an explicit instance-list
    /// side-channel.
    ///
    /// Cost: `O(nodes.len() + total_children + Σ mesh_vertex_count_for_reachable_nodes)`,
    /// where the per-mesh cost is the one
    /// [`crate::Mesh::bounding_box`] iteration. Allocates one
    /// `Vec<Option<BoundingBox>>` of length `nodes.len()` plus the
    /// DFS stack.
    pub fn world_node_bounds(&self) -> Vec<Option<BoundingBox>> {
        let n_nodes = self.nodes.len();
        let mut out: Vec<Option<BoundingBox>> = vec![None; n_nodes];
        if n_nodes == 0 {
            return out;
        }
        let n_meshes = self.meshes.len();
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // `visited` is keyed on reachability so a node-with-no-mesh
        // (returning `None` in `out`) is still detected as visited and
        // not re-walked through a cycle.
        let mut visited = vec![false; n_nodes];
        // Iterative depth-first walk; stack carries (node_id, ancestor_matrix).
        // Roots pushed in reverse so the LIFO pop visits the leftmost
        // root first — matching `world_node_transforms`'s ordering.
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().rev().map(|r| (*r, identity)).collect();
        while let Some((nid, parent)) = stack.pop() {
            let idx = nid.0 as usize;
            if idx >= n_nodes || visited[idx] {
                continue;
            }
            visited[idx] = true;
            let node = &self.nodes[idx];
            let world = mat4_mul(parent, node.transform.to_matrix());
            if let Some(m) = node.mesh {
                if n_meshes > 0 {
                    if let Some(mesh) = self.meshes.get(m.0 as usize) {
                        if let Some(local) = mesh.bounding_box() {
                            out[idx] = Some(local.transform(world));
                        }
                    }
                }
            }
            // Walk children in reverse so leftmost child is popped first
            // (deterministic ordering for snapshot consumers).
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        out
    }

    /// Convenience wrapper around [`crate::InstanceBvh::build`].
    ///
    /// Builds a scene-level bounding-volume hierarchy over every
    /// reachable node-mesh instance — the next acceleration layer
    /// above [`crate::Bvh::intersect_ray`] / per-instance
    /// [`Mesh::intersect_ray`]. The same per-instance walk
    /// [`Scene3D::intersect_ray`] performs becomes a
    /// `O(log reachable_instance_count)` median-split traversal once
    /// the tree is built. Cache the build alongside the scene; rebuild
    /// when any node transform or mesh AABB changes.
    pub fn build_instance_bvh(&self) -> Option<crate::InstanceBvh> {
        crate::InstanceBvh::build(self)
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

    /// Sum of every mesh primitive's [`Primitive::surface_area`] in the
    /// scene's local unit-squared (matching [`Scene3D::unit`]). This
    /// does *not* apply node transforms — primitives instanced by
    /// multiple nodes contribute their unscaled area once per mesh,
    /// not once per node. For a transform-aware total, walk
    /// [`Scene3D::world_node_transforms`] and apply the per-node
    /// scale's determinant per primitive instance.
    pub fn surface_area(&self) -> f64 {
        self.meshes.iter().map(|m| m.surface_area()).sum()
    }

    /// Area-weighted surface centroid across every mesh in the scene
    /// — the area-weighted combination of every mesh's own
    /// [`crate::Mesh::surface_centroid`]. This walks meshes once, not
    /// node instances: a mesh instanced by multiple reachable nodes
    /// contributes its centroid (with its area as the weight) once,
    /// not once per node. For a transform-aware, per-instance
    /// centroid total, walk [`Scene3D::world_node_transforms`]
    /// alongside [`crate::Primitive::surface_centroid`] and combine
    /// the per-instance centroids with their post-transform areas
    /// (`|(M_3·E1) × (M_3·E2)|/2`) as weights — the local centroid
    /// transforms by `world * [c, 1]` so each per-instance numerator
    /// is `(world * centroid) * post_transform_area`.
    ///
    /// Returns `None` when every contained mesh returns `None` (or
    /// the scene holds zero meshes). Coordinates are in the scene's
    /// local frame ([`Scene3D::unit`]); contract matches
    /// [`crate::Primitive::surface_centroid`] for finiteness,
    /// degenerate-skipping, and out-of-range / NaN handling.
    pub fn surface_centroid(&self) -> Option<[f64; 3]> {
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_area = 0.0_f64;
        for m in &self.meshes {
            let area = m.surface_area();
            if area == 0.0 || !area.is_finite() {
                continue;
            }
            if let Some(c) = m.surface_centroid() {
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

    /// Sum of every mesh primitive's
    /// [`crate::Primitive::signed_volume`] in the scene's local
    /// unit-cubed (matching [`Scene3D::unit`]). This does *not* apply
    /// node transforms — primitives instanced by multiple nodes
    /// contribute their unscaled volume once per mesh, not once per
    /// node. For a transform-aware total, walk
    /// [`Scene3D::world_node_transforms`] and apply the per-node
    /// scale's signed determinant per primitive instance (a negative
    /// scale flips winding and so flips the sign of the enclosed
    /// volume).
    ///
    /// **Only physically meaningful when each contained mesh is a
    /// closed two-manifold surface.** See
    /// [`crate::Primitive::is_closed_manifold`] /
    /// [`crate::Primitive::edge_manifold_report`].
    pub fn signed_volume(&self) -> f64 {
        self.meshes.iter().map(|m| m.signed_volume()).sum()
    }

    /// Unsigned `|signed_volume()|` across the scene. Same
    /// shell-cancellation caveat as [`crate::Mesh::volume`]: this is
    /// `|Σ signed|`, not `Σ |signed|`. For a multi-shell scene where
    /// individual shells may differ in sign, prefer summing each mesh's
    /// [`crate::Mesh::volume`] separately.
    pub fn volume(&self) -> f64 {
        self.signed_volume().abs()
    }

    /// Volume-weighted centroid (centre of mass) across every mesh in
    /// the scene — the signed-volume-weighted combination of every
    /// mesh's own [`crate::Mesh::volume_centroid`]. This walks meshes
    /// once, not node instances: a mesh instanced by multiple reachable
    /// nodes contributes its centroid (with its signed volume as the
    /// weight) once, not once per node. For a transform-aware
    /// per-instance centroid total, walk
    /// [`Scene3D::world_node_transforms`] alongside
    /// [`crate::Primitive::volume_centroid`] and combine the
    /// per-instance centroids with their post-transform signed volumes
    /// (`det(M_3x3) · V_local` for each closed-mesh instance) as
    /// weights.
    ///
    /// **Only physically meaningful when each contained mesh is a
    /// closed two-manifold surface.** See
    /// [`crate::Primitive::is_closed_manifold`] /
    /// [`crate::Primitive::edge_manifold_report`]. An open patch (a
    /// hemisphere, a plane) gives an answer that depends on where the
    /// origin sits because the surface-cancellation argument no longer
    /// applies; for those callers should use
    /// [`Scene3D::surface_centroid`].
    ///
    /// Returns `None` when every contained mesh returns `None` (or the
    /// scene holds zero meshes), or when the accumulated signed volume
    /// is `0.0` / non-finite (a flat sheet, or perfectly cancelling
    /// inside-out shells). Coordinates are in the scene's local frame
    /// ([`Scene3D::unit`]); contract matches
    /// [`crate::Primitive::volume_centroid`] for finiteness,
    /// degenerate-skipping, and out-of-range / NaN handling.
    pub fn volume_centroid(&self) -> Option<[f64; 3]> {
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_v = 0.0_f64;
        for m in &self.meshes {
            let v = m.signed_volume();
            if v == 0.0 || !v.is_finite() {
                continue;
            }
            if let Some(c) = m.volume_centroid() {
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

    /// Transform-aware total surface area across every node-instantiated
    /// mesh in the scene, in world units squared (matching
    /// [`Scene3D::unit`]² when the scene's root has identity transform).
    ///
    /// Whereas [`Scene3D::surface_area`] sums each *mesh resource* once
    /// regardless of how many nodes carry it (the geometric-content
    /// total), `world_surface_area` walks the [`Scene3D::roots`] forest
    /// the same way [`Scene3D::bounding_box`] does, applies each
    /// reachable node's full ancestor-chain world matrix to its
    /// primitive's triangle vertices, and sums the post-transform
    /// triangle areas. A mesh instanced under two nodes therefore
    /// contributes twice (once per instance), and each instance's
    /// contribution reflects the world-space scale (and any
    /// non-uniform skew) on the path to that node.
    ///
    /// # Derivation
    ///
    /// For a triangle `(P_a, P_b, P_c)` mapped through the affine world
    /// matrix `M`, the post-transform edge vectors are
    /// `M_3·(P_b - P_a)` and `M_3·(P_c - P_a)` (the translation row
    /// cancels in the difference; `M_3` is the upper-left 3x3). The
    /// transformed triangle's area is
    ///
    /// ```text
    /// A_world = |(M_3·E1) × (M_3·E2)| / 2.
    /// ```
    ///
    /// Under a uniform scale `s` the factor collapses to `s²`. Under a
    /// non-uniform diagonal scale `(sx, sy, sz)` the factor depends on
    /// the triangle's facing axis, so per-triangle evaluation — rather
    /// than a single det-based scale — is required for correctness.
    /// The translation column of `M` does not enter the area
    /// computation, so the result is translation-invariant per
    /// triangle (as expected for an intrinsic area metric).
    ///
    /// # Contract
    ///
    /// * Topology handling, degenerate-triangle skipping, NaN-guarding,
    ///   and out-of-range-index skipping all mirror
    ///   [`crate::Primitive::surface_area`]. Non-triangle topologies
    ///   contribute 0.0. Result is finite and non-negative for any
    ///   finite input.
    /// * Mesh resources not reachable from any [`Scene3D::roots`] node
    ///   contribute 0.0 — the count is per-instance over the
    ///   scene-graph, not per-resource. For a resource-level total see
    ///   [`Scene3D::surface_area`].
    /// * Cycles in the scene-graph are guarded the same way as
    ///   [`Scene3D::bounding_box`] / [`Scene3D::world_node_transforms`]:
    ///   each node is visited at most once. A node instanced under two
    ///   parents resolves to one world matrix (the first parent on the
    ///   DFS path); use an explicit instance side-table if your decoder
    ///   needs both.
    /// * Skin pose deformation, morph targets, and unit-axis conversion
    ///   are *not* applied — the static scene-graph transform is the
    ///   only thing folded in. For a pose-time area, apply the
    ///   animation pose before calling.
    /// * Cost `O(reachable_nodes + Σ triangle_count_per_reachable_mesh)`.
    ///   Allocates the DFS stack only; the per-triangle math is in
    ///   `f64` to avoid `f32` drift on dense meshes.
    pub fn world_surface_area(&self) -> f64 {
        let n_nodes = self.nodes.len();
        let n_meshes = self.meshes.len();
        if n_nodes == 0 || n_meshes == 0 {
            return 0.0;
        }
        let mut visited = vec![false; n_nodes];
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut total = 0.0_f64;
        // Push roots in reverse so the LIFO pop visits the leftmost
        // root first — matching `world_node_transforms`'s documented
        // single-resolution policy (a shared instance reachable from
        // two parents resolves via the first parent on the DFS path).
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().rev().map(|r| (*r, identity)).collect();
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
                    for prim in &mesh.primitives {
                        total += prim.world_surface_area(world);
                    }
                }
            }
            // Walk children in reverse so leftmost child is popped first.
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        total
    }

    /// Transform-aware total signed volume across every
    /// node-instantiated mesh, in world units cubed.
    ///
    /// Whereas [`Scene3D::signed_volume`] sums each *mesh resource* once
    /// in its local frame, `world_signed_volume` walks the
    /// [`Scene3D::roots`] forest, applies each reachable node's
    /// world-space transform to the underlying primitives, and
    /// accumulates the per-instance signed enclosed volume.
    ///
    /// # Derivation
    ///
    /// For a primitive with local signed volume
    /// `V_local = (1/6) Σ P_a · (P_b × P_c)` and an affine world
    /// transform `M` whose upper-left 3x3 is `M_3` with translation
    /// column `t`, every transformed corner is `M_3·P + t`. Expanding
    /// the per-triangle scalar triple product:
    ///
    /// ```text
    /// (M_3·P_a + t) · ((M_3·P_b + t) × (M_3·P_c + t))
    ///   = det(M_3) · (P_a · (P_b × P_c)) + boundary_terms(t).
    /// ```
    ///
    /// The `boundary_terms(t)` involve only the open-mesh boundary and
    /// vanish for a closed two-manifold (the same origin-cancellation
    /// that makes the local signed volume translation-invariant). For
    /// such a mesh the world signed volume reduces to
    ///
    /// ```text
    /// V_world = det(M_3) · V_local.
    /// ```
    ///
    /// `det(M_3)` is the *signed* 3x3 determinant: a uniform scale of
    /// `s` gives `s³`; a single-axis mirror (`-1` on one axis) gives
    /// `-1`, correctly flipping the enclosed-volume sign because the
    /// triangle winding flips with the mirror. For an open mesh, the
    /// translation-dependent boundary term means this scaling identity
    /// is only an approximation; the helper still returns the
    /// closed-form `det(M_3) · V_local` because that is the
    /// physically-meaningful summand whenever the per-instance mesh is
    /// itself a closed surface (the usual case for which the
    /// volume reduction is defined).
    ///
    /// # Contract
    ///
    /// * Reachability, cycle-guarding, and per-instance accumulation
    ///   match [`Scene3D::world_surface_area`].
    /// * Each node's world matrix is reduced to its upper-left 3x3
    ///   determinant; non-finite determinants (matrix corruption,
    ///   inf/NaN entries) skip the contribution.
    /// * Each mesh resource contributes once per reachable node that
    ///   references it. A two-node instance with mirrored scale
    ///   `[-1, 1, 1]` and an unmirrored sibling cancel each other in
    ///   the signed sum — that is the geometric truth.
    /// * Skin pose, morph targets, and unit-axis conversion are not
    ///   applied.
    /// * Returns `0.0` for an empty scene or one with no
    ///   reachable meshes.
    /// * Result is finite for any finite input; the accumulator is
    ///   `f64`.
    /// * Cost `O(reachable_nodes + Σ triangle_count_per_reachable_mesh)`.
    pub fn world_signed_volume(&self) -> f64 {
        let n_nodes = self.nodes.len();
        let n_meshes = self.meshes.len();
        if n_nodes == 0 || n_meshes == 0 {
            return 0.0;
        }
        let mut visited = vec![false; n_nodes];
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut total = 0.0_f64;
        // Push roots in reverse so the LIFO pop visits the leftmost
        // root first — matching `world_node_transforms`'s
        // single-resolution policy.
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().rev().map(|r| (*r, identity)).collect();
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
                    let det = mat3_det_of_world(world);
                    if det.is_finite() {
                        let local = mesh.signed_volume();
                        let scaled = det * local;
                        if scaled.is_finite() {
                            total += scaled;
                        }
                    }
                }
            }
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        total
    }

    /// Unsigned `|world_signed_volume()|` across the scene.
    ///
    /// Same shell-cancellation caveat as
    /// [`Scene3D::volume`] / [`crate::Mesh::volume`]: this is
    /// `|Σ signed_world|`, not `Σ |signed_world|`. For a scene where
    /// instances may carry mirrored scales (producing per-instance
    /// negative signed volumes), prefer summing each instance's
    /// `|det(M_3) · signed_volume|` separately.
    pub fn world_volume(&self) -> f64 {
        self.world_signed_volume().abs()
    }

    /// Transform-aware area-weighted surface centroid across every
    /// node-instantiated mesh in the scene, in world units.
    ///
    /// Whereas [`Scene3D::surface_centroid`] recombines each *mesh
    /// resource* once regardless of how many nodes carry it,
    /// `world_surface_centroid` walks the [`Scene3D::roots`] forest
    /// the same way [`Scene3D::world_surface_area`] does, applies each
    /// reachable node's full ancestor-chain world matrix to its
    /// primitive's triangle vertices, and recombines the post-
    /// transform per-instance centroids weighted by the per-instance
    /// post-transform surface area. A mesh instanced under two nodes
    /// therefore contributes twice (once per instance), and each
    /// instance's contribution reflects the world-space scale and skew
    /// on the path to that node.
    ///
    /// # Derivation
    ///
    /// Picking up where [`Scene3D::world_surface_area`] leaves off:
    /// for a triangle `(P_a, P_b, P_c)` mapped through the affine
    /// world matrix `M`, the post-transform centroid is `(M·P_a +
    /// M·P_b + M·P_c) / 3` and the post-transform area is
    /// `|(M_3·E1) × (M_3·E2)| / 2`. Substituting into the continuous
    /// identity `C = (Σ area_i · centroid_i) / Σ area_i` and
    /// accumulating across every reachable node's every primitive
    /// gives the world-frame centroid. The recombination across
    /// primitives (and across instances) is additivity of the surface
    /// integral over a union of patches — the same reasoning that
    /// makes [`Mesh::surface_centroid`] / [`Scene3D::surface_centroid`]
    /// well-defined.
    ///
    /// # Contract
    ///
    /// * Topology handling, degenerate-triangle skipping, NaN guards,
    ///   and out-of-range-index skipping all mirror
    ///   [`crate::Primitive::world_surface_centroid`].
    /// * Mesh resources not reachable from any [`Scene3D::roots`] node
    ///   contribute nothing — the count is per-instance over the
    ///   scene-graph, not per-resource. For a resource-level total see
    ///   [`Scene3D::surface_centroid`].
    /// * Cycles in the scene-graph are guarded the same way as
    ///   [`Scene3D::bounding_box`] / [`Scene3D::world_node_transforms`]
    ///   / [`Scene3D::world_surface_area`]: each node is visited at
    ///   most once. A node instanced under two parents resolves to one
    ///   world matrix (the first parent on the DFS path).
    /// * Returns `None` when no reachable triangle survives — empty
    ///   scene, no reachable mesh, every reachable mesh degenerate, or
    ///   every world transform collapsing the surface to zero area
    ///   under the transform.
    /// * Skin pose deformation, morph targets, and unit-axis conversion
    ///   are *not* applied — the static scene-graph transform is the
    ///   only thing folded in. For a pose-time centroid, apply the
    ///   animation pose before calling.
    /// * Cost `O(reachable_nodes + Σ triangle_count_per_reachable_mesh)`.
    ///   Allocates the DFS stack only; per-triangle math is in `f64` to
    ///   avoid `f32` drift on dense meshes.
    pub fn world_surface_centroid(&self) -> Option<[f64; 3]> {
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
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_area = 0.0_f64;
        // Push roots in reverse so the LIFO pop visits the leftmost
        // root first — matching `world_node_transforms`'s documented
        // single-resolution policy (a shared instance reachable from
        // two parents resolves via the first parent on the DFS path).
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().rev().map(|r| (*r, identity)).collect();
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
                    for prim in &mesh.primitives {
                        let area = prim.world_surface_area(world);
                        if area == 0.0 || !area.is_finite() {
                            continue;
                        }
                        if let Some(c) = prim.world_surface_centroid(world) {
                            sum_x += c[0] * area;
                            sum_y += c[1] * area;
                            sum_z += c[2] * area;
                            sum_area += area;
                        }
                    }
                }
            }
            // Walk children in reverse so leftmost child is popped first.
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        if sum_area == 0.0 || !sum_area.is_finite() {
            return None;
        }
        let inv = 1.0 / sum_area;
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Transform-aware volume-weighted centroid (centre of mass) across
    /// every node-instantiated mesh in the scene, in world units.
    ///
    /// Whereas [`Scene3D::volume_centroid`] recombines each *mesh
    /// resource* once regardless of how many nodes carry it,
    /// `world_volume_centroid` walks the [`Scene3D::roots`] forest the
    /// same way [`Scene3D::world_signed_volume`] /
    /// [`Scene3D::world_surface_centroid`] do, applies each reachable
    /// node's full ancestor-chain world matrix to its primitive's
    /// triangle vertices, and recombines the post-transform per-instance
    /// centroids weighted by the per-instance post-transform signed
    /// volume. A mesh instanced under two nodes therefore contributes
    /// twice (once per instance), and each instance's contribution
    /// reflects the world-space scale, skew, *and* translation on the
    /// path to that node — unlike the surface variants, the per-
    /// instance volume integral picks up the translation column too
    /// (the origin-anchored tet sum is not translation-invariant).
    ///
    /// # Derivation
    ///
    /// Picking up where [`Scene3D::world_volume`] leaves off: for a
    /// closed mesh under affine `M = [M_3 | t]`, the per-instance
    /// signed volume is `det(M_3) · V_local` and the per-instance
    /// centroid is `M · C_local = M_3 · C_local + t`. The
    /// signed-volume-weighted recombination across instances is then
    /// additivity of the volume integral over a union of solid bodies
    /// — the same reasoning that fixes [`Mesh::volume_centroid`] /
    /// [`Scene3D::volume_centroid`] in the local frame. For an open
    /// patch the recombination still goes through, but the per-
    /// instance signed volume is no longer `det(M_3) · V_local` — the
    /// origin-anchored tet sum picks up a translation-dependent
    /// boundary term — so the helper computes both the per-primitive
    /// centroid and signed volume in the transformed frame
    /// independently and feeds them through the
    /// `Σ V_i · C_i / Σ V_i` recombination directly.
    ///
    /// # Contract
    ///
    /// * Topology handling, degenerate / NaN guards, and out-of-range-
    ///   index skipping all mirror
    ///   [`crate::Primitive::world_volume_centroid`].
    /// * Mesh resources not reachable from any [`Scene3D::roots`] node
    ///   contribute nothing — the count is per-instance over the
    ///   scene-graph, not per-resource. For a resource-level total see
    ///   [`Scene3D::volume_centroid`].
    /// * Cycles in the scene-graph are guarded the same way as
    ///   [`Scene3D::bounding_box`] / [`Scene3D::world_node_transforms`]
    ///   / [`Scene3D::world_surface_centroid`]: each node is visited at
    ///   most once. A node instanced under two parents resolves to one
    ///   world matrix (the first parent on the DFS path).
    /// * Returns `None` when no reachable instance contributes —
    ///   empty scene, no reachable mesh, every reachable mesh
    ///   non-triangle / degenerate, or every world transform
    ///   collapsing every tet to zero signed volume.
    /// * Skin pose deformation, morph targets, and unit-axis conversion
    ///   are *not* applied — the static scene-graph transform is the
    ///   only thing folded in. For a pose-time centroid, apply the
    ///   animation pose before calling.
    /// * Only physically meaningful when each reachable mesh is a
    ///   closed two-manifold surface (see
    ///   [`crate::Primitive::edge_manifold_report`]). For an open patch
    ///   the result depends on where the origin sits in the
    ///   transformed frame — same caveat as
    ///   [`crate::Primitive::world_volume_centroid`].
    /// * Cost `O(reachable_nodes + Σ triangle_count_per_reachable_mesh)`.
    ///   Allocates the DFS stack only; per-triangle math is in `f64`.
    pub fn world_volume_centroid(&self) -> Option<[f64; 3]> {
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
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_v = 0.0_f64;
        // Push roots in reverse so the LIFO pop visits the leftmost
        // root first — matching `world_node_transforms`'s documented
        // single-resolution policy (a shared instance reachable from
        // two parents resolves via the first parent on the DFS path).
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().rev().map(|r| (*r, identity)).collect();
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
                    for prim in &mesh.primitives {
                        let v = prim.world_signed_volume(world);
                        if v == 0.0 || !v.is_finite() {
                            continue;
                        }
                        if let Some(c) = prim.world_volume_centroid(world) {
                            sum_x += c[0] * v;
                            sum_y += c[1] * v;
                            sum_z += c[2] * v;
                            sum_v += v;
                        }
                    }
                }
            }
            // Walk children in reverse so leftmost child is popped first.
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        if sum_v == 0.0 || !sum_v.is_finite() {
            return None;
        }
        let inv = 1.0 / sum_v;
        Some([sum_x * inv, sum_y * inv, sum_z * inv])
    }

    /// Closest-hit ray query across every reachable node-mesh
    /// instance in world space.
    ///
    /// Walks the [`Scene3D::roots`] forest with the same DFS shape as
    /// [`Scene3D::world_node_transforms`] / [`Scene3D::world_surface_area`].
    /// At each reachable node carrying a mesh, the world ray is
    /// transformed into the mesh's local frame via the inverse of the
    /// node's world matrix, [`crate::Mesh::intersect_ray`] runs in that
    /// frame, and the returned ray-parameter `t` is reported back
    /// verbatim — affine change-of-frame leaves the `t` value
    /// invariant (`P_world = M · P_local = M · (O_local + t · D_local) =
    /// O_world + t · D_world`).
    ///
    /// Each hit shrinks the search bound (`t_max`) before the next
    /// node is tested, so a scene with many instances pays the
    /// per-instance test only until the closest hit is fixed; later
    /// instances behind that hit do triangle-level work only if their
    /// transformed bound still satisfies the surviving `t_max`. That
    /// pruning matches the per-primitive shrinking inside
    /// [`crate::Mesh::intersect_ray`] and the
    /// per-leaf shrinking inside [`crate::Bvh::intersect_ray`].
    ///
    /// Returns `None` when the scene has no reachable mesh node, or
    /// when no triangle on any reachable mesh is struck within
    /// `t_max`.
    ///
    /// # Returned hit
    ///
    /// The [`SceneRayHit`] carries the `NodeId` that produced the hit,
    /// the primitive index within that node's mesh, and the
    /// mesh-local [`crate::ray::RayHit`] (barycentric, triangle index,
    /// front-face flag, and the world-space `t`). The triangle index
    /// indexes [`crate::Primitive::triangle_indices`] of the named
    /// primitive — callers needing world-space corner positions
    /// look up the local positions, then push them through
    /// [`Scene3D::world_node_transforms`]`[node]`.
    ///
    /// # Cycle / reachability contract
    ///
    /// Each reachable node is visited at most once; a node listed as
    /// its own descendant resolves only via the first DFS arrival
    /// (same convention as [`Scene3D::world_node_transforms`]).
    /// Detached mesh resources (not referenced from any root-reachable
    /// node) are not queried; the caller drives those directly through
    /// [`crate::Mesh::intersect_ray`] if needed.
    ///
    /// # Singular instance transforms
    ///
    /// If a node's world matrix is non-affine, contains non-finite
    /// entries, or has a singular linear part (zero determinant —
    /// e.g. a degenerate scale collapsing one axis to zero), that
    /// instance is silently skipped. The surrounding scene still
    /// produces hits where it can. The skip is the geometrically
    /// honest answer — a degenerate transform projects the mesh onto
    /// a sub-plane / sub-line whose ray intersection is undefined
    /// without a regularised limit.
    ///
    /// # Cost
    ///
    /// `O(reachable_nodes + Σ instance_triangle_tests)`. For ray
    /// budgets dominated by triangle-level work, pair this with a
    /// per-primitive [`crate::Bvh`] cached on each instance for the
    /// `O(log triangle_count)` per ray narrowing — see the
    /// `Bvh::build` builder. Scene-level BVH-of-instances is a
    /// candidate for a later round; the current walk is the
    /// reference brute-force baseline.
    ///
    /// # Degenerate ray
    ///
    /// A zero-direction or non-finite ray reaches
    /// [`crate::ray::intersect_triangle`] / [`crate::ray::intersect_aabb`]
    /// unchanged after the local-frame transform; both helpers reject
    /// such inputs with `None` (the slab test's `1/0` produces `Inf`,
    /// the cross-product `det` collapses to zero or `NaN`, and the
    /// existing finite-check guards short-circuit the test).
    pub fn intersect_ray(&self, ray: crate::ray::Ray, t_max: f32) -> Option<SceneRayHit> {
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
        let mut best: Option<SceneRayHit> = None;
        let mut best_t = t_max;
        // Push roots in reverse so the LIFO pop visits the leftmost
        // root first — matching world_node_transforms's deterministic
        // ordering. The deterministic walk order matters when two
        // instances tie on `t` exactly (e.g. two coincident mirrored
        // copies); the leftmost-first convention picks the same
        // winner across runs.
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().rev().map(|r| (*r, identity)).collect();
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
                    if let Some(world_inv) = mat4_affine_inverse(world) {
                        let local_ray = ray_into_local(world_inv, ray);
                        if let Some((prim_idx, hit)) = mesh.intersect_ray(local_ray, best_t) {
                            // hit.t is in the local-frame ray
                            // parameter, which equals the world-frame
                            // ray parameter (affine change of frame is
                            // parameter-preserving). Shrink best_t.
                            // A later instance whose hit ties exactly
                            // (`hit.t == best_t`) is not allowed to
                            // override the existing winner — the
                            // earlier-visited (leftmost-first DFS)
                            // instance is the deterministic winner.
                            if best.is_none() || hit.t < best_t {
                                best_t = hit.t;
                                best = Some(SceneRayHit {
                                    node: nid,
                                    primitive_index: prim_idx,
                                    hit,
                                });
                            }
                        }
                    }
                }
            }
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        best
    }

    /// Any-hit (shadow-ray) world-space query over the same reachable
    /// node-mesh instances as [`Scene3D::intersect_ray`].
    ///
    /// Returns `true` as soon as **any** reachable node-mesh instance
    /// reports a hit within `t_max` for the ray, transformed into
    /// each instance's local frame the same way
    /// [`Scene3D::intersect_ray`] does. Returns `false` only after
    /// exhausting every reachable instance without a hit.
    ///
    /// Used for shadow rays / occlusion queries: the caller needs to
    /// know whether *something* blocks the segment from the surface
    /// hit point to the light, not which thing or where. The
    /// short-circuit lets the walk skip the rest of the scene as soon
    /// as the answer is decided.
    ///
    /// Reachability, cycle-guarding, singular-transform skipping, and
    /// degenerate-ray handling match [`Scene3D::intersect_ray`].
    ///
    /// # Determinism
    ///
    /// The walk visits instances in the same DFS order as
    /// [`Scene3D::intersect_ray`], but the answer (`true` / `false`)
    /// does not depend on visit order — the existence of a blocker
    /// is order-invariant. Visit order only changes which instance
    /// is the *first* blocker discovered, never the return value.
    pub fn any_ray_intersection(&self, ray: crate::ray::Ray, t_max: f32) -> bool {
        let n_nodes = self.nodes.len();
        let n_meshes = self.meshes.len();
        if n_nodes == 0 || n_meshes == 0 {
            return false;
        }
        let mut visited = vec![false; n_nodes];
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().rev().map(|r| (*r, identity)).collect();
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
                    if let Some(world_inv) = mat4_affine_inverse(world) {
                        let local_ray = ray_into_local(world_inv, ray);
                        // We reuse the closest-hit primitive walk —
                        // it has the same finite-time termination
                        // contract and short-circuits at the first
                        // primitive hit within `t_max`. A dedicated
                        // any-hit `Mesh::any_ray_intersection`
                        // wouldn't change the answer; the closest-hit
                        // walk still examines every primitive in
                        // `mesh` because each primitive's hit might
                        // be closer than the last, but it returns
                        // `Some(_)` whenever any does.
                        if mesh.intersect_ray(local_ray, t_max).is_some() {
                            return true;
                        }
                    }
                }
            }
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        false
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
