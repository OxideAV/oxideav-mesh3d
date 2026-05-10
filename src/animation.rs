//! Keyframed property animation.
//!
//! Modelled on glTF 2.0 §3.6 animation: each [`Animation`] is a bag
//! of [`AnimationChannel`]s. A channel binds an [`AnimationSampler`]
//! (time → value series) to one [`AnimationTarget`] (a node + which
//! property of that node — translation, rotation, scale, or morph
//! weights). All keyframe times are seconds from `t = 0`.

use crate::scene::NodeId;

/// Property of a node that an animation channel drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
    /// Per-morph-target weight vector. Sample length must equal the
    /// number of morph targets on the bound mesh.
    MorphWeights,
}

/// What a channel writes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnimationTarget {
    pub node: NodeId,
    pub property: AnimationProperty,
}

/// How values between keyframes are interpolated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// Hold the previous keyframe's value until the next one.
    Step,
    /// Linear interpolation. For quaternions, use NLERP (or SLERP if
    /// the renderer prefers); the typed model carries the raw
    /// keyframe values either way.
    Linear,
    /// glTF cubic-spline: each keyframe stores `(in_tangent, value, out_tangent)`,
    /// so the values vector is 3x as long as the keyframes vector.
    /// The format crate is responsible for laying that triple out.
    CubicSpline,
}

/// Time-aligned value series for one channel.
#[derive(Clone, Debug)]
pub struct AnimationSampler {
    /// Keyframe times in seconds — strictly increasing.
    pub keyframes: Vec<f32>,
    pub values: AnimationValues,
    pub interpolation: Interpolation,
}

/// Type of each keyframe value. The variant must match the bound
/// [`AnimationProperty`]: `Translation`/`Scale` → [`Vec3`](AnimationValues::Vec3),
/// `Rotation` → [`Quat`](AnimationValues::Quat), `MorphWeights` →
/// [`Scalar`](AnimationValues::Scalar) (concatenated weight vectors).
#[derive(Clone, Debug, PartialEq)]
pub enum AnimationValues {
    Vec3(Vec<[f32; 3]>),
    Quat(Vec<[f32; 4]>),
    Scalar(Vec<f32>),
}

impl AnimationValues {
    /// Number of stored values, regardless of variant arity.
    pub fn len(&self) -> usize {
        match self {
            Self::Vec3(v) => v.len(),
            Self::Quat(v) => v.len(),
            Self::Scalar(v) => v.len(),
        }
    }

    /// `true` if no values are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One animated property — `(target, sampler)`.
#[derive(Clone, Debug)]
pub struct AnimationChannel {
    pub target: AnimationTarget,
    pub sampler: AnimationSampler,
}

/// Named animation — a bag of channels played back together.
#[derive(Clone, Debug, Default)]
pub struct Animation {
    pub name: Option<String>,
    pub channels: Vec<AnimationChannel>,
}

impl Animation {
    /// Empty animation with the given name.
    pub fn new(name: impl Into<Option<String>>) -> Self {
        Self {
            name: name.into(),
            channels: Vec::new(),
        }
    }
}
