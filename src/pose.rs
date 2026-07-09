//! Sampling an [`Animation`] into a scene pose.
//!
//! An [`Animation`] is a bag of channels, each driving one node
//! property through time. A [`Pose`] is the *evaluated* snapshot at a
//! single timestamp `t`: per node, the sampled translation / rotation
//! / scale overrides plus the sampled morph-weight vector — exactly
//! the data a renderer (or a baking exporter) needs to reposition the
//! scene graph for one frame.
//!
//! The pipeline:
//!
//! ```text
//! Animation::sample_pose(t)          →  Pose (sparse TRS + weights)
//! Scene3D::posed_node_transforms(&pose) →  posed world matrices
//! Scene3D::joint_matrices_with / world_mesh_with →  deformed geometry
//! ```
//!
//! Sampler semantics are those of [`AnimationSampler::sample`] (glTF
//! 2.0 Appendix C): clamp outside the keyframe range, Step / Linear
//! (SLERP for rotations) / cubic-spline Hermite in between. On top of
//! that, `sample_pose` **renormalises rotation quaternions** — the
//! spec's Appendix C.5 implementation note requires the result of
//! cubic-spline rotation interpolation to be normalised before use,
//! and the typed sampler deliberately returns the raw blend.
//!
//! A property an animation does not drive keeps the node's rest value:
//! [`Pose::local_transform`] merges the overrides *component-wise*
//! into the node's rest TRS (a matrix rest transform is decomposed
//! first — glTF §3.9 forbids `matrix` on animated nodes, so the
//! decomposition only kicks in for out-of-spec inputs and is
//! best-effort per [`Transform::from_matrix`]).

use crate::animation::{Animation, AnimationProperty, SampledValue};
use crate::mesh::Mesh;
use crate::scene::{mat4_mul, NodeId, Scene3D, Transform};

/// One animation evaluated at one timestamp — per-node TRS overrides
/// plus morph-weight vectors, indexed by `NodeId.0`.
///
/// Sparse: a slot is `None` when no channel drove that property for
/// that node. Built by [`Animation::sample_pose`]; consumed by
/// [`Pose::local_transform`] and [`Scene3D::posed_node_transforms`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pose {
    /// Sampled `Translation` per node.
    pub translations: Vec<Option<[f32; 3]>>,
    /// Sampled `Rotation` per node (unit quaternion, xyzw —
    /// renormalised after sampling).
    pub rotations: Vec<Option<[f32; 4]>>,
    /// Sampled `Scale` per node.
    pub scales: Vec<Option<[f32; 3]>>,
    /// Sampled `MorphWeights` per node (one weight per morph target
    /// of the mesh the node instantiates).
    pub morph_weights: Vec<Option<Vec<f32>>>,
}

impl Pose {
    /// Empty pose over `node_count` nodes — no overrides anywhere.
    pub fn new(node_count: usize) -> Self {
        Self {
            translations: vec![None; node_count],
            rotations: vec![None; node_count],
            scales: vec![None; node_count],
            morph_weights: vec![None; node_count],
        }
    }

    /// Number of node slots this pose covers.
    pub fn node_count(&self) -> usize {
        self.translations.len()
    }

    /// `true` when no channel wrote any override into this pose.
    pub fn is_empty(&self) -> bool {
        self.translations.iter().all(Option::is_none)
            && self.rotations.iter().all(Option::is_none)
            && self.scales.iter().all(Option::is_none)
            && self.morph_weights.iter().all(Option::is_none)
    }

    /// The node's local transform under this pose: the rest transform
    /// `base` with every component this pose overrides replaced.
    ///
    /// * No override for `node` → `*base` unchanged (including a
    ///   `Matrix` rest transform, kept verbatim).
    /// * Any override → the rest transform is brought into TRS form
    ///   ([`Transform::from_matrix`] decomposition when the rest is a
    ///   `Matrix` — best-effort for shear, exact for pure TRS) and the
    ///   driven components are substituted. Undriven components keep
    ///   their rest values, per glTF 2.0 §3.6 (each channel targets
    ///   one property; the others are unaffected).
    ///
    /// A `node` beyond [`Pose::node_count`] returns `*base`.
    pub fn local_transform(&self, node: NodeId, base: &Transform) -> Transform {
        let i = node.0 as usize;
        let t = self.translations.get(i).copied().flatten();
        let r = self.rotations.get(i).copied().flatten();
        let s = self.scales.get(i).copied().flatten();
        if t.is_none() && r.is_none() && s.is_none() {
            return *base;
        }
        let (bt, br, bs) = match *base {
            Transform::Trs {
                translation,
                rotation,
                scale,
            } => (translation, rotation, scale),
            Transform::Matrix(m) => match Transform::from_matrix(m) {
                Transform::Trs {
                    translation,
                    rotation,
                    scale,
                } => (translation, rotation, scale),
                // from_matrix always returns Trs; keep the fallback
                // total anyway.
                Transform::Matrix(_) => ([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
            },
        };
        Transform::Trs {
            translation: t.unwrap_or(bt),
            rotation: r.unwrap_or(br),
            scale: s.unwrap_or(bs),
        }
    }
}

impl Animation {
    /// Length of this animation in seconds — the largest keyframe
    /// timestamp across every channel (`0.0` for an empty animation).
    /// The natural playback range is `0.0..=duration()` with sampling
    /// clamped at both ends (Appendix C.1).
    pub fn duration(&self) -> f32 {
        self.channels
            .iter()
            .filter_map(|ch| ch.sampler.keyframes.last().copied())
            .fold(0.0, f32::max)
    }

    /// Evaluate every channel at time `t` (seconds) into a [`Pose`]
    /// over `node_count` nodes (pass `scene.nodes.len()`).
    ///
    /// Per channel, [`AnimationSampler::sample`] supplies the value
    /// (clamped outside the keyframe range per Appendix C.1) and the
    /// result lands in the slot named by the channel's target —
    /// provided the target node is `< node_count` and the sampled
    /// value's variant matches the property (`Vec3` for translation /
    /// scale, `Quat` for rotation, `Scalar` for morph weights; a
    /// mismatched or malformed sampler is skipped —
    /// [`Scene3D::validate`] reports those). Rotation samples are
    /// renormalised (required after cubic-spline blending per the
    /// Appendix C.5 note; harmless for the already-unit SLERP path).
    /// A zero-norm / non-finite rotation sample is skipped rather
    /// than poisoning the pose.
    ///
    /// glTF 2.0 §5.6 forbids two channels of one animation targeting
    /// the same node + property; if an out-of-spec input does anyway,
    /// the later channel wins.
    ///
    /// [`AnimationSampler::sample`]: crate::AnimationSampler::sample
    pub fn sample_pose(&self, t: f32, node_count: usize) -> Pose {
        let mut pose = Pose::new(node_count);
        for ch in &self.channels {
            let idx = ch.target.node.0 as usize;
            if idx >= node_count {
                continue;
            }
            let Some(value) = ch.sampler.sample(t) else {
                continue;
            };
            match (ch.target.property, value) {
                (AnimationProperty::Translation, SampledValue::Vec3(v)) => {
                    pose.translations[idx] = Some(v);
                }
                (AnimationProperty::Scale, SampledValue::Vec3(v)) => {
                    pose.scales[idx] = Some(v);
                }
                (AnimationProperty::Rotation, SampledValue::Quat(q)) => {
                    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                    if n.is_finite() && n > 0.0 {
                        pose.rotations[idx] = Some([q[0] / n, q[1] / n, q[2] / n, q[3] / n]);
                    }
                }
                (AnimationProperty::MorphWeights, SampledValue::Scalar(w)) => {
                    pose.morph_weights[idx] = Some(w);
                }
                // Variant/property mismatch: malformed channel, skip
                // (validate() reports AnimationValueVariantMismatch).
                _ => {}
            }
        }
        pose
    }
}

impl Scene3D {
    /// World matrix per node under a [`Pose`] — the posed counterpart
    /// of [`Scene3D::world_node_transforms`], with each node's local
    /// transform replaced by [`Pose::local_transform`] before the
    /// parent-chain multiply.
    ///
    /// Same walk contract as the rest-pose version: depth-first from
    /// [`Scene3D::roots`] in order, children in source order,
    /// first-arrival wins on shared/cyclic references, unreachable
    /// nodes yield `None`, out-of-range ids are skipped. Feed the
    /// result into [`Scene3D::joint_matrices_with`] /
    /// [`Scene3D::world_mesh_with`] to deform geometry under the pose.
    pub fn posed_node_transforms(&self, pose: &Pose) -> Vec<Option<[[f32; 4]; 4]>> {
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
        let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
            self.roots.iter().rev().map(|r| (*r, identity)).collect();
        while let Some((nid, parent)) = stack.pop() {
            let idx = nid.0 as usize;
            if idx >= n_nodes || out[idx].is_some() {
                continue;
            }
            let node = &self.nodes[idx];
            let local = pose.local_transform(nid, &node.transform);
            let world = mat4_mul(parent, local.to_matrix());
            out[idx] = Some(world);
            for child in node.children.iter().rev() {
                stack.push((*child, world));
            }
        }
        out
    }

    /// Render-ready world-space geometry for one node under an
    /// animation at time `t` — the animated counterpart of
    /// [`Scene3D::world_mesh`], running the full frame pipeline:
    ///
    /// 1. `animation.sample_pose(t)` — evaluate every channel;
    /// 2. [`Scene3D::posed_node_transforms`] — posed world matrices;
    /// 3. [`Scene3D::world_mesh_with`]-style instantiation against
    ///    those matrices, except that a `MorphWeights` channel
    ///    targeting `node` **overrides** the mesh's default weights
    ///    (glTF 2.0 weight precedence: animated weights beat the
    ///    static defaults).
    ///
    /// Skinned nodes deform by the posed joints (their own transform
    /// ignored per §3.7.3.2); unskinned nodes bake their posed world
    /// matrix. The `None` conditions of [`Scene3D::world_mesh`]
    /// apply. Sampling outside `0.0..=animation.duration()` clamps
    /// per Appendix C.1.
    ///
    /// Instantiating many nodes at one timestamp? Build the pose and
    /// posed worlds once and drive [`Scene3D::world_mesh_with`] /
    /// [`Scene3D::joint_matrices_with`] yourself — this convenience
    /// resamples per call. Note the manual route uses the mesh's
    /// *default* morph weights; the per-node animated override is
    /// what this method adds on top.
    pub fn world_mesh_at(&self, animation: &Animation, t: f32, node: NodeId) -> Option<Mesh> {
        let pose = animation.sample_pose(t, self.nodes.len());
        let worlds = self.posed_node_transforms(&pose);
        let weight_override = pose
            .morph_weights
            .get(node.0 as usize)
            .and_then(|w| w.as_deref());
        self.world_mesh_impl(node, &worlds, weight_override)
    }

    /// Bake an animation frame into a copy of the scene: every node's
    /// transform becomes its [`Pose::local_transform`] under
    /// `animation` at time `t`, and every mesh instantiated by a node
    /// with an animated `MorphWeights` channel gets that sampled
    /// vector written to [`Mesh::weights`](crate::Mesh::weights).
    /// `self` is untouched.
    ///
    /// This is the exporter's flatten: the returned scene *is* the
    /// frame — `posed(anim, t).world_node_transforms()` equals
    /// `posed_node_transforms(&anim.sample_pose(t, …))`, and the
    /// downstream rest-pose tooling ([`Scene3D::world_mesh`],
    /// [`Scene3D::bake_transforms`], metrics, exporters) now sees the
    /// animated frame as its rest pose.
    ///
    /// Two structural caveats, both inherent to baking:
    ///
    /// * Driven nodes get `Trs` transforms (a `Matrix` rest transform
    ///   is decomposed first — see [`Pose::local_transform`]);
    ///   undriven nodes keep their rest transform bit-for-bit.
    /// * [`Mesh::weights`](crate::Mesh::weights) lives on the *mesh*,
    ///   so when two nodes with different animated weight vectors
    ///   share one mesh, the higher-indexed node wins (the same
    ///   last-wins rule as duplicate channels). Per-node divergence
    ///   needs per-node instantiation — [`Scene3D::world_mesh_at`].
    ///
    /// Animations (including `animation` itself) are carried over
    /// unchanged: they still describe motion relative to the original
    /// rest pose, so re-baking a baked scene compounds only if the
    /// caller asks it to.
    pub fn posed(&self, animation: &Animation, t: f32) -> Scene3D {
        let pose = animation.sample_pose(t, self.nodes.len());
        let mut out = self.clone();
        for (i, node) in out.nodes.iter_mut().enumerate() {
            node.transform = pose.local_transform(NodeId(i as u32), &node.transform);
        }
        for (i, weights) in pose.morph_weights.iter().enumerate() {
            let (Some(w), Some(node)) = (weights, self.nodes.get(i)) else {
                continue;
            };
            let Some(mesh) = node.mesh.and_then(|m| out.meshes.get_mut(m.0 as usize)) else {
                continue;
            };
            mesh.weights = w.clone();
        }
        out
    }
}
