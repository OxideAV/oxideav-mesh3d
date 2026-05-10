//! Skeleton + Skin — rigging primitives for skeletal animation.
//!
//! A [`Skeleton`] is the static rest-pose: an ordered list of joint
//! [`NodeId`]s plus the inverse bind matrix that takes a vertex from
//! mesh-local space into joint-local space at rest. The render-time
//! transform per vertex is `sum_i weight[i] * joint_world[i] * inverse_bind[i] * pos`.
//!
//! A [`Skin`] is the binding of one skeleton to one mesh — it holds
//! the skeleton id and an optional explicit root node within the
//! scene-graph (some formats name the bone root, others infer it
//! from the lowest common ancestor of the joint set).

use crate::scene::{NodeId, SkeletonId};

/// Static rest-pose of a skeletal rig.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub name: Option<String>,
    /// Joint nodes in skeleton order — index `i` of this list
    /// corresponds to joint index `i` in `Primitive::joints`.
    pub joints: Vec<NodeId>,
    /// One 4x4 inverse-bind matrix per joint, same ordering as `joints`.
    pub inverse_bind_matrices: Vec<[[f32; 4]; 4]>,
}

impl Skeleton {
    /// Empty skeleton.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Binding of one skeleton to one mesh.
///
/// `root_node` is the skeleton's scene-graph root when the format
/// stores it explicitly (most do). When `None`, the renderer finds
/// the lowest-common-ancestor of `Skeleton::joints` itself.
#[derive(Clone, Copy, Debug)]
pub struct Skin {
    pub skeleton: SkeletonId,
    pub root_node: Option<NodeId>,
}

impl Skin {
    /// Build a skin pointing at a skeleton with no explicit root.
    pub fn new(skeleton: SkeletonId) -> Self {
        Self {
            skeleton,
            root_node: None,
        }
    }

    /// Builder-style explicit-root setter.
    pub fn with_root(mut self, root: NodeId) -> Self {
        self.root_node = Some(root);
        self
    }
}
