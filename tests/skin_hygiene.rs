//! Tests for skinning data hygiene:
//!
//! * `Primitive::normalize_joint_weights` — the explicit weight-row
//!   repair (clamp negatives / non-finite to 0, renormalise to sum 1,
//!   keep all-zero rows).
//! * The `Scene3D::validate` skinning checks — `JointWeightInvalid`,
//!   `JointIndexOutOfRange`, `AnimationMorphStrideMismatch`, and the
//!   spec-aligned inverse-bind count rule (`>=` joints is conforming,
//!   glTF 2.0 §3.7.3.1).

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, Interpolation, Mesh, MorphTarget, Node, Primitive, Scene3D, Skeleton, Skin,
    Topology, ValidationError,
};

fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn skinned_triangle() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.joints = Some(vec![[0, 0, 0, 0]; 3]);
    p.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]);
    p
}

/// Scene binding `prim` to a 1-joint skeleton via a skinned node.
fn one_joint_scene(prim: Primitive) -> Scene3D {
    let mut s = Scene3D::new();
    let joint = s.add_node(Node::new());
    s.add_root(joint);
    let skel = s.add_skeleton(Skeleton {
        name: None,
        joints: vec![joint],
        inverse_bind_matrices: vec![identity()],
    });
    let skin = s.add_skin(Skin::new(skel));
    let mid = s.add_mesh(Mesh::new(None).with_primitive(prim));
    let mut mesh_node = Node::new().with_mesh(mid);
    mesh_node.skin = Some(skin);
    let nid = s.add_node(mesh_node);
    s.add_root(nid);
    s
}

// --- normalize_joint_weights -------------------------------------------

#[test]
fn rows_are_renormalised_to_sum_one() {
    let mut prim = skinned_triangle();
    prim.weights = Some(vec![
        [0.5, 0.25, 0.0, 0.0],    // sums to 0.75
        [2.0, 2.0, 0.0, 0.0],     // sums to 4
        [0.25, 0.25, 0.25, 0.25], // already 1
    ]);
    let fixed = prim.normalize_joint_weights();
    let w = fixed.weights.as_ref().unwrap();
    assert_eq!(w[0], [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]);
    assert_eq!(w[1], [0.5, 0.5, 0.0, 0.0]);
    assert_eq!(w[2], [0.25, 0.25, 0.25, 0.25]);
    // Pure: the source keeps its sloppy rows.
    assert_eq!(prim.weights.as_ref().unwrap()[0], [0.5, 0.25, 0.0, 0.0]);
}

#[test]
fn negative_and_non_finite_components_are_zeroed_first() {
    let mut prim = skinned_triangle();
    prim.weights = Some(vec![
        [0.5, -0.5, 0.0, 0.0],
        [f32::NAN, 0.5, 0.0, 0.0],
        [f32::INFINITY, 0.25, 0.25, 0.0],
    ]);
    let fixed = prim.normalize_joint_weights();
    let w = fixed.weights.as_ref().unwrap();
    assert_eq!(w[0], [1.0, 0.0, 0.0, 0.0], "negative dropped, rest scaled");
    assert_eq!(w[1], [0.0, 1.0, 0.0, 0.0], "NaN dropped");
    assert_eq!(w[2], [0.0, 0.5, 0.5, 0.0], "Inf dropped");
}

#[test]
fn all_zero_rows_stay_all_zero() {
    let mut prim = skinned_triangle();
    prim.weights = Some(vec![
        [0.0, 0.0, 0.0, 0.0],
        [-1.0, f32::NAN, 0.0, 0.0], // cleans to all-zero
        [1.0, 0.0, 0.0, 0.0],
    ]);
    let fixed = prim.normalize_joint_weights();
    let w = fixed.weights.as_ref().unwrap();
    assert_eq!(w[0], [0.0; 4], "unskinned vertex untouched");
    assert_eq!(w[1], [0.0; 4], "cleaned to unskinned");
    assert_eq!(w[2], [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn normalization_is_idempotent_and_respects_missing_buffer() {
    let mut prim = skinned_triangle();
    prim.weights = Some(vec![[0.3, 0.5, 0.1, 0.0]; 3]);
    let once = prim.normalize_joint_weights();
    let twice = once.normalize_joint_weights();
    // Idempotent up to one float rounding step: the repaired row sums
    // to 1 within one ulp, so the second pass divides by ≈1.
    for (a, b) in once
        .weights
        .as_ref()
        .unwrap()
        .iter()
        .zip(twice.weights.as_ref().unwrap())
    {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() <= f32::EPSILON, "{x} vs {y}");
        }
    }
    let sum: f32 = once.weights.as_ref().unwrap()[0].iter().sum();
    assert!((sum - 1.0).abs() <= f32::EPSILON, "row sums to ~1: {sum}");

    prim.weights = None;
    let unchanged = prim.normalize_joint_weights();
    assert!(unchanged.weights.is_none());
}

#[test]
fn normalized_weights_make_skinning_scale_invariant() {
    // Doubling every weight is a no-op after normalisation.
    let translate5 = {
        let mut m = identity();
        m[0][3] = 5.0;
        m
    };
    let mut a = skinned_triangle();
    a.joints = Some(vec![[0, 1, 0, 0]; 3]);
    a.weights = Some(vec![[0.5, 0.5, 0.0, 0.0]; 3]);
    let mut b = a.clone();
    b.weights = Some(vec![[1.0, 1.0, 0.0, 0.0]; 3]);
    let palette = [identity(), translate5];
    let pa = a.normalize_joint_weights().skinned(&palette);
    let pb = b.normalize_joint_weights().skinned(&palette);
    assert_eq!(pa.positions, pb.positions);
}

// --- validate: joint weights ------------------------------------------

#[test]
fn negative_weight_is_reported_once_per_primitive() {
    let mut prim = skinned_triangle();
    prim.weights = Some(vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.5, -0.5, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0], // second offender, same primitive
    ]);
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None).with_primitive(prim));
    let errs = s.validate().unwrap_err();
    let hits: Vec<_> = errs
        .iter()
        .filter(|e| matches!(e, ValidationError::JointWeightInvalid { .. }))
        .collect();
    assert_eq!(hits.len(), 1, "anti-flood: first offender only: {errs:?}");
    assert!(matches!(
        hits[0],
        ValidationError::JointWeightInvalid { value, location }
            if *value == -0.5 && location == "meshes[0].primitives[0].weights[1][1]"
    ));
}

#[test]
fn nan_weight_is_reported() {
    let mut prim = skinned_triangle();
    prim.weights = Some(vec![
        [1.0, 0.0, 0.0, 0.0],
        [f32::NAN, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]);
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None).with_primitive(prim));
    let errs = s.validate().unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::JointWeightInvalid { .. })));
}

#[test]
fn clean_weights_validate_ok() {
    let s = one_joint_scene(skinned_triangle());
    assert!(s.validate().is_ok(), "{:?}", s.validate());
}

// --- validate: joint index range ----------------------------------------

#[test]
fn joint_index_beyond_skeleton_is_reported() {
    let mut prim = skinned_triangle();
    prim.joints = Some(vec![[0, 0, 0, 0], [3, 0, 0, 0], [7, 0, 0, 0]]);
    let s = one_joint_scene(prim);
    let errs = s.validate().unwrap_err();
    let hits: Vec<_> = errs
        .iter()
        .filter(|e| matches!(e, ValidationError::JointIndexOutOfRange { .. }))
        .collect();
    assert_eq!(hits.len(), 1, "first offender only: {errs:?}");
    assert!(matches!(
        hits[0],
        ValidationError::JointIndexOutOfRange {
            joint: 3,
            joint_count: 1,
            ..
        }
    ));
}

#[test]
fn joint_range_is_per_binding_not_per_mesh() {
    // The same joint indices are fine on an unbound mesh: without a
    // node → skin binding there is no skeleton to range-check against.
    let mut prim = skinned_triangle();
    prim.joints = Some(vec![[5, 0, 0, 0]; 3]);
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None).with_primitive(prim));
    assert!(s.validate().is_ok(), "{:?}", s.validate());
}

// --- validate: inverse-bind count (spec >= rule) -------------------------

#[test]
fn extra_inverse_bind_matrices_are_conforming() {
    let mut s = one_joint_scene(skinned_triangle());
    // 1 joint, 3 IBMs: MUST be >= joints, so this validates clean.
    s.skeletons[0].inverse_bind_matrices = vec![identity(), identity(), identity()];
    assert!(s.validate().is_ok(), "{:?}", s.validate());
}

#[test]
fn short_inverse_bind_matrices_still_fail() {
    let mut s = one_joint_scene(skinned_triangle());
    let extra = s.add_node(Node::new());
    s.add_root(extra);
    s.skeletons[0].joints.push(extra); // 2 joints, 1 IBM
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::SkeletonBindMatrixCountMismatch {
            joints: 2,
            inverse_bind_matrices: 1,
            ..
        }
    )));
}

// --- validate: MorphWeights stride ---------------------------------------

/// Scene with one morphable mesh node (`n_targets` targets) and one
/// MorphWeights channel carrying `stride` weights per keyframe.
fn morph_anim_scene(n_targets: usize, stride: usize) -> Scene3D {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    prim.targets = (0..n_targets)
        .map(|_| MorphTarget {
            position: Some(vec![[0.0, 0.0, 1.0]; 3]),
            ..Default::default()
        })
        .collect();
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(prim));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: nid,
            property: AnimationProperty::MorphWeights,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Scalar(vec![0.0; 2 * stride]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);
    s
}

#[test]
fn matching_morph_stride_validates_ok() {
    let s = morph_anim_scene(2, 2);
    assert!(s.validate().is_ok(), "{:?}", s.validate());
}

#[test]
fn mismatched_morph_stride_is_reported() {
    let s = morph_anim_scene(2, 3);
    let errs = s.validate().unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ValidationError::AnimationMorphStrideMismatch {
                stride: 3,
                targets: 2,
                ..
            }
        )),
        "{errs:?}"
    );
}
