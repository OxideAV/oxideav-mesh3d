//! Tests for `Animation::sample_pose` / `Animation::duration` /
//! `Pose::local_transform` / `Scene3D::posed_node_transforms` —
//! evaluating keyframed channels into a scene pose (glTF 2.0 §3.6 +
//! Appendix C semantics, quaternion renormalisation per the C.5 note).

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, Interpolation, Node, NodeId, Pose, Scene3D, Transform,
};

const EPS: f32 = 1e-5;

fn assert_vec3_eq(a: [f32; 3], b: [f32; 3], ctx: &str) {
    for i in 0..3 {
        assert!(
            (a[i] - b[i]).abs() < EPS,
            "{ctx}: component {i}: {a:?} vs {b:?}"
        );
    }
}

fn channel(
    node: u32,
    property: AnimationProperty,
    keyframes: Vec<f32>,
    values: AnimationValues,
    interpolation: Interpolation,
) -> AnimationChannel {
    AnimationChannel {
        target: AnimationTarget {
            node: NodeId(node),
            property,
        },
        sampler: AnimationSampler {
            keyframes,
            values,
            interpolation,
        },
    }
}

fn translation_anim(node: u32) -> Animation {
    let mut a = Animation::new(None);
    a.channels.push(channel(
        node,
        AnimationProperty::Translation,
        vec![0.0, 2.0],
        AnimationValues::Vec3(vec![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        Interpolation::Linear,
    ));
    a
}

// --- duration -----------------------------------------------------------

#[test]
fn duration_is_last_keyframe_across_channels() {
    let mut a = translation_anim(0); // ends at 2.0
    a.channels.push(channel(
        1,
        AnimationProperty::Scale,
        vec![0.0, 5.5],
        AnimationValues::Vec3(vec![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        Interpolation::Step,
    ));
    assert_eq!(a.duration(), 5.5);
    assert_eq!(Animation::new(None).duration(), 0.0, "empty animation");
}

// --- sample_pose ----------------------------------------------------------

#[test]
fn linear_translation_midpoint() {
    let pose = translation_anim(0).sample_pose(1.0, 1);
    assert_vec3_eq(pose.translations[0].unwrap(), [2.0, 0.0, 0.0], "midpoint");
    assert!(pose.rotations[0].is_none(), "undriven property untouched");
    assert!(pose.scales[0].is_none());
    assert!(!pose.is_empty());
}

#[test]
fn sampling_clamps_outside_the_keyframe_range() {
    let anim = translation_anim(0);
    let before = anim.sample_pose(-10.0, 1);
    let after = anim.sample_pose(99.0, 1);
    assert_vec3_eq(before.translations[0].unwrap(), [0.0, 0.0, 0.0], "clamp lo");
    assert_vec3_eq(after.translations[0].unwrap(), [4.0, 0.0, 0.0], "clamp hi");
}

#[test]
fn step_holds_previous_keyframe() {
    let mut a = Animation::new(None);
    a.channels.push(channel(
        0,
        AnimationProperty::Translation,
        vec![0.0, 1.0],
        AnimationValues::Vec3(vec![[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        Interpolation::Step,
    ));
    let pose = a.sample_pose(0.999, 1);
    assert_vec3_eq(pose.translations[0].unwrap(), [0.0, 0.0, 0.0], "held");
}

#[test]
fn rotation_slerp_midpoint_is_unit_and_half_angle() {
    // Identity → 90° about Z; midpoint must be 45° about Z:
    // (0, 0, sin 22.5°, cos 22.5°).
    let q90 = [
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ];
    let mut a = Animation::new(None);
    a.channels.push(channel(
        0,
        AnimationProperty::Rotation,
        vec![0.0, 1.0],
        AnimationValues::Quat(vec![[0.0, 0.0, 0.0, 1.0], q90]),
        Interpolation::Linear,
    ));
    let pose = a.sample_pose(0.5, 1);
    let q = pose.rotations[0].unwrap();
    let a225 = 22.5f32.to_radians();
    assert!((q[2] - a225.sin()).abs() < EPS, "sin 22.5: {q:?}");
    assert!((q[3] - a225.cos()).abs() < EPS, "cos 22.5: {q:?}");
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    assert!((norm - 1.0).abs() < EPS, "unit quaternion");
}

#[test]
fn cubic_spline_rotation_is_renormalised() {
    // Two identical keyframes with huge tangents: the raw Hermite
    // blend leaves the unit sphere mid-segment; sample_pose must hand
    // back a unit quaternion anyway (Appendix C.5 note).
    let mut a = Animation::new(None);
    a.channels.push(channel(
        0,
        AnimationProperty::Rotation,
        vec![0.0, 1.0],
        AnimationValues::Quat(vec![
            [0.0, 0.0, 0.0, 0.0],  // in-tangent k0
            [0.0, 0.0, 0.0, 1.0],  // value k0
            [3.0, 0.0, 0.0, 0.0],  // out-tangent k0
            [-3.0, 0.0, 0.0, 0.0], // in-tangent k1 (opposite sign so the
            // two tangent terms add at u=0.5: c_b = -c_a there)
            [0.0, 0.0, 0.0, 1.0], // value k1
            [0.0, 0.0, 0.0, 0.0], // out-tangent k1
        ]),
        Interpolation::CubicSpline,
    ));
    let q = a.sample_pose(0.5, 1).rotations[0].unwrap();
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    assert!((norm - 1.0).abs() < EPS, "renormalised: {q:?}");
    assert!(q[0] > 0.0, "tangents actually bent the curve: {q:?}");
}

#[test]
fn morph_weights_land_in_the_pose() {
    let mut a = Animation::new(None);
    a.channels.push(channel(
        0,
        AnimationProperty::MorphWeights,
        vec![0.0, 1.0],
        // 2 targets per keyframe.
        AnimationValues::Scalar(vec![0.0, 1.0, 1.0, 0.0]),
        Interpolation::Linear,
    ));
    let pose = a.sample_pose(0.5, 1);
    let w = pose.morph_weights[0].as_ref().unwrap();
    assert_eq!(w.len(), 2);
    assert!(
        (w[0] - 0.5).abs() < EPS && (w[1] - 0.5).abs() < EPS,
        "{w:?}"
    );
}

#[test]
fn out_of_range_target_and_malformed_samplers_are_skipped() {
    let mut a = translation_anim(7); // node 7, but node_count = 1
    a.channels.push(channel(
        0,
        AnimationProperty::Rotation,
        vec![0.0], // Vec3 values on a Rotation channel: mismatched
        AnimationValues::Vec3(vec![[1.0, 2.0, 3.0]]),
        Interpolation::Linear,
    ));
    a.channels.push(channel(
        0,
        AnimationProperty::Scale,
        Vec::new(), // empty keyframes: sampler yields None
        AnimationValues::Vec3(Vec::new()),
        Interpolation::Linear,
    ));
    let pose = a.sample_pose(0.5, 1);
    assert!(pose.is_empty(), "{pose:?}");
}

#[test]
fn later_channel_wins_on_duplicate_target() {
    let mut a = translation_anim(0);
    a.channels.push(channel(
        0,
        AnimationProperty::Translation,
        vec![0.0],
        AnimationValues::Vec3(vec![[9.0, 9.0, 9.0]]),
        Interpolation::Step,
    ));
    let pose = a.sample_pose(1.0, 1);
    assert_vec3_eq(pose.translations[0].unwrap(), [9.0, 9.0, 9.0], "last wins");
}

// --- Pose::local_transform -------------------------------------------------

#[test]
fn undriven_node_keeps_rest_transform_verbatim() {
    let pose = Pose::new(2);
    let m = [
        [1.0, 0.0, 0.0, 3.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let base = Transform::Matrix(m);
    assert_eq!(pose.local_transform(NodeId(0), &base), base, "Matrix kept");
    assert_eq!(
        pose.local_transform(NodeId(99), &base),
        base,
        "out-of-range node falls back to rest"
    );
}

#[test]
fn overrides_merge_componentwise_into_rest_trs() {
    let mut pose = Pose::new(1);
    pose.translations[0] = Some([5.0, 0.0, 0.0]);
    let base = Transform::Trs {
        translation: [1.0, 1.0, 1.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [2.0, 2.0, 2.0],
    };
    let posed = pose.local_transform(NodeId(0), &base);
    match posed {
        Transform::Trs {
            translation,
            rotation,
            scale,
        } => {
            assert_eq!(translation, [5.0, 0.0, 0.0], "driven");
            assert_eq!(rotation, [0.0, 0.0, 0.0, 1.0], "rest kept");
            assert_eq!(scale, [2.0, 2.0, 2.0], "rest kept");
        }
        other => panic!("expected Trs, got {other:?}"),
    }
}

#[test]
fn matrix_rest_transform_is_decomposed_when_driven() {
    let mut pose = Pose::new(1);
    pose.scales[0] = Some([3.0, 3.0, 3.0]);
    let base = Transform::Matrix([
        [1.0, 0.0, 0.0, 7.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    match pose.local_transform(NodeId(0), &base) {
        Transform::Trs {
            translation, scale, ..
        } => {
            assert_vec3_eq(translation, [7.0, 0.0, 0.0], "decomposed T kept");
            assert_eq!(scale, [3.0, 3.0, 3.0], "driven scale");
        }
        other => panic!("expected Trs, got {other:?}"),
    }
}

// --- posed world transforms -------------------------------------------------

#[test]
fn posed_transforms_track_the_animated_chain() {
    // parent → child; animate the parent's translation and check the
    // child's world matrix follows.
    let mut s = Scene3D::new();
    let child = s.add_node(Node::new().with_transform(Transform::Trs {
        translation: [0.0, 1.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }));
    let mut parent = Node::new();
    parent.children.push(child);
    let pid = s.add_node(parent);
    s.add_root(pid);

    let anim = translation_anim(pid.0);
    let pose = anim.sample_pose(2.0, s.nodes.len()); // parent at (4,0,0)
    let worlds = s.posed_node_transforms(&pose);
    let cw = worlds[child.0 as usize].expect("child world");
    assert_vec3_eq(
        [cw[0][3], cw[1][3], cw[2][3]],
        [4.0, 1.0, 0.0],
        "parent pose + child rest",
    );

    // Rest pose for comparison: identity parent.
    let rest = s.world_node_transforms()[child.0 as usize].unwrap();
    assert_vec3_eq(
        [rest[0][3], rest[1][3], rest[2][3]],
        [0.0, 1.0, 0.0],
        "rest",
    );
}

#[test]
fn empty_pose_reproduces_rest_world_transforms() {
    let mut s = Scene3D::new();
    let a = s.add_node(Node::new().with_transform(Transform::Trs {
        translation: [1.0, 2.0, 3.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }));
    s.add_root(a);
    let pose = Pose::new(s.nodes.len());
    assert_eq!(
        s.posed_node_transforms(&pose),
        s.world_node_transforms(),
        "no overrides ⇒ identical walk"
    );
}
