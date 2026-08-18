//! Tests for the sampled-`MorphWeights` synthesis path:
//! [`AnimationSampler::morph_weights`] / [`morph_weights_cubic`] and
//! their lossless read-back accessors, plus the
//! [`AnimationChannel::new`] / [`Animation::with_channel`] /
//! [`Animation::channel_for`] conveniences.
//!
//! Truth ladder: glTF 2.0 §3.6 (animation samplers — strictly
//! increasing timestamps, cubic `(in, value, out)` keyframe triples,
//! weights samplers carrying `count(targets)` floats per keyframe) and
//! Appendix C (interpolation). Spec mirrored at
//! `docs/3d/gltf/gltf-2.0-spec.html`.
//!
//! [`morph_weights_cubic`]: AnimationSampler::morph_weights_cubic

use oxideav_mesh3d::{
    Animation, AnimationProperty, AnimationSampler, AnimationValues, Interpolation, Mesh,
    MorphTarget, NodeId, Primitive, SampledValue, Scene3D, Topology,
};

fn unwrap_scalar(s: SampledValue) -> Vec<f32> {
    match s {
        SampledValue::Scalar(v) => v,
        other => panic!("expected Scalar, got {other:?}"),
    }
}

fn close(a: &[f32], b: &[f32], eps: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
}

// ---------------------------------------------------------------- //
// morph_weights (Step / Linear)                                    //
// ---------------------------------------------------------------- //

#[test]
fn linear_constructor_samples_exact_keys_and_midpoint() {
    let s = AnimationSampler::morph_weights(
        vec![0.0, 1.0, 2.0],
        vec![vec![0.0, 1.0], vec![1.0, 0.5], vec![0.0, 0.0]],
        Interpolation::Linear,
    )
    .expect("well-formed sampler");
    assert_eq!(unwrap_scalar(s.sample(0.0).unwrap()), vec![0.0, 1.0]);
    assert_eq!(unwrap_scalar(s.sample(1.0).unwrap()), vec![1.0, 0.5]);
    assert_eq!(unwrap_scalar(s.sample(2.0).unwrap()), vec![0.0, 0.0]);
    // C.3 linear midpoint of frames 0 and 1.
    let mid = unwrap_scalar(s.sample(0.5).unwrap());
    assert!(close(&mid, &[0.5, 0.75], 1e-6));
}

#[test]
fn step_constructor_holds_previous_key() {
    let s = AnimationSampler::morph_weights(
        vec![0.0, 1.0],
        vec![vec![0.25], vec![0.75]],
        Interpolation::Step,
    )
    .unwrap();
    assert_eq!(unwrap_scalar(s.sample(0.99).unwrap()), vec![0.25]);
    assert_eq!(unwrap_scalar(s.sample(1.0).unwrap()), vec![0.75]);
}

#[test]
fn single_keyframe_is_well_formed() {
    let s = AnimationSampler::morph_weights(
        vec![0.5],
        vec![vec![0.1, 0.2, 0.3]],
        Interpolation::Linear,
    )
    .unwrap();
    // Clamps everywhere (C.1).
    assert_eq!(unwrap_scalar(s.sample(-3.0).unwrap()), vec![0.1, 0.2, 0.3]);
    assert_eq!(unwrap_scalar(s.sample(7.0).unwrap()), vec![0.1, 0.2, 0.3]);
}

#[test]
fn weight_values_pass_through_verbatim() {
    // Negative and >1 weights are meaningful morph inputs.
    let s = AnimationSampler::morph_weights(vec![0.0], vec![vec![-2.5, 3.0]], Interpolation::Step)
        .unwrap();
    assert_eq!(unwrap_scalar(s.sample(0.0).unwrap()), vec![-2.5, 3.0]);
}

#[test]
fn constructor_rejects_structural_malformations() {
    // Empty keyframes.
    assert!(AnimationSampler::morph_weights(vec![], vec![], Interpolation::Linear).is_none());
    // Frame count mismatch.
    assert!(AnimationSampler::morph_weights(
        vec![0.0, 1.0],
        vec![vec![0.0]],
        Interpolation::Linear
    )
    .is_none());
    // Ragged stride.
    assert!(AnimationSampler::morph_weights(
        vec![0.0, 1.0],
        vec![vec![0.0, 0.0], vec![1.0]],
        Interpolation::Linear
    )
    .is_none());
    // Zero stride.
    assert!(AnimationSampler::morph_weights(
        vec![0.0, 1.0],
        vec![vec![], vec![]],
        Interpolation::Linear
    )
    .is_none());
    // Non-increasing timestamps (§3.6 strictly increasing).
    assert!(AnimationSampler::morph_weights(
        vec![0.0, 0.0],
        vec![vec![0.0], vec![1.0]],
        Interpolation::Linear
    )
    .is_none());
    assert!(AnimationSampler::morph_weights(
        vec![1.0, 0.5],
        vec![vec![0.0], vec![1.0]],
        Interpolation::Linear
    )
    .is_none());
    // Non-finite timestamp.
    assert!(AnimationSampler::morph_weights(
        vec![0.0, f32::NAN],
        vec![vec![0.0], vec![1.0]],
        Interpolation::Linear
    )
    .is_none());
    // CubicSpline needs the tangent-triple constructor.
    assert!(AnimationSampler::morph_weights(
        vec![0.0, 1.0],
        vec![vec![0.0], vec![1.0]],
        Interpolation::CubicSpline
    )
    .is_none());
}

// ---------------------------------------------------------------- //
// morph_weights_cubic                                              //
// ---------------------------------------------------------------- //

#[test]
fn cubic_constructor_lays_out_interleaved_triples() {
    let s = AnimationSampler::morph_weights_cubic(
        vec![0.0, 1.0],
        vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![vec![0.5, 0.6], vec![0.7, 0.8]],
    )
    .expect("well-formed cubic sampler");
    assert_eq!(s.interpolation, Interpolation::CubicSpline);
    // Interleaved [a_0, v_0, b_0, a_1, v_1, b_1], stride 2.
    match &s.values {
        AnimationValues::Scalar(v) => assert_eq!(
            v,
            &vec![0.1, 0.2, 1.0, 2.0, 0.5, 0.6, 0.3, 0.4, 3.0, 4.0, 0.7, 0.8]
        ),
        other => panic!("expected Scalar, got {other:?}"),
    }
    // Exact keyframes return the centre value (C.1).
    assert_eq!(unwrap_scalar(s.sample(0.0).unwrap()), vec![1.0, 2.0]);
    assert_eq!(unwrap_scalar(s.sample(1.0).unwrap()), vec![3.0, 4.0]);
}

#[test]
fn cubic_midpoint_matches_manual_hermite() {
    // One weight, tangents chosen by hand; C.5 with t_d = 1, u = 0.5:
    //   v = 0.5*v_k + 0.125*b_k + 0.5*v_k1 - 0.125*a_k1
    let (v_k, b_k, v_k1, a_k1) = (0.0f32, 1.0f32, 2.0f32, -1.0f32);
    let s = AnimationSampler::morph_weights_cubic(
        vec![0.0, 1.0],
        vec![vec![9.0], vec![a_k1]], // in-tangent of frame 0 unused inside range
        vec![vec![v_k], vec![v_k1]],
        vec![vec![b_k], vec![9.0]], // out-tangent of frame 1 unused inside range
    )
    .unwrap();
    let expect = 0.5 * v_k + 0.125 * b_k + 0.5 * v_k1 - 0.125 * a_k1;
    let got = unwrap_scalar(s.sample(0.5).unwrap());
    assert!((got[0] - expect).abs() < 1e-6, "got {got:?}, want {expect}");
}

#[test]
fn cubic_constructor_rejects_table_mismatches() {
    // in_tangents short.
    assert!(AnimationSampler::morph_weights_cubic(
        vec![0.0, 1.0],
        vec![vec![0.0]],
        vec![vec![0.0], vec![1.0]],
        vec![vec![0.0], vec![0.0]],
    )
    .is_none());
    // Stride disagreement across tables.
    assert!(AnimationSampler::morph_weights_cubic(
        vec![0.0, 1.0],
        vec![vec![0.0], vec![0.0]],
        vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        vec![vec![0.0], vec![0.0]],
    )
    .is_none());
}

// ---------------------------------------------------------------- //
// Read-back accessors                                              //
// ---------------------------------------------------------------- //

#[test]
fn frames_round_trip_losslessly_linear() {
    let frames = vec![vec![0.0, 1.0, 0.5], vec![1.0, 0.0, 0.25]];
    let s = AnimationSampler::morph_weights(vec![0.0, 2.0], frames.clone(), Interpolation::Linear)
        .unwrap();
    assert_eq!(s.morph_weight_stride(), Some(3));
    assert_eq!(s.morph_weight_frame(0).unwrap(), frames[0].as_slice());
    assert_eq!(s.morph_weight_frame(1).unwrap(), frames[1].as_slice());
    assert!(s.morph_weight_frame(2).is_none());
    let all = s.morph_weight_frames().unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0], frames[0].as_slice());
    assert_eq!(all[1], frames[1].as_slice());
}

#[test]
fn frames_round_trip_losslessly_cubic() {
    let ins = vec![vec![0.1], vec![0.2]];
    let vals = vec![vec![1.5], vec![2.5]];
    let outs = vec![vec![0.3], vec![0.4]];
    let s = AnimationSampler::morph_weights_cubic(
        vec![0.0, 1.0],
        ins.clone(),
        vals.clone(),
        outs.clone(),
    )
    .unwrap();
    assert_eq!(s.morph_weight_stride(), Some(1));
    // Centre values via the plain frame accessor.
    assert_eq!(s.morph_weight_frame(0).unwrap(), vals[0].as_slice());
    assert_eq!(s.morph_weight_frame(1).unwrap(), vals[1].as_slice());
    // Full triples via the cubic accessor.
    let (a, v, b) = s.morph_weight_cubic_frame(0).unwrap();
    assert_eq!(
        (a, v, b),
        (ins[0].as_slice(), vals[0].as_slice(), outs[0].as_slice())
    );
    let (a, v, b) = s.morph_weight_cubic_frame(1).unwrap();
    assert_eq!(
        (a, v, b),
        (ins[1].as_slice(), vals[1].as_slice(), outs[1].as_slice())
    );
    assert!(s.morph_weight_cubic_frame(2).is_none());
}

#[test]
fn read_back_refuses_non_scalar_and_malformed() {
    let vec3 = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Vec3(vec![[0.0; 3], [1.0; 3]]),
        interpolation: Interpolation::Linear,
    };
    assert!(vec3.morph_weight_stride().is_none());
    assert!(vec3.morph_weight_frame(0).is_none());
    assert!(vec3.morph_weight_frames().is_none());
    assert!(vec3.morph_weight_cubic_frame(0).is_none());

    // Value table not a multiple of the keyframe count.
    let ragged = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Scalar(vec![0.0, 1.0, 2.0]),
        interpolation: Interpolation::Linear,
    };
    assert!(ragged.morph_weight_stride().is_none());
    assert!(ragged.morph_weight_frames().is_none());

    // Cubic accessor on a non-cubic sampler.
    let linear = AnimationSampler::morph_weights(
        vec![0.0, 1.0],
        vec![vec![0.0], vec![1.0]],
        Interpolation::Linear,
    )
    .unwrap();
    assert!(linear.morph_weight_cubic_frame(0).is_none());
}

// ---------------------------------------------------------------- //
// Channel / Animation conveniences                                 //
// ---------------------------------------------------------------- //

#[test]
fn with_channel_and_channel_for() {
    let s = AnimationSampler::morph_weights(
        vec![0.0, 1.0],
        vec![vec![0.0], vec![1.0]],
        Interpolation::Linear,
    )
    .unwrap();
    let anim = Animation::new("a".to_owned())
        .with_channel(NodeId(3), AnimationProperty::MorphWeights, s.clone())
        .with_channel(NodeId(4), AnimationProperty::Translation, {
            AnimationSampler {
                keyframes: vec![0.0],
                values: AnimationValues::Vec3(vec![[1.0, 0.0, 0.0]]),
                interpolation: Interpolation::Step,
            }
        });
    assert_eq!(anim.channels.len(), 2);
    let ch = anim
        .channel_for(NodeId(3), AnimationProperty::MorphWeights)
        .expect("channel present");
    assert_eq!(ch.target.node, NodeId(3));
    assert!(anim
        .channel_for(NodeId(3), AnimationProperty::Rotation)
        .is_none());
    assert!(anim
        .channel_for(NodeId(9), AnimationProperty::MorphWeights)
        .is_none());
}

#[test]
fn channel_for_later_wins_on_out_of_spec_duplicates() {
    let first =
        AnimationSampler::morph_weights(vec![0.0], vec![vec![0.1]], Interpolation::Step).unwrap();
    let second =
        AnimationSampler::morph_weights(vec![0.0], vec![vec![0.9]], Interpolation::Step).unwrap();
    let anim = Animation::new(None)
        .with_channel(NodeId(0), AnimationProperty::MorphWeights, first)
        .with_channel(NodeId(0), AnimationProperty::MorphWeights, second);
    let ch = anim
        .channel_for(NodeId(0), AnimationProperty::MorphWeights)
        .unwrap();
    // Matches sample_pose's later-channel-wins rule.
    assert_eq!(ch.sampler.morph_weight_frame(0).unwrap(), &[0.9][..]);
    let pose = anim.sample_pose(0.0, 1);
    assert_eq!(pose.morph_weights[0].as_deref(), Some(&[0.9f32][..]));
}

// ---------------------------------------------------------------- //
// End-to-end: synthesized channel through the scene pipeline        //
// ---------------------------------------------------------------- //

/// One-triangle primitive with two position morph targets.
fn morph_primitive() -> Primitive {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut t0 = MorphTarget::new();
    t0.position = Some(vec![[0.0, 0.0, 1.0]; 3]);
    let mut t1 = MorphTarget::new();
    t1.position = Some(vec![[0.0, 2.0, 0.0]; 3]);
    prim.targets = vec![t0, t1];
    prim
}

#[test]
fn synthesized_channel_validates_and_drives_world_mesh_at() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(Mesh::new(None).with_primitive(morph_primitive()));
    let node = scene.add_node(oxideav_mesh3d::Node::new().with_name("n").with_mesh(mesh));
    scene.roots.push(node);

    // Stride 2 == morph-target count, so validate() stays clean.
    let sampler = AnimationSampler::morph_weights(
        vec![0.0, 1.0],
        vec![vec![0.0, 0.0], vec![1.0, 0.5]],
        Interpolation::Linear,
    )
    .unwrap();
    scene
        .animations
        .push(Animation::new("w".to_owned()).with_channel(
            node,
            AnimationProperty::MorphWeights,
            sampler,
        ));
    assert_eq!(scene.validate(), Ok(()));

    // At t = 1 the blend is base + 1.0*t0 + 0.5*t1.
    let anim = scene.animations[0].clone();
    let baked = scene.world_mesh_at(&anim, 1.0, node).expect("instantiable");
    let p = &baked.primitives[0].positions;
    assert!((p[0][1] - 1.0).abs() < 1e-6, "0.5 * t1 delta y=2");
    assert!((p[0][2] - 1.0).abs() < 1e-6, "1.0 * t0 delta z=1");
    // Morph state consumed by instantiation.
    assert!(baked.primitives[0].targets.is_empty());
}

#[test]
fn synthesized_stride_mismatch_is_reported_by_validate() {
    let mut scene = Scene3D::new();
    let mesh = scene.add_mesh(Mesh::new(None).with_primitive(morph_primitive()));
    let node = scene.add_node(oxideav_mesh3d::Node::new().with_mesh(mesh));
    scene.roots.push(node);

    // Stride 3 against a 2-target mesh: structurally fine as a
    // sampler, semantically wrong for this binding.
    let sampler =
        AnimationSampler::morph_weights(vec![0.0], vec![vec![0.0, 0.0, 0.0]], Interpolation::Step)
            .unwrap();
    scene.animations.push(Animation::new(None).with_channel(
        node,
        AnimationProperty::MorphWeights,
        sampler,
    ));
    let errs = scene.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        oxideav_mesh3d::ValidationError::AnimationMorphStrideMismatch {
            stride: 3,
            targets: 2,
            ..
        }
    )));
}
