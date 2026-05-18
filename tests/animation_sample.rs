//! Tests for [`AnimationSampler::sample`].
//!
//! Truth ladder: glTF 2.0 Appendix C (Animation Sampler Interpolation
//! Modes) §C.1 clamping / exact-keyframe; §C.2 STEP; §C.3 LINEAR;
//! §C.4 SLERP for rotation/LINEAR; §C.5 CUBICSPLINE Hermite blend.
//! Spec mirrored at `docs/3d/gltf/gltf-2.0-spec.html`.

use oxideav_mesh3d::{AnimationSampler, AnimationValues, Interpolation, SampledValue};

/// Approximate-equality on a 3-vector, suitable for f32 arithmetic.
fn vec3_close(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
    (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
}

fn quat_close(a: [f32; 4], b: [f32; 4], eps: f32) -> bool {
    (a[0] - b[0]).abs() < eps
        && (a[1] - b[1]).abs() < eps
        && (a[2] - b[2]).abs() < eps
        && (a[3] - b[3]).abs() < eps
}

fn unwrap_vec3(s: SampledValue) -> [f32; 3] {
    match s {
        SampledValue::Vec3(v) => v,
        other => panic!("expected Vec3, got {other:?}"),
    }
}

fn unwrap_quat(s: SampledValue) -> [f32; 4] {
    match s {
        SampledValue::Quat(v) => v,
        other => panic!("expected Quat, got {other:?}"),
    }
}

fn unwrap_scalar(s: SampledValue) -> Vec<f32> {
    match s {
        SampledValue::Scalar(v) => v,
        other => panic!("expected Scalar, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// C.1 — preconditions and clamping
// ---------------------------------------------------------------------------

#[test]
fn empty_sampler_returns_none() {
    let s = AnimationSampler {
        keyframes: vec![],
        values: AnimationValues::Vec3(vec![]),
        interpolation: Interpolation::Linear,
    };
    assert!(s.sample(0.0).is_none());
}

#[test]
fn mismatched_value_count_returns_none() {
    // 2 keyframes but only 1 Vec3 value → fails the divisibility check.
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Vec3(vec![[1.0, 2.0, 3.0]]),
        interpolation: Interpolation::Linear,
    };
    assert!(s.sample(0.5).is_none());
}

#[test]
fn pre_first_keyframe_clamps_to_first_value() {
    let s = AnimationSampler {
        keyframes: vec![1.0, 2.0],
        values: AnimationValues::Vec3(vec![[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        interpolation: Interpolation::Linear,
    };
    assert_eq!(unwrap_vec3(s.sample(0.0).unwrap()), [10.0, 0.0, 0.0]);
    assert_eq!(unwrap_vec3(s.sample(-1000.0).unwrap()), [10.0, 0.0, 0.0]);
}

#[test]
fn post_last_keyframe_clamps_to_last_value() {
    let s = AnimationSampler {
        keyframes: vec![1.0, 2.0],
        values: AnimationValues::Vec3(vec![[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        interpolation: Interpolation::Linear,
    };
    assert_eq!(unwrap_vec3(s.sample(2.0).unwrap()), [20.0, 0.0, 0.0]);
    assert_eq!(unwrap_vec3(s.sample(99.0).unwrap()), [20.0, 0.0, 0.0]);
}

#[test]
fn exact_keyframe_match_returns_value_as_is() {
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0, 2.0],
        values: AnimationValues::Vec3(vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        interpolation: Interpolation::Linear,
    };
    assert_eq!(unwrap_vec3(s.sample(0.0).unwrap()), [1.0, 0.0, 0.0]);
    assert_eq!(unwrap_vec3(s.sample(1.0).unwrap()), [2.0, 0.0, 0.0]);
    assert_eq!(unwrap_vec3(s.sample(2.0).unwrap()), [3.0, 0.0, 0.0]);
}

#[test]
fn single_keyframe_returns_that_value_everywhere() {
    let s = AnimationSampler {
        keyframes: vec![5.0],
        values: AnimationValues::Vec3(vec![[7.0, 8.0, 9.0]]),
        interpolation: Interpolation::Linear,
    };
    assert_eq!(unwrap_vec3(s.sample(0.0).unwrap()), [7.0, 8.0, 9.0]);
    assert_eq!(unwrap_vec3(s.sample(5.0).unwrap()), [7.0, 8.0, 9.0]);
    assert_eq!(unwrap_vec3(s.sample(99.0).unwrap()), [7.0, 8.0, 9.0]);
}

// ---------------------------------------------------------------------------
// C.2 — STEP
// ---------------------------------------------------------------------------

#[test]
fn step_holds_previous_keyframe_value() {
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0, 2.0],
        values: AnimationValues::Vec3(vec![[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        interpolation: Interpolation::Step,
    };
    // Anywhere inside [0, 1) is the value of keyframe 0.
    assert_eq!(unwrap_vec3(s.sample(0.5).unwrap()), [1.0, 0.0, 0.0]);
    assert_eq!(unwrap_vec3(s.sample(0.999).unwrap()), [1.0, 0.0, 0.0]);
    // Exact match jumps to the new value.
    assert_eq!(unwrap_vec3(s.sample(1.0).unwrap()), [5.0, 0.0, 0.0]);
    // Anywhere inside [1, 2) is the value of keyframe 1.
    assert_eq!(unwrap_vec3(s.sample(1.5).unwrap()), [5.0, 0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// C.3 — LINEAR (Vec3 / Scalar)
// ---------------------------------------------------------------------------

#[test]
fn linear_midpoint_interpolates_halfway() {
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Vec3(vec![[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]]),
        interpolation: Interpolation::Linear,
    };
    assert!(vec3_close(
        unwrap_vec3(s.sample(0.5).unwrap()),
        [5.0, 10.0, 15.0],
        1e-6,
    ));
}

#[test]
fn linear_quarter_interpolates() {
    let s = AnimationSampler {
        keyframes: vec![0.0, 4.0],
        values: AnimationValues::Vec3(vec![[0.0, 0.0, 0.0], [16.0, 0.0, 0.0]]),
        interpolation: Interpolation::Linear,
    };
    // u = 1 / 4 → x = 4
    assert!(vec3_close(
        unwrap_vec3(s.sample(1.0).unwrap()),
        [4.0, 0.0, 0.0],
        1e-6,
    ));
}

#[test]
fn linear_scalar_morph_weights_two_targets() {
    // Two morph targets per keyframe; per-frame stride is 2.
    // values layout: [w0_k0, w1_k0, w0_k1, w1_k1] = [0, 0, 1, 0.5]
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Scalar(vec![0.0, 0.0, 1.0, 0.5]),
        interpolation: Interpolation::Linear,
    };
    let mid = unwrap_scalar(s.sample(0.5).unwrap());
    assert_eq!(mid.len(), 2);
    assert!((mid[0] - 0.5).abs() < 1e-6);
    assert!((mid[1] - 0.25).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// C.4 — SLERP (Linear interpolation of rotation)
// ---------------------------------------------------------------------------

#[test]
fn slerp_identity_to_identity_returns_identity() {
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Quat(vec![[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        interpolation: Interpolation::Linear,
    };
    let mid = unwrap_quat(s.sample(0.5).unwrap());
    assert!(quat_close(mid, [0.0, 0.0, 0.0, 1.0], 1e-5));
}

#[test]
fn slerp_90deg_about_z_midpoint_is_45deg() {
    // q0 = identity, q1 = 90° about +Z = (0, 0, sin(45°), cos(45°)).
    // Midpoint must be 45° about +Z = (0, 0, sin(22.5°), cos(22.5°)).
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Quat(vec![
            [0.0, 0.0, 0.0, 1.0],
            [
                0.0,
                0.0,
                (45.0_f32.to_radians()).sin(),
                (45.0_f32.to_radians()).cos(),
            ],
        ]),
        interpolation: Interpolation::Linear,
    };
    let mid = unwrap_quat(s.sample(0.5).unwrap());
    let expected = [
        0.0,
        0.0,
        (22.5_f32.to_radians()).sin(),
        (22.5_f32.to_radians()).cos(),
    ];
    assert!(
        quat_close(mid, expected, 1e-5),
        "got {mid:?} expected {expected:?}"
    );
    // Result must still be a unit quaternion.
    let mag2 = mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2] + mid[3] * mid[3];
    assert!(
        (mag2 - 1.0).abs() < 1e-5,
        "non-unit slerp output, |q|^2 = {mag2}"
    );
}

#[test]
fn slerp_picks_short_arc_when_dot_is_negative() {
    // q0 = identity, q1 = -identity. These represent the same
    // rotation; spec implementation note: the result must follow the
    // short arc by flipping `q1`'s sign. After flipping, both
    // endpoints are identical, so the midpoint must equal identity
    // (not the zero quaternion).
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Quat(vec![[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]]),
        interpolation: Interpolation::Linear,
    };
    let mid = unwrap_quat(s.sample(0.5).unwrap());
    let mag2 = mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2] + mid[3] * mid[3];
    assert!(
        mag2 > 0.5,
        "short-arc collapsed to zero quaternion: {mid:?}"
    );
}

// ---------------------------------------------------------------------------
// C.5 — CUBICSPLINE
// ---------------------------------------------------------------------------

#[test]
fn cubic_at_keyframe_returns_value_not_tangent() {
    // Two keyframes, values [v0=1, v1=2], all tangents zero. Sampling
    // exactly at t=0 must return v0 (the centre triple), not the
    // in-tangent a0 — proving the storage layout `[in, value, out]`
    // is decoded correctly.
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Vec3(vec![
            [99.0, 0.0, 0.0], // a0 — must NOT be returned
            [1.0, 0.0, 0.0],  // v0
            [0.0, 0.0, 0.0],  // b0
            [0.0, 0.0, 0.0],  // a1
            [2.0, 0.0, 0.0],  // v1 — must NOT be returned at t=0
            [88.0, 0.0, 0.0], // b1
        ]),
        interpolation: Interpolation::CubicSpline,
    };
    assert_eq!(unwrap_vec3(s.sample(0.0).unwrap()), [1.0, 0.0, 0.0]);
    assert_eq!(unwrap_vec3(s.sample(1.0).unwrap()), [2.0, 0.0, 0.0]);
}

#[test]
fn cubic_with_zero_tangents_collapses_to_hermite_basis() {
    // With all tangents zero, the Hermite blend reduces to
    //   (2u^3 - 3u^2 + 1) v_k + (-2u^3 + 3u^2) v_{k+1}
    // At u = 0.5: 0.5 * v_k + 0.5 * v_{k+1}.
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Vec3(vec![
            [0.0, 0.0, 0.0],  // a0
            [0.0, 0.0, 0.0],  // v0
            [0.0, 0.0, 0.0],  // b0
            [0.0, 0.0, 0.0],  // a1
            [10.0, 0.0, 0.0], // v1
            [0.0, 0.0, 0.0],  // b1
        ]),
        interpolation: Interpolation::CubicSpline,
    };
    assert!(vec3_close(
        unwrap_vec3(s.sample(0.5).unwrap()),
        [5.0, 0.0, 0.0],
        1e-6,
    ));
}

#[test]
fn cubic_unit_tangent_pulls_midpoint_off_linear() {
    // Same v0/v1 as the previous test but with a non-zero out-tangent
    // on v0 — the midpoint must shift away from the pure-Hermite
    // (5, 0, 0) of the zero-tangent test. Verifies the b_k term is
    // wired up.
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0],
        values: AnimationValues::Vec3(vec![
            [0.0, 0.0, 0.0],  // a0
            [0.0, 0.0, 0.0],  // v0
            [10.0, 0.0, 0.0], // b0 — pushes positive on x at u=0.5
            [0.0, 0.0, 0.0],  // a1
            [10.0, 0.0, 0.0], // v1
            [0.0, 0.0, 0.0],  // b1
        ]),
        interpolation: Interpolation::CubicSpline,
    };
    let mid = unwrap_vec3(s.sample(0.5).unwrap());
    // c_b_k at u=0.5, t_d=1: 1*(0.125 - 0.5 + 0.5) = 0.125 → +1.25.
    // So x should land near 5 + 1.25 = 6.25 (not 5.0).
    assert!(
        (mid[0] - 6.25).abs() < 1e-5,
        "expected x ≈ 6.25, got {mid:?}"
    );
}

#[test]
fn cubic_pre_post_clamp_returns_centre_value() {
    // Even with CUBICSPLINE storage, the spec's C.1 clamping rule
    // applies — at t < first keyframe / t > last, return that
    // keyframe's centre `value`, not its in/out tangent.
    let s = AnimationSampler {
        keyframes: vec![1.0, 2.0],
        values: AnimationValues::Vec3(vec![
            [50.0, 0.0, 0.0], // a0 — must NOT leak
            [1.0, 0.0, 0.0],  // v0
            [60.0, 0.0, 0.0], // b0
            [70.0, 0.0, 0.0], // a1
            [2.0, 0.0, 0.0],  // v1
            [80.0, 0.0, 0.0], // b1 — must NOT leak
        ]),
        interpolation: Interpolation::CubicSpline,
    };
    assert_eq!(unwrap_vec3(s.sample(-99.0).unwrap()), [1.0, 0.0, 0.0]);
    assert_eq!(unwrap_vec3(s.sample(99.0).unwrap()), [2.0, 0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// Multi-segment walk — confirms the binary-search segment-pick is right.
// ---------------------------------------------------------------------------

#[test]
fn linear_multi_segment_walk() {
    // Four keyframes, three segments. Sample in each.
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0, 2.0, 3.0],
        values: AnimationValues::Vec3(vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 5.0, 0.0],
            [20.0, 5.0, 0.0],
        ]),
        interpolation: Interpolation::Linear,
    };
    // First segment: x ramps 0→10.
    assert!(vec3_close(
        unwrap_vec3(s.sample(0.25).unwrap()),
        [2.5, 0.0, 0.0],
        1e-6
    ));
    // Second segment: x stays 10, y ramps 0→5.
    assert!(vec3_close(
        unwrap_vec3(s.sample(1.5).unwrap()),
        [10.0, 2.5, 0.0],
        1e-6
    ));
    // Third segment: x ramps 10→20.
    assert!(vec3_close(
        unwrap_vec3(s.sample(2.25).unwrap()),
        [12.5, 5.0, 0.0],
        1e-6
    ));
}

#[test]
fn step_morph_weights_strides_correctly() {
    // Two morph targets per keyframe, three keyframes — exercises the
    // (k * factor + centre) * stride arithmetic in value_at for Scalar.
    let s = AnimationSampler {
        keyframes: vec![0.0, 1.0, 2.0],
        values: AnimationValues::Scalar(vec![
            1.0, 2.0, // keyframe 0
            3.0, 4.0, // keyframe 1
            5.0, 6.0, // keyframe 2
        ]),
        interpolation: Interpolation::Step,
    };
    assert_eq!(unwrap_scalar(s.sample(0.5).unwrap()), vec![1.0, 2.0]);
    assert_eq!(unwrap_scalar(s.sample(1.5).unwrap()), vec![3.0, 4.0]);
    assert_eq!(unwrap_scalar(s.sample(2.5).unwrap()), vec![5.0, 6.0]);
}
