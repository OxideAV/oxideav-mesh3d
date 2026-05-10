//! Transform Matrix ↔ TRS conversion round-trips.

use oxideav_mesh3d::Transform;

const TOL: f32 = 1e-5;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol + tol * a.abs().max(b.abs())
}

fn quat_close(a: [f32; 4], b: [f32; 4]) -> bool {
    // Quaternions are double-cover: `q` and `-q` represent the same rotation.
    let same = a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, TOL));
    let neg = a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, -*y, TOL));
    same || neg
}

#[test]
fn identity_round_trips() {
    let id = Transform::identity();
    let m = id.to_matrix();
    let back = Transform::from_matrix(m);
    let Transform::Trs {
        translation,
        rotation,
        scale,
    } = back
    else {
        panic!("expected TRS");
    };
    assert!(translation.iter().all(|c| approx_eq(*c, 0.0, TOL)));
    assert!(scale.iter().all(|c| approx_eq(*c, 1.0, TOL)));
    assert!(quat_close(rotation, [0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn pure_translation_round_trips() {
    let trs = Transform::Trs {
        translation: [1.5, -2.25, 3.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    };
    let m = trs.to_matrix();
    assert!(approx_eq(m[0][3], 1.5, TOL));
    assert!(approx_eq(m[1][3], -2.25, TOL));
    assert!(approx_eq(m[2][3], 3.0, TOL));

    let back = Transform::from_matrix(m);
    let Transform::Trs {
        translation,
        rotation,
        scale,
    } = back
    else {
        panic!("expected TRS");
    };
    assert!(approx_eq(translation[0], 1.5, TOL));
    assert!(approx_eq(translation[1], -2.25, TOL));
    assert!(approx_eq(translation[2], 3.0, TOL));
    assert!(scale.iter().all(|c| approx_eq(*c, 1.0, TOL)));
    assert!(quat_close(rotation, [0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn pure_scale_round_trips() {
    let trs = Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [2.0, 3.0, 0.5],
    };
    let m = trs.to_matrix();
    let back = Transform::from_matrix(m);
    let Transform::Trs { scale, .. } = back else {
        panic!("expected TRS");
    };
    assert!(approx_eq(scale[0], 2.0, TOL));
    assert!(approx_eq(scale[1], 3.0, TOL));
    assert!(approx_eq(scale[2], 0.5, TOL));
}

#[test]
fn rotation_around_y_round_trips() {
    // 90 degrees about Y. Quaternion = (0, sin(45), 0, cos(45)).
    let half_pi = std::f32::consts::FRAC_PI_2;
    let s = (half_pi / 2.0).sin();
    let c = (half_pi / 2.0).cos();
    let trs_in = Transform::Trs {
        translation: [0.5, 0.0, -1.0],
        rotation: [0.0, s, 0.0, c],
        scale: [1.0, 1.0, 1.0],
    };
    let m = trs_in.to_matrix();
    let back = Transform::from_matrix(m);
    let Transform::Trs {
        translation,
        rotation,
        scale,
    } = back
    else {
        panic!("expected TRS");
    };
    assert!(quat_close(rotation, [0.0, s, 0.0, c]));
    assert!(approx_eq(translation[0], 0.5, TOL));
    assert!(approx_eq(translation[2], -1.0, TOL));
    assert!(scale.iter().all(|c| approx_eq(*c, 1.0, TOL)));
}

#[test]
fn combined_trs_round_trips() {
    // Arbitrary rotation about (1,2,3), normalised; non-uniform scale; non-zero T.
    let axis = {
        let v = [1.0_f32, 2.0, 3.0];
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let theta = 1.2_f32;
    let s = (theta / 2.0).sin();
    let q = [axis[0] * s, axis[1] * s, axis[2] * s, (theta / 2.0).cos()];
    let original = Transform::Trs {
        translation: [4.0, -1.5, 0.25],
        rotation: q,
        scale: [1.5, 2.0, 0.75],
    };
    let m = original.to_matrix();
    let back = Transform::from_matrix(m);
    let m2 = back.to_matrix();
    // Re-encoding to a matrix and comparing avoids the quaternion
    // double-cover ambiguity entirely.
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                approx_eq(m[i][j], m2[i][j], TOL),
                "mismatch at [{i}][{j}]: {} vs {}",
                m[i][j],
                m2[i][j]
            );
        }
    }
}

#[test]
fn matrix_variant_to_matrix_is_identity_passthrough() {
    let m = [
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 6.0],
        [0.0, 0.0, 1.0, 7.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let t = Transform::Matrix(m);
    let m2 = t.to_matrix();
    assert_eq!(m, m2);
}
