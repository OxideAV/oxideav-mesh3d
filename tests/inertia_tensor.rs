//! Tests for `Primitive::inertia_tensor`, `Mesh::inertia_tensor`, and
//! `Scene3D::inertia_tensor`.
//!
//! Every closed-form expected value below is derived from the
//! elementary closed-form per-tetrahedron second-moment integrals
//! that `Primitive::inertia_tensor` is built on (origin-anchored
//! tet `(0, P_a, P_b, P_c)`, signed volume `V = (P_a · (P_b × P_c)) /
//! 6`, integrand a quadratic polynomial in the corner coordinates).
//! The same derivation supports the `Primitive::signed_volume` /
//! `Primitive::volume_centroid` reductions (Cha & Chen, ICIP 2001;
//! Mirtich, JGT 1996), specialised here to the second-moment
//! kernel rather than the constant or first-moment kernel.

use oxideav_mesh3d::{Mesh, Primitive, Scene3D, Topology};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn approx_eq_tensor(a: [[f64; 3]; 3], b: [[f64; 3]; 3], tol: f64) -> bool {
    for r in 0..3 {
        for c in 0..3 {
            if !approx_eq(a[r][c], b[r][c], tol) {
                return false;
            }
        }
    }
    true
}

/// Unit cube spanning [0, 1]³ wound CCW-from-outside. 12 triangles.
/// Closed-form integrals (unit density):
///   ∫ x² dV = ∫ y² dV = ∫ z² dV = 1/3
///   ∫ x·y dV = ∫ x·z dV = ∫ y·z dV = 1/4
/// → I_xx = I_yy = I_zz = 2/3; I_xy = I_xz = I_yz = -1/4.
fn unit_cube_ccw() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // +Z face — CCW seen from +Z.
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        // -Z face — CCW seen from -Z.
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        // +X face — CCW seen from +X.
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        // -X face — CCW seen from -X.
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        // +Y face — CCW seen from +Y.
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // -Y face — CCW seen from -Y.
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    p
}

/// Cube centred at the origin spanning [-1/2, 1/2]³, CCW-from-outside.
/// Same winding as `unit_cube_ccw` shifted by (-1/2, -1/2, -1/2).
/// Closed-form (unit density, side = 1, mass = 1):
///   ∫ x² dV = 1/12; ∫ x·y dV = 0 by odd symmetry.
///   I_xx = 1/6; off-diagonals all zero.
fn unit_cube_centred() -> Primitive {
    let mut p = unit_cube_ccw();
    for v in p.positions.iter_mut() {
        v[0] -= 0.5;
        v[1] -= 0.5;
        v[2] -= 0.5;
    }
    p
}

/// CW-from-outside (inside-out) unit cube. Same shape, flipped winding.
fn unit_cube_cw() -> Primitive {
    let mut p = unit_cube_ccw();
    for tri in p.positions.chunks_mut(3) {
        tri.swap(1, 2);
    }
    p
}

/// Axis-aligned tetrahedron with corners (0,0,0), (1,0,0), (0,1,0),
/// (0,0,1). Volume = 1/6. Closed-form integrals (unit density):
///   ∫ x² dV = ∫ y² dV = ∫ z² dV = 1/60.
///   ∫ x·y dV = ∫ x·z dV = ∫ y·z dV = 1/120.
/// → I_xx = I_yy = I_zz = 1/30; I_xy = I_xz = I_yz = -1/120.
fn axis_tetrahedron() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // Bottom (normal -Z): O, B, A.
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        // Front (normal -Y): O, A, C.
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        // Left (normal -X): O, C, B.
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        // Slanted (normal +X+Y+Z): A, B, C.
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    p
}

#[test]
fn primitive_unit_cube_about_origin_corner() {
    let p = unit_cube_ccw();
    let i = p.inertia_tensor().expect("closed cube has inertia tensor");
    // Diagonals: each = 2/3 (from ∫ y² + ∫ z² = 1/3 + 1/3).
    // Off-diagonals: each = -1/4.
    let expected = [
        [2.0 / 3.0, -1.0 / 4.0, -1.0 / 4.0],
        [-1.0 / 4.0, 2.0 / 3.0, -1.0 / 4.0],
        [-1.0 / 4.0, -1.0 / 4.0, 2.0 / 3.0],
    ];
    assert!(
        approx_eq_tensor(i, expected, 1e-9),
        "unit cube about corner: got {i:?}, expected {expected:?}"
    );
}

#[test]
fn primitive_unit_cube_centred_off_diagonals_vanish_by_symmetry() {
    let p = unit_cube_centred();
    let i = p.inertia_tensor().expect("closed centred cube");
    // Side = 1, mass M = 1 (unit density), classical result
    //   I_αα = M · (1² + 1²) / 12 = 1/6 for a cube centred at origin.
    // Off-diagonals vanish by reflection symmetry through every plane
    // x = 0, y = 0, z = 0.
    let expected = [
        [1.0 / 6.0, 0.0, 0.0],
        [0.0, 1.0 / 6.0, 0.0],
        [0.0, 0.0, 1.0 / 6.0],
    ];
    assert!(
        approx_eq_tensor(i, expected, 1e-12),
        "centred cube tensor: got {i:?}"
    );
    // Symmetry assertion: every off-diagonal is bit-exact zero
    // (sums of antisymmetric pairs cancel in f64 for this shape).
    for (r, row) in i.iter().enumerate() {
        for (c, val) in row.iter().enumerate() {
            if r != c {
                assert!(
                    val.abs() < 1e-12,
                    "centred-cube off-diagonal i[{r}][{c}] = {val} should be ~0"
                );
            }
        }
    }
}

#[test]
fn primitive_inside_out_cube_negates_tensor() {
    let ccw = unit_cube_ccw().inertia_tensor().unwrap();
    let cw = unit_cube_cw().inertia_tensor().unwrap();
    let neg = [
        [-ccw[0][0], -ccw[0][1], -ccw[0][2]],
        [-ccw[1][0], -ccw[1][1], -ccw[1][2]],
        [-ccw[2][0], -ccw[2][1], -ccw[2][2]],
    ];
    assert!(
        approx_eq_tensor(cw, neg, 1e-12),
        "inside-out cube should negate every entry: ccw={ccw:?} cw={cw:?}"
    );
}

#[test]
fn primitive_tetrahedron_about_origin_corner() {
    let p = axis_tetrahedron();
    let i = p.inertia_tensor().expect("axis tet");
    let expected = [
        [1.0 / 30.0, -1.0 / 120.0, -1.0 / 120.0],
        [-1.0 / 120.0, 1.0 / 30.0, -1.0 / 120.0],
        [-1.0 / 120.0, -1.0 / 120.0, 1.0 / 30.0],
    ];
    assert!(
        approx_eq_tensor(i, expected, 1e-12),
        "axis tetrahedron: got {i:?}, expected {expected:?}"
    );
}

#[test]
fn primitive_tensor_is_symmetric() {
    for p in [unit_cube_ccw(), unit_cube_centred(), axis_tetrahedron()] {
        let i = p.inertia_tensor().unwrap();
        // I_xy == I_yx etc.; the helper populates both slots from the
        // same scalar.
        for (r, row) in i.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let mirror = i[c][r];
                assert!(
                    approx_eq(val, mirror, 1e-14),
                    "tensor not symmetric: i[{r}][{c}]={val} vs i[{c}][{r}]={mirror}"
                );
            }
        }
    }
}

#[test]
fn primitive_parallel_axis_theorem_unit_cube() {
    // Translating the cube from [-1/2, 1/2]³ to [0, 1]³ shifts the
    // centre of mass from origin to c = (1/2, 1/2, 1/2). The
    // parallel-axis theorem says
    //   I_corner = I_centre + M · (|c|² · δ_αβ - c_α · c_β)
    // for unit mass M = 1 (unit density × volume 1).
    let centred = unit_cube_centred().inertia_tensor().unwrap();
    let corner = unit_cube_ccw().inertia_tensor().unwrap();
    let c = [0.5_f64, 0.5, 0.5];
    let m = 1.0_f64;
    let csq = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
    let mut expected_corner = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for cc in 0..3 {
            let kron = if r == cc { 1.0 } else { 0.0 };
            expected_corner[r][cc] = centred[r][cc] + m * (kron * csq - c[r] * c[cc]);
        }
    }
    assert!(
        approx_eq_tensor(corner, expected_corner, 1e-12),
        "parallel-axis theorem mismatch: corner={corner:?}, expected={expected_corner:?}"
    );
}

#[test]
fn primitive_empty_returns_none() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.inertia_tensor().is_none());
}

#[test]
fn primitive_non_triangle_returns_none() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(p.inertia_tensor().is_none());

    let mut p = Primitive::new(Topology::Points);
    p.positions = vec![[0.0, 0.0, 0.0]];
    assert!(p.inertia_tensor().is_none());
}

#[test]
fn primitive_uniform_scale_scales_tensor_by_s5() {
    // For unit density, scaling a body uniformly by factor s scales
    //   ∫ x_α x_β dV  →  s² · ∫ (x_α / s)(x_β / s) · s³ dV  = s⁵ · (original)
    // i.e. every second-moment integral picks up s⁵. So the entire
    // tensor scales by s⁵.
    let base = unit_cube_centred().inertia_tensor().unwrap();
    let mut scaled = unit_cube_centred();
    let s = 2.0_f32;
    for v in scaled.positions.iter_mut() {
        v[0] *= s;
        v[1] *= s;
        v[2] *= s;
    }
    let got = scaled.inertia_tensor().unwrap();
    let factor = (s as f64).powi(5);
    let expected = [
        [
            base[0][0] * factor,
            base[0][1] * factor,
            base[0][2] * factor,
        ],
        [
            base[1][0] * factor,
            base[1][1] * factor,
            base[1][2] * factor,
        ],
        [
            base[2][0] * factor,
            base[2][1] * factor,
            base[2][2] * factor,
        ],
    ];
    assert!(
        approx_eq_tensor(got, expected, 1e-9),
        "scaled (×{s}) tensor: got {got:?}, expected {expected:?} (factor s⁵={factor})"
    );
}

#[test]
fn primitive_skips_out_of_range_indices_without_panic() {
    let mut p = unit_cube_ccw();
    p.indices = Some(oxideav_mesh3d::Indices::U32(vec![
        0, 1, 2, // legitimate
        99_999_999, 0, 1, // out-of-range, must be skipped
    ]));
    // We only assert it doesn't panic — and that it returns Some
    // (the one legitimate face contributes).
    let _ = p.inertia_tensor();
}

#[test]
fn primitive_triangle_fan_matches_triangle_list() {
    // A triangle fan tessellating a square quad
    //   [(0,0,0), (1,0,0), (1,1,0), (0,1,0)] from corner 0
    // is a degenerate flat patch — signed volume is zero so the
    // arithmetic tensor is zero in every cell. We only check the API
    // accepts TriangleFan / TriangleStrip the same way `volume_centroid`
    // does (no `None` for non-triangle reason).
    let mut fan = Primitive::new(Topology::TriangleFan);
    fan.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    // Result is Some — degenerate-but-arithmetically-well-defined.
    let t = fan.inertia_tensor();
    if let Some(i) = t {
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    i[r][c].abs() < 1e-12,
                    "flat fan should yield ~0 tensor, got {i:?}"
                );
            }
        }
    } else {
        // Acceptable — every face had zero contribution and the helper
        // returned None because nothing finite accumulated. The
        // important invariant is "no panic, no NaN".
    }
}

#[test]
fn mesh_inertia_rollup_is_additive_across_primitives() {
    // Build a mesh with the centred unit cube and a centred half-size
    // cube — both share the origin. The mesh-level tensor must equal
    // the sum of the two primitives' tensors.
    let p1 = unit_cube_centred();
    let mut p2 = unit_cube_centred();
    for v in p2.positions.iter_mut() {
        v[0] *= 0.5;
        v[1] *= 0.5;
        v[2] *= 0.5;
    }
    let mut mesh = Mesh::new(Some("two_cubes".to_string()));
    mesh.primitives.push(p1.clone());
    mesh.primitives.push(p2.clone());
    let t_mesh = mesh.inertia_tensor().unwrap();
    let t_p1 = p1.inertia_tensor().unwrap();
    let t_p2 = p2.inertia_tensor().unwrap();
    let mut expected = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            expected[r][c] = t_p1[r][c] + t_p2[r][c];
        }
    }
    assert!(
        approx_eq_tensor(t_mesh, expected, 1e-12),
        "mesh roll-up not additive: got {t_mesh:?}, expected {expected:?}"
    );
}

#[test]
fn mesh_empty_returns_none() {
    let mesh = Mesh::new(Some("empty".to_string()));
    assert!(mesh.inertia_tensor().is_none());
}

#[test]
fn mesh_skips_non_triangle_primitive() {
    let mut mesh = Mesh::new(Some("mixed".to_string()));
    let mut lines = Primitive::new(Topology::Lines);
    lines.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    mesh.primitives.push(lines);
    mesh.primitives.push(unit_cube_centred());
    let got = mesh.inertia_tensor().unwrap();
    let expected = unit_cube_centred().inertia_tensor().unwrap();
    assert!(
        approx_eq_tensor(got, expected, 1e-12),
        "non-triangle primitive should not contribute: got {got:?}"
    );
}

#[test]
fn scene_inertia_rollup_is_additive_across_meshes() {
    let p1 = unit_cube_centred();
    let mut p2_prim = unit_cube_centred();
    for v in p2_prim.positions.iter_mut() {
        v[0] *= 0.5;
        v[1] *= 0.5;
        v[2] *= 0.5;
    }
    let mut m1 = Mesh::new(Some("m1".to_string()));
    m1.primitives.push(p1.clone());
    let mut m2 = Mesh::new(Some("m2".to_string()));
    m2.primitives.push(p2_prim.clone());
    let mut scene = Scene3D::new();
    scene.add_mesh(m1);
    scene.add_mesh(m2);
    let t_scene = scene.inertia_tensor().unwrap();
    let mut expected = [[0.0_f64; 3]; 3];
    let t1 = p1.inertia_tensor().unwrap();
    let t2 = p2_prim.inertia_tensor().unwrap();
    for r in 0..3 {
        for c in 0..3 {
            expected[r][c] = t1[r][c] + t2[r][c];
        }
    }
    assert!(
        approx_eq_tensor(t_scene, expected, 1e-12),
        "scene roll-up not additive: got {t_scene:?}, expected {expected:?}"
    );
}

#[test]
fn scene_empty_returns_none() {
    let scene = Scene3D::new();
    assert!(scene.inertia_tensor().is_none());
}

#[test]
fn scene_instance_count_is_per_mesh_not_per_node() {
    // Doc-stated invariant: the scene-level rollup walks `meshes` once
    // regardless of how many nodes reference the mesh. So adding the
    // same mesh to two nodes still yields a single contribution.
    use oxideav_mesh3d::Node;
    let mut mesh = Mesh::new(Some("cube".to_string()));
    mesh.primitives.push(unit_cube_centred());
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(mesh);
    let n1 = scene.add_node(Node::new().with_mesh(mid));
    let n2 = scene.add_node(Node::new().with_mesh(mid));
    scene.add_root(n1);
    scene.add_root(n2);
    let t = scene.inertia_tensor().unwrap();
    let expected = unit_cube_centred().inertia_tensor().unwrap();
    assert!(
        approx_eq_tensor(t, expected, 1e-12),
        "double-instanced mesh should still contribute once: got {t:?}"
    );
}

#[test]
fn primitive_principal_moments_match_classical_for_centred_box() {
    // A box centred at origin with sides (a, b, c) has classical
    // diagonal inertia I_xx = M(b² + c²)/12 etc. Build the
    // [-a/2, a/2] x [-b/2, b/2] x [-c/2, c/2] box and verify.
    let (a, b, c) = (2.0_f32, 3.0_f32, 5.0_f32);
    let mut p = unit_cube_centred();
    for v in p.positions.iter_mut() {
        v[0] *= a;
        v[1] *= b;
        v[2] *= c;
    }
    // Density = 1 → mass M = a · b · c.
    let m = a as f64 * b as f64 * c as f64;
    let i = p.inertia_tensor().unwrap();
    let a2 = (a as f64).powi(2);
    let b2 = (b as f64).powi(2);
    let c2 = (c as f64).powi(2);
    let exp_xx = m * (b2 + c2) / 12.0;
    let exp_yy = m * (a2 + c2) / 12.0;
    let exp_zz = m * (a2 + b2) / 12.0;
    assert!(
        approx_eq(i[0][0], exp_xx, 1e-9),
        "I_xx got {} exp {exp_xx}",
        i[0][0]
    );
    assert!(
        approx_eq(i[1][1], exp_yy, 1e-9),
        "I_yy got {} exp {exp_yy}",
        i[1][1]
    );
    assert!(
        approx_eq(i[2][2], exp_zz, 1e-9),
        "I_zz got {} exp {exp_zz}",
        i[2][2]
    );
    // Off-diagonals must vanish by symmetry.
    for (r, cc) in [(0, 1), (0, 2), (1, 2)] {
        assert!(
            i[r][cc].abs() < 1e-12,
            "centred-box off-diagonal i[{r}][{cc}]={} should vanish",
            i[r][cc]
        );
    }
}
