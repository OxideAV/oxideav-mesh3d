//! Tests for `Primitive::world_inertia_tensor`,
//! `Mesh::world_inertia_tensor`, and `Scene3D::world_inertia_tensor` —
//! the transform-aware (world-frame) unit-density inertia tensor that
//! closes the per-instance gap round 259 flagged.
//!
//! Every expected value below is derived from first principles: the
//! origin-anchored per-tetrahedron second-moment integrals
//! `Primitive::inertia_tensor` rests on (Mirtich, JGT 1996; Cha & Chen,
//! ICIP 2001), evaluated in the world frame after the affine map is
//! applied to every corner. The cross-checks are the classical
//! rigid-body identities: parallel-axis theorem under translation,
//! `R · I · Rᵀ` similarity under rotation, `s⁵` scaling under uniform
//! scale, and sign-flip under a mirror.

use oxideav_mesh3d::{Mesh, Node, Primitive, Scene3D, Topology, Transform};

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

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn translation_matrix(t: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn scale_matrix(s: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [s[0], 0.0, 0.0, 0.0],
        [0.0, s[1], 0.0, 0.0],
        [0.0, 0.0, s[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Rotation about +Z by `theta` (radians), as a row-major
/// column-vector affine 4x4. The upper-left 3x3 is the f64 rotation
/// `r` returned separately for the `R · I · Rᵀ` cross-check.
fn rot_z(theta: f64) -> ([[f32; 4]; 4], [[f64; 3]; 3]) {
    let (s, c) = (theta.sin(), theta.cos());
    let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    let m = [
        [c as f32, -s as f32, 0.0, 0.0],
        [s as f32, c as f32, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    (m, r)
}

/// `R · I · Rᵀ` for a 3x3 `R` and symmetric 3x3 `I`.
fn conjugate(r: [[f64; 3]; 3], i: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // tmp = R · I
    let mut tmp = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += r[a][k] * i[k][b];
            }
            tmp[a][b] = s;
        }
    }
    // out = tmp · Rᵀ
    let mut out = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += tmp[a][k] * r[b][k];
            }
            out[a][b] = s;
        }
    }
    out
}

/// CCW-from-outside unit cube spanning [0, 1]³ (12 triangles, closed
/// two-manifold). Same layout as `tests/inertia_tensor.rs`.
fn unit_cube_ccw() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // +Z
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        // -Z
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        // +X
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        // -X
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        // +Y
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // -Y
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
fn unit_cube_centred() -> Primitive {
    let mut p = unit_cube_ccw();
    for v in p.positions.iter_mut() {
        v[0] -= 0.5;
        v[1] -= 0.5;
        v[2] -= 0.5;
    }
    p
}

// ---------------------------------------------------------------------
// Primitive level
// ---------------------------------------------------------------------

#[test]
fn primitive_identity_matches_local() {
    let p = unit_cube_ccw();
    let local = p.inertia_tensor().unwrap();
    let world = p.world_inertia_tensor(IDENTITY).unwrap();
    assert!(
        approx_eq_tensor(world, local, 1e-12),
        "identity world transform must reproduce the local tensor: world={world:?} local={local:?}"
    );
}

#[test]
fn primitive_translation_obeys_parallel_axis() {
    // The centred cube has centre of mass at the origin and unit mass
    // (unit density × volume 1). Translating by `t` moves the centre of
    // mass to `c = t`. The inertia about the *world origin* is then the
    // parallel-axis shift of the centred tensor:
    //   I_origin = I_centred + M · (|c|² δ_αβ - c_α c_β).
    let p = unit_cube_centred();
    let centred = p.inertia_tensor().unwrap();
    let t = [3.0_f32, -2.0, 1.5];
    let world = p.world_inertia_tensor(translation_matrix(t)).unwrap();
    let c = [t[0] as f64, t[1] as f64, t[2] as f64];
    let m = 1.0_f64; // unit density, volume 1
    let csq = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
    let mut expected = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for cc in 0..3 {
            let kron = if r == cc { 1.0 } else { 0.0 };
            expected[r][cc] = centred[r][cc] + m * (kron * csq - c[r] * c[cc]);
        }
    }
    assert!(
        approx_eq_tensor(world, expected, 1e-9),
        "translation must obey the parallel-axis theorem: world={world:?} expected={expected:?}"
    );
}

#[test]
fn primitive_uniform_scale_scales_by_s5() {
    // A uniform scale `s` multiplies every second-moment integral
    // (two position powers + three volume powers) by s⁵, so the whole
    // tensor scales by s⁵. Centred so translation does not interfere.
    let p = unit_cube_centred();
    let base = p.inertia_tensor().unwrap();
    let s = 2.0_f32;
    let world = p.world_inertia_tensor(scale_matrix([s, s, s])).unwrap();
    let f = (s as f64).powi(5);
    let mut expected = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            expected[r][c] = base[r][c] * f;
        }
    }
    assert!(
        approx_eq_tensor(world, expected, 1e-9),
        "uniform scale s must scale the tensor by s⁵: world={world:?} expected={expected:?}"
    );
}

#[test]
fn primitive_rotation_is_similarity_transform() {
    // A pure rotation about the origin leaves the body's mass and
    // origin-distance distribution intact, so the inertia tensor
    // transforms by the similarity `I_world = R · I_local · Rᵀ`.
    let p = unit_cube_ccw();
    let local = p.inertia_tensor().unwrap();
    let (m, r) = rot_z(0.7);
    let world = p.world_inertia_tensor(m).unwrap();
    let expected = conjugate(r, local);
    // The world matrix is stored as `f32`, so `sin`/`cos` are truncated
    // before the integral — agreement is to `f32` precision (~1e-6),
    // not the `f64` integral precision the translation tests enjoy.
    assert!(
        approx_eq_tensor(world, expected, 1e-5),
        "rotation must act as R·I·Rᵀ: world={world:?} expected={expected:?}"
    );
}

#[test]
fn primitive_rotation_preserves_trace() {
    // The trace of the inertia tensor (`2 ∫ |x|² dV`) is rotation-
    // invariant about the origin — a sanity cross-check independent of
    // the full similarity test.
    let p = unit_cube_ccw();
    let local = p.inertia_tensor().unwrap();
    let (m, _) = rot_z(1.3);
    let world = p.world_inertia_tensor(m).unwrap();
    let tr_local = local[0][0] + local[1][1] + local[2][2];
    let tr_world = world[0][0] + world[1][1] + world[2][2];
    // `f32`-stored rotation matrix → trace agreement to `f32` precision.
    assert!(
        approx_eq(tr_local, tr_world, 1e-5),
        "rotation about origin must preserve the trace: local={tr_local} world={tr_world}"
    );
}

#[test]
fn primitive_mirror_negates_tensor() {
    // A single-axis mirror flips the triangle winding (det(M_3) < 0),
    // so the signed second-moment integrals — and hence the whole
    // tensor — negate, the same way `world_signed_volume` does.
    let p = unit_cube_centred();
    let base = p.inertia_tensor().unwrap();
    let world = p
        .world_inertia_tensor(scale_matrix([-1.0, 1.0, 1.0]))
        .unwrap();
    let mut expected = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            expected[r][c] = -base[r][c];
        }
    }
    assert!(
        approx_eq_tensor(world, expected, 1e-12),
        "single-axis mirror must negate the tensor: world={world:?} expected={expected:?}"
    );
}

#[test]
fn primitive_result_is_symmetric() {
    let p = unit_cube_ccw();
    let (m, _) = rot_z(0.4);
    let world = p
        .world_inertia_tensor(translation_matrix([1.0, 2.0, 3.0]))
        .unwrap();
    let rotated = p.world_inertia_tensor(m).unwrap();
    for i in [world, rotated] {
        for (r, row) in i.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let mirror = i[c][r];
                assert!(
                    approx_eq(val, mirror, 1e-12),
                    "tensor not symmetric: i[{r}][{c}]={val} vs i[{c}][{r}]={mirror}"
                );
            }
        }
    }
}

#[test]
fn primitive_non_triangle_returns_none() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(p.world_inertia_tensor(IDENTITY).is_none());

    let mut p = Primitive::new(Topology::Points);
    p.positions = vec![[0.0, 0.0, 0.0]];
    assert!(p.world_inertia_tensor(IDENTITY).is_none());
}

#[test]
fn primitive_empty_returns_none() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.world_inertia_tensor(IDENTITY).is_none());
}

#[test]
fn primitive_nan_matrix_returns_none() {
    let p = unit_cube_ccw();
    let mut m = IDENTITY;
    m[0][0] = f32::NAN;
    assert!(
        p.world_inertia_tensor(m).is_none(),
        "a NaN in the world matrix collapses every transformed corner → None"
    );
}

#[test]
fn primitive_out_of_range_indices_skipped() {
    // An index buffer pointing past the vertex pool must be skipped,
    // not panic. A single valid cube prim with a bogus trailing index
    // triple still yields the cube tensor.
    let mut p = unit_cube_ccw();
    // Append an out-of-range triple to a fresh U32 index buffer that
    // first enumerates the implicit order, then the bogus triple.
    let n = p.positions.len() as u32;
    let mut idx: Vec<u32> = (0..n).collect();
    idx.extend_from_slice(&[n + 5, n + 6, n + 7]);
    p.indices = Some(oxideav_mesh3d::Indices::U32(idx));
    let got = p.world_inertia_tensor(IDENTITY).unwrap();
    let expected = unit_cube_ccw().inertia_tensor().unwrap();
    assert!(
        approx_eq_tensor(got, expected, 1e-12),
        "out-of-range index triple must be skipped: got {got:?}"
    );
}

#[test]
fn primitive_strip_matches_list() {
    // A triangle strip and the equivalent triangle list must produce
    // the same world tensor (the alternating-winding de-strip feeds the
    // same triangles in).
    let list = unit_cube_ccw();
    // Build a 2-triangle strip quad and the equivalent list; compare.
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let mut listq = Primitive::new(Topology::Triangles);
    // Strip (0,1,2,3) expands to triangles (0,1,2) and (2,1,3).
    listq.positions = strip.positions.clone();
    listq.indices = Some(oxideav_mesh3d::Indices::U32(vec![0, 1, 2, 2, 1, 3]));
    let m = translation_matrix([2.0, 0.0, 0.0]);
    let a = strip.world_inertia_tensor(m).unwrap();
    let b = listq.world_inertia_tensor(m).unwrap();
    assert!(
        approx_eq_tensor(a, b, 1e-12),
        "strip and equivalent list must match: strip={a:?} list={b:?}"
    );
    // Cross-check the closed-cube list path runs too (no panic).
    let _ = list.world_inertia_tensor(m).unwrap();
}

// ---------------------------------------------------------------------
// Mesh level
// ---------------------------------------------------------------------

#[test]
fn mesh_rollup_is_additive() {
    let p1 = unit_cube_centred();
    let mut p2 = unit_cube_centred();
    for v in p2.positions.iter_mut() {
        v[0] *= 0.5;
        v[1] *= 0.5;
        v[2] *= 0.5;
    }
    let mut mesh = Mesh::new(Some("two".to_string()));
    mesh.primitives.push(p1.clone());
    mesh.primitives.push(p2.clone());
    let m = translation_matrix([1.0, 2.0, -1.0]);
    let got = mesh.world_inertia_tensor(m).unwrap();
    let t1 = p1.world_inertia_tensor(m).unwrap();
    let t2 = p2.world_inertia_tensor(m).unwrap();
    let mut expected = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            expected[r][c] = t1[r][c] + t2[r][c];
        }
    }
    assert!(
        approx_eq_tensor(got, expected, 1e-12),
        "mesh world roll-up not additive: got {got:?} expected {expected:?}"
    );
}

#[test]
fn mesh_identity_matches_local_rollup() {
    let mut mesh = Mesh::new(Some("cube".to_string()));
    mesh.primitives.push(unit_cube_ccw());
    let got = mesh.world_inertia_tensor(IDENTITY).unwrap();
    let expected = mesh.inertia_tensor().unwrap();
    assert!(approx_eq_tensor(got, expected, 1e-12));
}

#[test]
fn mesh_skips_non_triangle_primitive() {
    let mut mesh = Mesh::new(Some("mixed".to_string()));
    let mut lines = Primitive::new(Topology::Lines);
    lines.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    mesh.primitives.push(lines);
    mesh.primitives.push(unit_cube_centred());
    let m = scale_matrix([2.0, 2.0, 2.0]);
    let got = mesh.world_inertia_tensor(m).unwrap();
    let expected = unit_cube_centred().world_inertia_tensor(m).unwrap();
    assert!(
        approx_eq_tensor(got, expected, 1e-12),
        "non-triangle primitive must not contribute: got {got:?}"
    );
}

#[test]
fn mesh_empty_returns_none() {
    let mesh = Mesh::new(Some("empty".to_string()));
    assert!(mesh.world_inertia_tensor(IDENTITY).is_none());
}

// ---------------------------------------------------------------------
// Scene level
// ---------------------------------------------------------------------

fn translation_trs(t: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: t,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn scale_trs(s: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: s,
    }
}

#[test]
fn scene_identity_root_matches_resource_rollup() {
    // A single mesh under one identity-transform root reproduces the
    // resource-level `Scene3D::inertia_tensor` (local frame == world).
    let mut mesh = Mesh::new(Some("cube".to_string()));
    mesh.primitives.push(unit_cube_ccw());
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(mesh);
    let nid = scene.add_node(Node::new().with_mesh(mid));
    scene.add_root(nid);
    let world = scene.world_inertia_tensor().unwrap();
    let local = scene.inertia_tensor().unwrap();
    assert!(
        approx_eq_tensor(world, local, 1e-12),
        "identity-rooted scene must match the resource rollup: world={world:?} local={local:?}"
    );
}

#[test]
fn scene_translated_node_obeys_parallel_axis() {
    // One centred cube under a translated node: the world tensor about
    // the world origin is the parallel-axis shift by the node's
    // translation.
    let mut mesh = Mesh::new(Some("cube".to_string()));
    mesh.primitives.push(unit_cube_centred());
    let centred = unit_cube_centred().inertia_tensor().unwrap();
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(mesh);
    let t = [4.0_f32, 1.0, -3.0];
    let nid = scene.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_trs(t)),
    );
    scene.add_root(nid);
    let world = scene.world_inertia_tensor().unwrap();
    let c = [t[0] as f64, t[1] as f64, t[2] as f64];
    let csq = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
    let mut expected = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for cc in 0..3 {
            let kron = if r == cc { 1.0 } else { 0.0 };
            expected[r][cc] = centred[r][cc] + (kron * csq - c[r] * c[cc]);
        }
    }
    assert!(
        approx_eq_tensor(world, expected, 1e-9),
        "translated node must obey parallel-axis: world={world:?} expected={expected:?}"
    );
}

#[test]
fn scene_double_instanced_mesh_contributes_twice() {
    // Unlike `Scene3D::inertia_tensor` (resource-once), the world walk
    // is per-instance: two nodes carrying the same mesh contribute two
    // tensors. Two identity instances → 2× the single-instance tensor.
    let mut mesh = Mesh::new(Some("cube".to_string()));
    mesh.primitives.push(unit_cube_centred());
    let single = unit_cube_centred().inertia_tensor().unwrap();
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(mesh);
    let n1 = scene.add_node(Node::new().with_mesh(mid));
    let n2 = scene.add_node(Node::new().with_mesh(mid));
    scene.add_root(n1);
    scene.add_root(n2);
    let world = scene.world_inertia_tensor().unwrap();
    let mut expected = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            expected[r][c] = 2.0 * single[r][c];
        }
    }
    assert!(
        approx_eq_tensor(world, expected, 1e-12),
        "two identity instances must double the tensor: world={world:?}"
    );
}

#[test]
fn scene_ancestor_chain_composes() {
    // A child cube under an intermediate scale under a translated root:
    // the composed world matrix is translation ∘ scale. Cross-check
    // against the primitive helper fed the same composed matrix.
    let mut mesh = Mesh::new(Some("cube".to_string()));
    mesh.primitives.push(unit_cube_centred());
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(mesh);
    let leaf = scene.add_node(Node::new().with_mesh(mid));
    let mut inter = Node::new().with_transform(scale_trs([2.0, 1.0, 1.0]));
    inter.children.push(leaf);
    let inter_id = scene.add_node(inter);
    let mut root = Node::new().with_transform(translation_trs([3.0, 0.0, 0.0]));
    root.children.push(inter_id);
    let root_id = scene.add_node(root);
    scene.add_root(root_id);
    let world = scene.world_inertia_tensor().unwrap();
    // Composed matrix: translate(3,0,0) · scale(2,1,1).
    let composed = [
        [2.0_f32, 0.0, 0.0, 3.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let expected = unit_cube_centred().world_inertia_tensor(composed).unwrap();
    assert!(
        approx_eq_tensor(world, expected, 1e-9),
        "ancestor chain must compose: world={world:?} expected={expected:?}"
    );
}

#[test]
fn scene_detached_mesh_skipped() {
    // A mesh not referenced from any reachable root contributes nothing;
    // the scene returns None when there is no reachable mesh node.
    let mut mesh = Mesh::new(Some("cube".to_string()));
    mesh.primitives.push(unit_cube_ccw());
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(mesh);
    let _unreachable = scene.add_node(Node::new().with_mesh(mid));
    // No root added → nothing reachable.
    assert!(
        scene.world_inertia_tensor().is_none(),
        "no reachable mesh node → None"
    );
}

#[test]
fn scene_empty_returns_none() {
    let scene = Scene3D::new();
    assert!(scene.world_inertia_tensor().is_none());
}

#[test]
fn scene_cycle_is_visited_once() {
    // A node listed as its own descendant must be visited once (cycle
    // guard); a self-cycling single-instance cube yields exactly the
    // single-instance tensor, not an infinite accumulation.
    let mut mesh = Mesh::new(Some("cube".to_string()));
    mesh.primitives.push(unit_cube_centred());
    let single = unit_cube_centred().inertia_tensor().unwrap();
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(mesh);
    let nid = scene.add_node(Node::new().with_mesh(mid));
    // Make the node its own child.
    scene.nodes[nid.0 as usize].children.push(nid);
    scene.add_root(nid);
    let world = scene.world_inertia_tensor().unwrap();
    assert!(
        approx_eq_tensor(world, single, 1e-12),
        "self-cycle node must be visited once: world={world:?} single={single:?}"
    );
}

#[test]
fn scene_two_translated_instances_sum() {
    // Two cubes at distinct world positions: the scene tensor is the
    // sum of the two parallel-axis-shifted per-instance tensors.
    let mut mesh = Mesh::new(Some("cube".to_string()));
    mesh.primitives.push(unit_cube_centred());
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(mesh);
    let ta = [5.0_f32, 0.0, 0.0];
    let tb = [0.0_f32, 7.0, 0.0];
    let na = scene.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_trs(ta)),
    );
    let nb = scene.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_trs(tb)),
    );
    scene.add_root(na);
    scene.add_root(nb);
    let world = scene.world_inertia_tensor().unwrap();
    let ia = unit_cube_centred()
        .world_inertia_tensor(translation_matrix(ta))
        .unwrap();
    let ib = unit_cube_centred()
        .world_inertia_tensor(translation_matrix(tb))
        .unwrap();
    let mut expected = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            expected[r][c] = ia[r][c] + ib[r][c];
        }
    }
    assert!(
        approx_eq_tensor(world, expected, 1e-9),
        "two translated instances must sum: world={world:?} expected={expected:?}"
    );
}
