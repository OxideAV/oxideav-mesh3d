//! Property tests over randomized rigs and animations — seeded LCG,
//! fully deterministic, no external dependencies.
//!
//! Each test generates a few hundred randomized inputs and checks an
//! algebraic invariant of the skinning / pose pipeline rather than a
//! hand-computed value:
//!
//! * identity palettes are geometric no-ops;
//! * rigid (single-joint) skinning equals `transformed()`;
//! * skinning is invariant under uniform weight scaling after
//!   `normalize_joint_weights`;
//! * repaired weight rows sum to 1 (or stay all-zero);
//! * sampled rotations are always unit quaternions;
//! * an empty pose walks the exact rest world transforms on random
//!   trees, and posed walks are deterministic;
//! * `skinned` preserves buffer shapes and non-geometric attributes;
//! * a `Node::weights` override instantiates exactly like the same
//!   vector stored as `Mesh::weights` (two levels, one rung);
//! * `posed(a, t).world_mesh(n)` equals `world_mesh_at(a, t, n)` over
//!   random shared-mesh scenes with mixed weight/transform channels;
//! * `Primitive::morphed` folds exactly `apply_morph_weights`
//!   (including wrong-length soft-skips) and consumes the roster.

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, Interpolation, Node, NodeId, Pose, Primitive, Scene3D, Topology, Transform,
};

/// Minimal deterministic LCG (numerical-recipes constants).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [lo, hi).
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next_u32() as f32 / u32::MAX as f32)
    }
    fn usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n.max(1)
    }
    fn vec3(&mut self, lo: f32, hi: f32) -> [f32; 3] {
        [self.f32(lo, hi), self.f32(lo, hi), self.f32(lo, hi)]
    }
    /// Random unit quaternion (normalised 4-cube sample — biased but
    /// fine for coverage).
    fn quat(&mut self) -> [f32; 4] {
        loop {
            let q = [
                self.f32(-1.0, 1.0),
                self.f32(-1.0, 1.0),
                self.f32(-1.0, 1.0),
                self.f32(-1.0, 1.0),
            ];
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if n > 0.1 {
                return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
            }
        }
    }
    fn affine(&mut self) -> [[f32; 4]; 4] {
        Transform::Trs {
            translation: self.vec3(-5.0, 5.0),
            rotation: self.quat(),
            scale: [self.f32(0.2, 3.0), self.f32(0.2, 3.0), self.f32(0.2, 3.0)],
        }
        .to_matrix()
    }
}

fn random_skinned_primitive(rng: &mut Lcg, n_verts: usize, palette_len: usize) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    for _ in 0..n_verts {
        p.positions.push(rng.vec3(-10.0, 10.0));
    }
    let mut normals = Vec::with_capacity(n_verts);
    let mut tangents = Vec::with_capacity(n_verts);
    let mut joints = Vec::with_capacity(n_verts);
    let mut weights = Vec::with_capacity(n_verts);
    for _ in 0..n_verts {
        let n = rng.vec3(-1.0, 1.0);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(0.1);
        normals.push([n[0] / len, n[1] / len, n[2] / len]);
        tangents.push([1.0, 0.0, 0.0, if rng.usize(2) == 0 { 1.0 } else { -1.0 }]);
        joints.push([
            rng.usize(palette_len) as u16,
            rng.usize(palette_len) as u16,
            rng.usize(palette_len) as u16,
            rng.usize(palette_len) as u16,
        ]);
        // Random positive weights, deliberately not normalised.
        weights.push([
            rng.f32(0.0, 1.0),
            rng.f32(0.0, 1.0),
            rng.f32(0.0, 1.0),
            rng.f32(0.0, 1.0),
        ]);
    }
    p.normals = Some(normals);
    p.tangents = Some(tangents);
    p.joints = Some(joints);
    p.weights = Some(weights);
    p
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

#[test]
fn identity_palette_is_a_no_op_on_random_primitives() {
    let mut rng = Lcg::new(0xC0FFEE);
    for _ in 0..50 {
        let mut prim = random_skinned_primitive(&mut rng, 12, 4);
        prim = prim.normalize_joint_weights(); // rows sum to 1 ⇒ blend == I
        let skinned = prim.skinned(&[IDENTITY; 4]);
        for (a, b) in skinned.positions.iter().zip(prim.positions.iter()) {
            for i in 0..3 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-4,
                    "identity blend moved a vertex: {a:?} vs {b:?}"
                );
            }
        }
    }
}

#[test]
fn rigid_binding_matches_transformed_on_random_matrices() {
    let mut rng = Lcg::new(0xB04E);
    for _ in 0..50 {
        let m = rng.affine();
        let mut prim = random_skinned_primitive(&mut rng, 9, 3);
        let j = rng.usize(3) as u16;
        prim.joints = Some(vec![[j, 0, 0, 0]; 9]);
        prim.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]; 9]);
        let palette = [rng.affine(), rng.affine(), rng.affine()];
        let mut palette = palette;
        palette[j as usize] = m;
        let skinned = prim.skinned(&palette);
        let baked = prim.transformed(m);
        for v in 0..9 {
            for i in 0..3 {
                assert!(
                    (skinned.positions[v][i] - baked.positions[v][i]).abs() < 1e-3,
                    "positions diverge at v{v}: {:?} vs {:?}",
                    skinned.positions[v],
                    baked.positions[v]
                );
                let sn = skinned.normals.as_ref().unwrap()[v];
                let bn = baked.normals.as_ref().unwrap()[v];
                assert!(
                    (sn[i] - bn[i]).abs() < 1e-3,
                    "normals diverge at v{v}: {sn:?} vs {bn:?}"
                );
            }
            assert_eq!(
                skinned.tangents.as_ref().unwrap()[v][3],
                baked.tangents.as_ref().unwrap()[v][3],
                "handedness diverges at v{v}"
            );
        }
    }
}

#[test]
fn skinning_is_scale_invariant_after_normalization() {
    let mut rng = Lcg::new(0x5EED);
    for _ in 0..50 {
        let prim = random_skinned_primitive(&mut rng, 8, 4);
        let scale = rng.f32(0.25, 8.0);
        let mut scaled = prim.clone();
        if let Some(w) = scaled.weights.as_mut() {
            for row in w.iter_mut() {
                for c in row.iter_mut() {
                    *c *= scale;
                }
            }
        }
        let palette = [rng.affine(), rng.affine(), rng.affine(), rng.affine()];
        let a = prim.normalize_joint_weights().skinned(&palette);
        let b = scaled.normalize_joint_weights().skinned(&palette);
        for (pa, pb) in a.positions.iter().zip(b.positions.iter()) {
            for i in 0..3 {
                assert!(
                    (pa[i] - pb[i]).abs() < 1e-3,
                    "scaled weights diverge: {pa:?} vs {pb:?}"
                );
            }
        }
    }
}

#[test]
fn repaired_rows_sum_to_one_or_stay_zero() {
    let mut rng = Lcg::new(0xDEADBEEF);
    for _ in 0..100 {
        let mut prim = random_skinned_primitive(&mut rng, 6, 4);
        // Poison some rows with negatives / NaN / all-zero.
        if let Some(w) = prim.weights.as_mut() {
            w[0] = [-1.0, 0.5, 0.0, 0.0];
            w[1] = [f32::NAN, f32::NAN, f32::NAN, f32::NAN];
            w[2] = [0.0; 4];
        }
        let fixed = prim.normalize_joint_weights();
        for row in fixed.weights.as_ref().unwrap() {
            let sum: f32 = row.iter().sum();
            assert!(
                sum == 0.0 || (sum - 1.0).abs() <= 4.0 * f32::EPSILON,
                "row sums to {sum}: {row:?}"
            );
            assert!(row.iter().all(|w| *w >= 0.0 && w.is_finite()), "{row:?}");
        }
    }
}

#[test]
fn sampled_rotations_are_always_unit() {
    let mut rng = Lcg::new(0x0451);
    for _ in 0..40 {
        let n_keys = 2 + rng.usize(4);
        let keyframes: Vec<f32> = (0..n_keys).map(|k| k as f32).collect();
        let (values, interpolation) = if rng.usize(2) == 0 {
            (
                AnimationValues::Quat((0..n_keys).map(|_| rng.quat()).collect()),
                Interpolation::Linear,
            )
        } else {
            // CubicSpline: [in, value, out] per keyframe, tangents
            // arbitrary (unnormalised on purpose).
            let mut v = Vec::new();
            for _ in 0..n_keys {
                v.push(rng.quat().map(|c| c * rng.f32(-3.0, 3.0)));
                v.push(rng.quat());
                v.push(rng.quat().map(|c| c * rng.f32(-3.0, 3.0)));
            }
            (AnimationValues::Quat(v), Interpolation::CubicSpline)
        };
        let mut anim = Animation::new(None);
        anim.channels.push(AnimationChannel {
            target: AnimationTarget {
                node: NodeId(0),
                property: AnimationProperty::Rotation,
            },
            sampler: AnimationSampler {
                keyframes,
                values,
                interpolation,
            },
        });
        for step in 0..20 {
            let t = rng.f32(-1.0, n_keys as f32);
            let pose = anim.sample_pose(t, 1);
            if let Some(q) = pose.rotations[0] {
                let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                assert!(
                    (n - 1.0).abs() < 1e-4,
                    "non-unit rotation at t={t} step {step}: {q:?} (|q|={n})"
                );
            }
        }
    }
}

/// Random node forest: every node's parent index is below its own, a
/// random subset of low-index nodes are roots.
fn random_tree(rng: &mut Lcg, n: usize) -> Scene3D {
    let mut s = Scene3D::new();
    for i in 0..n {
        let node = Node::new().with_transform(Transform::Trs {
            translation: rng.vec3(-2.0, 2.0),
            rotation: rng.quat(),
            scale: [rng.f32(0.5, 2.0); 3],
        });
        let id = s.add_node(node);
        if i == 0 || rng.usize(5) == 0 {
            s.add_root(id);
        } else {
            let parent = rng.usize(i);
            s.nodes[parent].children.push(id);
        }
    }
    s
}

#[test]
fn empty_pose_walk_equals_rest_walk_on_random_trees() {
    let mut rng = Lcg::new(0x7EE);
    for _ in 0..30 {
        let n = 1 + rng.usize(20);
        let s = random_tree(&mut rng, n);
        let pose = Pose::new(s.nodes.len());
        assert_eq!(s.posed_node_transforms(&pose), s.world_node_transforms());
    }
}

#[test]
fn posed_walk_is_deterministic() {
    let mut rng = Lcg::new(0xABCD);
    let s = random_tree(&mut rng, 15);
    let mut anim = Animation::new(None);
    for i in 0..15 {
        if rng.usize(2) == 0 {
            anim.channels.push(AnimationChannel {
                target: AnimationTarget {
                    node: NodeId(i),
                    property: AnimationProperty::Rotation,
                },
                sampler: AnimationSampler {
                    keyframes: vec![0.0, 1.0],
                    values: AnimationValues::Quat(vec![rng.quat(), rng.quat()]),
                    interpolation: Interpolation::Linear,
                },
            });
        }
    }
    let p1 = anim.sample_pose(0.37, s.nodes.len());
    let p2 = anim.sample_pose(0.37, s.nodes.len());
    assert_eq!(p1, p2, "sampling is deterministic");
    assert_eq!(
        s.posed_node_transforms(&p1),
        s.posed_node_transforms(&p2),
        "walk is deterministic"
    );
}

#[test]
fn skinned_preserves_shapes_and_non_geometric_attributes() {
    let mut rng = Lcg::new(0x91117);
    for _ in 0..30 {
        let mut prim = random_skinned_primitive(&mut rng, 10, 4);
        prim.uvs = vec![vec![[0.5, 0.5]; 10]];
        prim.colors = vec![vec![[1.0, 0.0, 0.0, 1.0]; 10]];
        prim.indices = Some(oxideav_mesh3d::Indices::U32(vec![0, 1, 2, 3, 4, 5]));
        let palette = [rng.affine(), rng.affine(), rng.affine(), rng.affine()];
        let skinned = prim.skinned(&palette);
        assert_eq!(skinned.positions.len(), prim.positions.len());
        assert_eq!(
            skinned.normals.as_ref().map(Vec::len),
            prim.normals.as_ref().map(Vec::len)
        );
        assert_eq!(skinned.uvs, prim.uvs, "UVs pass through untouched");
        assert_eq!(skinned.colors, prim.colors, "colours pass through");
        assert_eq!(skinned.indices, prim.indices, "indices pass through");
        assert_eq!(skinned.topology, prim.topology);
        // Normals stay unit length.
        for n in skinned.normals.as_ref().unwrap() {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "normal not unit: {n:?}");
        }
    }
}

#[test]
fn mesh_normalize_joint_weights_maps_every_primitive() {
    let mut rng = Lcg::new(0x3333);
    let a = random_skinned_primitive(&mut rng, 5, 2);
    let b = random_skinned_primitive(&mut rng, 7, 2);
    let mesh = oxideav_mesh3d::Mesh::new(Some("rig".to_string()))
        .with_primitive(a)
        .with_primitive(b)
        .with_weights(vec![0.5]);
    let fixed = mesh.normalize_joint_weights();
    assert_eq!(fixed.name.as_deref(), Some("rig"));
    assert_eq!(fixed.weights, vec![0.5], "morph weights untouched");
    for prim in &fixed.primitives {
        for row in prim.weights.as_ref().unwrap() {
            let sum: f32 = row.iter().sum();
            assert!(
                sum == 0.0 || (sum - 1.0).abs() <= 4.0 * f32::EPSILON,
                "{row:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Weight-precedence chain properties (node.weights / posed / morphed)
// ---------------------------------------------------------------------------

fn random_morphable_primitive(rng: &mut Lcg, n_verts: usize, n_targets: usize) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    let mut normals = Vec::with_capacity(n_verts);
    for _ in 0..n_verts {
        p.positions.push(rng.vec3(-10.0, 10.0));
        let n = rng.vec3(-1.0, 1.0);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(0.1);
        normals.push([n[0] / len, n[1] / len, n[2] / len]);
    }
    p.normals = Some(normals);
    for _ in 0..n_targets {
        let position = Some((0..n_verts).map(|_| rng.vec3(-2.0, 2.0)).collect());
        // Half the targets also displace normals — per-attribute
        // opt-in must hold on both sides of every equivalence.
        let normal = if rng.usize(2) == 0 {
            Some((0..n_verts).map(|_| rng.vec3(-0.5, 0.5)).collect())
        } else {
            None
        };
        p.targets.push(oxideav_mesh3d::MorphTarget {
            position,
            normal,
            tangent: None,
        });
    }
    p
}

fn morph_channel(rng: &mut Lcg, node: NodeId, n_targets: usize) -> AnimationChannel {
    let values: Vec<f32> = (0..2 * n_targets).map(|_| rng.f32(-1.0, 2.0)).collect();
    AnimationChannel {
        target: AnimationTarget {
            node,
            property: AnimationProperty::MorphWeights,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Scalar(values),
            interpolation: Interpolation::Linear,
        },
    }
}

/// The node override and the mesh default are the same rung expressed
/// at two levels: instantiating `node.weights = w` must equal
/// instantiating `mesh.weights = w` exactly (identical blend path).
#[test]
fn node_override_equals_mesh_defaults_carrying_the_same_vector() {
    let mut rng = Lcg::new(0x0DDBA11);
    for _ in 0..200 {
        let n_targets = 1 + rng.usize(3);
        let n_verts = 3 + rng.usize(8);
        let prim = random_morphable_primitive(&mut rng, n_verts, n_targets);
        let w: Vec<f32> = (0..n_targets).map(|_| rng.f32(-1.0, 2.0)).collect();

        let mut sa = Scene3D::new();
        let ma = sa.add_mesh(oxideav_mesh3d::Mesh::new(None).with_primitive(prim.clone()));
        let na = sa.add_node(Node::new().with_mesh(ma).with_weights(w.clone()));
        sa.add_root(na);

        let mut sb = Scene3D::new();
        let mb = sb.add_mesh(
            oxideav_mesh3d::Mesh::new(None)
                .with_primitive(prim.clone())
                .with_weights(w.clone()),
        );
        let nb = sb.add_node(Node::new().with_mesh(mb));
        sb.add_root(nb);

        let a = sa.world_mesh(na).expect("override side");
        let b = sb.world_mesh(nb).expect("default side");
        assert_eq!(a.primitives[0].positions, b.primitives[0].positions);
        assert_eq!(a.primitives[0].normals, b.primitives[0].normals);
    }
}

/// The doc contract of `posed`: `posed(a, t).world_mesh(n)` equals
/// `world_mesh_at(a, t, n)` for every node — over random scenes where
/// several nodes share one mesh, only some of them weight-driven, and
/// transforms animate too.
#[test]
fn posed_bake_equals_animated_instantiation_on_shared_meshes() {
    let mut rng = Lcg::new(0xBAAB);
    for _ in 0..100 {
        let n_targets = 1 + rng.usize(3);
        let n_verts = 3 + rng.usize(6);
        let prim = random_morphable_primitive(&mut rng, n_verts, n_targets);
        let defaults: Vec<f32> = (0..n_targets).map(|_| rng.f32(0.0, 1.0)).collect();
        let mut s = Scene3D::new();
        let mid = s.add_mesh(
            oxideav_mesh3d::Mesh::new(None)
                .with_primitive(prim)
                .with_weights(defaults),
        );
        let mut anim = Animation::new(None);
        let mut nodes = Vec::new();
        for _ in 0..2 + rng.usize(3) {
            let node = Node::new().with_mesh(mid).with_transform(Transform::Trs {
                translation: rng.vec3(-5.0, 5.0),
                rotation: rng.quat(),
                scale: [1.0, 1.0, 1.0],
            });
            let nid = s.add_node(node);
            s.add_root(nid);
            if rng.usize(2) == 0 {
                anim.channels.push(morph_channel(&mut rng, nid, n_targets));
            }
            if rng.usize(2) == 0 {
                anim.channels.push(AnimationChannel {
                    target: AnimationTarget {
                        node: nid,
                        property: AnimationProperty::Translation,
                    },
                    sampler: AnimationSampler {
                        keyframes: vec![0.0, 1.0],
                        values: AnimationValues::Vec3(vec![
                            rng.vec3(-3.0, 3.0),
                            rng.vec3(-3.0, 3.0),
                        ]),
                        interpolation: Interpolation::Linear,
                    },
                });
            }
            nodes.push(nid);
        }
        let t = rng.f32(0.0, 1.0);
        let baked = s.posed(&anim, t);
        for &nid in &nodes {
            let via_bake = baked.world_mesh(nid).expect("baked");
            let via_anim = s.world_mesh_at(&anim, t, nid).expect("animated");
            assert_eq!(
                via_bake.primitives[0].positions, via_anim.primitives[0].positions,
                "node {nid:?}"
            );
        }
    }
}

/// `Primitive::morphed` is exactly `apply_morph_weights` folded in
/// place — including the soft-skip behaviour on wrong-length weight
/// vectors — with the target roster consumed.
#[test]
fn morphed_lift_matches_apply_morph_weights_on_random_inputs() {
    let mut rng = Lcg::new(0x5EED);
    for _ in 0..200 {
        let n_targets = rng.usize(4);
        let n_verts = 3 + rng.usize(8);
        let prim = random_morphable_primitive(&mut rng, n_verts, n_targets);
        // Deliberately allow mismatched weight-vector lengths.
        let w: Vec<f32> = (0..rng.usize(6)).map(|_| rng.f32(-1.0, 2.0)).collect();
        let folded = prim.morphed(&w);
        let blended = prim.apply_morph_weights(&w);
        assert_eq!(folded.positions, blended.positions);
        assert_eq!(folded.normals, blended.normals);
        assert_eq!(folded.tangents, blended.tangents);
        assert!(folded.targets.is_empty());
    }
}
