//! Tests for the animated instantiation pipeline —
//! `Scene3D::world_mesh_at(animation, t, node)` and
//! `Scene3D::posed(animation, t)` — driving skinning, morphing, and
//! node transforms from sampled animation channels end-to-end.

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, Interpolation, Mesh, MorphTarget, Node, NodeId, Primitive, Scene3D, Skeleton,
    Skin, Topology, Transform,
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

fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn translate(t: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn translation_channel(node: NodeId, from: [f32; 3], to: [f32; 3], end: f32) -> AnimationChannel {
    AnimationChannel {
        target: AnimationTarget {
            node,
            property: AnimationProperty::Translation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, end],
            values: AnimationValues::Vec3(vec![from, to]),
            interpolation: Interpolation::Linear,
        },
    }
}

/// Two-joint rig: root joint at origin, elbow joint rest at (1,0,0)
/// with the matching inverse bind. Vertices along +X: base bound to
/// joint 0, outer two bound to joint 1. Returns (scene, mesh node,
/// elbow joint node).
fn two_bone_arm() -> (Scene3D, NodeId, NodeId) {
    let mut s = Scene3D::new();
    let root = s.add_node(Node::new());
    let elbow = s.add_node(Node::new().with_transform(Transform::Trs {
        translation: [1.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }));
    s.nodes[root.0 as usize].children.push(elbow);
    s.add_root(root);
    let skel = s.add_skeleton(Skeleton {
        name: None,
        joints: vec![root, elbow],
        inverse_bind_matrices: vec![identity(), translate([-1.0, 0.0, 0.0])],
    });
    let skin = s.add_skin(Skin::new(skel));
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    prim.joints = Some(vec![[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]);
    prim.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]);
    let mid = s.add_mesh(Mesh::new(None).with_primitive(prim));
    let mut mesh_node = Node::new().with_mesh(mid);
    mesh_node.skin = Some(skin);
    let nid = s.add_node(mesh_node);
    s.add_root(nid);
    (s, nid, elbow)
}

/// 90° about Z as an xyzw quaternion.
fn quat_z_90() -> [f32; 4] {
    [
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ]
}

fn elbow_bend_animation(elbow: NodeId) -> Animation {
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: elbow,
            property: AnimationProperty::Rotation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Quat(vec![[0.0, 0.0, 0.0, 1.0], quat_z_90()]),
            interpolation: Interpolation::Linear,
        },
    });
    anim
}

// --- world_mesh_at: skinning driven by animation -------------------------

#[test]
fn rest_frame_matches_world_mesh() {
    let (s, nid, elbow) = two_bone_arm();
    let anim = elbow_bend_animation(elbow);
    let at_rest = s.world_mesh_at(&anim, 0.0, nid).expect("frame 0");
    let rest = s.world_mesh(nid).expect("rest");
    for (a, b) in at_rest.primitives[0]
        .positions
        .iter()
        .zip(rest.primitives[0].positions.iter())
    {
        assert_vec3_eq(*a, *b, "t=0 equals rest pose");
    }
}

#[test]
fn elbow_bend_swings_the_forearm() {
    let (s, nid, elbow) = two_bone_arm();
    let anim = elbow_bend_animation(elbow);
    let bent = s.world_mesh_at(&anim, 1.0, nid).expect("frame 1");
    let p = &bent.primitives[0].positions;
    assert_vec3_eq(p[0], [0.0, 0.0, 0.0], "root vertex fixed");
    assert_vec3_eq(p[1], [1.0, 0.0, 0.0], "elbow pivot fixed");
    assert_vec3_eq(p[2], [1.0, 1.0, 0.0], "forearm bent up 90°");
}

#[test]
fn sampling_clamps_beyond_duration() {
    let (s, nid, elbow) = two_bone_arm();
    let anim = elbow_bend_animation(elbow);
    let over = s.world_mesh_at(&anim, 42.0, nid).expect("clamped");
    let at_end = s.world_mesh_at(&anim, 1.0, nid).expect("end");
    assert_eq!(
        over.primitives[0].positions, at_end.primitives[0].positions,
        "Appendix C.1 clamp"
    );
}

// --- world_mesh_at: morph weights driven by animation ---------------------

fn morphable_scene() -> (Scene3D, NodeId) {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    prim.targets = vec![MorphTarget::with_deltas(
        Some(vec![[0.0, 0.0, 4.0]; 3]),
        None,
        None,
    )];
    let mut s = Scene3D::new();
    let mid = s.add_mesh(
        Mesh::new(None)
            .with_primitive(prim)
            .with_weights(vec![0.25]), // static default
    );
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    (s, nid)
}

fn weight_ramp_animation(node: NodeId) -> Animation {
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node,
            property: AnimationProperty::MorphWeights,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Scalar(vec![0.0, 1.0]),
            interpolation: Interpolation::Linear,
        },
    });
    anim
}

#[test]
fn animated_morph_weights_override_static_defaults() {
    let (s, nid) = morphable_scene();
    let anim = weight_ramp_animation(nid);
    // t = 0.5 → weight 0.5, NOT the static default 0.25.
    let frame = s.world_mesh_at(&anim, 0.5, nid).expect("frame");
    assert_vec3_eq(
        frame.primitives[0].positions[0],
        [0.0, 0.0, 2.0],
        "animated weight 0.5 · delta 4",
    );
    // An animation with no MorphWeights channel falls back to the
    // static default.
    let unrelated = Animation::new(None);
    let rest = s.world_mesh_at(&unrelated, 0.5, nid).expect("rest frame");
    assert_vec3_eq(
        rest.primitives[0].positions[0],
        [0.0, 0.0, 1.0],
        "static default 0.25 · delta 4",
    );
}

// --- world_mesh_at: node transform driven by animation --------------------

#[test]
fn unskinned_node_follows_animated_transform() {
    let mut s = Scene3D::new();
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mid = s.add_mesh(Mesh::new(None).with_primitive(prim));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    let mut anim = Animation::new(None);
    anim.channels
        .push(translation_channel(nid, [0.0; 3], [0.0, 6.0, 0.0], 2.0));
    let frame = s.world_mesh_at(&anim, 1.0, nid).expect("frame");
    assert_vec3_eq(
        frame.primitives[0].positions[0],
        [0.0, 3.0, 0.0],
        "halfway up",
    );
}

// --- posed: baking a frame into a scene copy -------------------------------

#[test]
fn posed_scene_reproduces_posed_world_transforms() {
    let (s, _nid, elbow) = two_bone_arm();
    let anim = elbow_bend_animation(elbow);
    let baked = s.posed(&anim, 1.0);
    let expected = s.posed_node_transforms(&anim.sample_pose(1.0, s.nodes.len()));
    assert_eq!(baked.world_node_transforms(), expected);
    // Original untouched.
    assert_eq!(
        s.nodes[elbow.0 as usize].transform,
        Transform::Trs {
            translation: [1.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    );
}

#[test]
fn posed_writes_animated_morph_weights_onto_the_node() {
    let (s, nid) = morphable_scene();
    let anim = weight_ramp_animation(nid);
    let baked = s.posed(&anim, 1.0);
    // The sampled vector lands on the node (glTF `node.weights`), not
    // in the shared mesh arena.
    assert_eq!(
        baked.nodes[nid.0 as usize].weights,
        vec![1.0],
        "animated weight baked onto the node"
    );
    assert_eq!(
        baked.meshes[0].weights,
        vec![0.25],
        "mesh defaults untouched in the baked scene"
    );
    assert!(
        s.nodes[nid.0 as usize].weights.is_empty(),
        "original untouched"
    );
    // The baked scene's rest-pose world_mesh now IS the frame (the
    // node override beats the mesh default).
    let frame = baked.world_mesh(nid).expect("baked frame");
    assert_vec3_eq(
        frame.primitives[0].positions[0],
        [0.0, 0.0, 4.0],
        "weight 1 · delta 4",
    );
}

#[test]
fn posed_keeps_undriven_nodes_bit_for_bit() {
    let mut s = Scene3D::new();
    let m = Transform::Matrix([
        [1.0, 0.0, 0.0, 1.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    let still = s.add_node(Node::new().with_transform(m));
    let driven = s.add_node(Node::new());
    s.add_root(still);
    s.add_root(driven);
    let mut anim = Animation::new(None);
    anim.channels
        .push(translation_channel(driven, [0.0; 3], [5.0, 0.0, 0.0], 1.0));
    let baked = s.posed(&anim, 1.0);
    assert_eq!(
        baked.nodes[still.0 as usize].transform, m,
        "undriven Matrix node is not decomposed"
    );
    assert_eq!(
        baked.nodes[driven.0 as usize].transform,
        Transform::Trs {
            translation: [5.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    );
}

#[test]
fn posed_carries_animations_over_unchanged() {
    let (s, _nid, elbow) = two_bone_arm();
    let mut s = s;
    s.add_animation(elbow_bend_animation(elbow));
    let anim = s.animations[0].clone();
    let baked = s.posed(&anim, 0.5);
    assert_eq!(baked.animations.len(), 1, "animation arena preserved");
    assert_eq!(
        baked.animations[0].channels.len(),
        anim.channels.len(),
        "channels intact"
    );
}
