//! Tests for node-level morph-weight overrides (glTF 2.0
//! `node.weights`) — the `Node::weights` field, the
//! `Scene3D::effective_morph_weights` precedence resolver, the
//! instantiation pipeline's *animation > node > mesh* weight
//! precedence, per-node baking in `Scene3D::posed`, the two new
//! `Scene3D::validate` checks, and verbatim carry through
//! `Scene3D::append`.

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, Interpolation, Mesh, MeshId, MorphTarget, Node, NodeId, Primitive, Scene3D,
    Skeleton, Skin, Topology, Transform, ValidationError,
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

/// One triangle with a single morph target displacing every vertex by
/// +4 in Z.
fn morphable_primitive() -> Primitive {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    prim.targets = vec![MorphTarget::with_deltas(
        Some(vec![[0.0, 0.0, 4.0]; 3]),
        None,
        None,
    )];
    prim
}

/// Scene with one morphable mesh (static default weight 0.25) under
/// one root node.
fn morphable_scene() -> (Scene3D, NodeId) {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(
        Mesh::new(None)
            .with_primitive(morphable_primitive())
            .with_weights(vec![0.25]),
    );
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    (s, nid)
}

fn weight_channel(node: NodeId, from: f32, to: f32) -> AnimationChannel {
    AnimationChannel {
        target: AnimationTarget {
            node,
            property: AnimationProperty::MorphWeights,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Scalar(vec![from, to]),
            interpolation: Interpolation::Linear,
        },
    }
}

// --- model surface ----------------------------------------------------------

#[test]
fn node_weights_default_empty_and_builder_sets() {
    assert!(Node::new().weights.is_empty(), "no override by default");
    assert!(Node::default().weights.is_empty());
    let n = Node::new().with_weights(vec![0.5, 0.25]);
    assert_eq!(n.weights, vec![0.5, 0.25]);
}

// --- effective_morph_weights: the static (node > mesh) precedence ----------

#[test]
fn effective_weights_prefer_the_node_override() {
    let (mut s, nid) = morphable_scene();
    s.nodes[nid.0 as usize].weights = vec![0.75];
    assert_eq!(s.effective_morph_weights(nid), Some(&[0.75f32][..]));
}

#[test]
fn effective_weights_fall_back_to_mesh_defaults() {
    let (s, nid) = morphable_scene();
    assert_eq!(s.effective_morph_weights(nid), Some(&[0.25f32][..]));
}

#[test]
fn effective_weights_empty_when_neither_side_stores_any() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(morphable_primitive()));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    assert_eq!(s.effective_morph_weights(nid), Some(&[][..]));
}

#[test]
fn effective_weights_none_without_a_resolvable_mesh() {
    let mut s = Scene3D::new();
    let bare = s.add_node(Node::new());
    let dangling = s.add_node(Node::new().with_mesh(MeshId(9)));
    assert_eq!(s.effective_morph_weights(bare), None, "no mesh");
    assert_eq!(s.effective_morph_weights(dangling), None, "dangling mesh");
    assert_eq!(
        s.effective_morph_weights(NodeId(42)),
        None,
        "node out of range"
    );
}

// --- world_mesh: node override beats mesh defaults --------------------------

#[test]
fn node_override_beats_mesh_defaults_in_world_mesh() {
    let (mut s, nid) = morphable_scene();
    s.nodes[nid.0 as usize].weights = vec![1.0];
    let m = s.world_mesh(nid).expect("instantiate");
    assert_vec3_eq(
        m.primitives[0].positions[0],
        [0.0, 0.0, 4.0],
        "node weight 1 · delta 4, not the mesh's 0.25",
    );
}

#[test]
fn empty_node_weights_defer_to_mesh_defaults() {
    let (s, nid) = morphable_scene();
    assert!(s.nodes[nid.0 as usize].weights.is_empty());
    let m = s.world_mesh(nid).expect("instantiate");
    assert_vec3_eq(
        m.primitives[0].positions[0],
        [0.0, 0.0, 1.0],
        "mesh default 0.25 · delta 4",
    );
}

#[test]
fn node_override_works_without_mesh_defaults() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(morphable_primitive()));
    let nid = s.add_node(Node::new().with_mesh(mid).with_weights(vec![0.5]));
    s.add_root(nid);
    let m = s.world_mesh(nid).expect("instantiate");
    assert_vec3_eq(
        m.primitives[0].positions[0],
        [0.0, 0.0, 2.0],
        "node weight 0.5 · delta 4 with an empty mesh default",
    );
}

#[test]
fn two_nodes_sharing_one_mesh_instantiate_divergent_blend_states() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(
        Mesh::new(None)
            .with_primitive(morphable_primitive())
            .with_weights(vec![0.25]),
    );
    let a = s.add_node(Node::new().with_mesh(mid).with_weights(vec![1.0]));
    let b = s.add_node(Node::new().with_mesh(mid));
    s.add_root(a);
    s.add_root(b);
    let ma = s.world_mesh(a).expect("a");
    let mb = s.world_mesh(b).expect("b");
    assert_vec3_eq(
        ma.primitives[0].positions[0],
        [0.0, 0.0, 4.0],
        "node a: override 1.0",
    );
    assert_vec3_eq(
        mb.primitives[0].positions[0],
        [0.0, 0.0, 1.0],
        "node b: mesh default 0.25",
    );
}

// --- world_mesh_at: the animated rung stays on top ---------------------------

#[test]
fn animated_weights_beat_the_node_override() {
    let (mut s, nid) = morphable_scene();
    s.nodes[nid.0 as usize].weights = vec![1.0];
    let mut anim = Animation::new(None);
    anim.channels.push(weight_channel(nid, 0.0, 1.0));
    // t = 0.5 → animated 0.5 wins over node 1.0 and mesh 0.25.
    let frame = s.world_mesh_at(&anim, 0.5, nid).expect("frame");
    assert_vec3_eq(
        frame.primitives[0].positions[0],
        [0.0, 0.0, 2.0],
        "animated 0.5 · delta 4",
    );
}

#[test]
fn undriven_animated_frame_falls_back_to_the_node_override() {
    let (mut s, nid) = morphable_scene();
    s.nodes[nid.0 as usize].weights = vec![1.0];
    // No MorphWeights channel: the node override is the next rung.
    let unrelated = Animation::new(None);
    let frame = s.world_mesh_at(&unrelated, 0.5, nid).expect("frame");
    assert_vec3_eq(
        frame.primitives[0].positions[0],
        [0.0, 0.0, 4.0],
        "node override 1.0 · delta 4",
    );
}

// --- skin path: morph (node weights) folds in before skinning ---------------

#[test]
fn node_override_morphs_before_skinning() {
    let mut s = Scene3D::new();
    let joint = s.add_node(Node::new().with_transform(Transform::Trs {
        translation: [2.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }));
    s.add_root(joint);
    let skel = s.add_skeleton(Skeleton {
        name: None,
        joints: vec![joint],
        inverse_bind_matrices: vec![identity()],
    });
    let skin = s.add_skin(Skin::new(skel));
    let mut prim = morphable_primitive();
    prim.joints = Some(vec![[0, 0, 0, 0]; 3]);
    prim.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]);
    let mid = s.add_mesh(Mesh::new(None).with_primitive(prim));
    let mut mesh_node = Node::new().with_mesh(mid).with_weights(vec![0.5]);
    mesh_node.skin = Some(skin);
    let nid = s.add_node(mesh_node);
    s.add_root(nid);
    let m = s.world_mesh(nid).expect("skinned instantiation");
    // Morph first (z += 0.5 · 4 = 2), then the rigid joint translate
    // (x += 2).
    assert_vec3_eq(
        m.primitives[0].positions[0],
        [2.0, 0.0, 2.0],
        "morph folds in before the skin palette applies",
    );
}

// --- posed: per-node baking, no shared-mesh clobber --------------------------

#[test]
fn posed_bakes_divergent_weights_per_node_on_a_shared_mesh() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(
        Mesh::new(None)
            .with_primitive(morphable_primitive())
            .with_weights(vec![0.25]),
    );
    let a = s.add_node(Node::new().with_mesh(mid));
    let b = s.add_node(Node::new().with_mesh(mid));
    s.add_root(a);
    s.add_root(b);
    let mut anim = Animation::new(None);
    anim.channels.push(weight_channel(a, 0.0, 1.0));
    anim.channels.push(weight_channel(b, 0.0, 0.5));
    let baked = s.posed(&anim, 1.0);
    assert_eq!(
        baked.nodes[a.0 as usize].weights,
        vec![1.0],
        "node a keeps its own sampled vector"
    );
    assert_eq!(
        baked.nodes[b.0 as usize].weights,
        vec![0.5],
        "node b keeps its own sampled vector"
    );
    assert_eq!(
        baked.meshes[0].weights,
        vec![0.25],
        "the shared mesh's defaults are never clobbered"
    );
    // The baked scene reproduces the animated frame node-for-node.
    for nid in [a, b] {
        let via_bake = baked.world_mesh(nid).expect("baked");
        let via_anim = s.world_mesh_at(&anim, 1.0, nid).expect("animated");
        for (p, q) in via_bake.primitives[0]
            .positions
            .iter()
            .zip(via_anim.primitives[0].positions.iter())
        {
            assert_vec3_eq(*p, *q, "posed().world_mesh == world_mesh_at");
        }
    }
}

#[test]
fn posed_drops_weight_vectors_targeting_meshless_or_dangling_nodes() {
    let mut s = Scene3D::new();
    let bare = s.add_node(Node::new());
    let dangling = s.add_node(Node::new().with_mesh(MeshId(9)));
    s.add_root(bare);
    s.add_root(dangling);
    let mut anim = Animation::new(None);
    anim.channels.push(weight_channel(bare, 0.0, 1.0));
    anim.channels.push(weight_channel(dangling, 0.0, 1.0));
    let baked = s.posed(&anim, 1.0);
    assert!(
        baked.nodes[bare.0 as usize].weights.is_empty(),
        "no mesh, nothing to blend"
    );
    assert!(
        baked.nodes[dangling.0 as usize].weights.is_empty(),
        "dangling mesh id stays weightless"
    );
}

// --- validate ---------------------------------------------------------------

#[test]
fn matching_node_weights_validate_clean() {
    let (mut s, nid) = morphable_scene();
    s.nodes[nid.0 as usize].weights = vec![0.75];
    assert!(s.validate().is_ok());
}

#[test]
fn node_weight_count_mismatch_is_reported() {
    let (mut s, nid) = morphable_scene();
    s.nodes[nid.0 as usize].weights = vec![0.5, 0.5];
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::NodeMorphWeightCountMismatch {
            node_weights: 2,
            primitive_targets: 1,
            ..
        }
    )));
}

#[test]
fn node_weights_on_a_targetless_mesh_are_reported() {
    let mut s = Scene3D::new();
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mid = s.add_mesh(Mesh::new(None).with_primitive(prim));
    let nid = s.add_node(Node::new().with_mesh(mid).with_weights(vec![1.0]));
    s.add_root(nid);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::NodeMorphWeightCountMismatch {
            node_weights: 1,
            primitive_targets: 0,
            ..
        }
    )));
}

#[test]
fn node_weights_without_a_mesh_are_reported() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new().with_weights(vec![0.5]));
    s.add_root(nid);
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    assert!(matches!(
        &errs[0],
        ValidationError::NodeMorphWeightsWithoutMesh { location } if location == "nodes[0].weights"
    ));
}

#[test]
fn node_weights_on_a_dangling_mesh_report_only_the_dangling_id() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new().with_mesh(MeshId(3)).with_weights(vec![0.5]));
    s.add_root(nid);
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1, "count check is skipped on a dangling mesh");
    assert!(matches!(
        &errs[0],
        ValidationError::DanglingId {
            arena: "meshes",
            ..
        }
    ));
}

// --- append ------------------------------------------------------------------

#[test]
fn append_carries_node_weight_overrides_verbatim() {
    let (dst_src, _) = morphable_scene();
    let mut dst = dst_src;
    let mut src = Scene3D::new();
    let mid = src.add_mesh(
        Mesh::new(None)
            .with_primitive(morphable_primitive())
            .with_weights(vec![0.25]),
    );
    let nid = src.add_node(Node::new().with_mesh(mid).with_weights(vec![0.5]));
    src.add_root(nid);
    let off = dst.append(&src);
    let relocated = &dst.nodes[(off.nodes + nid.0) as usize];
    assert_eq!(relocated.weights, vec![0.5], "override travels verbatim");
    assert!(dst.validate().is_ok(), "relocated ids still line up");
    let m = dst
        .world_mesh(NodeId(off.nodes + nid.0))
        .expect("relocated instantiation");
    assert_vec3_eq(
        m.primitives[0].positions[0],
        [0.0, 0.0, 2.0],
        "relocated node still blends its own override",
    );
}
