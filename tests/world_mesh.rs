//! Tests for `Scene3D::world_mesh` / `world_mesh_with` — the glTF 2.0
//! §3.7.4 instantiation pipeline (morph → skin-or-transform) baked
//! into a static world-space mesh.

use oxideav_mesh3d::{
    Mesh, MeshId, MorphTarget, Node, NodeId, Primitive, Scene3D, Skeleton, Skin, Topology,
    Transform,
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

fn triangle() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    p
}

// --- unskinned path ----------------------------------------------------

#[test]
fn unskinned_node_bakes_world_matrix() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(triangle()));
    let nid = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(Transform::Matrix(translate([0.0, 0.0, 4.0]))),
    );
    s.add_root(nid);
    let baked = s.world_mesh(nid).expect("world mesh");
    assert_vec3_eq(baked.primitives[0].positions[0], [0.0, 0.0, 4.0], "moved");
    assert_vec3_eq(baked.primitives[0].positions[1], [1.0, 0.0, 4.0], "moved");
}

#[test]
fn ancestor_chain_is_folded_in() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(triangle()));
    let child = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(Transform::Matrix(translate([0.0, 1.0, 0.0]))),
    );
    let mut parent = Node::new().with_transform(Transform::Matrix(translate([1.0, 0.0, 0.0])));
    parent.children.push(child);
    let pid = s.add_node(parent);
    s.add_root(pid);
    let baked = s.world_mesh(child).expect("world mesh");
    assert_vec3_eq(
        baked.primitives[0].positions[0],
        [1.0, 1.0, 0.0],
        "parent · child",
    );
}

#[test]
fn morph_weights_fold_before_transform() {
    let mut prim = triangle();
    prim.targets = vec![MorphTarget::with_deltas(
        Some(vec![[0.0, 0.0, 1.0]; 3]),
        None,
        None,
    )];
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(prim).with_weights(vec![0.5]));
    let nid = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(Transform::Matrix(translate([10.0, 0.0, 0.0]))),
    );
    s.add_root(nid);
    let baked = s.world_mesh(nid).expect("world mesh");
    // base (0,0,0) + 0.5·(0,0,1) = (0,0,0.5), then +10 in x.
    assert_vec3_eq(
        baked.primitives[0].positions[0],
        [10.0, 0.0, 0.5],
        "morph then transform",
    );
    assert!(baked.primitives[0].targets.is_empty(), "targets baked away");
    assert!(baked.weights.is_empty(), "weights baked away");
}

#[test]
fn empty_weights_instantiate_the_non_morphed_state() {
    let mut prim = triangle();
    prim.targets = vec![MorphTarget::with_deltas(
        Some(vec![[0.0, 0.0, 1.0]; 3]),
        None,
        None,
    )];
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(prim));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    let baked = s.world_mesh(nid).expect("world mesh");
    assert_vec3_eq(baked.primitives[0].positions[0], [0.0, 0.0, 0.0], "base");
    assert!(baked.primitives[0].targets.is_empty(), "targets cleared");
}

// --- error paths ---------------------------------------------------------

#[test]
fn none_for_missing_or_unreachable_nodes() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(triangle()));
    let detached = s.add_node(Node::new().with_mesh(mid)); // never rooted
    let no_mesh = s.add_node(Node::new());
    s.add_root(no_mesh);
    assert!(s.world_mesh(NodeId(99)).is_none(), "out of range");
    assert!(s.world_mesh(no_mesh).is_none(), "no mesh attached");
    assert!(s.world_mesh(detached).is_none(), "unreachable");

    let mut dangling = Node::new();
    dangling.mesh = Some(MeshId(42));
    let d = s.add_node(dangling);
    s.add_root(d);
    assert!(s.world_mesh(d).is_none(), "dangling mesh id");
}

// --- skinned path ----------------------------------------------------------

/// One-joint rig: joint node 0 posed by `joint_pose`, identity IBM,
/// mesh node carrying `prim` + the skin, with its own (ignored)
/// transform set to something loud.
fn one_joint_scene(joint_pose: [[f32; 4]; 4], prim: Primitive) -> (Scene3D, NodeId) {
    let mut s = Scene3D::new();
    let joint = s.add_node(Node::new().with_transform(Transform::Matrix(joint_pose)));
    s.add_root(joint);
    let skel = s.add_skeleton(Skeleton {
        name: None,
        joints: vec![joint],
        inverse_bind_matrices: vec![identity()],
    });
    let skin = s.add_skin(Skin::new(skel));
    let mid = s.add_mesh(Mesh::new(None).with_primitive(prim));
    let mut mesh_node = Node::new()
        .with_mesh(mid)
        .with_transform(Transform::Matrix(translate([100.0, 100.0, 100.0])));
    mesh_node.skin = Some(skin);
    let nid = s.add_node(mesh_node);
    s.add_root(nid);
    (s, nid)
}

fn skinned_triangle() -> Primitive {
    let mut p = triangle();
    p.joints = Some(vec![[0, 0, 0, 0]; 3]);
    p.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]);
    p
}

#[test]
fn skinned_node_ignores_its_own_transform() {
    let (s, nid) = one_joint_scene(translate([0.0, 2.0, 0.0]), skinned_triangle());
    let baked = s.world_mesh(nid).expect("world mesh");
    // Joint moves the vertices by (0,2,0); the mesh node's own
    // (100,100,100) translation must NOT appear.
    assert_vec3_eq(
        baked.primitives[0].positions[0],
        [0.0, 2.0, 0.0],
        "joint only",
    );
    assert!(baked.primitives[0].joints.is_none(), "influences consumed");
    assert!(baked.primitives[0].weights.is_none(), "influences consumed");
}

#[test]
fn morph_folds_before_skinning() {
    let mut prim = skinned_triangle();
    prim.targets = vec![MorphTarget::with_deltas(
        Some(vec![[0.0, 0.0, 2.0]; 3]),
        None,
        None,
    )];
    let (mut s, nid) = one_joint_scene(translate([5.0, 0.0, 0.0]), prim);
    let mesh_id = s.node(nid).and_then(|n| n.mesh).unwrap();
    s.meshes[mesh_id.0 as usize].weights = vec![1.0];
    let baked = s.world_mesh(nid).expect("world mesh");
    // base (0,0,0) + 1.0·(0,0,2) = (0,0,2); joint translates +5 x.
    assert_vec3_eq(
        baked.primitives[0].positions[0],
        [5.0, 0.0, 2.0],
        "morph then skin",
    );
}

#[test]
fn uninfluenced_primitive_in_skinned_mesh_falls_back_to_world() {
    // Invalid per §3.7.3.3 but robustly handled: the primitive with no
    // joints/weights follows the node's world matrix so both
    // primitives land in one coordinate space.
    let (mut s, nid) = one_joint_scene(translate([0.0, 2.0, 0.0]), skinned_triangle());
    let mesh_id = s.node(nid).and_then(|n| n.mesh).unwrap();
    s.meshes[mesh_id.0 as usize].primitives.push(triangle());
    let baked = s.world_mesh(nid).expect("world mesh");
    assert_vec3_eq(
        baked.primitives[0].positions[0],
        [0.0, 2.0, 0.0],
        "skinned primitive",
    );
    assert_vec3_eq(
        baked.primitives[1].positions[0],
        [100.0, 100.0, 100.0],
        "rigid fallback follows node world",
    );
}

#[test]
fn broken_palette_yields_none_not_rigid_fallback() {
    let (mut s, nid) = one_joint_scene(translate([0.0, 2.0, 0.0]), skinned_triangle());
    // Sever the joint from the roots: palette can no longer be built.
    s.roots.retain(|r| *r != NodeId(0));
    // Keep the mesh node reachable.
    assert!(s.world_mesh(nid).is_none());
}

#[test]
fn world_mesh_with_accepts_posed_worlds() {
    let (s, nid) = one_joint_scene(identity(), skinned_triangle());
    let mut worlds = s.world_node_transforms();
    // Pose the joint at (0, 0, 9) without touching the scene.
    worlds[0] = Some(translate([0.0, 0.0, 9.0]));
    let baked = s.world_mesh_with(nid, &worlds).expect("world mesh");
    assert_vec3_eq(
        baked.primitives[0].positions[0],
        [0.0, 0.0, 9.0],
        "posed joint",
    );
}
