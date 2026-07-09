//! Tests for linear-blend skinning — `Scene3D::joint_matrices` /
//! `joint_matrices_with` + `Primitive::skinned` / `Mesh::skinned`.
//!
//! Every expectation is closed-form: joint matrices are checked
//! against hand-multiplied `world · inverseBind` products (glTF 2.0
//! §3.7.3), and blends against hand-evaluated `Σ wᵢ·Mᵢ` per-vertex
//! matrices (§3.7.3.3). The rest scene graph supplies the joint world
//! transforms; the skinned-mesh node's own transform must be ignored
//! (§3.7.3.2).

use oxideav_mesh3d::{
    Mesh, Node, NodeId, Primitive, Scene3D, Skeleton, Skin, SkinId, Topology, Transform,
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

fn assert_mat4_eq(a: [[f32; 4]; 4], b: [[f32; 4]; 4], ctx: &str) {
    for r in 0..4 {
        for c in 0..4 {
            assert!(
                (a[r][c] - b[r][c]).abs() < EPS,
                "{ctx}: [{r}][{c}]: {a:?} vs {b:?}"
            );
        }
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

/// +90° about Z (column-vector convention): x̂ → ŷ.
fn rot_z_90() -> [[f32; 4]; 4] {
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn scale(s: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [s[0], 0.0, 0.0, 0.0],
        [0.0, s[1], 0.0, 0.0],
        [0.0, 0.0, s[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// One skinnable triangle: positions on the XY plane, +Z normals,
/// +X tangents, all four weight slots on joint 0.
fn skinnable_triangle() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    p.tangents = Some(vec![[1.0, 0.0, 0.0, 1.0]; 3]);
    p.joints = Some(vec![[0, 0, 0, 0]; 3]);
    p.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]);
    p
}

/// Scene with `n_joints` root-level joint nodes (each carrying the
/// given local transform), one skeleton over them (given IBMs), one
/// skin, and one mesh node bound to the skin. Returns the scene and
/// the mesh node's id.
fn skinned_scene(
    joint_transforms: &[Transform],
    ibms: Vec<[[f32; 4]; 4]>,
    prim: Primitive,
) -> (Scene3D, NodeId) {
    let mut s = Scene3D::new();
    let mut joints = Vec::new();
    for t in joint_transforms {
        let id = s.add_node(Node::new().with_transform(*t));
        s.add_root(id);
        joints.push(id);
    }
    let skel = s.add_skeleton(Skeleton {
        name: None,
        joints,
        inverse_bind_matrices: ibms,
    });
    let skin = s.add_skin(Skin::new(skel));
    let mesh = s.add_mesh(Mesh::new(None).with_primitive(prim));
    let mut mesh_node = Node::new().with_mesh(mesh);
    mesh_node.skin = Some(skin);
    let nid = s.add_node(mesh_node);
    s.add_root(nid);
    (s, nid)
}

// --- joint_matrices --------------------------------------------------

#[test]
fn identity_rig_yields_identity_palette() {
    let (s, nid) = skinned_scene(
        &[Transform::identity(), Transform::identity()],
        vec![identity(), identity()],
        skinnable_triangle(),
    );
    let palette = s.joint_matrices(nid).expect("palette");
    assert_eq!(palette.len(), 2);
    assert_mat4_eq(palette[0], identity(), "joint 0");
    assert_mat4_eq(palette[1], identity(), "joint 1");
}

#[test]
fn joint_matrix_is_world_times_inverse_bind() {
    // Joint rest pose at (0, 1, 0); IBM = translate(0, -1, 0) undoes it,
    // so the rest palette is the identity. Move the joint to (0, 2, 0)
    // and the palette becomes translate(0, 1, 0): a bound vertex
    // follows the joint by exactly the joint's displacement.
    let (mut s, nid) = skinned_scene(
        &[Transform::Matrix(translate([0.0, 1.0, 0.0]))],
        vec![translate([0.0, -1.0, 0.0])],
        skinnable_triangle(),
    );
    let rest = s.joint_matrices(nid).expect("rest palette");
    assert_mat4_eq(rest[0], identity(), "rest");

    s.nodes[0].transform = Transform::Matrix(translate([0.0, 2.0, 0.0]));
    let posed = s.joint_matrices(nid).expect("posed palette");
    assert_mat4_eq(posed[0], translate([0.0, 1.0, 0.0]), "posed");
}

#[test]
fn joint_world_includes_ancestor_chain() {
    // parent (translate x+1) → child joint (translate y+1):
    // world(child) = T(1,0,0)·T(0,1,0) = T(1,1,0).
    let mut s = Scene3D::new();
    let child =
        s.add_node(Node::new().with_transform(Transform::Matrix(translate([0.0, 1.0, 0.0]))));
    let mut parent = Node::new().with_transform(Transform::Matrix(translate([1.0, 0.0, 0.0])));
    parent.children.push(child);
    let parent = s.add_node(parent);
    s.add_root(parent);
    let skel = s.add_skeleton(Skeleton {
        name: None,
        joints: vec![child],
        inverse_bind_matrices: vec![identity()],
    });
    let skin = s.add_skin(Skin::new(skel));
    let mut mesh_node = Node::new().with_mesh(s.add_mesh(Mesh::new(None)));
    mesh_node.skin = Some(skin);
    let nid = s.add_node(mesh_node);
    s.add_root(nid);

    let palette = s.joint_matrices(nid).expect("palette");
    assert_mat4_eq(palette[0], translate([1.0, 1.0, 0.0]), "chained world");
}

#[test]
fn mesh_node_transform_is_ignored() {
    // §3.7.3.2: only the joint transforms apply; the skinned-mesh
    // node's transform must not leak into the palette.
    let (mut s, nid) = skinned_scene(
        &[Transform::Matrix(translate([0.0, 3.0, 0.0]))],
        vec![identity()],
        skinnable_triangle(),
    );
    let before = s.joint_matrices(nid).expect("before");
    s.nodes[nid.0 as usize].transform = Transform::Matrix(translate([100.0, -50.0, 7.0]));
    let after = s.joint_matrices(nid).expect("after");
    assert_mat4_eq(before[0], after[0], "palette must not move");
    assert_mat4_eq(after[0], translate([0.0, 3.0, 0.0]), "joint world · I");
}

#[test]
fn empty_inverse_bind_matrices_default_to_identity() {
    let (s, nid) = skinned_scene(
        &[Transform::Matrix(translate([2.0, 0.0, 0.0]))],
        Vec::new(),
        skinnable_triangle(),
    );
    let palette = s.joint_matrices(nid).expect("palette");
    assert_mat4_eq(palette[0], translate([2.0, 0.0, 0.0]), "world · I");
}

#[test]
fn extra_inverse_bind_matrices_are_allowed_short_ones_are_not() {
    // §3.7.3.1: ibm count MUST be >= joint count.
    let (mut s, nid) = skinned_scene(
        &[Transform::identity()],
        vec![identity(), translate([9.0, 9.0, 9.0])], // 2 ibms, 1 joint
        skinnable_triangle(),
    );
    let palette = s.joint_matrices(nid).expect("extra ibms are fine");
    assert_eq!(palette.len(), 1);
    assert_mat4_eq(palette[0], identity(), "first ibm used");

    // Now 2 joints, 1 ibm → malformed.
    let extra_joint = s.add_node(Node::new());
    s.add_root(extra_joint);
    s.skeletons[0].joints.push(extra_joint);
    s.skeletons[0].inverse_bind_matrices = vec![identity()];
    assert!(s.joint_matrices(nid).is_none(), "short ibms must fail");
}

#[test]
fn unreachable_joint_yields_none() {
    let (mut s, nid) = skinned_scene(
        &[Transform::identity()],
        vec![identity()],
        skinnable_triangle(),
    );
    // Detach the joint node from the roots: no world transform exists.
    s.roots.retain(|r| *r != NodeId(0));
    assert!(s.joint_matrices(nid).is_none());
}

#[test]
fn node_without_skin_yields_none() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    assert!(s.joint_matrices(nid).is_none());
    assert!(s.joint_matrices(NodeId(55)).is_none(), "out of range");
}

#[test]
fn dangling_skin_or_skeleton_yields_none() {
    let mut s = Scene3D::new();
    let mut n = Node::new();
    n.skin = Some(SkinId(3)); // no skins in the arena
    let nid = s.add_node(n);
    s.add_root(nid);
    assert!(s.joint_matrices(nid).is_none());
}

#[test]
fn joint_matrices_with_uses_supplied_worlds() {
    let (s, nid) = skinned_scene(
        &[Transform::identity()],
        vec![translate([0.0, -1.0, 0.0])],
        skinnable_triangle(),
    );
    // Caller-supplied (e.g. posed) world for the joint node.
    let mut worlds = vec![None; s.nodes.len()];
    worlds[0] = Some(rot_z_90());
    worlds[nid.0 as usize] = Some(identity());
    let palette = s.joint_matrices_with(nid, &worlds).expect("palette");
    // rot_z_90 · translate(0,-1,0): translation column rotates to (1, 0, 0).
    let mut expected = rot_z_90();
    expected[0][3] = 1.0;
    expected[1][3] = 0.0;
    assert_mat4_eq(palette[0], expected, "posed world · ibm");

    // Missing world slot for the joint → None.
    let empty: Vec<Option<[[f32; 4]; 4]>> = vec![None; s.nodes.len()];
    assert!(s.joint_matrices_with(nid, &empty).is_none());
}

// --- Primitive::skinned ----------------------------------------------

#[test]
fn rigid_single_joint_matches_transformed() {
    // All weight on one joint ⇒ skinning degenerates to a rigid
    // transform; positions/normals/tangents must match the
    // `transformed()` baking exactly.
    let prim = skinnable_triangle();
    let m = rot_z_90();
    let skinned = prim.skinned(&[m]);
    let baked = prim.transformed(m);
    for v in 0..3 {
        assert_vec3_eq(skinned.positions[v], baked.positions[v], "position");
        assert_vec3_eq(
            skinned.normals.as_ref().unwrap()[v],
            baked.normals.as_ref().unwrap()[v],
            "normal",
        );
        let st = skinned.tangents.as_ref().unwrap()[v];
        let bt = baked.tangents.as_ref().unwrap()[v];
        assert_vec3_eq([st[0], st[1], st[2]], [bt[0], bt[1], bt[2]], "tangent");
        assert_eq!(st[3], bt[3], "handedness");
    }
    assert!(skinned.joints.is_none(), "joints consumed");
    assert!(skinned.weights.is_none(), "weights consumed");
}

#[test]
fn identity_palette_is_a_geometric_no_op() {
    let prim = skinnable_triangle();
    let skinned = prim.skinned(&[identity()]);
    for v in 0..3 {
        assert_vec3_eq(skinned.positions[v], prim.positions[v], "position");
        assert_vec3_eq(
            skinned.normals.as_ref().unwrap()[v],
            prim.normals.as_ref().unwrap()[v],
            "normal",
        );
    }
}

#[test]
fn translation_blend_is_weighted_average() {
    // 50/50 between identity and translate(0,0,2):
    // M = 0.5·I + 0.5·T ⇒ p' = p + (0, 0, 1).
    let mut prim = skinnable_triangle();
    prim.joints = Some(vec![[0, 1, 0, 0]; 3]);
    prim.weights = Some(vec![[0.5, 0.5, 0.0, 0.0]; 3]);
    let skinned = prim.skinned(&[identity(), translate([0.0, 0.0, 2.0])]);
    for v in 0..3 {
        let p = prim.positions[v];
        assert_vec3_eq(
            skinned.positions[v],
            [p[0], p[1], p[2] + 1.0],
            "blended translation",
        );
    }
}

#[test]
fn rotation_blend_shrinks_toward_axis() {
    // The classic candy-wrapper closed form: blending I with a +90° Z
    // rotation at 50/50 gives linear part
    //   [[0.5, -0.5, 0], [0.5, 0.5, 0], [0, 0, 1]]
    // so (1, 0, 0) lands at (0.5, 0.5, 0) — shorter than unit length,
    // exactly what LBS does (it blends matrices, not rotations).
    let mut prim = skinnable_triangle();
    prim.positions = vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    prim.joints = Some(vec![[0, 1, 0, 0]; 3]);
    prim.weights = Some(vec![[0.5, 0.5, 0.0, 0.0]; 3]);
    let skinned = prim.skinned(&[identity(), rot_z_90()]);
    assert_vec3_eq(skinned.positions[0], [0.5, 0.5, 0.0], "candy wrapper");
    assert_vec3_eq(skinned.positions[1], [1.0, 1.0, 0.0], "scales linearly");
    assert_vec3_eq(skinned.positions[2], [1.5, 1.5, 0.0], "scales linearly");
}

#[test]
fn zero_weight_vertex_keeps_rest_pose() {
    let mut prim = skinnable_triangle();
    prim.weights = Some(vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0], // unskinned per spec shape
        [1.0, 0.0, 0.0, 0.0],
    ]);
    let skinned = prim.skinned(&[translate([5.0, 0.0, 0.0])]);
    assert_vec3_eq(skinned.positions[0], [5.0, 0.0, 0.0], "skinned");
    assert_vec3_eq(skinned.positions[1], [1.0, 0.0, 0.0], "rest pose kept");
    assert_vec3_eq(skinned.positions[2], [5.0, 1.0, 0.0], "skinned");
}

#[test]
fn out_of_palette_joint_contributes_nothing() {
    let mut prim = skinnable_triangle();
    prim.joints = Some(vec![[0, 200, 0, 0]; 3]);
    prim.weights = Some(vec![[0.5, 0.5, 0.0, 0.0]; 3]);
    let skinned = prim.skinned(&[identity()]);
    // Only the 0.5·I influence survives; weights are used as stored,
    // so the vertex contracts halfway toward the origin.
    assert_vec3_eq(skinned.positions[1], [0.5, 0.0, 0.0], "0.5·I only");
}

#[test]
fn non_finite_and_negative_weights_are_skipped() {
    let mut prim = skinnable_triangle();
    prim.weights = Some(vec![
        [f32::NAN, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0, 0.0],
    ]);
    let skinned = prim.skinned(&[translate([5.0, 0.0, 0.0])]);
    for v in 0..3 {
        assert_vec3_eq(skinned.positions[v], prim.positions[v], "rest pose kept");
    }
}

#[test]
fn primitive_without_influences_is_returned_unchanged() {
    let mut prim = skinnable_triangle();
    prim.joints = None;
    let skinned = prim.skinned(&[translate([5.0, 0.0, 0.0])]);
    assert_eq!(skinned.positions, prim.positions);
    assert!(skinned.weights.is_some(), "no influence data consumed");
}

#[test]
fn normals_use_per_vertex_inverse_transpose() {
    // Joint 0 stretches x by 2. A normal along (1, 1, 0)/√2 must
    // transform by the inverse-transpose diag(0.5, 1, 1) and
    // renormalise: (0.5, 1, 0) → (0.4472, 0.8944, 0).
    let mut prim = skinnable_triangle();
    let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
    prim.normals = Some(vec![[inv_sqrt2, inv_sqrt2, 0.0]; 3]);
    let skinned = prim.skinned(&[scale([2.0, 1.0, 1.0])]);
    let expected = {
        let len = (0.25f32 + 1.0).sqrt();
        [0.5 / len, 1.0 / len, 0.0]
    };
    for v in 0..3 {
        assert_vec3_eq(
            skinned.normals.as_ref().unwrap()[v],
            expected,
            "inverse-transpose normal",
        );
    }
}

#[test]
fn mirroring_blend_flips_tangent_handedness() {
    let prim = skinnable_triangle();
    let skinned = prim.skinned(&[scale([-1.0, 1.0, 1.0])]);
    for t in skinned.tangents.as_ref().unwrap() {
        assert_eq!(t[3], -1.0, "handedness flipped under mirror");
    }
}

#[test]
fn singular_blend_leaves_normal_untouched_but_moves_position() {
    let prim = skinnable_triangle();
    let skinned = prim.skinned(&[scale([0.0, 1.0, 1.0])]);
    assert_vec3_eq(skinned.positions[1], [0.0, 0.0, 0.0], "x collapsed");
    assert_vec3_eq(
        skinned.normals.as_ref().unwrap()[1],
        [0.0, 0.0, 1.0],
        "normal fallback",
    );
}

#[test]
fn skinned_drops_morph_targets_and_documents_order() {
    let mut prim = skinnable_triangle();
    prim.targets = vec![oxideav_mesh3d::MorphTarget {
        position: Some(vec![[0.0, 0.0, 1.0]; 3]),
        ..Default::default()
    }];
    let skinned = prim.skinned(&[identity()]);
    assert!(skinned.targets.is_empty(), "targets consumed/dropped");
}

// --- Mesh::skinned -----------------------------------------------------

#[test]
fn mesh_skinned_maps_every_primitive_and_clears_default_weights() {
    let mut a = skinnable_triangle();
    a.targets = vec![oxideav_mesh3d::MorphTarget::default()];
    let b = skinnable_triangle();
    let mesh = Mesh::new(Some("rig".to_string()))
        .with_primitive(a)
        .with_primitive(b)
        .with_weights(vec![0.25]);
    let skinned = mesh.skinned(&[translate([0.0, 5.0, 0.0])]);
    assert_eq!(skinned.name.as_deref(), Some("rig"));
    assert!(skinned.weights.is_empty(), "morph weights cleared");
    for prim in &skinned.primitives {
        assert_vec3_eq(prim.positions[0], [0.0, 5.0, 0.0], "primitive skinned");
        assert!(prim.targets.is_empty());
    }
}

// --- end-to-end through the scene --------------------------------------

#[test]
fn scene_palette_drives_primitive_deformation() {
    // Two-bone rig: joint 0 fixed at origin, joint 1 rest at (1, 0, 0)
    // with IBM = translate(-1, 0, 0). Vertices at x = 0 (bound to j0),
    // x = 1 and x = 2 (bound to j1). Posing joint 1 with a +90° Z
    // rotation *about its pivot* (T(1,0,0)·R) swings the far vertex
    // from (2, 0, 0) to (1, 1, 0) — the textbook forearm bend.
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    prim.joints = Some(vec![[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]);
    prim.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]);

    let mut pose1 = rot_z_90();
    pose1[0][3] = 1.0; // T(1,0,0) · R_z(90°)
    let (s, nid) = skinned_scene(
        &[Transform::identity(), Transform::Matrix(pose1)],
        vec![identity(), translate([-1.0, 0.0, 0.0])],
        prim,
    );
    let palette = s.joint_matrices(nid).expect("palette");
    let mesh = s.node(nid).and_then(|n| n.mesh).unwrap();
    let skinned = s.meshes[mesh.0 as usize].primitives[0].skinned(&palette);
    assert_vec3_eq(skinned.positions[0], [0.0, 0.0, 0.0], "root vertex");
    assert_vec3_eq(skinned.positions[1], [1.0, 0.0, 0.0], "elbow pivot");
    assert_vec3_eq(skinned.positions[2], [1.0, 1.0, 0.0], "forearm bent up");
}
