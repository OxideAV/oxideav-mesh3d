//! Tests for [`Scene3D::world_node_transforms`] — per-node world-space
//! 4x4 matrix snapshot, indexed by [`NodeId`], covering reachability,
//! ancestor-chain composition, cycle guarding, shared-instance
//! single-resolution, determinism, and the
//! pure-vs-`bounding_box` cross-check that the same DFS underlies
//! both helpers.

use oxideav_mesh3d::{Node, NodeId, Scene3D, Transform};

const TOL: f32 = 1e-5;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOL + TOL * a.abs().max(b.abs())
}

fn mat_close(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(ra, rb)| ra.iter().zip(rb.iter()).all(|(x, y)| approx_eq(*x, *y)))
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn translation_only(t: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: t,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn scale_only(s: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: s,
    }
}

// ---- Empty / trivial scenes ----------------------------------------------

#[test]
fn empty_scene_returns_empty_vec() {
    let scene = Scene3D::new();
    let xforms = scene.world_node_transforms();
    assert!(xforms.is_empty());
}

#[test]
fn nodes_with_no_roots_are_all_none() {
    // Three nodes pushed but none promoted to root — nothing is reachable.
    let mut scene = Scene3D::new();
    scene.add_node(Node::new().with_transform(translation_only([1.0, 2.0, 3.0])));
    scene.add_node(Node::new());
    scene.add_node(Node::new());
    let xforms = scene.world_node_transforms();
    assert_eq!(xforms.len(), 3);
    assert!(xforms.iter().all(|t| t.is_none()));
}

#[test]
fn single_root_identity_is_identity() {
    let mut scene = Scene3D::new();
    let n = scene.add_node(Node::new());
    scene.add_root(n);
    let xforms = scene.world_node_transforms();
    assert_eq!(xforms.len(), 1);
    assert!(mat_close(xforms[0].unwrap(), IDENTITY));
}

#[test]
fn single_root_translation_is_passthrough() {
    let mut scene = Scene3D::new();
    let n = scene.add_node(Node::new().with_transform(translation_only([4.0, -2.0, 7.5])));
    scene.add_root(n);
    let xforms = scene.world_node_transforms();
    let m = xforms[0].unwrap();
    assert!(approx_eq(m[0][3], 4.0));
    assert!(approx_eq(m[1][3], -2.0));
    assert!(approx_eq(m[2][3], 7.5));
    // No rotation / scale → upper-left 3x3 is identity.
    for (i, row) in m.iter().enumerate().take(3) {
        for (j, &cell) in row.iter().enumerate().take(3) {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(approx_eq(cell, want), "[{i}][{j}] = {cell}");
        }
    }
}

// ---- Ancestor-chain composition ------------------------------------------

#[test]
fn child_inherits_parent_translation() {
    // Root at (10, 0, 0); child at local (1, 0, 0). Expected world (11, 0, 0).
    let mut scene = Scene3D::new();
    let parent = scene.add_node(Node::new().with_transform(translation_only([10.0, 0.0, 0.0])));
    let child = scene.add_node(Node::new().with_transform(translation_only([1.0, 0.0, 0.0])));
    scene.nodes[parent.0 as usize].children.push(child);
    scene.add_root(parent);

    let xforms = scene.world_node_transforms();
    let pm = xforms[parent.0 as usize].unwrap();
    let cm = xforms[child.0 as usize].unwrap();
    assert!(approx_eq(pm[0][3], 10.0));
    assert!(approx_eq(cm[0][3], 11.0));
}

#[test]
fn grandchild_translation_chain_accumulates() {
    let mut scene = Scene3D::new();
    let a = scene.add_node(Node::new().with_transform(translation_only([1.0, 0.0, 0.0])));
    let b = scene.add_node(Node::new().with_transform(translation_only([0.0, 2.0, 0.0])));
    let c = scene.add_node(Node::new().with_transform(translation_only([0.0, 0.0, 3.0])));
    scene.nodes[a.0 as usize].children.push(b);
    scene.nodes[b.0 as usize].children.push(c);
    scene.add_root(a);

    let xforms = scene.world_node_transforms();
    let am = xforms[a.0 as usize].unwrap();
    let bm = xforms[b.0 as usize].unwrap();
    let cm = xforms[c.0 as usize].unwrap();
    assert!(approx_eq(am[0][3], 1.0));
    assert!(approx_eq(am[1][3], 0.0));
    assert!(approx_eq(am[2][3], 0.0));

    assert!(approx_eq(bm[0][3], 1.0));
    assert!(approx_eq(bm[1][3], 2.0));
    assert!(approx_eq(bm[2][3], 0.0));

    assert!(approx_eq(cm[0][3], 1.0));
    assert!(approx_eq(cm[1][3], 2.0));
    assert!(approx_eq(cm[2][3], 3.0));
}

#[test]
fn nested_scale_multiplies_componentwise() {
    let mut scene = Scene3D::new();
    let parent = scene.add_node(Node::new().with_transform(scale_only([2.0, 1.0, 1.0])));
    let child = scene.add_node(Node::new().with_transform(scale_only([1.0, 3.0, 4.0])));
    scene.nodes[parent.0 as usize].children.push(child);
    scene.add_root(parent);

    let xforms = scene.world_node_transforms();
    let cm = xforms[child.0 as usize].unwrap();
    // Effective world scale is (2*1, 1*3, 1*4) = (2, 3, 4).
    assert!(approx_eq(cm[0][0], 2.0));
    assert!(approx_eq(cm[1][1], 3.0));
    assert!(approx_eq(cm[2][2], 4.0));
}

#[test]
fn parent_scale_modulates_child_translation() {
    // A child's local translation gets scaled by the parent's scale
    // (the canonical glTF semantic — node transforms compose, child's
    // local offset is expressed in the parent's coordinate frame).
    let mut scene = Scene3D::new();
    let parent = scene.add_node(Node::new().with_transform(scale_only([10.0, 10.0, 10.0])));
    let child = scene.add_node(Node::new().with_transform(translation_only([1.0, 0.0, 0.0])));
    scene.nodes[parent.0 as usize].children.push(child);
    scene.add_root(parent);

    let xforms = scene.world_node_transforms();
    let cm = xforms[child.0 as usize].unwrap();
    assert!(approx_eq(cm[0][3], 10.0));
}

// ---- Matrix Transform variant -------------------------------------------

#[test]
fn matrix_transform_variant_passes_through() {
    let m: [[f32; 4]; 4] = [
        [2.0, 0.0, 0.0, 5.0],
        [0.0, 3.0, 0.0, -1.0],
        [0.0, 0.0, 4.0, 7.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut scene = Scene3D::new();
    let n = scene.add_node(Node::new().with_transform(Transform::Matrix(m)));
    scene.add_root(n);
    let xforms = scene.world_node_transforms();
    assert_eq!(xforms[0].unwrap(), m);
}

// ---- Forest with multiple roots / detached subtrees ---------------------

#[test]
fn multiple_roots_each_get_local_chain() {
    let mut scene = Scene3D::new();
    let r0 = scene.add_node(Node::new().with_transform(translation_only([100.0, 0.0, 0.0])));
    let r1 = scene.add_node(Node::new().with_transform(translation_only([0.0, 200.0, 0.0])));
    let c0 = scene.add_node(Node::new().with_transform(translation_only([1.0, 0.0, 0.0])));
    let c1 = scene.add_node(Node::new().with_transform(translation_only([0.0, 1.0, 0.0])));
    scene.nodes[r0.0 as usize].children.push(c0);
    scene.nodes[r1.0 as usize].children.push(c1);
    scene.add_root(r0);
    scene.add_root(r1);

    let xforms = scene.world_node_transforms();
    assert!(approx_eq(xforms[c0.0 as usize].unwrap()[0][3], 101.0));
    assert!(approx_eq(xforms[c1.0 as usize].unwrap()[1][3], 201.0));
}

#[test]
fn detached_node_reports_none() {
    let mut scene = Scene3D::new();
    let r = scene.add_node(Node::new());
    let detached = scene.add_node(Node::new().with_transform(translation_only([99.0, 99.0, 99.0])));
    scene.add_root(r);
    // `detached` is never linked into the tree.

    let xforms = scene.world_node_transforms();
    assert!(xforms[r.0 as usize].is_some());
    assert!(xforms[detached.0 as usize].is_none());
}

#[test]
fn vec_length_matches_nodes_len() {
    let mut scene = Scene3D::new();
    for _ in 0..7 {
        scene.add_node(Node::new());
    }
    scene.add_root(NodeId(0));
    let xforms = scene.world_node_transforms();
    assert_eq!(xforms.len(), 7);
}

// ---- Cycle / shared-instance guard --------------------------------------

#[test]
fn cycle_visits_each_node_once() {
    // Build A → B → A. The walk must not loop forever and must report
    // exactly one matrix per node.
    let mut scene = Scene3D::new();
    let a = scene.add_node(Node::new().with_transform(translation_only([10.0, 0.0, 0.0])));
    let b = scene.add_node(Node::new().with_transform(translation_only([1.0, 0.0, 0.0])));
    scene.nodes[a.0 as usize].children.push(b);
    scene.nodes[b.0 as usize].children.push(a); // cycle
    scene.add_root(a);

    let xforms = scene.world_node_transforms();
    // A is the root: world = T(10).
    // B inherits from A: world = T(11).
    // The back-edge from B → A is suppressed because A is already
    // resolved.
    assert!(approx_eq(xforms[a.0 as usize].unwrap()[0][3], 10.0));
    assert!(approx_eq(xforms[b.0 as usize].unwrap()[0][3], 11.0));
}

#[test]
fn shared_child_resolves_to_first_parents_chain() {
    // Two roots both list the same child node. The first root encountered
    // on the DFS wins (deterministic single-resolution policy).
    let mut scene = Scene3D::new();
    let r0 = scene.add_node(Node::new().with_transform(translation_only([100.0, 0.0, 0.0])));
    let r1 = scene.add_node(Node::new().with_transform(translation_only([0.0, 100.0, 0.0])));
    let shared = scene.add_node(Node::new().with_transform(translation_only([1.0, 0.0, 0.0])));
    scene.nodes[r0.0 as usize].children.push(shared);
    scene.nodes[r1.0 as usize].children.push(shared);
    scene.add_root(r0);
    scene.add_root(r1);

    let xforms = scene.world_node_transforms();
    // r0 is pushed first, so the shared child resolves via r0's chain
    // (world translation 101 along X, not 1 along X with 100 along Y).
    let sm = xforms[shared.0 as usize].unwrap();
    assert!(approx_eq(sm[0][3], 101.0));
    assert!(approx_eq(sm[1][3], 0.0));
}

#[test]
fn self_cycle_root_is_resolved_once() {
    // A node that lists itself as a child — pathological but must not
    // loop and must return its identity-chain world matrix.
    let mut scene = Scene3D::new();
    let a = scene.add_node(Node::new().with_transform(translation_only([5.0, 0.0, 0.0])));
    scene.nodes[a.0 as usize].children.push(a);
    scene.add_root(a);
    let xforms = scene.world_node_transforms();
    assert!(approx_eq(xforms[a.0 as usize].unwrap()[0][3], 5.0));
}

// ---- Out-of-range / dangling references ---------------------------------

#[test]
fn out_of_range_root_is_skipped() {
    let mut scene = Scene3D::new();
    let valid = scene.add_node(Node::new());
    scene.add_root(NodeId(42)); // dangling
    scene.add_root(valid);
    let xforms = scene.world_node_transforms();
    assert_eq!(xforms.len(), 1);
    assert!(xforms[0].is_some());
}

#[test]
fn out_of_range_child_is_skipped() {
    let mut scene = Scene3D::new();
    let r = scene.add_node(Node::new());
    scene.nodes[r.0 as usize].children.push(NodeId(999));
    scene.add_root(r);
    let xforms = scene.world_node_transforms();
    assert!(xforms[r.0 as usize].is_some());
    // Walk completes without panicking.
}

// ---- Cross-check with Scene3D::bounding_box ------------------------------
// The two helpers walk the same forest with the same identity-root
// composition rule; on a scene whose only resource is one mesh attached
// to one node, `world_node_transforms[node.0]` applied to the local
// bounding box must equal what `bounding_box()` returns.

#[test]
fn matches_bounding_box_traversal_for_translated_mesh() {
    use oxideav_mesh3d::{Mesh, Primitive, Topology};
    let mut scene = Scene3D::new();
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(prim));
    let n = scene.add_node(
        Node::new()
            .with_transform(translation_only([10.0, 0.0, 0.0]))
            .with_mesh(mid),
    );
    scene.add_root(n);

    let xforms = scene.world_node_transforms();
    let m = xforms[n.0 as usize].unwrap();
    assert!(approx_eq(m[0][3], 10.0));

    let bbox = scene.bounding_box().unwrap();
    // Local bbox was [0..1] on x; translated by +10 → [10..11].
    assert!(approx_eq(bbox.min[0], 10.0));
    assert!(approx_eq(bbox.max[0], 11.0));
}

// ---- Determinism --------------------------------------------------------

#[test]
fn output_is_deterministic_across_calls() {
    let mut scene = Scene3D::new();
    let a = scene.add_node(Node::new().with_transform(translation_only([1.0, 2.0, 3.0])));
    let b = scene.add_node(Node::new().with_transform(scale_only([2.0, 2.0, 2.0])));
    let c = scene.add_node(Node::new().with_transform(translation_only([0.5, 0.0, 0.0])));
    scene.nodes[a.0 as usize].children.push(b);
    scene.nodes[b.0 as usize].children.push(c);
    scene.add_root(a);

    let x1 = scene.world_node_transforms();
    let x2 = scene.world_node_transforms();
    let x3 = scene.world_node_transforms();
    assert_eq!(x1, x2);
    assert_eq!(x2, x3);
}

// ---- Larger forest (sanity) ---------------------------------------------

#[test]
fn balanced_binary_tree_all_nodes_resolved() {
    // Build a depth-3 binary tree: 1 root + 2 + 4 = 7 nodes, each with a
    // tiny per-node translation. Verify every node gets a Some matrix
    // and child[k] = parent + child's local.
    let mut scene = Scene3D::new();
    let r = scene.add_node(Node::new().with_transform(translation_only([1.0, 0.0, 0.0])));
    let l1 = scene.add_node(Node::new().with_transform(translation_only([0.0, 1.0, 0.0])));
    let r1 = scene.add_node(Node::new().with_transform(translation_only([0.0, -1.0, 0.0])));
    let ll = scene.add_node(Node::new().with_transform(translation_only([0.0, 0.0, 1.0])));
    let lr = scene.add_node(Node::new().with_transform(translation_only([0.0, 0.0, -1.0])));
    let rl = scene.add_node(Node::new().with_transform(translation_only([0.0, 0.0, 2.0])));
    let rr = scene.add_node(Node::new().with_transform(translation_only([0.0, 0.0, -2.0])));
    scene.nodes[r.0 as usize]
        .children
        .extend_from_slice(&[l1, r1]);
    scene.nodes[l1.0 as usize]
        .children
        .extend_from_slice(&[ll, lr]);
    scene.nodes[r1.0 as usize]
        .children
        .extend_from_slice(&[rl, rr]);
    scene.add_root(r);

    let xforms = scene.world_node_transforms();
    assert_eq!(xforms.len(), 7);
    assert!(xforms.iter().all(|t| t.is_some()));

    // Spot-check one of the deeper paths: r → l1 → ll
    //   r:  T(1,0,0)
    //   l1: + T(0,1,0)  → (1,1,0)
    //   ll: + T(0,0,1)  → (1,1,1)
    let m = xforms[ll.0 as usize].unwrap();
    assert!(approx_eq(m[0][3], 1.0));
    assert!(approx_eq(m[1][3], 1.0));
    assert!(approx_eq(m[2][3], 1.0));

    // And the opposite extreme: r → r1 → rr → (1, -1, -2)
    let m = xforms[rr.0 as usize].unwrap();
    assert!(approx_eq(m[0][3], 1.0));
    assert!(approx_eq(m[1][3], -1.0));
    assert!(approx_eq(m[2][3], -2.0));
}

// ---- Mixed Trs + Matrix transforms --------------------------------------

#[test]
fn matrix_parent_trs_child_compose() {
    // Parent stores a non-identity Matrix variant; child stores a TRS
    // translation. Verify the composition matches an equivalent TRS
    // parent.
    let parent_mat: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut scene = Scene3D::new();
    let p = scene.add_node(Node::new().with_transform(Transform::Matrix(parent_mat)));
    let c = scene.add_node(Node::new().with_transform(translation_only([0.0, 3.0, 0.0])));
    scene.nodes[p.0 as usize].children.push(c);
    scene.add_root(p);

    let xforms = scene.world_node_transforms();
    let cm = xforms[c.0 as usize].unwrap();
    assert!(approx_eq(cm[0][3], 5.0));
    assert!(approx_eq(cm[1][3], 3.0));
    assert!(approx_eq(cm[2][3], 0.0));
}
