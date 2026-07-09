//! Tests for `Scene3D::skin_root` — explicit skin root passthrough +
//! lowest-common-ancestor fallback over the joint set (glTF 2.0
//! §3.7.3.2 common-root semantics).

use oxideav_mesh3d::{Node, NodeId, Scene3D, Skeleton, Skin, SkinId};

/// Build a scene from a parent table: `spec[i] = Some(p)` makes node
/// `i` a child of `p`; `None` makes it a root. Parents must precede
/// children.
fn scene_from_parents(spec: &[Option<usize>]) -> Scene3D {
    let mut s = Scene3D::new();
    for (i, parent) in spec.iter().enumerate() {
        let id = s.add_node(Node::new());
        assert_eq!(id.0 as usize, i);
        match parent {
            Some(p) => s.nodes[*p].children.push(id),
            None => s.add_root(id),
        }
    }
    s
}

fn add_skin(s: &mut Scene3D, joints: Vec<u32>, root: Option<u32>) -> SkinId {
    let skel = s.add_skeleton(Skeleton {
        name: None,
        joints: joints.into_iter().map(NodeId).collect(),
        inverse_bind_matrices: Vec::new(),
    });
    let mut skin = Skin::new(skel);
    if let Some(r) = root {
        skin = skin.with_root(NodeId(r));
    }
    s.add_skin(skin)
}

#[test]
fn explicit_root_wins_over_lca() {
    //      0
    //     / \
    //    1   2
    let mut s = scene_from_parents(&[None, Some(0), Some(0)]);
    let skin = add_skin(&mut s, vec![1, 2], Some(2));
    assert_eq!(s.skin_root(skin), Some(NodeId(2)), "author's pivot wins");
}

#[test]
fn dangling_explicit_root_yields_none() {
    let mut s = scene_from_parents(&[None]);
    let skin = add_skin(&mut s, vec![0], Some(77));
    assert_eq!(s.skin_root(skin), None);
}

#[test]
fn lca_of_siblings_is_their_parent() {
    //      0
    //     / \
    //    1   2
    //   /     \
    //  3       4
    let mut s = scene_from_parents(&[None, Some(0), Some(0), Some(1), Some(2)]);
    let skin = add_skin(&mut s, vec![3, 4], None);
    assert_eq!(s.skin_root(skin), Some(NodeId(0)), "grandparent LCA");
}

#[test]
fn lca_of_chain_is_the_top_joint() {
    // 0 → 1 → 2 → 3 chain; joints {1, 2, 3}: the common root is
    // joint 1 itself (spec: the common root may be a joint node).
    let mut s = scene_from_parents(&[None, Some(0), Some(1), Some(2)]);
    let skin = add_skin(&mut s, vec![3, 1, 2], None);
    assert_eq!(s.skin_root(skin), Some(NodeId(1)), "ancestor joint is LCA");
}

#[test]
fn single_joint_is_its_own_root() {
    let mut s = scene_from_parents(&[None, Some(0)]);
    let skin = add_skin(&mut s, vec![1], None);
    assert_eq!(s.skin_root(skin), Some(NodeId(1)));
}

#[test]
fn joints_in_disjoint_trees_yield_none() {
    // Two separate roots.
    let mut s = scene_from_parents(&[None, None, Some(0), Some(1)]);
    let skin = add_skin(&mut s, vec![2, 3], None);
    assert_eq!(s.skin_root(skin), None, "§3.7.3.2 violated: no common root");
}

#[test]
fn deep_asymmetric_depths_resolve() {
    // 0 → 1 → 2 → 3 → 4 and 0 → 5; joints {4, 5} at depths 4 and 1.
    let mut s = scene_from_parents(&[None, Some(0), Some(1), Some(2), Some(3), Some(0)]);
    let skin = add_skin(&mut s, vec![4, 5], None);
    assert_eq!(s.skin_root(skin), Some(NodeId(0)));
}

#[test]
fn malformed_skins_yield_none() {
    let mut s = scene_from_parents(&[None]);
    // Out-of-range skin id.
    assert_eq!(s.skin_root(SkinId(9)), None);
    // Empty joint list.
    let empty = add_skin(&mut s, vec![], None);
    assert_eq!(s.skin_root(empty), None);
    // Out-of-range joint id.
    let dangling = add_skin(&mut s, vec![42], None);
    assert_eq!(s.skin_root(dangling), None);
}

#[test]
fn lca_covers_the_whole_rig_subtree() {
    // The computed root's descendants must contain every joint — the
    // property that makes skin_root usable for one-subtree culling.
    let mut s = scene_from_parents(&[None, Some(0), Some(1), Some(1), Some(2), Some(3)]);
    let skin = add_skin(&mut s, vec![4, 5, 2], None);
    let root = s.skin_root(skin).expect("common root");
    assert_eq!(root, NodeId(1));
    let subtree = s.descendants(root);
    for j in [NodeId(4), NodeId(5), NodeId(2)] {
        assert!(subtree.contains(&j), "joint {j:?} outside {root:?} subtree");
    }
}
