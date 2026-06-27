//! Scene-graph navigation + transform baking: parent/ancestor/
//! descendant queries and the world-preserving bake.

use oxideav_mesh3d::{Node, NodeId, Scene3D, Transform};

/// Build a chain root → child → grandchild, each translated by +1 on x,
/// plus a second root. Layout of node ids:
///   0: root A   (translate +1 x), child = 1
///   1: child    (translate +1 x), child = 2
///   2: grandchild (translate +1 x)
///   3: root B   (translate +10 y)
fn chain_scene() -> Scene3D {
    let mut s = Scene3D::new();
    let tx = |x: f32| Transform::Trs {
        translation: [x, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    };
    let a = s.add_node(Node::new().with_transform(tx(1.0)));
    let c = s.add_node(Node::new().with_transform(tx(1.0)));
    let g = s.add_node(Node::new().with_transform(tx(1.0)));
    let b = s.add_node(Node::new().with_transform(Transform::Trs {
        translation: [0.0, 10.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }));
    s.nodes[a.0 as usize].children = vec![c];
    s.nodes[c.0 as usize].children = vec![g];
    s.add_root(a);
    s.add_root(b);
    s
}

#[test]
fn parents_map_reflects_hierarchy() {
    let s = chain_scene();
    let p = s.parents();
    assert_eq!(p[0], None); // root A
    assert_eq!(p[1], Some(NodeId(0))); // child of A
    assert_eq!(p[2], Some(NodeId(1))); // grandchild
    assert_eq!(p[3], None); // root B
}

#[test]
fn ancestors_runs_root_to_parent() {
    let s = chain_scene();
    assert_eq!(s.ancestors(NodeId(2)), vec![NodeId(0), NodeId(1)]);
    assert_eq!(s.ancestors(NodeId(1)), vec![NodeId(0)]);
    assert!(s.ancestors(NodeId(0)).is_empty()); // root has none
    assert!(s.ancestors(NodeId(3)).is_empty());
}

#[test]
fn descendants_includes_self_and_subtree() {
    let s = chain_scene();
    assert_eq!(
        s.descendants(NodeId(0)),
        vec![NodeId(0), NodeId(1), NodeId(2)]
    );
    assert_eq!(s.descendants(NodeId(1)), vec![NodeId(1), NodeId(2)]);
    assert_eq!(s.descendants(NodeId(2)), vec![NodeId(2)]);
    assert_eq!(s.descendants(NodeId(3)), vec![NodeId(3)]);
}

#[test]
fn is_ancestor_of_is_proper() {
    let s = chain_scene();
    assert!(s.is_ancestor_of(NodeId(0), NodeId(2)));
    assert!(s.is_ancestor_of(NodeId(1), NodeId(2)));
    assert!(!s.is_ancestor_of(NodeId(2), NodeId(2))); // not its own ancestor
    assert!(!s.is_ancestor_of(NodeId(3), NodeId(2))); // other root
    assert!(!s.is_ancestor_of(NodeId(2), NodeId(0))); // wrong direction
}

#[test]
fn out_of_range_queries_are_empty() {
    let s = chain_scene();
    assert!(s.ancestors(NodeId(99)).is_empty());
    assert!(s.descendants(NodeId(99)).is_empty());
    assert!(!s.is_ancestor_of(NodeId(99), NodeId(0)));
}

#[test]
fn bake_preserves_world_transforms() {
    let s = chain_scene();
    let before = s.world_node_transforms();
    let baked = s.bake_transforms();
    let after = baked.world_node_transforms();

    // Every reachable node's world matrix is unchanged by baking.
    for i in 0..before.len() {
        match (before[i], after[i]) {
            (Some(b), Some(a)) => {
                for r in 0..4 {
                    for c in 0..4 {
                        assert!(
                            (b[r][c] - a[r][c]).abs() < 1e-5,
                            "node {i} world entry [{r}][{c}] drifted: {} vs {}",
                            b[r][c],
                            a[r][c]
                        );
                    }
                }
            }
            (None, None) => {}
            _ => panic!("reachability changed for node {i}"),
        }
    }
}

#[test]
fn bake_flattens_hierarchy() {
    let s = chain_scene();
    let baked = s.bake_transforms();

    // Same node count; every node is now a root with no children.
    assert_eq!(baked.nodes.len(), s.nodes.len());
    for node in &baked.nodes {
        assert!(node.children.is_empty());
        // Each baked transform is the Matrix form.
        assert!(matches!(node.transform, Transform::Matrix(_)));
    }
    // Reachable nodes (all four) are roots in DFS order: A, child,
    // grandchild, then root B.
    assert_eq!(
        baked.roots,
        vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)]
    );
}

#[test]
fn bake_grandchild_world_is_translation_sum() {
    // Grandchild's world translation is +3 on x (three +1 hops).
    let s = chain_scene();
    let baked = s.bake_transforms();
    if let Transform::Matrix(m) = baked.nodes[2].transform {
        assert!((m[0][3] - 3.0).abs() < 1e-5, "x = {}", m[0][3]);
    } else {
        panic!("expected baked Matrix");
    }
}

#[test]
fn unreachable_node_is_not_promoted_to_root() {
    let mut s = chain_scene();
    // Add an orphan node not referenced by any root or child.
    let orphan = s.add_node(Node::new().with_transform(Transform::Trs {
        translation: [5.0, 5.0, 5.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }));
    let baked = s.bake_transforms();
    assert!(!baked.roots.contains(&orphan));
    // Orphan keeps its original local transform (not baked).
    assert!(matches!(
        baked.nodes[orphan.0 as usize].transform,
        Transform::Trs { .. }
    ));
}

#[test]
fn node_local_matrix_matches_transform() {
    let n = Node::new().with_transform(Transform::Trs {
        translation: [2.0, 3.0, 4.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    });
    let m = n.local_matrix();
    assert_eq!(m[0][3], 2.0);
    assert_eq!(m[1][3], 3.0);
    assert_eq!(m[2][3], 4.0);
}
