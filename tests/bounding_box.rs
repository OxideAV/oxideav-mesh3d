//! Tests for [`BoundingBox`] + [`Scene3D::bounding_box`] +
//! [`Mesh::bounding_box`] + [`Primitive::bounding_box`].
//!
//! Round-next polish: a renderer wants the scene-graph world-space
//! extent for default camera framing / shadow-cascade sizing without
//! having to walk the typed model itself.

use oxideav_mesh3d::{BoundingBox, Mesh, Node, NodeId, Primitive, Scene3D, Topology, Transform};

fn unit_cube_prim() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    p
}

// ---- BoundingBox basics --------------------------------------------------

#[test]
fn from_point_is_degenerate_zero_size() {
    let b = BoundingBox::from_point([3.0, -1.0, 2.5]);
    assert_eq!(b.min, [3.0, -1.0, 2.5]);
    assert_eq!(b.max, [3.0, -1.0, 2.5]);
    assert_eq!(b.size(), [0.0, 0.0, 0.0]);
    assert_eq!(b.center(), [3.0, -1.0, 2.5]);
    assert!(b.is_valid());
}

#[test]
fn from_points_empty_iterator_yields_none() {
    let pts: Vec<[f32; 3]> = vec![];
    assert!(BoundingBox::from_points(pts).is_none());
}

#[test]
fn from_points_skips_nan_components() {
    let pts = vec![
        [0.0, 0.0, 0.0],
        [f32::NAN, 1.0, 1.0], // dropped
        [1.0, 1.0, 1.0],
    ];
    let b = BoundingBox::from_points(pts).unwrap();
    assert_eq!(b.min, [0.0, 0.0, 0.0]);
    assert_eq!(b.max, [1.0, 1.0, 1.0]);
}

#[test]
fn from_points_all_nan_yields_none() {
    let pts = vec![[f32::NAN, 0.0, 0.0], [0.0, f32::NAN, 0.0]];
    assert!(BoundingBox::from_points(pts).is_none());
}

#[test]
fn unit_cube_aabb_is_zero_to_one() {
    let b = BoundingBox::from_points(unit_cube_prim().positions).unwrap();
    assert_eq!(b.min, [0.0; 3]);
    assert_eq!(b.max, [1.0; 3]);
    assert_eq!(b.size(), [1.0; 3]);
    assert_eq!(b.center(), [0.5; 3]);
}

#[test]
fn expand_grows_corners() {
    let b = BoundingBox::from_point([0.0; 3]).expand([2.0, -1.0, 0.5]);
    assert_eq!(b.min, [0.0, -1.0, 0.0]);
    assert_eq!(b.max, [2.0, 0.0, 0.5]);
}

#[test]
fn union_is_componentwise_min_max() {
    let a = BoundingBox {
        min: [0.0, 0.0, 0.0],
        max: [1.0, 1.0, 1.0],
    };
    let b = BoundingBox {
        min: [-1.0, 0.5, 2.0],
        max: [0.5, 3.0, 5.0],
    };
    let u = a.union(b);
    assert_eq!(u.min, [-1.0, 0.0, 0.0]);
    assert_eq!(u.max, [1.0, 3.0, 5.0]);
}

#[test]
fn transform_translation_shifts_box() {
    let b = BoundingBox::from_point([1.0, 2.0, 3.0]).expand([2.0, 3.0, 4.0]);
    // Pure translation matrix (move by (10, 20, 30)).
    let m = [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let xf = b.transform(m);
    assert_eq!(xf.min, [11.0, 22.0, 33.0]);
    assert_eq!(xf.max, [12.0, 23.0, 34.0]);
}

#[test]
fn transform_rotation_90_around_y_swaps_x_z_extents() {
    // 90 deg rotation around +Y: (x, y, z) → (z, y, -x).
    let b = BoundingBox {
        min: [0.0, 0.0, 0.0],
        max: [1.0, 1.0, 2.0],
    };
    let m = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let xf = b.transform(m);
    // Old X extent was [0,1] → maps to Z = -[0,1] = [-1,0].
    // Old Z extent was [0,2] → maps to X = [0,2].
    assert!((xf.min[0] - 0.0).abs() < 1e-5);
    assert!((xf.max[0] - 2.0).abs() < 1e-5);
    assert!((xf.min[1] - 0.0).abs() < 1e-5);
    assert!((xf.max[1] - 1.0).abs() < 1e-5);
    assert!((xf.min[2] - -1.0).abs() < 1e-5);
    assert!((xf.max[2] - 0.0).abs() < 1e-5);
}

#[test]
fn transform_scale_two_x_doubles_x_extent() {
    let b = BoundingBox::from_point([1.0, 1.0, 1.0]).expand([3.0, 2.0, 2.0]);
    let m = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let xf = b.transform(m);
    assert_eq!(xf.min[0], 2.0);
    assert_eq!(xf.max[0], 6.0);
    assert_eq!(xf.size(), [4.0, 1.0, 1.0]);
}

// ---- Primitive::bounding_box --------------------------------------------

#[test]
fn empty_primitive_has_no_bbox() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.bounding_box().is_none());
}

#[test]
fn primitive_bbox_matches_positions() {
    let p = unit_cube_prim();
    let b = p.bounding_box().unwrap();
    assert_eq!(b.min, [0.0; 3]);
    assert_eq!(b.max, [1.0; 3]);
}

// ---- Mesh::bounding_box -------------------------------------------------

#[test]
fn empty_mesh_has_no_bbox() {
    let m = Mesh::new(None);
    assert!(m.bounding_box().is_none());
}

#[test]
fn mesh_with_two_primitives_unions_extents() {
    let mut p1 = Primitive::new(Topology::Triangles);
    p1.positions = vec![[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]];
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]];
    let m = Mesh::new(None).with_primitive(p1).with_primitive(p2);
    let b = m.bounding_box().unwrap();
    assert_eq!(b.min, [-1.0, -1.0, -1.0]);
    assert_eq!(b.max, [3.0, 4.0, 5.0]);
}

// ---- Scene3D::bounding_box ----------------------------------------------

#[test]
fn empty_scene_has_no_bbox() {
    assert!(Scene3D::new().bounding_box().is_none());
}

#[test]
fn scene_without_meshes_has_no_bbox() {
    let mut s = Scene3D::new();
    let n = s.add_node(Node::new());
    s.add_root(n);
    assert!(s.bounding_box().is_none());
}

#[test]
fn scene_with_one_mesh_at_identity_matches_mesh_bbox() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(unit_cube_prim()));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    let b = s.bounding_box().unwrap();
    assert_eq!(b.min, [0.0; 3]);
    assert_eq!(b.max, [1.0; 3]);
}

#[test]
fn scene_node_translation_is_applied() {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(unit_cube_prim()));
    let nid = s.add_node(Node::new().with_mesh(mid).with_transform(Transform::Trs {
        translation: [10.0, 20.0, 30.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }));
    s.add_root(nid);
    let b = s.bounding_box().unwrap();
    assert_eq!(b.min, [10.0, 20.0, 30.0]);
    assert_eq!(b.max, [11.0, 21.0, 31.0]);
}

#[test]
fn scene_parent_child_transforms_compose() {
    // Parent translates by (10, 0, 0); child translates by (0, 5, 0) and
    // carries the cube. Total expected: cube shifted to (10..11, 5..6, 0..1).
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(unit_cube_prim()));
    let child = s.add_node(Node::new().with_mesh(mid).with_transform(Transform::Trs {
        translation: [0.0, 5.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }));
    let mut parent = Node::new().with_transform(Transform::Trs {
        translation: [10.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    });
    parent.children.push(child);
    let pid = s.add_node(parent);
    s.add_root(pid);
    let b = s.bounding_box().unwrap();
    assert!((b.min[0] - 10.0).abs() < 1e-5, "min was {:?}", b.min);
    assert!((b.min[1] - 5.0).abs() < 1e-5, "min was {:?}", b.min);
    assert!((b.min[2] - 0.0).abs() < 1e-5);
    assert!((b.max[0] - 11.0).abs() < 1e-5);
    assert!((b.max[1] - 6.0).abs() < 1e-5);
    assert!((b.max[2] - 1.0).abs() < 1e-5);
}

#[test]
fn scene_skips_unreachable_meshes() {
    // Two meshes: one attached to a rooted node, one to a detached node.
    let mut s = Scene3D::new();
    let m_rooted = s.add_mesh(Mesh::new(None).with_primitive(unit_cube_prim()));
    let mut other = Primitive::new(Topology::Triangles);
    other.positions = vec![[100.0; 3], [200.0; 3]];
    let m_detached = s.add_mesh(Mesh::new(None).with_primitive(other));
    let nid = s.add_node(Node::new().with_mesh(m_rooted));
    s.add_root(nid);
    // detached node, not added to roots
    let _ = s.add_node(Node::new().with_mesh(m_detached));
    let b = s.bounding_box().unwrap();
    assert_eq!(b.max, [1.0; 3], "detached mesh leaked into bbox");
}

#[test]
fn scene_unions_multiple_root_meshes() {
    let mut s = Scene3D::new();
    let m1 = s.add_mesh(Mesh::new(None).with_primitive(unit_cube_prim()));
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]];
    let m2 = s.add_mesh(Mesh::new(None).with_primitive(p2));
    let n1 = s.add_node(Node::new().with_mesh(m1));
    let n2 = s.add_node(Node::new().with_mesh(m2));
    s.add_root(n1);
    s.add_root(n2);
    let b = s.bounding_box().unwrap();
    assert_eq!(b.min, [0.0; 3]);
    assert_eq!(b.max, [6.0; 3]);
}

#[test]
fn scene_handles_self_cycle_without_infinite_recursion() {
    // Pathological: a node lists itself as its own child. The walk's
    // visited set must catch this; otherwise the test hangs.
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(unit_cube_prim()));
    let nid = s.add_node(Node::new().with_mesh(mid));
    // Mutate to add self-cycle.
    s.node_mut(nid).unwrap().children.push(nid);
    s.add_root(nid);
    let b = s.bounding_box().unwrap();
    assert_eq!(b.min, [0.0; 3]);
    assert_eq!(b.max, [1.0; 3]);
}

#[test]
fn scene_handles_dangling_root_node_id_gracefully() {
    let mut s = Scene3D::new();
    s.add_root(NodeId(99)); // dangling
    assert!(s.bounding_box().is_none());
}
