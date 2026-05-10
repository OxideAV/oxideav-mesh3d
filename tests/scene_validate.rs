//! Tests for [`Scene3D::validate`].
//!
//! Round 7 polish: a defensive cross-collection consistency check
//! intended for fuzz harnesses + codec authors. The scene-graph
//! invariants come from glTF 2.0 §3.5 (id resolution) and §3.7.2
//! (per-primitive attribute length parity).

use oxideav_mesh3d::{
    Indices, Mesh, MeshId, MorphTarget, Node, NodeId, Primitive, Scene3D, Topology, ValidationError,
};

fn one_triangle_primitive() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p
}

#[test]
fn empty_scene_is_valid() {
    let s = Scene3D::new();
    assert!(s.validate().is_ok());
}

#[test]
fn one_triangle_with_normals_validates() {
    let mut p = one_triangle_primitive();
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]];
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(p));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    assert!(s.validate().is_ok());
}

#[test]
fn dangling_root_is_reported() {
    let mut s = Scene3D::new();
    s.add_root(NodeId(0));
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    assert!(matches!(
        &errs[0],
        ValidationError::DanglingId {
            id: 0,
            arena: "nodes",
            ..
        }
    ));
}

#[test]
fn dangling_mesh_id_on_node_is_reported() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new().with_mesh(MeshId(7)));
    s.add_root(nid);
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    assert!(matches!(
        &errs[0],
        ValidationError::DanglingId {
            id: 7,
            arena: "meshes",
            ..
        }
    ));
}

#[test]
fn attribute_length_mismatch_reported() {
    let mut p = one_triangle_primitive();
    // 3 positions but only 2 normals — broken.
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 2]);
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None).with_primitive(p));
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    assert!(matches!(
        &errs[0],
        ValidationError::AttributeLengthMismatch {
            expected: 3,
            actual: 2,
            ..
        }
    ));
}

#[test]
fn uv_set_mismatch_reported_with_set_index() {
    let mut p = one_triangle_primitive();
    p.uvs = vec![
        vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], // ok
        vec![[0.0, 0.0], [1.0, 0.0]],             // broken
    ];
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None).with_primitive(p));
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    if let ValidationError::AttributeLengthMismatch { location, .. } = &errs[0] {
        assert!(location.contains("uvs[1]"), "location was {location}");
    } else {
        panic!("wrong variant: {:?}", errs[0]);
    }
}

#[test]
fn out_of_range_index_reported() {
    let mut p = one_triangle_primitive();
    p.indices = Some(Indices::U16(vec![0, 1, 99])); // 99 >= 3
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None).with_primitive(p));
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::IndexOutOfRange {
            vertex_count: 3,
            ..
        }
    )));
}

#[test]
fn morph_target_length_mismatch_reported() {
    let mut p = one_triangle_primitive();
    p.targets = vec![MorphTarget {
        position: Some(vec![[0.1, 0.0, 0.0]; 2]), // 2 vs 3 positions
        normal: None,
        tangent: None,
    }];
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None).with_primitive(p));
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    if let ValidationError::AttributeLengthMismatch { location, .. } = &errs[0] {
        assert!(
            location.contains("targets[0].position"),
            "location was {location}"
        );
    } else {
        panic!("wrong variant: {:?}", errs[0]);
    }
}

#[test]
fn mesh_weight_vs_primitive_target_count_mismatch_reported() {
    let p = one_triangle_primitive(); // zero targets
    let m = Mesh::new(None)
        .with_primitive(p)
        .with_weights(vec![1.0, 0.5]);
    let mut s = Scene3D::new();
    s.add_mesh(m);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::MorphWeightCountMismatch {
            mesh_weights: 2,
            primitive_targets: 0,
            ..
        }
    )));
}

#[test]
fn validate_collects_multiple_errors_in_one_pass() {
    let mut p = one_triangle_primitive();
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 1]); // wrong length
    p.indices = Some(Indices::U32(vec![0, 1, 100])); // out of range
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None).with_primitive(p));
    s.add_root(NodeId(42)); // dangling root
    let errs = s.validate().unwrap_err();
    // Three independent issues — the walk doesn't short-circuit.
    assert!(errs.len() >= 3, "got {} errors: {:?}", errs.len(), errs);
}

#[test]
fn validation_error_display_carries_location() {
    let mut s = Scene3D::new();
    s.add_root(NodeId(3));
    let errs = s.validate().unwrap_err();
    let msg = format!("{}", errs[0]);
    assert!(msg.contains("roots[0]"), "got: {msg}");
    assert!(msg.contains("nodes"), "got: {msg}");
}
