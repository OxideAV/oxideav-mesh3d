//! Tests for [`Scene3D::validate`].
//!
//! Round 7 polish: a defensive cross-collection consistency check
//! intended for fuzz harnesses + codec authors. The scene-graph
//! invariants come from glTF 2.0 §3.5 (id resolution) and §3.7.2
//! (per-primitive attribute length parity).

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, AudioEmitter, AudioSourceId, Indices, Interpolation, Material, Mesh, MeshId,
    MorphTarget, Node, NodeId, Primitive, Scene3D, Skeleton, SkeletonId, Skin, TextureId,
    TextureRef, Topology, ValidationError,
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

// ---- round-next: material / skin / animation / audio rules ---------------

#[test]
fn dangling_material_base_color_texture_reported() {
    let mut m = Material::new();
    m.base_color_texture = Some(TextureRef::new(TextureId(5)));
    let mut s = Scene3D::new();
    s.add_material(m);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId {
            id: 5,
            arena: "textures",
            location,
        } if location == "materials[0].base_color_texture"
    )));
}

#[test]
fn dangling_material_emissive_texture_reported() {
    let mut m = Material::new();
    m.emissive_texture = Some(TextureRef::new(TextureId(99)));
    let mut s = Scene3D::new();
    s.add_material(m);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId {
            arena: "textures",
            location,
            ..
        } if location.contains("emissive_texture")
    )));
}

#[test]
fn dangling_skeleton_joint_reported() {
    let mut skel = Skeleton::new();
    skel.joints = vec![NodeId(0), NodeId(7)];
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    s.add_skeleton(skel);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId {
            id: 7,
            arena: "nodes",
            location,
        } if location == "skeletons[0].joints[1]"
    )));
}

#[test]
fn skeleton_inverse_bind_matrix_mismatch_reported() {
    let mut skel = Skeleton::new();
    skel.joints = vec![NodeId(0), NodeId(0)];
    skel.inverse_bind_matrices = vec![[[0.0; 4]; 4]]; // 1 vs 2 joints
    let mut s = Scene3D::new();
    s.add_node(Node::new());
    s.add_skeleton(skel);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::SkeletonBindMatrixCountMismatch {
            joints: 2,
            inverse_bind_matrices: 1,
            ..
        }
    )));
}

#[test]
fn skeleton_empty_inverse_bind_matrices_is_allowed() {
    // glTF lets the field be omitted entirely; we keep that escape hatch.
    let mut skel = Skeleton::new();
    skel.joints = vec![NodeId(0)];
    // inverse_bind_matrices left empty
    let mut s = Scene3D::new();
    s.add_node(Node::new());
    s.add_skeleton(skel);
    assert!(s.validate().is_ok());
}

#[test]
fn dangling_skin_skeleton_reported() {
    let mut s = Scene3D::new();
    s.add_skin(Skin::new(SkeletonId(9)));
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId {
            id: 9,
            arena: "skeletons",
            ..
        }
    )));
}

#[test]
fn dangling_skin_root_node_reported() {
    let mut s = Scene3D::new();
    let sk = s.add_skeleton(Skeleton::new());
    s.add_skin(Skin::new(sk).with_root(NodeId(42)));
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId {
            id: 42,
            arena: "nodes",
            location,
        } if location == "skins[0].root_node"
    )));
}

#[test]
fn dangling_audio_emitter_source_reported() {
    let mut s = Scene3D::new();
    s.add_audio_emitter(AudioEmitter::new(AudioSourceId(3)));
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId {
            id: 3,
            arena: "audio_sources",
            ..
        }
    )));
}

#[test]
fn animation_channel_dangling_target_node_reported() {
    let mut s = Scene3D::new();
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: NodeId(11),
            property: AnimationProperty::Translation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Vec3(vec![[0.0; 3], [1.0, 0.0, 0.0]]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId {
            id: 11,
            arena: "nodes",
            location,
        } if location.ends_with(".target.node")
    )));
}

#[test]
fn animation_sampler_empty_keyframes_reported() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: nid,
            property: AnimationProperty::Translation,
        },
        sampler: AnimationSampler {
            keyframes: vec![],
            values: AnimationValues::Vec3(vec![]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);
    let errs = s.validate().unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::AnimationSamplerEmpty { .. })));
}

#[test]
fn animation_sampler_keyframes_must_strictly_increase() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: nid,
            property: AnimationProperty::Translation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0, 0.5], // 0.5 < 1.0
            values: AnimationValues::Vec3(vec![[0.0; 3]; 3]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::AnimationKeyframesNotStrictlyIncreasing { .. }
    )));
}

#[test]
fn animation_rotation_with_vec3_values_is_variant_mismatch() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: nid,
            property: AnimationProperty::Rotation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Vec3(vec![[0.0; 3], [1.0, 0.0, 0.0]]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::AnimationValueVariantMismatch {
            property: "Rotation",
            expected_variant: "Quat",
            actual_variant: "Vec3",
            ..
        }
    )));
}

#[test]
fn animation_cubicspline_values_must_be_three_times_keyframes() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: nid,
            property: AnimationProperty::Translation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0, 2.0], // 3 keyframes → need 9 vec3 values
            values: AnimationValues::Vec3(vec![[0.0; 3]; 3]), // only 3
            interpolation: Interpolation::CubicSpline,
        },
    });
    s.add_animation(anim);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::AnimationSamplerLengthMismatch {
            keyframes: 3,
            values: 3,
            interpolation: "CubicSpline",
            ..
        }
    )));
}

#[test]
fn animation_linear_translation_passes_when_lengths_match() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: nid,
            property: AnimationProperty::Translation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0, 2.0],
            values: AnimationValues::Vec3(vec![[0.0; 3], [1.0; 3], [2.0; 3]]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);
    assert!(s.validate().is_ok());
}

#[test]
fn animation_morph_weights_must_be_multiple_of_keyframes() {
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    let mut anim = Animation::new(None);
    // 2 keyframes, 2 morph targets on bound mesh ⇒ 4 scalars expected.
    // We give 3 — not a multiple of keyframes count 2 ⇒ rejected.
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: nid,
            property: AnimationProperty::MorphWeights,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Scalar(vec![0.0, 1.0, 0.5]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);
    let errs = s.validate().unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::AnimationSamplerLengthMismatch { .. })));
}

#[test]
fn animation_morph_weights_multiple_passes() {
    // 2 keyframes × 3 targets = 6 scalars ⇒ OK.
    let mut s = Scene3D::new();
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: nid,
            property: AnimationProperty::MorphWeights,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Scalar(vec![0.0, 0.5, 1.0, 1.0, 0.5, 0.0]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);
    assert!(s.validate().is_ok());
}

#[test]
fn full_valid_scene_with_all_resource_kinds_passes() {
    // Self-contained correctness oracle: builds a scene exercising
    // every newly-checked relation and asserts validate() is Ok.
    let mut s = Scene3D::new();
    let tid = s.add_texture(oxideav_mesh3d::Texture::from_uri("img.png"));
    let mut mat = Material::new();
    mat.base_color_texture = Some(TextureRef::new(tid));
    mat.normal_texture = Some(TextureRef::new(tid));
    let _midx = s.add_material(mat);
    let nid = s.add_node(Node::new());
    s.add_root(nid);
    let mut skel = Skeleton::new();
    skel.joints = vec![nid];
    skel.inverse_bind_matrices = vec![[[0.0; 4]; 4]];
    let skel_id = s.add_skeleton(skel);
    s.add_skin(Skin::new(skel_id).with_root(nid));
    let mut anim = Animation::new(None);
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: nid,
            property: AnimationProperty::Rotation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Quat(vec![[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);
    assert!(s.validate().is_ok(), "{:?}", s.validate().err());
}
