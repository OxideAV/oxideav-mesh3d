//! Tests for [`Scene3D::validate`].
//!
//! Round 7 polish: a defensive cross-collection consistency check
//! intended for fuzz harnesses + codec authors. The scene-graph
//! invariants come from glTF 2.0 §3.5 (id resolution) and §3.7.2
//! (per-primitive attribute length parity).

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, AudioEmitter, AudioSourceId, Indices, Interpolation, Material, Mesh, MeshId,
    MorphTarget, Node, NodeId, Primitive, Scene3D, Skeleton, SkeletonId, Skin, Specular, TextureId,
    TextureRef, TextureTransform, Topology, ValidationError,
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
    p.targets = vec![MorphTarget::with_deltas(
        Some(vec![[0.1, 0.0, 0.0]; 2]),
        None,
        None,
    )];
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
fn dangling_specular_textures_reported() {
    let mut m = Material::new();
    m.ext.specular = Some(Specular {
        factor: 1.0,
        factor_texture: Some(TextureRef::new(TextureId(7))),
        color_factor: [1.0, 1.0, 1.0],
        color_texture: Some(TextureRef::new(TextureId(8))),
    });
    let mut s = Scene3D::new();
    s.add_material(m);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId { id: 7, arena: "textures", location }
            if location == "materials[0].ext.specular.factor_texture"
    )));
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId { id: 8, arena: "textures", location }
            if location == "materials[0].ext.specular.color_texture"
    )));
}

/// The layered extension maps (clearcoat / sheen / transmission /
/// volume / iridescence / anisotropy) must be dangling-checked too —
/// validation walks `Material::texture_refs`, which enumerates every
/// slot the model exposes.
#[test]
fn dangling_layered_extension_textures_reported() {
    let mut m = Material::new();
    m.ext.clearcoat = Some(oxideav_mesh3d::Clearcoat {
        factor_texture: Some(TextureRef::new(TextureId(20))),
        roughness_texture: Some(TextureRef::new(TextureId(21))),
        normal_texture: Some(TextureRef::new(TextureId(22))),
        ..Default::default()
    });
    m.ext.sheen = Some(oxideav_mesh3d::Sheen {
        color_texture: Some(TextureRef::new(TextureId(23))),
        roughness_texture: Some(TextureRef::new(TextureId(24))),
        ..Default::default()
    });
    m.ext.transmission = Some(oxideav_mesh3d::Transmission {
        factor_texture: Some(TextureRef::new(TextureId(25))),
        ..Default::default()
    });
    m.ext.volume = Some(oxideav_mesh3d::Volume {
        thickness_texture: Some(TextureRef::new(TextureId(26))),
        ..Default::default()
    });
    m.ext.iridescence = Some(oxideav_mesh3d::Iridescence {
        factor_texture: Some(TextureRef::new(TextureId(27))),
        thickness_texture: Some(TextureRef::new(TextureId(28))),
        ..Default::default()
    });
    m.ext.anisotropy = Some(oxideav_mesh3d::Anisotropy {
        texture: Some(TextureRef::new(TextureId(29))),
        ..Default::default()
    });
    m.ext.diffuse_transmission = Some(oxideav_mesh3d::DiffuseTransmission {
        factor_texture: Some(TextureRef::new(TextureId(30))),
        color_texture: Some(TextureRef::new(TextureId(31))),
        ..Default::default()
    });
    let mut s = Scene3D::new();
    s.add_material(m);
    let errs = s.validate().unwrap_err();
    for (id, slot) in [
        (20, "materials[0].ext.clearcoat.factor_texture"),
        (21, "materials[0].ext.clearcoat.roughness_texture"),
        (22, "materials[0].ext.clearcoat.normal_texture"),
        (23, "materials[0].ext.sheen.color_texture"),
        (24, "materials[0].ext.sheen.roughness_texture"),
        (25, "materials[0].ext.transmission.factor_texture"),
        (26, "materials[0].ext.volume.thickness_texture"),
        (27, "materials[0].ext.iridescence.factor_texture"),
        (28, "materials[0].ext.iridescence.thickness_texture"),
        (29, "materials[0].ext.anisotropy.texture"),
        (30, "materials[0].ext.diffuse_transmission.factor_texture"),
        (31, "materials[0].ext.diffuse_transmission.color_texture"),
    ] {
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ValidationError::DanglingId { id: got, arena: "textures", location }
                    if *got == id && location == slot
            )),
            "missing dangling report for {slot}"
        );
    }
}

/// Extension texture slots that reference **live** textures must not
/// be flagged.
#[test]
fn live_layered_extension_textures_validate_clean() {
    let mut s = Scene3D::new();
    let tid = s.add_texture(oxideav_mesh3d::Texture::from_uri("lacquer.png"));
    let mut m = Material::new();
    m.ext.clearcoat = Some(oxideav_mesh3d::Clearcoat {
        factor_texture: Some(TextureRef::new(tid)),
        normal_texture: Some(TextureRef::new(tid)),
        ..Default::default()
    });
    m.ext.anisotropy = Some(oxideav_mesh3d::Anisotropy {
        texture: Some(TextureRef::new(tid)),
        ..Default::default()
    });
    s.add_material(m);
    assert!(s.validate().is_ok());
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
    // Affine identity IBM — fourth row is [0, 0, 0, 1] per glTF §5.28.1.
    skel.inverse_bind_matrices = vec![[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]];
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

#[test]
fn skeleton_ibm_last_row_must_be_affine() {
    // glTF 2.0 §5.28.1: the fourth row of every inverse-bind matrix
    // MUST be [0, 0, 0, 1]. A zero last row is the most common bug
    // (forgetting to set the homogeneous component).
    let mut skel = Skeleton::new();
    skel.joints = vec![NodeId(0)];
    skel.inverse_bind_matrices = vec![[[0.0; 4]; 4]]; // last row = [0,0,0,0]
    let mut s = Scene3D::new();
    s.add_node(Node::new());
    s.add_skeleton(skel);
    let errs = s.validate().unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ValidationError::SkeletonBindMatrixNotAffine {
                last_row: [0.0, 0.0, 0.0, 0.0],
                ..
            }
        )),
        "expected SkeletonBindMatrixNotAffine, got {errs:?}"
    );
}

#[test]
fn skeleton_ibm_projective_last_row_reported() {
    // A non-(0,0,0,1) last row that's not all-zero — would imply a
    // perspective projection getting smuggled into the skinning math.
    let mut skel = Skeleton::new();
    skel.joints = vec![NodeId(0)];
    skel.inverse_bind_matrices = vec![[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5, 1.0], // perspective-ish entry
    ]];
    let mut s = Scene3D::new();
    s.add_node(Node::new());
    s.add_skeleton(skel);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::SkeletonBindMatrixNotAffine {
            last_row: [0.0, 0.0, 0.5, 1.0],
            ..
        }
    )));
}

#[test]
fn skeleton_ibm_affine_identity_passes() {
    let mut skel = Skeleton::new();
    skel.joints = vec![NodeId(0)];
    skel.inverse_bind_matrices = vec![[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]];
    let mut s = Scene3D::new();
    s.add_node(Node::new());
    s.add_skeleton(skel);
    assert!(s.validate().is_ok());
}

#[test]
fn skeleton_ibm_display_breadcrumb() {
    let mut skel = Skeleton::new();
    skel.joints = vec![NodeId(0)];
    skel.inverse_bind_matrices = vec![[[0.0; 4]; 4]];
    let mut s = Scene3D::new();
    s.add_node(Node::new());
    s.add_skeleton(skel);
    let errs = s.validate().unwrap_err();
    let msg = errs
        .iter()
        .find(|e| matches!(e, ValidationError::SkeletonBindMatrixNotAffine { .. }))
        .unwrap()
        .to_string();
    assert!(
        msg.contains("skeletons[0].inverse_bind_matrices[0]"),
        "{msg}"
    );
    assert!(
        msg.contains("[0.0, 0.0, 0.0, 0.0]") || msg.contains("[0, 0, 0, 0]"),
        "{msg}"
    );
}

// ---------------------------------------------------------------
// UV-set coverage + texture-transform finiteness

/// One textured material bound to a primitive carrying `n_uv_sets`
/// UV channels; the material's base-colour reference is `r`.
fn textured_prim_scene(r: TextureRef, n_uv_sets: usize) -> Scene3D {
    let mut s = Scene3D::new();
    s.add_texture(oxideav_mesh3d::Texture::from_uri("t.png"));
    let mut mat = Material::new();
    mat.base_color_texture = Some(r);
    let matid = s.add_material(mat);
    let mut p = one_triangle_primitive();
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]; n_uv_sets];
    p.material = Some(matid);
    let mid = s.add_mesh(Mesh::new(None).with_primitive(p));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    s
}

#[test]
fn covered_uv_set_validates_clean() {
    let s = textured_prim_scene(TextureRef::new(TextureId(0)), 1);
    assert!(s.validate().is_ok());
}

#[test]
fn uv_set_beyond_primitive_channels_reported() {
    let s = textured_prim_scene(TextureRef::new(TextureId(0)).with_uv_set(1), 1);
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    let ValidationError::UvSetOutOfRange {
        location,
        uv_set,
        available,
    } = &errs[0]
    else {
        panic!("wrong variant: {:?}", errs[0]);
    };
    assert_eq!(*uv_set, 1);
    assert_eq!(*available, 1);
    assert!(
        location.contains("meshes[0].primitives[0]")
            && location.contains("materials[0].base_color_texture"),
        "location was {location}"
    );
}

#[test]
fn texture_on_uvless_primitive_reported() {
    let s = textured_prim_scene(TextureRef::new(TextureId(0)), 0);
    let errs = s.validate().unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::UvSetOutOfRange { available: 0, .. })));
}

#[test]
fn transform_uv_set_override_is_the_checked_value() {
    // Base uv_set 0 would be fine, but the KHR_texture_transform
    // texCoord override re-targets set 3 — which doesn't exist.
    let r = TextureRef::new(TextureId(0)).with_transform(TextureTransform::new().with_uv_set(3));
    let s = textured_prim_scene(r, 1);
    let errs = s.validate().unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::UvSetOutOfRange { uv_set: 3, .. })));

    // And the converse: an out-of-range base set redeemed by an
    // in-range override validates clean.
    let r = TextureRef::new(TextureId(0))
        .with_uv_set(7)
        .with_transform(TextureTransform::new().with_uv_set(0));
    let s = textured_prim_scene(r, 1);
    assert!(s.validate().is_ok());
}

#[test]
fn unused_material_uv_set_not_checked() {
    // The material is in the arena but no primitive applies it — the
    // coverage rule is per applied primitive, so nothing fires.
    let mut s = Scene3D::new();
    s.add_texture(oxideav_mesh3d::Texture::from_uri("t.png"));
    let mut mat = Material::new();
    mat.base_color_texture = Some(TextureRef::new(TextureId(0)).with_uv_set(9));
    s.add_material(mat);
    assert!(s.validate().is_ok());
}

#[test]
fn variant_mapping_material_uv_set_checked() {
    let mut s = Scene3D::new();
    s.add_texture(oxideav_mesh3d::Texture::from_uri("t.png"));
    let base = s.add_material(Material::new());
    let mut alt = Material::new();
    alt.emissive_texture = Some(TextureRef::new(TextureId(0)).with_uv_set(2));
    let altid = s.add_material(alt);
    let red = s.add_material_variant("Red");
    let mut p = one_triangle_primitive();
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]];
    p.material = Some(base);
    p.variant_mappings = vec![oxideav_mesh3d::VariantMapping {
        material: altid,
        variants: vec![red],
    }];
    let mid = s.add_mesh(Mesh::new(None).with_primitive(p));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    let ValidationError::UvSetOutOfRange { location, .. } = &errs[0] else {
        panic!("wrong variant: {:?}", errs[0]);
    };
    assert!(
        location.contains("variant_mappings[0]") && location.contains("materials[1]"),
        "location was {location}"
    );
}

#[test]
fn shared_base_and_mapping_material_reports_once() {
    // The same out-of-coverage material both as base and as a
    // mapping override — one report, not two.
    let mut s = Scene3D::new();
    s.add_texture(oxideav_mesh3d::Texture::from_uri("t.png"));
    let mut mat = Material::new();
    mat.base_color_texture = Some(TextureRef::new(TextureId(0)).with_uv_set(1));
    let matid = s.add_material(mat);
    let red = s.add_material_variant("Red");
    let mut p = one_triangle_primitive();
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]];
    p.material = Some(matid);
    p.variant_mappings = vec![oxideav_mesh3d::VariantMapping {
        material: matid,
        variants: vec![red],
    }];
    s.add_mesh(Mesh::new(None).with_primitive(p));
    let errs = s.validate().unwrap_err();
    let uv_errs = errs
        .iter()
        .filter(|e| matches!(e, ValidationError::UvSetOutOfRange { .. }))
        .count();
    assert_eq!(uv_errs, 1);
}

#[test]
fn dangling_applied_material_does_not_double_report() {
    // A primitive applying a dangling material id: the DanglingId
    // report stands alone — the uv-set rule skips unresolvable ids.
    let mut p = one_triangle_primitive();
    p.material = Some(oxideav_mesh3d::MaterialId(4));
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None).with_primitive(p));
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    assert!(matches!(
        &errs[0],
        ValidationError::DanglingId {
            arena: "materials",
            ..
        }
    ));
}

#[test]
fn non_finite_texture_transform_reported() {
    let r = TextureRef::new(TextureId(0))
        .with_transform(TextureTransform::new().with_rotation(f32::NAN));
    let s = textured_prim_scene(r, 1);
    let errs = s.validate().unwrap_err();
    assert_eq!(errs.len(), 1);
    let ValidationError::TextureTransformNotFinite { location } = &errs[0] else {
        panic!("wrong variant: {:?}", errs[0]);
    };
    assert!(
        location.contains("materials[0].base_color_texture.transform"),
        "location was {location}"
    );
}

#[test]
fn finite_texture_transform_validates_clean() {
    let r = TextureRef::new(TextureId(0)).with_transform(
        TextureTransform::new()
            .with_offset([0.5, 0.5])
            .with_rotation(1.0)
            .with_scale([2.0, -1.0]),
    );
    let s = textured_prim_scene(r, 1);
    assert!(s.validate().is_ok());
}
