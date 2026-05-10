//! Type-model coverage tests:
//! * `extras` HashMap round-trips through serde_json
//! * Builder helpers populate a 1-triangle scene + verify counts
//! * AlphaMode default + Mask{cutoff}
//! * Camera::Perspective default znear/zfar handling
//! * Animation values type matches property
//! * Indices length helpers

use oxideav_mesh3d::{
    AlphaMode, Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, Camera, Indices, Interpolation, Light, MagFilter, Material, Mesh, MinFilter,
    NodeId, Primitive, Sampler, Scene3D, Texture, Topology, WrapMode,
};
use serde_json::json;

#[test]
fn extras_roundtrip_serde_json() {
    let mut mat = Material::new();
    mat.extras
        .insert("fbx_lambert_diffuse".into(), json!([0.8, 0.4, 0.2]));
    mat.extras.insert("usd_kind".into(), json!("subcomponent"));
    mat.extras.insert("any_int".into(), json!(42));

    let blob = serde_json::to_string(&mat.extras).expect("ser");
    let parsed: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(&blob).expect("de");
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed["any_int"].as_i64(), Some(42));
    assert_eq!(parsed["usd_kind"].as_str(), Some("subcomponent"));
    assert_eq!(parsed["fbx_lambert_diffuse"][2].as_f64(), Some(0.2));
}

#[test]
fn builder_one_triangle_scene_counts_match() {
    let mut scene = Scene3D::new();

    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    let mesh = Mesh::new(Some("triangle".to_string())).with_primitive(prim);

    let mid = scene.add_mesh(mesh);
    let nid = scene.add_node(oxideav_mesh3d::Node::new().with_mesh(mid));
    scene.add_root(nid);

    assert_eq!(scene.triangle_count(), 1);
    assert_eq!(scene.vertex_count(), 3);
    assert_eq!(scene.roots, vec![nid]);
    assert_eq!(scene.node(nid).unwrap().mesh, Some(mid));
}

#[test]
fn triangle_count_handles_indexed_and_strip() {
    // Indexed triangle list — 6 indices = 2 triangles even though
    // there are 4 unique vertex positions.
    let mut prim_indexed = Primitive::new(Topology::Triangles);
    prim_indexed.positions = vec![[0.; 3]; 4];
    prim_indexed.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
    assert_eq!(prim_indexed.triangle_count(), 2);

    // Triangle strip — 5 vertices makes 3 triangles.
    let mut prim_strip = Primitive::new(Topology::TriangleStrip);
    prim_strip.positions = vec![[0.; 3]; 5];
    assert_eq!(prim_strip.triangle_count(), 3);

    // Lines / points contribute 0 triangles.
    let mut prim_lines = Primitive::new(Topology::Lines);
    prim_lines.positions = vec![[0.; 3]; 8];
    assert_eq!(prim_lines.triangle_count(), 0);
}

#[test]
fn alpha_mode_default_is_opaque() {
    let mat = Material::new();
    assert_eq!(mat.alpha_mode, AlphaMode::Opaque);
    assert!(matches!(AlphaMode::default(), AlphaMode::Opaque));
}

#[test]
fn alpha_mode_mask_cutoff() {
    let mut mat = Material::new();
    mat.alpha_mode = AlphaMode::Mask { cutoff: 0.5 };
    match mat.alpha_mode {
        AlphaMode::Mask { cutoff } => assert!((cutoff - 0.5).abs() < f32::EPSILON),
        _ => panic!("expected Mask"),
    }
}

#[test]
fn perspective_camera_defaults_have_infinite_far() {
    let c = Camera::perspective(std::f32::consts::FRAC_PI_3, 0.1);
    match c {
        Camera::Perspective {
            aspect_ratio,
            yfov,
            znear,
            zfar,
        } => {
            assert!(aspect_ratio.is_none(), "aspect None means use framebuffer");
            assert!(zfar.is_none(), "default zfar is None (infinite far plane)");
            assert!((znear - 0.1).abs() < f32::EPSILON);
            assert!((yfov - std::f32::consts::FRAC_PI_3).abs() < f32::EPSILON);
        }
        _ => panic!("expected Perspective"),
    }
}

#[test]
fn orthographic_camera_carries_explicit_planes() {
    let c = Camera::orthographic(2.0, 1.5, 0.1, 100.0);
    match c {
        Camera::Orthographic {
            xmag,
            ymag,
            znear,
            zfar,
        } => {
            assert!((xmag - 2.0).abs() < f32::EPSILON);
            assert!((ymag - 1.5).abs() < f32::EPSILON);
            assert!((znear - 0.1).abs() < f32::EPSILON);
            assert!((zfar - 100.0).abs() < f32::EPSILON);
        }
        _ => panic!("expected Orthographic"),
    }
}

#[test]
fn animation_value_kinds_pair_with_property() {
    let chan = AnimationChannel {
        target: AnimationTarget {
            node: NodeId(0),
            property: AnimationProperty::Translation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Vec3(vec![[0.0; 3], [1.0; 3]]),
            interpolation: Interpolation::Linear,
        },
    };
    let mut anim = Animation::new(Some("idle".to_string()));
    anim.channels.push(chan);
    assert_eq!(anim.channels.len(), 1);
    assert_eq!(anim.channels[0].sampler.values.len(), 2);
    assert!(!anim.channels[0].sampler.values.is_empty());
}

#[test]
fn indices_helpers_match_underlying_len() {
    let i16 = Indices::U16(vec![0, 1, 2, 3]);
    let i32 = Indices::U32(vec![100_000, 200_000]);
    assert_eq!(i16.len(), 4);
    assert_eq!(i32.len(), 2);
    assert!(!i16.is_empty());
    assert!(!Indices::U32(vec![1]).is_empty());
    assert!(Indices::U16(vec![]).is_empty());
}

#[test]
fn light_variants_round_trip_via_clone() {
    let lights = [
        Light::Directional {
            color: [1.0, 0.95, 0.9],
            intensity: 50_000.0,
        },
        Light::Point {
            color: [1.0, 1.0, 1.0],
            intensity: 800.0,
            range: Some(20.0),
        },
        Light::Spot {
            color: [0.5, 0.5, 1.0],
            intensity: 1200.0,
            range: None,
            inner_cone_angle: 0.1,
            outer_cone_angle: 0.5,
        },
    ];
    for l in lights {
        let cloned = l;
        // Pattern-match enforces the variant matches.
        match (l, cloned) {
            (Light::Directional { .. }, Light::Directional { .. })
            | (Light::Point { .. }, Light::Point { .. })
            | (Light::Spot { .. }, Light::Spot { .. }) => (),
            _ => panic!("clone changed variant"),
        }
    }
}

#[test]
fn texture_constructors_default_to_gltf_sampler() {
    let t = Texture::from_uri("file://foo.png");
    assert_eq!(t.sampler.mag_filter, MagFilter::Linear);
    assert_eq!(t.sampler.min_filter, MinFilter::LinearMipLinear);
    assert_eq!(t.sampler.wrap_s, WrapMode::Repeat);
    assert_eq!(t.sampler.wrap_t, WrapMode::Repeat);

    let s = Sampler::default_sampler();
    assert_eq!(s.mag_filter, MagFilter::Linear);
}
