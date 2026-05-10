//! Scene3D::add_* helpers, ID issuance, lookups, and skinning fixture
//! invariants (4-joint single-vertex weights sum to 1.0).

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, Axis, Camera, Interpolation, Light, Material, Mesh, Mesh3DRegistry, Node,
    NodeId, Primitive, Scene3D, Skeleton, Skin, Texture, Topology, Transform, Unit,
};

#[test]
fn fresh_scene_uses_gltf_defaults() {
    let s = Scene3D::new();
    assert_eq!(s.up_axis, Axis::PosY);
    assert_eq!(s.front_axis, Axis::NegZ);
    assert_eq!(s.unit, Unit::Metres);
    assert!(s.nodes.is_empty());
    assert!(s.roots.is_empty());
}

#[test]
fn unit_to_metres_table() {
    assert!((Unit::Metres.to_metres() - 1.0).abs() < f32::EPSILON);
    assert!((Unit::Centimetres.to_metres() - 0.01).abs() < f32::EPSILON);
    assert!((Unit::Millimetres.to_metres() - 0.001).abs() < f32::EPSILON);
    assert!((Unit::Inches.to_metres() - 0.0254).abs() < f32::EPSILON);
    assert!((Unit::Feet.to_metres() - 0.3048).abs() < 1e-7);
    assert!((Unit::Yards.to_metres() - 0.9144).abs() < 1e-7);
}

#[test]
fn add_node_issues_sequential_ids() {
    let mut s = Scene3D::new();
    let a = s.add_node(Node::new().with_name("a"));
    let b = s.add_node(Node::new().with_name("b"));
    let c = s.add_node(Node::new().with_name("c"));
    assert_eq!(a, NodeId(0));
    assert_eq!(b, NodeId(1));
    assert_eq!(c, NodeId(2));
    assert_eq!(s.node(a).unwrap().name.as_deref(), Some("a"));
    assert_eq!(s.node(c).unwrap().name.as_deref(), Some("c"));
}

#[test]
fn lookup_out_of_range_returns_none() {
    let s = Scene3D::new();
    assert!(s.node(NodeId(0)).is_none());
    assert!(s.mesh(oxideav_mesh3d::MeshId(99)).is_none());
}

#[test]
fn skinning_fixture_weights_sum_to_one() {
    // Four-joint single-vertex fixture — verify the weight-sum
    // invariant the GPU expects.
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![[0.0, 0.0, 0.0]];
    prim.joints = Some(vec![[0, 1, 2, 3]]);
    prim.weights = Some(vec![[0.4, 0.3, 0.2, 0.1]]);
    let weights = prim.weights.as_ref().unwrap();
    assert_eq!(weights.len(), 1);
    let sum: f32 = weights[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "weights sum = {sum}");
}

#[test]
fn skin_skeleton_and_animation_added() {
    let mut s = Scene3D::new();

    // 3-joint skeleton.
    let j0 = s.add_node(Node::new().with_name("root_bone"));
    let j1 = s.add_node(Node::new().with_name("mid_bone"));
    let j2 = s.add_node(Node::new().with_name("tip_bone"));
    let identity4 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut sk = Skeleton::new();
    sk.joints = vec![j0, j1, j2];
    sk.inverse_bind_matrices = vec![identity4; 3];
    let sk_id = s.add_skeleton(sk);

    let skin = Skin::new(sk_id).with_root(j0);
    let skin_id = s.add_skin(skin);
    assert_eq!(s.skins[skin_id.0 as usize].root_node, Some(j0));

    // Translation animation on j1.
    let mut anim = Animation::new(Some("wave".to_string()));
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: j1,
            property: AnimationProperty::Translation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 0.5, 1.0],
            values: AnimationValues::Vec3(vec![[0.; 3], [0.5, 0., 0.], [0.; 3]]),
            interpolation: Interpolation::Linear,
        },
    });
    let anim_idx = s.add_animation(anim);
    assert_eq!(anim_idx, 0);
    assert_eq!(s.animations.len(), 1);
}

#[test]
fn full_scene_aggregate_roundtrip() {
    // Build one of every kind, add as roots, verify the scene
    // counts/lookups stay consistent.
    let mut s = Scene3D::new();

    let mat = s.add_material(Material::new().with_name("m"));
    let _tex = s.add_texture(Texture::from_uri("foo.png"));
    let cam = s.add_camera(Camera::perspective(1.0, 0.1));
    let lit = s.add_light(Light::Directional {
        color: [1.; 3],
        intensity: 10.0,
    });

    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.; 3], [1., 0., 0.], [0., 1., 0.]];
    prim.material = Some(mat);
    let mesh = Mesh::new(Some("tri".to_string())).with_primitive(prim);
    let mesh_id = s.add_mesh(mesh);

    let n_mesh = s.add_node(Node::new().with_mesh(mesh_id));
    let n_cam = s.add_node(Node {
        camera: Some(cam),
        ..Node::new()
    });
    let n_lit = s.add_node(Node {
        light: Some(lit),
        ..Node::new()
    });
    s.add_root(n_mesh);
    s.add_root(n_cam);
    s.add_root(n_lit);

    assert_eq!(s.materials.len(), 1);
    assert_eq!(s.textures.len(), 1);
    assert_eq!(s.cameras.len(), 1);
    assert_eq!(s.lights.len(), 1);
    assert_eq!(s.meshes.len(), 1);
    assert_eq!(s.nodes.len(), 3);
    assert_eq!(s.roots.len(), 3);
    assert_eq!(s.triangle_count(), 1);
    assert_eq!(s.vertex_count(), 3);
    assert_eq!(s.mesh(mesh_id).unwrap().primitives[0].material, Some(mat));
    assert_eq!(s.node(n_mesh).unwrap().mesh, Some(mesh_id));
    let _: Transform = s.node(n_cam).unwrap().transform;
}

#[test]
fn registry_dispatches_by_extension_and_format() {
    use oxideav_mesh3d::{Mesh3DDecoder, Mesh3DEncoder, Result};

    struct DummyDec;
    impl Mesh3DDecoder for DummyDec {
        fn decode(&mut self, _: &[u8]) -> Result<Scene3D> {
            Ok(Scene3D::new())
        }
    }
    struct DummyEnc;
    impl Mesh3DEncoder for DummyEnc {
        fn encode(&mut self, _: &Scene3D) -> Result<Vec<u8>> {
            Ok(vec![1, 2, 3])
        }
    }

    let mut reg = Mesh3DRegistry::new();
    reg.register_decoder("stl", &["stl"], Box::new(|| Box::new(DummyDec)));
    reg.register_encoder("stl", &["stl"], Box::new(|| Box::new(DummyEnc)));
    reg.register_decoder("gltf", &["gltf", "glb"], Box::new(|| Box::new(DummyDec)));

    // Case-insensitive ext lookup.
    let mut d = reg.decoder_for_extension("STL").expect("decoder by ext");
    let scene = d.decode(b"").unwrap();
    assert_eq!(scene.nodes.len(), 0);

    // Format-id route works too.
    assert!(reg.decoder_for_format("GLTF").is_some());
    assert!(reg.decoder_for_extension("glb").is_some());

    // Unknown returns None.
    assert!(reg.decoder_for_extension("xyz").is_none());

    // Encoder round-trip.
    let mut e = reg.encoder_for_extension("stl").expect("encoder by ext");
    let bytes = e.encode(&Scene3D::new()).unwrap();
    assert_eq!(bytes, vec![1, 2, 3]);

    // Extensions reverse-lookup.
    assert_eq!(
        reg.decoder_extensions("gltf"),
        Some(["gltf".to_string(), "glb".to_string()].as_slice())
    );

    // Format listings include both.
    let mut decs: Vec<&str> = reg.decoder_formats().collect();
    decs.sort();
    assert_eq!(decs, vec!["gltf", "stl"]);
}
