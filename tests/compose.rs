//! Scene3D::append coverage: cross-arena id remapping, root merge, and
//! validity of the composed scene.

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AudioEmitter, AudioSource, Indices, Interpolation, Material, Mesh, Node, Primitive, Scene3D,
    Skeleton, Skin, Texture, TextureRef, Topology,
};

/// A minimal scene with one textured material, one mesh referencing it,
/// and one root node carrying that mesh. The `x` offset places the
/// geometry so two appended copies are spatially distinct.
fn textured_scene(x: f32) -> Scene3D {
    let mut s = Scene3D::new();
    let tex = s.add_texture(Texture::from_uri("t.png"));
    let mut mat = Material::new();
    mat.base_color_texture = Some(TextureRef::new(tex));
    let matid = s.add_material(mat);

    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]];
    // The textured material samples UV set 0, so the primitive must
    // carry that channel (validate() enforces the coverage rule).
    prim.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]];
    prim.indices = Some(Indices::U32(vec![0, 1, 2]));
    prim.material = Some(matid);
    let mesh = Mesh::new(Some("m".to_owned())).with_primitive(prim);
    let meshid = s.add_mesh(mesh);

    let node = Node::new().with_mesh(meshid);
    let nid = s.add_node(node);
    s.add_root(nid);
    s
}

#[test]
fn append_offsets_are_pre_append_lengths() {
    let mut dst = textured_scene(0.0);
    let src = textured_scene(5.0);
    let off = dst.append(&src);
    // dst had 1 of each before the append.
    assert_eq!(off.nodes, 1);
    assert_eq!(off.meshes, 1);
    assert_eq!(off.materials, 1);
    assert_eq!(off.textures, 1);
}

#[test]
fn arenas_grow_by_the_source_size() {
    let mut dst = textured_scene(0.0);
    let src = textured_scene(5.0);
    dst.append(&src);
    assert_eq!(dst.nodes.len(), 2);
    assert_eq!(dst.meshes.len(), 2);
    assert_eq!(dst.materials.len(), 2);
    assert_eq!(dst.textures.len(), 2);
    assert_eq!(dst.roots.len(), 2);
}

#[test]
fn relocated_node_references_relocated_mesh() {
    let mut dst = textured_scene(0.0);
    let src = textured_scene(5.0);
    let off = dst.append(&src);
    // The source node landed at index off.nodes; its mesh ref must point
    // at the relocated mesh (off.meshes + 0), not the destination's.
    let relocated = &dst.nodes[off.nodes as usize];
    assert_eq!(relocated.mesh.unwrap().0, off.meshes);
}

#[test]
fn relocated_mesh_references_relocated_material() {
    let mut dst = textured_scene(0.0);
    let src = textured_scene(5.0);
    let off = dst.append(&src);
    let mesh = &dst.meshes[off.meshes as usize];
    assert_eq!(mesh.primitives[0].material.unwrap().0, off.materials);
}

#[test]
fn relocated_material_references_relocated_texture() {
    let mut dst = textured_scene(0.0);
    let src = textured_scene(5.0);
    let off = dst.append(&src);
    let mat = &dst.materials[off.materials as usize];
    let tr = mat.base_color_texture.expect("base color texture");
    assert_eq!(tr.texture.0, off.textures);
}

#[test]
fn composed_scene_validates() {
    let mut dst = textured_scene(0.0);
    let src = textured_scene(5.0);
    dst.append(&src);
    // Every remapped reference is in range → the scene validates.
    assert!(dst.validate().is_ok(), "composed scene failed validation");
}

#[test]
fn composed_scene_bounds_union_both() {
    let mut dst = textured_scene(0.0);
    let src = textured_scene(5.0);
    dst.append(&src);
    // The scene-wide AABB now spans both triangles (x in [0, 6]).
    let bb = dst.bounding_box().expect("non-empty scene");
    assert!(bb.min[0] <= 0.0 + 1e-6);
    assert!(bb.max[0] >= 6.0 - 1e-6);
}

#[test]
fn append_does_not_mutate_source() {
    let mut dst = textured_scene(0.0);
    let src = textured_scene(5.0);
    let src_nodes_before = src.nodes.len();
    let src_root_before = src.roots[0];
    dst.append(&src);
    assert_eq!(src.nodes.len(), src_nodes_before);
    assert_eq!(src.roots[0], src_root_before);
    // The source node still references its own mesh id 0, unmodified.
    assert_eq!(src.nodes[0].mesh.unwrap().0, 0);
}

#[test]
fn destination_metadata_is_kept() {
    let mut dst = Scene3D::new();
    dst.unit = oxideav_mesh3d::Unit::Centimetres;
    let src = textured_scene(0.0); // metres
    dst.append(&src);
    // Destination's unit wins.
    assert_eq!(dst.unit, oxideav_mesh3d::Unit::Centimetres);
}

/// A scene exercising the id-bearing arenas that the textured scene
/// doesn't: a one-joint skeleton, a skin, an animation channel
/// targeting the joint node, and an audio source + emitter.
fn rigged_scene() -> Scene3D {
    let mut s = Scene3D::new();
    let joint = s.add_node(Node::new().with_name("joint"));
    let mut skel = Skeleton::new();
    skel.joints = vec![joint];
    skel.inverse_bind_matrices = vec![[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]];
    let skelid = s.add_skeleton(skel);
    let skinid = s.add_skin(Skin::new(skelid).with_root(joint));

    let mut anim = Animation::new(Some("wiggle".to_owned()));
    anim.channels.push(AnimationChannel {
        target: AnimationTarget {
            node: joint,
            property: AnimationProperty::Translation,
        },
        sampler: AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: oxideav_mesh3d::AnimationValues::Vec3(vec![[0.0; 3], [1.0, 0.0, 0.0]]),
            interpolation: Interpolation::Linear,
        },
    });
    s.add_animation(anim);

    let src = s.add_audio_source(AudioSource::from_uri("beep.ogg"));
    let emitter = s.add_audio_emitter(AudioEmitter::new(src));
    // Attach the skin + emitter to the joint node, and root it.
    s.nodes[joint.0 as usize].skin = Some(skinid);
    s.nodes[joint.0 as usize].audio_emitter = Some(emitter);
    s.add_root(joint);
    s
}

#[test]
fn rigging_audio_and_animation_refs_are_remapped() {
    let mut dst = textured_scene(0.0);
    let src = rigged_scene();
    let off = dst.append(&src);

    // Skeleton joint node remapped.
    let skel = &dst.skeletons[off.skeletons as usize];
    assert_eq!(skel.joints[0].0, off.nodes);

    // Skin → skeleton + root node remapped.
    let skin = &dst.skins[off.skins as usize];
    assert_eq!(skin.skeleton.0, off.skeletons);
    assert_eq!(skin.root_node.unwrap().0, off.nodes);

    // Animation channel target node remapped.
    let anim = &dst.animations[off.animations as usize];
    assert_eq!(anim.channels[0].target.node.0, off.nodes);

    // Audio emitter → source remapped.
    let em = &dst.audio_emitters[off.audio_emitters as usize];
    assert_eq!(em.source.0, off.audio_sources);

    // Node's skin + emitter back-references remapped.
    let node = &dst.nodes[off.nodes as usize];
    assert_eq!(node.skin.unwrap().0, off.skins);
    assert_eq!(node.audio_emitter.unwrap().0, off.audio_emitters);

    // The whole composed scene is internally consistent.
    assert!(dst.validate().is_ok());
}

#[test]
fn appending_empty_scene_is_a_noop_on_resources() {
    let mut dst = textured_scene(0.0);
    let before = (dst.nodes.len(), dst.meshes.len(), dst.roots.len());
    let off = dst.append(&Scene3D::new());
    assert_eq!((dst.nodes.len(), dst.meshes.len(), dst.roots.len()), before);
    // Offsets equal the current sizes.
    assert_eq!(off.nodes, before.0 as u32);
}
