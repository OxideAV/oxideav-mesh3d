//! Tests for the typed morph-target names — [`Mesh::target_names`],
//! its builder/accessors, the [`Scene3D::validate`] length rule, and
//! the lifecycle interplay (append carry, morph/skin consumption,
//! merge/simplify preservation).
//!
//! Truth ladder: glTF 2.0 §3.7.2.2 implementation note (the de-facto
//! `mesh.extras.targetNames` convention — "The `targetNames` array and
//! all primitive `targets` arrays must have the same length"). Spec
//! mirrored at `docs/3d/gltf/gltf-2.0-spec.html`.

use oxideav_mesh3d::{Mesh, MorphTarget, Node, Primitive, Scene3D, Topology, ValidationError};

/// One-triangle primitive with `n` position morph targets.
fn morph_primitive(n: usize) -> Primitive {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    prim.targets = (0..n)
        .map(|i| {
            let mut t = MorphTarget::new();
            t.position = Some(vec![[i as f32 + 1.0, 0.0, 0.0]; 3]);
            t
        })
        .collect();
    prim
}

fn scene_with(mesh: Mesh) -> Scene3D {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(mesh);
    let node = scene.add_node(Node::new().with_mesh(mid));
    scene.roots.push(node);
    scene
}

// ---------------------------------------------------------------- //
// Builder + accessors                                              //
// ---------------------------------------------------------------- //

#[test]
fn builder_and_accessors() {
    let mesh = Mesh::new(Some("face".to_owned()))
        .with_primitive(morph_primitive(2))
        .with_target_names(["smile", "blink"]);
    assert_eq!(mesh.target_names, vec!["smile", "blink"]);
    assert_eq!(mesh.target_name(0), Some("smile"));
    assert_eq!(mesh.target_name(1), Some("blink"));
    assert_eq!(mesh.target_name(2), None);
    assert_eq!(mesh.find_target("blink"), Some(1));
    assert_eq!(mesh.find_target("frown"), None);
}

#[test]
fn unnamed_targets_have_no_names() {
    let mesh = Mesh::new(None).with_primitive(morph_primitive(2));
    assert!(mesh.target_names.is_empty());
    assert_eq!(mesh.target_name(0), None);
    assert_eq!(mesh.find_target("smile"), None);
}

#[test]
fn find_target_first_match_wins_on_duplicates() {
    let mesh = Mesh::new(None)
        .with_primitive(morph_primitive(3))
        .with_target_names(["a", "dup", "dup"]);
    assert_eq!(mesh.find_target("dup"), Some(1));
}

// ---------------------------------------------------------------- //
// validate()                                                       //
// ---------------------------------------------------------------- //

#[test]
fn validate_accepts_matching_and_empty_names() {
    // Names matching the target count.
    let scene = scene_with(
        Mesh::new(None)
            .with_primitive(morph_primitive(2))
            .with_target_names(["a", "b"]),
    );
    assert_eq!(scene.validate(), Ok(()));

    // Unnamed targets are always fine.
    let scene = scene_with(Mesh::new(None).with_primitive(morph_primitive(2)));
    assert_eq!(scene.validate(), Ok(()));

    // No targets, no names.
    let scene = scene_with(Mesh::new(None).with_primitive(morph_primitive(0)));
    assert_eq!(scene.validate(), Ok(()));
}

#[test]
fn validate_rejects_name_count_mismatch() {
    let scene = scene_with(
        Mesh::new(None)
            .with_primitive(morph_primitive(2))
            .with_target_names(["only-one"]),
    );
    let errs = scene.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::MorphTargetNameCountMismatch {
            target_names: 1,
            primitive_targets: 2,
            ..
        }
    )));
}

#[test]
fn validate_rejects_names_without_targets() {
    // Names on a morph-free mesh: 3 names vs 0 targets.
    let scene = scene_with(
        Mesh::new(None)
            .with_primitive(morph_primitive(0))
            .with_target_names(["a", "b", "c"]),
    );
    let errs = scene.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::MorphTargetNameCountMismatch {
            target_names: 3,
            primitive_targets: 0,
            ..
        }
    )));
}

#[test]
fn validate_reports_per_offending_primitive() {
    // Two primitives, only the second disagrees.
    let mesh = Mesh::new(None)
        .with_primitive(morph_primitive(2))
        .with_primitive(morph_primitive(1))
        .with_target_names(["a", "b"]);
    let scene = scene_with(mesh);
    let errs = scene.validate().unwrap_err();
    let hits: Vec<_> = errs
        .iter()
        .filter(|e| matches!(e, ValidationError::MorphTargetNameCountMismatch { .. }))
        .collect();
    assert_eq!(hits.len(), 1);
    assert!(format!("{}", hits[0]).contains("primitives[1]"));
}

// ---------------------------------------------------------------- //
// Lifecycle interplay                                              //
// ---------------------------------------------------------------- //

#[test]
fn append_carries_names_verbatim() {
    let mut dst = scene_with(
        Mesh::new(None)
            .with_primitive(morph_primitive(1))
            .with_target_names(["dst-pose"]),
    );
    let src = scene_with(
        Mesh::new(None)
            .with_primitive(morph_primitive(2))
            .with_target_names(["smile", "blink"]),
    );
    let off = dst.append(&src);
    assert_eq!(dst.meshes[0].target_names, vec!["dst-pose"]);
    let relocated = &dst.meshes[off.meshes as usize];
    assert_eq!(relocated.target_names, vec!["smile", "blink"]);
    assert_eq!(dst.validate(), Ok(()));
}

#[test]
fn morphed_clears_consumed_names() {
    let mesh = Mesh::new(Some("face".to_owned()))
        .with_primitive(morph_primitive(2))
        .with_weights([0.5, 0.5])
        .with_target_names(["a", "b"]);
    let flat = mesh.morphed(&mesh.weights);
    assert!(flat.primitives[0].targets.is_empty());
    assert!(flat.weights.is_empty());
    assert!(flat.target_names.is_empty(), "names name consumed slots");
    assert_eq!(flat.name.as_deref(), Some("face"));
    // The flattened mesh still validates.
    assert_eq!(scene_with(flat).validate(), Ok(()));
}

#[test]
fn skinned_clears_consumed_names() {
    let mesh = Mesh::new(None)
        .with_primitive(morph_primitive(1))
        .with_target_names(["pose"]);
    let identity = [[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]];
    let baked = mesh.skinned(&identity);
    assert!(baked.target_names.is_empty());
}

#[test]
fn world_mesh_output_carries_no_names() {
    let scene = scene_with(
        Mesh::new(None)
            .with_primitive(morph_primitive(1))
            .with_weights([1.0])
            .with_target_names(["pose"]),
    );
    let baked = scene.world_mesh(scene.roots[0]).expect("instantiable");
    assert!(baked.primitives[0].targets.is_empty());
    assert!(baked.target_names.is_empty());
    assert_eq!(scene_with(baked).validate(), Ok(()));
}

#[test]
fn simplify_rollups_preserve_names() {
    let mesh = Mesh::new(None)
        .with_primitive(morph_primitive(2))
        .with_target_names(["a", "b"]);
    assert_eq!(
        mesh.simplify_quadric(100).target_names,
        vec!["a", "b"],
        "targets survive quadric collapse, so names stay aligned"
    );
    assert_eq!(
        mesh.simplify_quadric_error(0.0).target_names,
        vec!["a", "b"]
    );
}

#[test]
fn merge_by_material_preserves_mesh_level_fields() {
    let mesh = Mesh::new(Some("m".to_owned()))
        .with_primitive(morph_primitive(1))
        .with_target_names(["pose"]);
    let merged = mesh.merge_primitives_by_material();
    // Same contract as `weights`: mesh-level morph fields are
    // preserved; the caller consuming morph state clears them.
    assert_eq!(merged.target_names, vec!["pose"]);
    assert_eq!(merged.name.as_deref(), Some("m"));
}
