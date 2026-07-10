//! Coverage for the `KHR_materials_variants` model: the scene-level
//! variant-name roster (`Scene3D::material_variants`), per-primitive
//! `VariantMapping`s, validation rules (live ids + at-most-once
//! variant use), name-unified `append`, and the draw-state partition
//! in `merge_primitives_by_material`.

use oxideav_mesh3d::{
    Material, MaterialId, MaterialVariantId, Mesh, Node, Primitive, Scene3D, Topology,
    ValidationError, VariantMapping,
};

fn one_triangle_primitive() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p
}

#[test]
fn scene_and_primitive_default_to_no_variants() {
    assert!(Scene3D::new().material_variants.is_empty());
    assert!(Primitive::new(Topology::Triangles)
        .variant_mappings
        .is_empty());
}

#[test]
fn add_and_find_or_add_material_variant() {
    let mut s = Scene3D::new();
    let red = s.add_material_variant("Red");
    let blue = s.add_material_variant("Blue");
    assert_eq!(red, MaterialVariantId(0));
    assert_eq!(blue, MaterialVariantId(1));
    assert_eq!(s.material_variants, vec!["Red", "Blue"]);
    // find_or_add reuses by exact name, appends otherwise.
    assert_eq!(s.find_or_add_material_variant("Blue"), blue);
    let green = s.find_or_add_material_variant("Green");
    assert_eq!(green, MaterialVariantId(2));
    assert_eq!(s.material_variants.len(), 3);
    // add_material_variant never dedups.
    let red2 = s.add_material_variant("Red");
    assert_eq!(red2, MaterialVariantId(3));
}

/// A one-primitive scene with two variants and one mapping, fully
/// live — must validate clean.
fn variant_scene() -> Scene3D {
    let mut s = Scene3D::new();
    let base = s.add_material(Material::new().with_name("base"));
    let alt = s.add_material(Material::new().with_name("alt"));
    let red = s.add_material_variant("Red");
    let blue = s.add_material_variant("Blue");
    let mut p = one_triangle_primitive();
    p.material = Some(base);
    p.variant_mappings = vec![VariantMapping {
        material: alt,
        variants: vec![red, blue],
    }];
    let mid = s.add_mesh(Mesh::new(None).with_primitive(p));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    s
}

#[test]
fn live_variant_mappings_validate_clean() {
    assert!(variant_scene().validate().is_ok());
}

#[test]
fn dangling_variant_mapping_material_reported() {
    let mut s = variant_scene();
    s.meshes[0].primitives[0].variant_mappings[0].material = MaterialId(99);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId { id: 99, arena: "materials", location }
            if location == "meshes[0].primitives[0].variant_mappings[0].material"
    )));
}

#[test]
fn dangling_variant_id_reported() {
    let mut s = variant_scene();
    s.meshes[0].primitives[0].variant_mappings[0].variants[1] = MaterialVariantId(7);
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DanglingId { id: 7, arena: "material_variants", location }
            if location == "meshes[0].primitives[0].variant_mappings[0].variants[1]"
    )));
}

#[test]
fn duplicate_variant_across_mappings_reported() {
    let mut s = variant_scene();
    // Claim "Red" (variant 0) from a second mapping too.
    let alt2 = s.add_material(Material::new());
    s.meshes[0].primitives[0]
        .variant_mappings
        .push(VariantMapping {
            material: alt2,
            variants: vec![MaterialVariantId(0)],
        });
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DuplicateVariantMapping { variant: 0, location }
            if location == "meshes[0].primitives[0].variant_mappings[1].variants[0]"
    )));
}

#[test]
fn duplicate_variant_within_one_mapping_reported() {
    let mut s = variant_scene();
    let m = &mut s.meshes[0].primitives[0].variant_mappings[0];
    m.variants = vec![MaterialVariantId(1), MaterialVariantId(1)];
    let errs = s.validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::DuplicateVariantMapping { variant: 1, .. }
    )));
}

#[test]
fn append_unifies_variant_rosters_by_name() {
    let mut dst = variant_scene(); // variants: Red(0), Blue(1)
                                   // Source scene declares Blue + Green — Blue must unify with dst's,
                                   // Green must join the roster.
    let mut src = Scene3D::new();
    let alt = src.add_material(Material::new().with_name("src-alt"));
    let blue = src.add_material_variant("Blue");
    let green = src.add_material_variant("Green");
    let mut p = one_triangle_primitive();
    p.variant_mappings = vec![VariantMapping {
        material: alt,
        variants: vec![blue, green],
    }];
    let mid = src.add_mesh(Mesh::new(None).with_primitive(p));
    let nid = src.add_node(Node::new().with_mesh(mid));
    src.add_root(nid);

    let off = dst.append(&src);
    // Roster: Red, Blue, Green — no duplicate Blue.
    assert_eq!(dst.material_variants, vec!["Red", "Blue", "Green"]);
    // The relocated mapping's material moved by the materials offset;
    // its variant ids were rewritten through the name map.
    let moved = &dst.meshes[1].primitives[0].variant_mappings[0];
    assert_eq!(moved.material, MaterialId(alt.0 + off.materials));
    assert_eq!(
        moved.variants,
        vec![MaterialVariantId(1), MaterialVariantId(2)]
    );
    // The merged scene still validates.
    assert!(dst.validate().is_ok());
}

#[test]
fn merge_by_material_keeps_variant_divergent_primitives_apart() {
    let mut s = Scene3D::new();
    let base = s.add_material(Material::new());
    let alt = s.add_material(Material::new());
    let red = s.add_material_variant("Red");

    let mut a = one_triangle_primitive();
    a.material = Some(base);
    let mut b = one_triangle_primitive();
    b.material = Some(base);
    b.variant_mappings = vec![VariantMapping {
        material: alt,
        variants: vec![red],
    }];
    // Same base material as `b` AND the same mappings: fuses with `b`.
    let mut c = one_triangle_primitive();
    c.material = Some(base);
    c.variant_mappings = b.variant_mappings.clone();

    let mesh = Mesh::new(None)
        .with_primitive(a)
        .with_primitive(b)
        .with_primitive(c);
    let merged = mesh.merge_primitives_by_material();
    // a alone; b+c fused (same draw state).
    assert_eq!(merged.primitives.len(), 2);
    assert!(merged.primitives[0].variant_mappings.is_empty());
    assert_eq!(merged.primitives[1].positions.len(), 6);
    assert_eq!(
        merged.primitives[1].variant_mappings,
        vec![VariantMapping {
            material: alt,
            variants: vec![red],
        }]
    );
}

#[test]
fn material_for_variant_activation_rule() {
    let base = MaterialId(0);
    let alt_a = MaterialId(1);
    let alt_b = MaterialId(2);
    let red = MaterialVariantId(0);
    let blue = MaterialVariantId(1);
    let green = MaterialVariantId(2);

    let mut p = one_triangle_primitive();
    p.material = Some(base);
    p.variant_mappings = vec![
        VariantMapping {
            material: alt_a,
            variants: vec![red, green],
        },
        VariantMapping {
            material: alt_b,
            variants: vec![blue],
        },
    ];

    // Active variant claimed by a mapping: that mapping's material.
    assert_eq!(p.material_for_variant(Some(red)), Some(alt_a));
    assert_eq!(p.material_for_variant(Some(green)), Some(alt_a));
    assert_eq!(p.material_for_variant(Some(blue)), Some(alt_b));
    // No active variant, or an unclaimed one: base material.
    assert_eq!(p.material_for_variant(None), Some(base));
    assert_eq!(
        p.material_for_variant(Some(MaterialVariantId(9))),
        Some(base)
    );

    // Unmaterialled primitive with no mappings: stays None.
    let bare = one_triangle_primitive();
    assert_eq!(bare.material_for_variant(Some(red)), None);
    assert_eq!(bare.material_for_variant(None), None);
}

#[test]
fn material_for_variant_first_mapping_wins_on_duplicates() {
    // Out-of-spec: two mappings both claim variant 0. Resolution must
    // be deterministic — first listing wins.
    let mut p = one_triangle_primitive();
    p.variant_mappings = vec![
        VariantMapping {
            material: MaterialId(1),
            variants: vec![MaterialVariantId(0)],
        },
        VariantMapping {
            material: MaterialId(2),
            variants: vec![MaterialVariantId(0)],
        },
    ];
    assert_eq!(
        p.material_for_variant(Some(MaterialVariantId(0))),
        Some(MaterialId(1))
    );
}

#[test]
fn transformed_preserves_variant_mappings() {
    let mut p = one_triangle_primitive();
    p.material = Some(MaterialId(0));
    p.variant_mappings = vec![VariantMapping {
        material: MaterialId(1),
        variants: vec![MaterialVariantId(0)],
    }];
    let scale2 = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let t = p.transformed(scale2);
    assert_eq!(t.variant_mappings, p.variant_mappings);
    // weld/dedup paths carry them too.
    let w = p.weld_vertices();
    assert_eq!(w.variant_mappings, p.variant_mappings);
}
