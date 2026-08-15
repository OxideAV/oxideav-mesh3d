//! Tests for the typed per-texture-reference UV transform
//! ([`TextureTransform`], `KHR_texture_transform`-aligned) and its
//! threading through [`TextureRef`] / [`Material`] / scene append.

use std::f32::consts::FRAC_PI_2;

use oxideav_mesh3d::{
    Indices, Material, Mesh, Node, Primitive, Scene3D, Texture, TextureId, TextureRef,
    TextureTransform, Topology,
};

fn assert_uv_eq(a: [f32; 2], b: [f32; 2]) {
    assert!(
        (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6,
        "{a:?} != {b:?}"
    );
}

// ---------------------------------------------------------------
// Identity / defaults

#[test]
fn default_is_identity() {
    let t = TextureTransform::default();
    assert_eq!(t, TextureTransform::IDENTITY);
    assert_eq!(t, TextureTransform::new());
    assert_eq!(t.offset, [0.0, 0.0]);
    assert_eq!(t.rotation, 0.0);
    assert_eq!(t.scale, [1.0, 1.0]);
    assert_eq!(t.uv_set, None);
    assert!(t.is_identity());
    assert!(t.is_finite());
}

#[test]
fn identity_apply_is_exact_no_op() {
    let t = TextureTransform::IDENTITY;
    for uv in [[0.0, 0.0], [1.0, 1.0], [0.25, 0.75], [-3.5, 7.25]] {
        assert_eq!(t.apply(uv), uv);
    }
}

#[test]
fn non_identity_fields_break_is_identity() {
    assert!(!TextureTransform::new()
        .with_offset([0.1, 0.0])
        .is_identity());
    assert!(!TextureTransform::new().with_rotation(0.1).is_identity());
    assert!(!TextureTransform::new().with_scale([2.0, 1.0]).is_identity());
    // A texCoord-only override still changes sampling, so it is not
    // an identity either.
    assert!(!TextureTransform::new().with_uv_set(0).is_identity());
}

// ---------------------------------------------------------------
// Component semantics

#[test]
fn offset_translates() {
    let t = TextureTransform::new().with_offset([0.25, -0.5]);
    assert_uv_eq(t.apply([0.5, 0.5]), [0.75, 0.0]);
}

#[test]
fn scale_scales_about_origin() {
    let t = TextureTransform::new().with_scale([0.5, 2.0]);
    assert_uv_eq(t.apply([1.0, 1.0]), [0.5, 2.0]);
    assert_uv_eq(t.apply([0.0, 0.0]), [0.0, 0.0]);
}

#[test]
fn rotation_is_counter_clockwise_about_origin() {
    // 90° CCW takes the +U axis onto the +V axis.
    let t = TextureTransform::new().with_rotation(FRAC_PI_2);
    assert_uv_eq(t.apply([1.0, 0.0]), [0.0, 1.0]);
    assert_uv_eq(t.apply([0.0, 1.0]), [-1.0, 0.0]);
}

#[test]
fn composition_order_is_scale_then_rotate_then_offset() {
    let t = TextureTransform {
        offset: [0.3, -0.2],
        rotation: 0.7,
        scale: [2.0, 0.5],
        uv_set: None,
    };
    let uv = [0.6, -1.1];
    // Manual chain: S, then R, then T.
    let scaled = [uv[0] * 2.0, uv[1] * 0.5];
    let (s, c) = 0.7_f32.sin_cos();
    let rotated = [c * scaled[0] - s * scaled[1], s * scaled[0] + c * scaled[1]];
    let expect = [rotated[0] + 0.3, rotated[1] - 0.2];
    assert_uv_eq(t.apply(uv), expect);
}

#[test]
fn to_matrix_agrees_with_apply() {
    let t = TextureTransform {
        offset: [-0.4, 1.2],
        rotation: -1.1,
        scale: [0.25, -3.0],
        uv_set: None,
    };
    let m = t.to_matrix();
    assert_eq!(m[2], [0.0, 0.0, 1.0]);
    for uv in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.3, 0.9]] {
        let via_matrix = [
            m[0][0] * uv[0] + m[0][1] * uv[1] + m[0][2],
            m[1][0] * uv[0] + m[1][1] * uv[1] + m[1][2],
        ];
        assert_uv_eq(t.apply(uv), via_matrix);
    }
}

// ---------------------------------------------------------------
// The extension's two worked examples

#[test]
fn quadrant_atlas_example() {
    // offset [0, 1], rotation π/2, scale [0.5, 0.5] — selects the
    // lower-left quadrant of the source image, rotated.
    let t = TextureTransform {
        offset: [0.0, 1.0],
        rotation: FRAC_PI_2,
        scale: [0.5, 0.5],
        uv_set: None,
    };
    // The UV origin maps straight onto the offset.
    assert_uv_eq(t.apply([0.0, 0.0]), [0.0, 1.0]);
    // (1, 0): scaled to (0.5, 0), rotated CCW onto (0, 0.5), offset
    // to (0, 1.5).
    assert_uv_eq(t.apply([1.0, 0.0]), [0.0, 1.5]);
    // (0, 1): scaled to (0, 0.5), rotated CCW onto (-0.5, 0), offset
    // to (-0.5, 1).
    assert_uv_eq(t.apply([0.0, 1.0]), [-0.5, 1.0]);
}

#[test]
fn t_axis_inversion_example() {
    // offset [0, 1], scale [1, -1] — inverts the T axis, defining a
    // bottom-left origin: (u, v) → (u, 1 − v).
    let t = TextureTransform {
        offset: [0.0, 1.0],
        rotation: 0.0,
        scale: [1.0, -1.0],
        uv_set: None,
    };
    for uv in [[0.0, 0.0], [1.0, 1.0], [0.3, 0.8], [0.5, 0.25]] {
        assert_uv_eq(t.apply(uv), [uv[0], 1.0 - uv[1]]);
    }
}

// ---------------------------------------------------------------
// Channel baking

#[test]
fn apply_channel_maps_every_element() {
    let t = TextureTransform::new().with_offset([1.0, 2.0]);
    let channel = vec![[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]];
    let baked = t.apply_channel(&channel);
    assert_eq!(baked.len(), channel.len());
    for (src, dst) in channel.iter().zip(&baked) {
        assert_uv_eq(*dst, t.apply(*src));
    }
    // Pure: the input channel is untouched.
    assert_eq!(channel[1], [0.5, 0.5]);
}

#[test]
fn apply_channel_empty_is_empty() {
    assert!(TextureTransform::IDENTITY.apply_channel(&[]).is_empty());
}

// ---------------------------------------------------------------
// Finiteness

#[test]
fn is_finite_rejects_each_non_finite_component() {
    for t in [
        TextureTransform::new().with_offset([f32::NAN, 0.0]),
        TextureTransform::new().with_offset([0.0, f32::INFINITY]),
        TextureTransform::new().with_rotation(f32::NEG_INFINITY),
        TextureTransform::new().with_scale([f32::NAN, 1.0]),
        TextureTransform::new().with_scale([1.0, f32::NAN]),
    ] {
        assert!(!t.is_finite(), "{t:?} should not be finite");
    }
    assert!(TextureTransform::new().is_finite());
}

// ---------------------------------------------------------------
// TextureRef threading

#[test]
fn texture_ref_new_has_no_transform() {
    let r = TextureRef::new(TextureId(3));
    assert_eq!(r.uv_set, 0);
    assert_eq!(r.transform, None);
    assert_eq!(r.effective_uv_set(), 0);
}

#[test]
fn effective_uv_set_resolution_chain() {
    // No transform: the reference's own set.
    let r = TextureRef::new(TextureId(0)).with_uv_set(2);
    assert_eq!(r.effective_uv_set(), 2);
    // Transform without override: still the reference's own set.
    let r = r.with_transform(TextureTransform::new().with_offset([0.1, 0.0]));
    assert_eq!(r.effective_uv_set(), 2);
    // Transform with override: the override wins.
    let r = TextureRef::new(TextureId(0))
        .with_uv_set(2)
        .with_transform(TextureTransform::new().with_uv_set(5));
    assert_eq!(r.effective_uv_set(), 5);
}

#[test]
fn map_texture_ids_preserves_uv_set_and_transform() {
    let mut m = Material::new();
    let t = TextureTransform::new()
        .with_offset([0.5, 0.5])
        .with_uv_set(1);
    m.base_color_texture = Some(
        TextureRef::new(TextureId(0))
            .with_uv_set(1)
            .with_transform(t),
    );
    m.map_texture_ids(|id| TextureId(id.0 + 10));
    let r = m.base_color_texture.unwrap();
    assert_eq!(r.texture, TextureId(10));
    assert_eq!(r.uv_set, 1);
    assert_eq!(r.transform, Some(t));
}

#[test]
fn map_texture_refs_can_clear_transforms_after_baking() {
    // The exporter-baking pattern: apply each transform into the UV
    // data, then clear the transforms in one walk.
    let mut m = Material::new();
    let t = TextureTransform::new().with_scale([2.0, 2.0]);
    m.base_color_texture = Some(TextureRef::new(TextureId(0)).with_transform(t));
    m.emissive_texture = Some(TextureRef::new(TextureId(1)).with_transform(t));
    m.map_texture_refs(|mut r| {
        r.transform = None;
        r
    });
    assert_eq!(m.base_color_texture.unwrap().transform, None);
    assert_eq!(m.emissive_texture.unwrap().transform, None);
    // Texture ids untouched.
    assert_eq!(m.base_color_texture.unwrap().texture, TextureId(0));
    assert_eq!(m.emissive_texture.unwrap().texture, TextureId(1));
}

#[test]
fn texture_refs_carries_the_transform() {
    let mut m = Material::new();
    let t = TextureTransform::new().with_rotation(1.0);
    m.normal_texture = Some(TextureRef::new(TextureId(4)).with_transform(t));
    let refs = m.texture_refs();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].0, "normal_texture");
    assert_eq!(refs[0].1.transform, Some(t));
}

// ---------------------------------------------------------------
// Scene append carries the transform verbatim (id-free payload)

#[test]
fn append_remaps_texture_id_but_keeps_transform() {
    fn scene_with_transform() -> Scene3D {
        let mut s = Scene3D::new();
        let tex = s.add_texture(Texture::from_uri("t.png"));
        let mut mat = Material::new();
        mat.base_color_texture = Some(
            TextureRef::new(tex).with_transform(
                TextureTransform::new()
                    .with_offset([0.25, 0.75])
                    .with_uv_set(0),
            ),
        );
        let matid = s.add_material(mat);
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        prim.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]];
        prim.indices = Some(Indices::U32(vec![0, 1, 2]));
        prim.material = Some(matid);
        let meshid = s.add_mesh(Mesh::new(None).with_primitive(prim));
        let nid = s.add_node(Node::new().with_mesh(meshid));
        s.add_root(nid);
        s
    }

    let mut dst = scene_with_transform();
    let src = scene_with_transform();
    dst.append(&src);
    assert!(dst.validate().is_ok());

    let relocated = dst.materials[1].base_color_texture.unwrap();
    // The texture id was re-based onto the destination arena…
    assert_eq!(relocated.texture, TextureId(1));
    // …and the id-free transform travelled verbatim.
    assert_eq!(
        relocated.transform,
        Some(
            TextureTransform::new()
                .with_offset([0.25, 0.75])
                .with_uv_set(0)
        )
    );
}
