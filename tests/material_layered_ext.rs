//! Coverage for the layered ratified-KHR material extensions modelled
//! on [`MaterialExt`]: clearcoat, sheen, transmission, volume,
//! iridescence, and anisotropy. Each is `Option`-shaped so an absent
//! extension stays distinguishable from its spec default, and each
//! struct's `Default` reproduces the extension's normative defaults.

use oxideav_mesh3d::{
    Anisotropy, Clearcoat, Iridescence, Material, Sheen, TextureId, TextureRef, Transmission,
    Volume,
};

#[test]
fn all_layered_extensions_absent_by_default() {
    let mat = Material::new();
    assert!(mat.ext.clearcoat.is_none());
    assert!(mat.ext.sheen.is_none());
    assert!(mat.ext.transmission.is_none());
    assert!(mat.ext.volume.is_none());
    assert!(mat.ext.iridescence.is_none());
    assert!(mat.ext.anisotropy.is_none());
}

#[test]
fn clearcoat_defaults_match_spec() {
    let c = Clearcoat::default();
    assert_eq!(c.factor, 0.0);
    assert_eq!(c.roughness, 0.0);
    assert_eq!(c.normal_scale, 1.0);
    assert!(c.factor_texture.is_none());
    assert!(c.roughness_texture.is_none());
    assert!(c.normal_texture.is_none());
}

#[test]
fn clearcoat_carries_factors_and_textures() {
    let c = Clearcoat {
        factor: 0.8,
        factor_texture: Some(TextureRef::new(TextureId(2))),
        roughness: 0.1,
        roughness_texture: Some(TextureRef::new(TextureId(3))),
        normal_texture: Some(TextureRef::new(TextureId(4))),
        normal_scale: 0.5,
    };
    let mat = Material::new().with_clearcoat(c);
    let got = mat.ext.clearcoat.expect("clearcoat set");
    assert!((got.factor - 0.8).abs() < f32::EPSILON);
    assert!((got.roughness - 0.1).abs() < f32::EPSILON);
    assert!((got.normal_scale - 0.5).abs() < f32::EPSILON);
    assert_eq!(got.factor_texture.unwrap().texture.0, 2);
    assert_eq!(got.roughness_texture.unwrap().texture.0, 3);
    assert_eq!(got.normal_texture.unwrap().texture.0, 4);
}

#[test]
fn sheen_defaults_match_spec() {
    let s = Sheen::default();
    assert_eq!(s.color_factor, [0.0, 0.0, 0.0]);
    assert_eq!(s.roughness, 0.0);
}

#[test]
fn sheen_carries_color_and_roughness() {
    let s = Sheen {
        color_factor: [0.4, 0.2, 0.1],
        color_texture: Some(TextureRef::new(TextureId(5))),
        roughness: 0.7,
        roughness_texture: Some(TextureRef::new(TextureId(6))),
    };
    let mat = Material::new().with_sheen(s);
    let got = mat.ext.sheen.expect("sheen set");
    assert_eq!(got.color_factor, [0.4, 0.2, 0.1]);
    assert!((got.roughness - 0.7).abs() < f32::EPSILON);
    assert_eq!(got.color_texture.unwrap().texture.0, 5);
}

#[test]
fn transmission_defaults_to_opaque() {
    let t = Transmission::default();
    assert_eq!(t.factor, 0.0);
    assert!(t.factor_texture.is_none());

    let mat = Material::new().with_transmission(Transmission {
        factor: 0.9,
        factor_texture: Some(TextureRef::new(TextureId(7))),
    });
    let got = mat.ext.transmission.unwrap();
    assert!((got.factor - 0.9).abs() < f32::EPSILON);
    assert_eq!(got.factor_texture.unwrap().texture.0, 7);
}

#[test]
fn volume_defaults_are_thin_walled_with_infinite_attenuation() {
    let v = Volume::default();
    assert_eq!(v.thickness, 0.0);
    assert_eq!(v.attenuation_color, [1.0, 1.0, 1.0]);
    // None encodes the spec's +Infinity default.
    assert!(v.attenuation_distance.is_none());
    assert!(v.effective_attenuation_distance().is_infinite());
    assert!(v.effective_attenuation_distance() > 0.0);
}

#[test]
fn volume_finite_attenuation_distance_round_trips() {
    let v = Volume {
        thickness: 2.0,
        thickness_texture: Some(TextureRef::new(TextureId(8))),
        attenuation_distance: Some(0.25),
        attenuation_color: [0.8, 0.9, 1.0],
    };
    let mat = Material::new().with_volume(v);
    let got = mat.ext.volume.unwrap();
    assert!((got.thickness - 2.0).abs() < f32::EPSILON);
    assert_eq!(got.attenuation_color, [0.8, 0.9, 1.0]);
    assert!((got.effective_attenuation_distance() - 0.25).abs() < f32::EPSILON);
}

#[test]
fn iridescence_defaults_match_spec() {
    let i = Iridescence::default();
    assert_eq!(i.factor, 0.0);
    assert!((i.ior - 1.3).abs() < f32::EPSILON);
    assert_eq!(i.thickness_min, 100.0);
    assert_eq!(i.thickness_max, 400.0);
}

#[test]
fn iridescence_carries_film_parameters() {
    let i = Iridescence {
        factor: 1.0,
        factor_texture: Some(TextureRef::new(TextureId(9))),
        ior: 1.8,
        thickness_min: 200.0,
        thickness_max: 800.0,
        thickness_texture: Some(TextureRef::new(TextureId(10))),
    };
    let mat = Material::new().with_iridescence(i);
    let got = mat.ext.iridescence.unwrap();
    assert!((got.ior - 1.8).abs() < f32::EPSILON);
    assert_eq!(got.thickness_min, 200.0);
    assert_eq!(got.thickness_max, 800.0);
    assert_eq!(got.thickness_texture.unwrap().texture.0, 10);
}

#[test]
fn anisotropy_defaults_to_isotropic() {
    let a = Anisotropy::default();
    assert_eq!(a.strength, 0.0);
    assert_eq!(a.rotation, 0.0);
    assert!(a.texture.is_none());
}

#[test]
fn anisotropy_carries_strength_rotation_texture() {
    let a = Anisotropy {
        strength: 0.6,
        rotation: std::f32::consts::FRAC_PI_4,
        texture: Some(TextureRef::new(TextureId(11))),
    };
    let mat = Material::new().with_anisotropy(a);
    let got = mat.ext.anisotropy.unwrap();
    assert!((got.strength - 0.6).abs() < f32::EPSILON);
    assert!((got.rotation - std::f32::consts::FRAC_PI_4).abs() < f32::EPSILON);
    assert_eq!(got.texture.unwrap().texture.0, 11);
}

/// A material with **every** texture slot occupied by a distinct id —
/// the five core maps + every extension map — used to prove the two
/// slot enumerations ([`Material::texture_refs`] and
/// [`Material::map_texture_ids`]) cannot drift apart.
fn fully_textured_material() -> Material {
    let mut mat = Material::new();
    mat.base_color_texture = Some(TextureRef::new(TextureId(0)));
    mat.metallic_roughness_texture = Some(TextureRef::new(TextureId(1)));
    mat.normal_texture = Some(TextureRef::new(TextureId(2)));
    mat.occlusion_texture = Some(TextureRef::new(TextureId(3)));
    mat.emissive_texture = Some(TextureRef::new(TextureId(4)));
    mat.ext.specular = Some(oxideav_mesh3d::Specular {
        factor_texture: Some(TextureRef::new(TextureId(5))),
        color_texture: Some(TextureRef::new(TextureId(6))),
        ..Default::default()
    });
    mat.ext.clearcoat = Some(Clearcoat {
        factor_texture: Some(TextureRef::new(TextureId(7))),
        roughness_texture: Some(TextureRef::new(TextureId(8))),
        normal_texture: Some(TextureRef::new(TextureId(9))),
        ..Default::default()
    });
    mat.ext.sheen = Some(Sheen {
        color_texture: Some(TextureRef::new(TextureId(10))),
        roughness_texture: Some(TextureRef::new(TextureId(11))),
        ..Default::default()
    });
    mat.ext.transmission = Some(Transmission {
        factor_texture: Some(TextureRef::new(TextureId(12))),
        ..Default::default()
    });
    mat.ext.volume = Some(Volume {
        thickness_texture: Some(TextureRef::new(TextureId(13))),
        ..Default::default()
    });
    mat.ext.iridescence = Some(Iridescence {
        factor_texture: Some(TextureRef::new(TextureId(14))),
        thickness_texture: Some(TextureRef::new(TextureId(15))),
        ..Default::default()
    });
    mat.ext.anisotropy = Some(Anisotropy {
        texture: Some(TextureRef::new(TextureId(16))),
        ..Default::default()
    });
    mat
}

/// Total number of texture slots the material model exposes (core +
/// every extension map). Bump when a new slot lands — the drift-guard
/// tests below then force `texture_refs` and `map_texture_ids` to
/// both cover it.
const TOTAL_TEXTURE_SLOTS: usize = 17;

#[test]
fn texture_refs_enumerates_every_slot_with_unique_names() {
    let mat = fully_textured_material();
    let refs = mat.texture_refs();
    assert_eq!(refs.len(), TOTAL_TEXTURE_SLOTS);
    // Slot names are unique diagnostics keys.
    let mut names: Vec<&str> = refs.iter().map(|(n, _)| *n).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), TOTAL_TEXTURE_SLOTS);
    // Every distinct id 0..N shows up exactly once.
    let mut ids: Vec<u32> = refs.iter().map(|(_, r)| r.texture.0).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..TOTAL_TEXTURE_SLOTS as u32).collect::<Vec<_>>());
}

#[test]
fn texture_refs_empty_on_untextured_material() {
    assert!(Material::new().texture_refs().is_empty());
}

#[test]
fn map_texture_ids_and_texture_refs_cover_the_same_slots() {
    // Shift every id by +100 through `map_texture_ids`, then read the
    // slots back through `texture_refs`: if either enumeration misses
    // a slot the other covers, an id stays unshifted (or a slot count
    // disagrees) and this fails.
    let mut mat = fully_textured_material();
    mat.map_texture_ids(|t| TextureId(t.0 + 100));
    let refs = mat.texture_refs();
    assert_eq!(refs.len(), TOTAL_TEXTURE_SLOTS);
    for (name, r) in refs {
        assert!(
            (100..100 + TOTAL_TEXTURE_SLOTS as u32).contains(&r.texture.0),
            "slot {name} escaped map_texture_ids: id {}",
            r.texture.0
        );
    }
}

#[test]
fn extensions_compose_independently_on_one_material() {
    let mat = Material::new()
        .with_clearcoat(Clearcoat {
            factor: 1.0,
            ..Default::default()
        })
        .with_transmission(Transmission {
            factor: 0.5,
            ..Default::default()
        })
        .with_anisotropy(Anisotropy {
            strength: 0.3,
            ..Default::default()
        });
    assert!(mat.ext.clearcoat.is_some());
    assert!(mat.ext.transmission.is_some());
    assert!(mat.ext.anisotropy.is_some());
    // Untouched extensions stay absent.
    assert!(mat.ext.sheen.is_none());
    assert!(mat.ext.volume.is_none());
    assert!(mat.ext.iridescence.is_none());
}
