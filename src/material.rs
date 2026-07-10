//! PBR material definition (glTF 2.0 metallic-roughness model).
//!
//! Every channel has a constant factor and an optional texture
//! reference; the runtime sample is `factor * sample(texture)` per
//! channel (RGB componentwise). This is the lowest-common-denominator
//! PBR shape that every modern engine supports — glTF, USD's
//! UsdPreviewSurface, FBX's PBR exporter chain, and Blender Principled
//! BSDF all map cleanly into it.
//!
//! Non-PBR formats (legacy OBJ MTL Phong, FBX Lambert) collapse into
//! the same shape with metallic = 0, roughness from `Ns`, and the
//! original parameters preserved in [`Material::extras`] for any
//! lossless round-trip.

use std::collections::HashMap;

use crate::scene::TextureId;

/// Reference to one [`Texture`](crate::Texture) along with which UV
/// set in the consuming primitive samples it.
///
/// `uv_set` indexes into [`Primitive::uvs`](crate::mesh::Primitive::uvs);
/// the default for most files is 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureRef {
    pub texture: TextureId,
    pub uv_set: u32,
}

impl TextureRef {
    /// Bind a texture to UV set 0 (the most common case).
    pub fn new(texture: TextureId) -> Self {
        Self { texture, uv_set: 0 }
    }
}

/// How the material's alpha channel composites against the framebuffer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AlphaMode {
    /// Alpha is ignored; the surface is fully opaque.
    #[default]
    Opaque,
    /// Pixels with alpha < `cutoff` are discarded; surviving pixels
    /// are rendered fully opaque (no blending). Standard glTF cutoff
    /// is 0.5.
    Mask { cutoff: f32 },
    /// Standard `src.a` over-blend.
    Blend,
}

/// Typed surface for the ratified KHR material extensions that refine
/// the core metallic-roughness dielectric BRDF.
///
/// Every field is `Option`/flag-shaped so the *absence* of an
/// extension is distinguishable from its spec default — a format crate
/// only sets a field when the file actually carries that extension, so
/// an encoder re-emits exactly the extension blocks the source
/// declared. The doc-comments record each parameter's normative
/// default so a consumer that wants "the value the renderer should
/// use" can substitute it for `None`.
///
/// The extensions modelled here are the simply-shaped dielectric
/// refinements — a scalar or a factor-plus-texture — drawn from the
/// Khronos KHR extension registry:
///
/// - **emissive strength** — a unitless multiplier on the core
///   `emissive_factor` / `emissive_texture` product, lifting emission
///   out of the core `[0,1]` clamp for HDR bloom.
/// - **index of refraction** — replaces the fixed dielectric IOR of
///   1.5 used by the core model.
/// - **specular** — strength + F0 colour of the dielectric specular
///   reflection, each a factor optionally modulated by a texture.
/// - **unlit** — a flag selecting a constant-shaded (lighting-
///   independent) model that uses only the base-colour term.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialExt {
    /// Multiplier on the material's emissive value
    /// (`KHR_materials_emissive_strength`). The shaded emission is
    /// `emissive_factor * sample(emissive_texture) * strength`. Spec
    /// default when the extension is absent: `1.0`. Values above `1.0`
    /// drive bloom / tonemapping in HDR pipelines. Mutually exclusive
    /// with [`unlit`](Self::unlit) per the extension's exclusions.
    pub emissive_strength: Option<f32>,
    /// Index of refraction of the dielectric BRDF
    /// (`KHR_materials_ior`). Spec default when absent: `1.5`
    /// (`dielectric_f0 = ((ior-1)/(ior+1))^2 = 0.04`). Valid values
    /// are `>= 1`, with the special case `0.0` permanently selecting
    /// the legacy specular-glossiness backwards-compatibility mode
    /// (effective IOR → +∞, Fresnel ≡ 1).
    pub ior: Option<f32>,
    /// Dielectric specular strength + F0 colour
    /// (`KHR_materials_specular`). `None` ⇒ the core model's implicit
    /// `specular = 1.0`, `specular_color = [1,1,1]`.
    pub specular: Option<Specular>,
    /// `KHR_materials_unlit` flag. When `true` the surface is
    /// constant-shaded from the base-colour term alone (factor ×
    /// texture × vertex colour); all lighting-dependent PBR inputs are
    /// ignored, though alpha coverage and `double_sided` still apply.
    pub unlit: bool,
    /// `KHR_materials_clearcoat` — a thin glossy lacquer layer over the
    /// base material. `None` ⇒ no clearcoat layer (factor `0.0`).
    pub clearcoat: Option<Clearcoat>,
    /// `KHR_materials_sheen` — a retroreflective cloth/fabric layer.
    /// `None` ⇒ no sheen layer (colour `[0,0,0]`).
    pub sheen: Option<Sheen>,
    /// `KHR_materials_transmission` — the fraction of light passing
    /// through the (thin) surface. `None` ⇒ opaque (factor `0.0`).
    pub transmission: Option<Transmission>,
    /// `KHR_materials_volume` — turns the surface into the boundary of
    /// a homogeneous absorbing medium (requires a manifold mesh).
    /// `None` ⇒ infinitely thin wall (thickness `0.0`).
    pub volume: Option<Volume>,
    /// `KHR_materials_iridescence` — a thin-film interference layer
    /// producing a view-dependent colour shift. `None` ⇒ no
    /// iridescence (factor `0.0`).
    pub iridescence: Option<Iridescence>,
    /// `KHR_materials_anisotropy` — direction-dependent roughness for
    /// brushed-metal / hair highlights. `None` ⇒ isotropic
    /// (strength `0.0`).
    pub anisotropy: Option<Anisotropy>,
}

/// `KHR_materials_clearcoat` parameters: a thin, glossy isotropic
/// lacquer layer applied on top of the base material. The clearcoat
/// has its own intensity, roughness, and (optionally) a dedicated
/// normal map so the lacquer can read as smooth over a bumpy base.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clearcoat {
    /// Clearcoat layer intensity (`clearcoatFactor`). Default `0.0`
    /// (no clearcoat); `1.0` is a full lacquer layer. Multiplied by
    /// the red channel of [`factor_texture`](Self::factor_texture).
    pub factor: f32,
    /// Clearcoat intensity texture (`clearcoatTexture`), sampled from
    /// the `R` channel.
    pub factor_texture: Option<TextureRef>,
    /// Clearcoat layer roughness (`clearcoatRoughnessFactor`). Default
    /// `0.0` (mirror-smooth lacquer). Multiplied by the green channel
    /// of [`roughness_texture`](Self::roughness_texture).
    pub roughness: f32,
    /// Clearcoat roughness texture (`clearcoatRoughnessTexture`),
    /// sampled from the `G` channel.
    pub roughness_texture: Option<TextureRef>,
    /// Dedicated clearcoat normal map (`clearcoatNormalTexture`);
    /// independent of the base material's normal map so a smooth coat
    /// can sit over a detailed base. Modulated by
    /// [`normal_scale`](Self::normal_scale).
    pub normal_texture: Option<TextureRef>,
    /// Scale applied to the clearcoat normal map's XY components
    /// (`clearcoatNormalTexture.scale`). Default `1.0`.
    pub normal_scale: f32,
}

impl Default for Clearcoat {
    fn default() -> Self {
        Self {
            factor: 0.0,
            factor_texture: None,
            roughness: 0.0,
            roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
        }
    }
}

/// `KHR_materials_sheen` parameters: a retroreflective microfibre
/// layer modelling cloth, velvet, and fabric. The sheen colour fades
/// the silhouette and the roughness controls the width of the
/// grazing-angle rim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sheen {
    /// Sheen colour in linear RGB (`sheenColorFactor`). Default
    /// `[0,0,0]` — black sheen means no effect. Multiplied by the
    /// `RGB` of [`color_texture`](Self::color_texture).
    pub color_factor: [f32; 3],
    /// Sheen colour texture (`sheenColorTexture`), `RGB` channels.
    pub color_texture: Option<TextureRef>,
    /// Sheen roughness (`sheenRoughnessFactor`). Default `0.0`.
    /// Multiplied by the alpha channel of
    /// [`roughness_texture`](Self::roughness_texture).
    pub roughness: f32,
    /// Sheen roughness texture (`sheenRoughnessTexture`), `A` channel.
    pub roughness_texture: Option<TextureRef>,
}

impl Default for Sheen {
    fn default() -> Self {
        Self {
            color_factor: [0.0, 0.0, 0.0],
            color_texture: None,
            roughness: 0.0,
            roughness_texture: None,
        }
    }
}

/// `KHR_materials_transmission` parameters: the fraction of light that
/// passes through the surface, modelling thin transparent dielectrics
/// (glass, water films) that still respect Fresnel and roughness —
/// distinct from alpha blending, which fades the whole BRDF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transmission {
    /// Base fraction of light transmitted through the surface
    /// (`transmissionFactor`). Default `0.0` (opaque). Multiplied by
    /// the red channel of [`factor_texture`](Self::factor_texture).
    pub factor: f32,
    /// Transmission texture (`transmissionTexture`), `R` channel.
    pub factor_texture: Option<TextureRef>,
}

impl Default for Transmission {
    fn default() -> Self {
        Self {
            factor: 0.0,
            factor_texture: None,
        }
    }
}

/// `KHR_materials_volume` parameters: turns the surface from an
/// infinitely thin wall into the boundary of a homogeneous absorbing
/// medium. Light travelling through the medium is attenuated towards
/// [`attenuation_color`](Self::attenuation_color) over
/// [`attenuation_distance`](Self::attenuation_distance). Requires a
/// manifold mesh and is typically paired with a non-zero
/// [`Transmission`] / [`MaterialExt::ior`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Volume {
    /// Thickness of the volume beneath the surface in the mesh's
    /// coordinate space (`thicknessFactor`). Default `0.0` —
    /// thin-walled, i.e. the extension has no effect until thickness
    /// is positive. Multiplied by the green channel of
    /// [`thickness_texture`](Self::thickness_texture).
    pub thickness: f32,
    /// Thickness texture (`thicknessTexture`), `G` channel.
    pub thickness_texture: Option<TextureRef>,
    /// Average distance light travels in the medium before interacting
    /// with a particle, in world space (`attenuationDistance`). The
    /// spec default is `+Infinity` (no absorption); represented here
    /// as `None` so the special "infinite" case stays distinct from a
    /// finite distance. See [`effective_attenuation_distance`].
    ///
    /// [`effective_attenuation_distance`]: Volume::effective_attenuation_distance
    pub attenuation_distance: Option<f32>,
    /// Colour white light turns into after travelling
    /// [`attenuation_distance`](Self::attenuation_distance) through the
    /// medium (`attenuationColor`), linear RGB. Default `[1,1,1]` (no
    /// tint).
    pub attenuation_color: [f32; 3],
}

impl Volume {
    /// The effective attenuation distance — the stored value when
    /// finite, else `f32::INFINITY` (the spec default).
    pub fn effective_attenuation_distance(&self) -> f32 {
        self.attenuation_distance.unwrap_or(f32::INFINITY)
    }
}

impl Default for Volume {
    fn default() -> Self {
        Self {
            thickness: 0.0,
            thickness_texture: None,
            attenuation_distance: None,
            attenuation_color: [1.0, 1.0, 1.0],
        }
    }
}

/// `KHR_materials_iridescence` parameters: a thin-film interference
/// layer whose perceived colour shifts with view angle, modelling soap
/// bubbles, oil films, and insect carapaces. The film's optical
/// thickness varies between [`thickness_min`](Self::thickness_min) and
/// [`thickness_max`](Self::thickness_max) nanometres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Iridescence {
    /// Iridescence intensity (`iridescenceFactor`). Default `0.0`
    /// (extension has no effect). Multiplied by
    /// [`factor_texture`](Self::factor_texture).
    pub factor: f32,
    /// Iridescence intensity texture (`iridescenceTexture`), single
    /// linear channel.
    pub factor_texture: Option<TextureRef>,
    /// Index of refraction of the thin-film layer (`iridescenceIor`).
    /// Default `1.3`.
    pub ior: f32,
    /// Minimum film thickness in nanometres
    /// (`iridescenceThicknessMinimum`). Default `100.0`. Selected
    /// where the thickness texture reads `0`.
    pub thickness_min: f32,
    /// Maximum film thickness in nanometres
    /// (`iridescenceThicknessMaximum`). Default `400.0`. Selected
    /// where the thickness texture reads `1`.
    pub thickness_max: f32,
    /// Thickness texture (`iridescenceThicknessTexture`), single linear
    /// channel interpolating min→max.
    pub thickness_texture: Option<TextureRef>,
}

impl Default for Iridescence {
    fn default() -> Self {
        Self {
            factor: 0.0,
            factor_texture: None,
            ior: 1.3,
            thickness_min: 100.0,
            thickness_max: 400.0,
            thickness_texture: None,
        }
    }
}

/// `KHR_materials_anisotropy` parameters: direction-dependent
/// roughness producing stretched highlights along the tangent frame,
/// modelling brushed metal, hair, and vinyl records. Requires the
/// primitive to carry (or be able to synthesise) a tangent basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anisotropy {
    /// Anisotropy strength in `[0,1]` (`anisotropyStrength`). Default
    /// `0.0` (isotropic). When the texture is present this is
    /// multiplied by the texture's blue channel.
    pub strength: f32,
    /// Rotation of the anisotropy direction in tangent/bitangent
    /// space, radians counter-clockwise from the tangent
    /// (`anisotropyRotation`). Default `0.0`. Additive with the
    /// per-texel direction when the texture is present.
    pub rotation: f32,
    /// Anisotropy texture (`anisotropyTexture`): `RG` encode the
    /// tangent-space direction in `[-1,1]`, `B` the strength in
    /// `[0,1]`.
    pub texture: Option<TextureRef>,
}

impl Default for Anisotropy {
    fn default() -> Self {
        Self {
            strength: 0.0,
            rotation: 0.0,
            texture: None,
        }
    }
}

/// `KHR_materials_specular` parameters: the strength and F0 colour of
/// the dielectric specular reflection. Each value is a constant factor
/// optionally multiplied by a texture sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Specular {
    /// Strength of the specular reflection (`specularFactor`).
    /// Default `1.0`; `0.0` disables the dielectric specular lobe,
    /// leaving a pure-diffuse dielectric. The metal BRDF is
    /// unaffected. Combined with [`factor_texture`](Self::factor_texture)
    /// by multiplication, sampling its alpha channel.
    pub factor: f32,
    /// Strength texture (`specularTexture`); the alpha (`A`) channel
    /// scales [`factor`](Self::factor).
    pub factor_texture: Option<TextureRef>,
    /// F0 colour of the dielectric reflection in linear RGB
    /// (`specularColorFactor`). Default `[1.0, 1.0, 1.0]`. May exceed
    /// `1.0`; the renderer clamps the product against IOR-derived F0.
    pub color_factor: [f32; 3],
    /// F0-colour texture (`specularColorTexture`); the `RGB` channels
    /// (sRGB-encoded) multiply [`color_factor`](Self::color_factor).
    pub color_texture: Option<TextureRef>,
}

impl Default for Specular {
    /// The spec defaults: full strength, white F0, no textures.
    fn default() -> Self {
        Self {
            factor: 1.0,
            factor_texture: None,
            color_factor: [1.0, 1.0, 1.0],
            color_texture: None,
        }
    }
}

/// PBR material.
///
/// Defaults match the glTF spec: white base colour, fully metallic /
/// fully rough (so a missing `metallic`/`roughness` field renders as
/// rough metal — the spec-mandated "this should look obviously
/// wrong" sentinel, not a sane shading default). Format crates
/// override as the file directs.
#[derive(Clone, Debug)]
pub struct Material {
    pub name: Option<String>,
    /// RGBA factor multiplied into the base-colour texture sample.
    pub base_color: [f32; 4],
    pub base_color_texture: Option<TextureRef>,
    /// `[0,1]`; default 1.0.
    pub metallic: f32,
    /// `[0,1]`; default 1.0.
    pub roughness: f32,
    /// Packed B = metallic, G = roughness per glTF KHR_pbr_metallic_roughness.
    pub metallic_roughness_texture: Option<TextureRef>,
    pub normal_texture: Option<TextureRef>,
    /// Multiplier applied to the normal-map XY components.
    pub normal_scale: f32,
    pub occlusion_texture: Option<TextureRef>,
    pub occlusion_strength: f32,
    /// RGB additive emission factor.
    pub emissive_factor: [f32; 3],
    pub emissive_texture: Option<TextureRef>,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
    /// Typed ratified-KHR extension refinements (emissive strength,
    /// IOR, specular, unlit). All absent / `false` by default, meaning
    /// "plain core metallic-roughness".
    pub ext: MaterialExt,
    /// Round-trip side-channel for non-glTF data (FBX/USD/OBJ
    /// extensions). Format crates that need to preserve exotic
    /// fields drop them here as `serde_json::Value` and the
    /// matching encoder pulls them back out.
    pub extras: HashMap<String, serde_json::Value>,
}

impl Material {
    /// Construct a default material — white base colour, no maps,
    /// fully metallic + fully rough (per glTF defaults).
    pub fn new() -> Self {
        Self {
            name: None,
            base_color: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: None,
            metallic: 1.0,
            roughness: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive_factor: [0.0, 0.0, 0.0],
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            ext: MaterialExt::default(),
            extras: HashMap::new(),
        }
    }

    /// Builder-style name setter.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Builder-style base-colour setter.
    pub fn with_base_color(mut self, rgba: [f32; 4]) -> Self {
        self.base_color = rgba;
        self
    }

    /// Builder-style `KHR_materials_emissive_strength` setter.
    pub fn with_emissive_strength(mut self, strength: f32) -> Self {
        self.ext.emissive_strength = Some(strength);
        self
    }

    /// Builder-style `KHR_materials_ior` setter.
    pub fn with_ior(mut self, ior: f32) -> Self {
        self.ext.ior = Some(ior);
        self
    }

    /// Builder-style `KHR_materials_specular` setter.
    pub fn with_specular(mut self, specular: Specular) -> Self {
        self.ext.specular = Some(specular);
        self
    }

    /// Builder-style `KHR_materials_unlit` flag setter.
    pub fn with_unlit(mut self, unlit: bool) -> Self {
        self.ext.unlit = unlit;
        self
    }

    /// Builder-style `KHR_materials_clearcoat` setter.
    pub fn with_clearcoat(mut self, clearcoat: Clearcoat) -> Self {
        self.ext.clearcoat = Some(clearcoat);
        self
    }

    /// Builder-style `KHR_materials_sheen` setter.
    pub fn with_sheen(mut self, sheen: Sheen) -> Self {
        self.ext.sheen = Some(sheen);
        self
    }

    /// Builder-style `KHR_materials_transmission` setter.
    pub fn with_transmission(mut self, transmission: Transmission) -> Self {
        self.ext.transmission = Some(transmission);
        self
    }

    /// Builder-style `KHR_materials_volume` setter.
    pub fn with_volume(mut self, volume: Volume) -> Self {
        self.ext.volume = Some(volume);
        self
    }

    /// Builder-style `KHR_materials_iridescence` setter.
    pub fn with_iridescence(mut self, iridescence: Iridescence) -> Self {
        self.ext.iridescence = Some(iridescence);
        self
    }

    /// Builder-style `KHR_materials_anisotropy` setter.
    pub fn with_anisotropy(mut self, anisotropy: Anisotropy) -> Self {
        self.ext.anisotropy = Some(anisotropy);
        self
    }

    /// The effective emissive strength the renderer should apply —
    /// the extension value when present, else the spec default `1.0`.
    pub fn effective_emissive_strength(&self) -> f32 {
        self.ext.emissive_strength.unwrap_or(1.0)
    }

    /// The effective index of refraction — the extension value when
    /// present, else the spec default `1.5`.
    pub fn effective_ior(&self) -> f32 {
        self.ext.ior.unwrap_or(1.5)
    }

    /// Every texture slot this material currently references, as
    /// `(slot_path, TextureRef)` pairs.
    ///
    /// Enumerates the core metallic-roughness maps (`base_color`,
    /// `metallic_roughness`, `normal`, `occlusion`, `emissive`) *and*
    /// every extension map on [`MaterialExt`] that is present. The
    /// slot path is a stable dotted identifier matching the field
    /// names (e.g. `"base_color_texture"`,
    /// `"ext.clearcoat.normal_texture"`), suitable for diagnostics.
    ///
    /// This is the read-side companion of
    /// [`map_texture_ids`](Self::map_texture_ids):
    /// [`Scene3D::validate`](crate::Scene3D::validate) walks this list
    /// to check every referenced [`TextureId`](crate::TextureId) is
    /// live, so a texture slot added to the model in the future cannot
    /// silently escape validation, and generic consumers (resource
    /// collectors, dependency walkers) never have to enumerate the
    /// slots by hand.
    pub fn texture_refs(&self) -> Vec<(&'static str, TextureRef)> {
        let mut out = Vec::new();
        let mut push = |name: &'static str, r: &Option<TextureRef>| {
            if let Some(tr) = r {
                out.push((name, *tr));
            }
        };
        push("base_color_texture", &self.base_color_texture);
        push(
            "metallic_roughness_texture",
            &self.metallic_roughness_texture,
        );
        push("normal_texture", &self.normal_texture);
        push("occlusion_texture", &self.occlusion_texture);
        push("emissive_texture", &self.emissive_texture);
        if let Some(s) = &self.ext.specular {
            push("ext.specular.factor_texture", &s.factor_texture);
            push("ext.specular.color_texture", &s.color_texture);
        }
        if let Some(c) = &self.ext.clearcoat {
            push("ext.clearcoat.factor_texture", &c.factor_texture);
            push("ext.clearcoat.roughness_texture", &c.roughness_texture);
            push("ext.clearcoat.normal_texture", &c.normal_texture);
        }
        if let Some(sh) = &self.ext.sheen {
            push("ext.sheen.color_texture", &sh.color_texture);
            push("ext.sheen.roughness_texture", &sh.roughness_texture);
        }
        if let Some(t) = &self.ext.transmission {
            push("ext.transmission.factor_texture", &t.factor_texture);
        }
        if let Some(v) = &self.ext.volume {
            push("ext.volume.thickness_texture", &v.thickness_texture);
        }
        if let Some(ir) = &self.ext.iridescence {
            push("ext.iridescence.factor_texture", &ir.factor_texture);
            push("ext.iridescence.thickness_texture", &ir.thickness_texture);
        }
        if let Some(an) = &self.ext.anisotropy {
            push("ext.anisotropy.texture", &an.texture);
        }
        out
    }

    /// Rewrite **every** [`TextureId`](crate::TextureId) this material
    /// references through `f`, in place.
    ///
    /// Walks all texture slots — the core metallic-roughness maps
    /// (`base_color`, `metallic_roughness`, `normal`, `occlusion`,
    /// `emissive`) *and* every extension map on [`MaterialExt`]
    /// (specular, clearcoat, sheen, transmission, volume, iridescence,
    /// anisotropy). The `uv_set` of each [`TextureRef`] is left
    /// untouched; only the texture index is remapped.
    ///
    /// This is the building block for re-indexing a material when its
    /// textures move to a new arena (scene composition / append), so a
    /// caller never has to enumerate the slot list by hand and risk
    /// missing a newly-added extension map.
    pub fn map_texture_ids(&mut self, f: impl Fn(crate::TextureId) -> crate::TextureId) {
        let remap = |r: &mut Option<TextureRef>| {
            if let Some(tr) = r {
                tr.texture = f(tr.texture);
            }
        };
        remap(&mut self.base_color_texture);
        remap(&mut self.metallic_roughness_texture);
        remap(&mut self.normal_texture);
        remap(&mut self.occlusion_texture);
        remap(&mut self.emissive_texture);
        if let Some(s) = &mut self.ext.specular {
            remap(&mut s.factor_texture);
            remap(&mut s.color_texture);
        }
        if let Some(c) = &mut self.ext.clearcoat {
            remap(&mut c.factor_texture);
            remap(&mut c.roughness_texture);
            remap(&mut c.normal_texture);
        }
        if let Some(sh) = &mut self.ext.sheen {
            remap(&mut sh.color_texture);
            remap(&mut sh.roughness_texture);
        }
        if let Some(t) = &mut self.ext.transmission {
            remap(&mut t.factor_texture);
        }
        if let Some(v) = &mut self.ext.volume {
            remap(&mut v.thickness_texture);
        }
        if let Some(ir) = &mut self.ext.iridescence {
            remap(&mut ir.factor_texture);
            remap(&mut ir.thickness_texture);
        }
        if let Some(an) = &mut self.ext.anisotropy {
            remap(&mut an.texture);
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new()
    }
}
