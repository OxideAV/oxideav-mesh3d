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
}

impl Default for Material {
    fn default() -> Self {
        Self::new()
    }
}
