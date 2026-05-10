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
}

impl Default for Material {
    fn default() -> Self {
        Self::new()
    }
}
