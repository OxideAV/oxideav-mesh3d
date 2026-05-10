//! Texture, image data, and sampler state.
//!
//! Three image-data shapes are supported so format crates can pick
//! the cheapest representation:
//!
//! * [`ImageData::Embedded`] — already-decoded pixels in an
//!   `oxideav_core::VideoFrame` (only available with the `registry`
//!   feature). Use this when the format crate decoded the image
//!   itself or already had pixel data on hand.
//! * [`ImageData::External`] — URI reference (`file://`, relative
//!   path, or `http(s)://`) plus optional MIME hint. The caller
//!   resolves and decodes lazily.
//! * [`ImageData::Encoded`] — raw bytes of a still-encoded image
//!   payload (PNG/JPEG/WebP/...). Use this when the format embeds
//!   the texture but the mesh3d crate doesn't want to pull in an
//!   image decoder.

use std::fmt;

/// Encoded or decoded texture pixels.
#[derive(Clone, Debug)]
pub enum ImageData {
    /// Already-decoded pixel buffer. Only available with the
    /// default-on `registry` feature (which pulls in
    /// `oxideav-core`'s [`VideoFrame`](oxideav_core::VideoFrame)).
    #[cfg(feature = "registry")]
    Embedded(oxideav_core::VideoFrame),
    /// URI to fetch and decode lazily. `mime` is a hint when known.
    External { uri: String, mime: Option<String> },
    /// Encoded image bytes (PNG/JPEG/WebP/...). `mime` identifies
    /// the codec so the caller can route to the right decoder.
    Encoded { mime: String, bytes: Vec<u8> },
}

/// Magnification filter — applied when one screen pixel covers less
/// than one texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MagFilter {
    Nearest,
    Linear,
}

/// Minification filter — applied when one screen pixel covers more
/// than one texel. The `Mip*` variants describe how mipmap levels
/// are picked and combined; matches glTF (and ultimately OpenGL)
/// names verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinFilter {
    Nearest,
    Linear,
    /// Pick nearest mip, sample nearest texel.
    NearestMipNearest,
    /// Pick nearest mip, sample linearly within it.
    LinearMipNearest,
    /// Linearly blend nearest mips, sample nearest within each.
    NearestMipLinear,
    /// Trilinear: linearly blend nearest mips, sample linearly within each.
    LinearMipLinear,
}

/// UV-coordinate behaviour outside `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WrapMode {
    /// Coordinates are clamped to the `[0, 1]` range.
    ClampToEdge,
    /// Tiling that mirrors every other tile.
    MirroredRepeat,
    /// Standard repeat — fractional part of the UV is sampled.
    Repeat,
}

/// Sampler state controlling how a texture is fetched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampler {
    pub mag_filter: MagFilter,
    pub min_filter: MinFilter,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
}

impl Sampler {
    /// glTF default sampler — linear/trilinear, repeating in both axes.
    pub fn default_sampler() -> Self {
        Self {
            mag_filter: MagFilter::Linear,
            min_filter: MinFilter::LinearMipLinear,
            wrap_s: WrapMode::Repeat,
            wrap_t: WrapMode::Repeat,
        }
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::default_sampler()
    }
}

/// A texture: image source + sampler state.
pub struct Texture {
    pub name: Option<String>,
    pub image: ImageData,
    pub sampler: Sampler,
}

impl fmt::Debug for Texture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Texture")
            .field("name", &self.name)
            .field("image", &self.image)
            .field("sampler", &self.sampler)
            .finish()
    }
}

impl Clone for Texture {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            image: self.image.clone(),
            sampler: self.sampler,
        }
    }
}

impl Texture {
    /// Construct a texture from an external URI with the default
    /// glTF sampler.
    pub fn from_uri(uri: impl Into<String>) -> Self {
        Self {
            name: None,
            image: ImageData::External {
                uri: uri.into(),
                mime: None,
            },
            sampler: Sampler::default_sampler(),
        }
    }

    /// Construct a texture from encoded bytes plus MIME with the
    /// default glTF sampler.
    pub fn from_encoded(mime: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            name: None,
            image: ImageData::Encoded {
                mime: mime.into(),
                bytes,
            },
            sampler: Sampler::default_sampler(),
        }
    }
}
