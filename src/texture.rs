//! Texture, image data, and sampler state.
//!
//! Three image-data shapes are supported so format crates can pick
//! the cheapest representation:
//!
//! * [`ImageData::Embedded`] — already-decoded pixels in an
//!   `oxideav_core::VideoFrame` (only available with the `registry`
//!   feature). Use this when the format crate decoded the image
//!   itself or already had pixel data on hand.
//! * [`ImageData::Source`] — lazy reference to an [`AssetSource`]
//!   that can be opened on demand and (optionally) supports
//!   zero-copy raw-storage pass-through. Replaces round-1's
//!   `Encoded { mime, bytes }` so massive scenes (USDZ archives in
//!   the hundreds of MB) don't have to be materialised in RAM.
//! * [`ImageData::External`] — URI reference (`file://`, relative
//!   path, or `http(s)://`) plus optional MIME hint. The caller
//!   resolves and decodes lazily.

use std::fmt;
use std::sync::Arc;

use crate::asset::{AssetSource, InMemoryAsset};

/// Encoded or decoded texture pixels.
#[derive(Clone, Debug)]
pub enum ImageData {
    /// Already-decoded pixel buffer. Only available with the
    /// default-on `registry` feature (which pulls in
    /// `oxideav-core`'s [`VideoFrame`](oxideav_core::VideoFrame)).
    #[cfg(feature = "registry")]
    Embedded(oxideav_core::VideoFrame),
    /// Lazy reference. The wrapped [`AssetSource`] streams bytes on
    /// demand (`open()`) and may expose `raw_storage()` so a writer
    /// targeting the same container scheme can pass the original
    /// payload through without re-encoding.
    Source(Arc<dyn AssetSource>),
    /// URI to fetch and decode lazily. `mime` is a hint when known.
    External { uri: String, mime: Option<String> },
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

    /// Construct a texture from any [`AssetSource`] implementor.
    /// Use this when the format crate already exposes its blob via
    /// a custom `AssetSource` (USDZ ZIP entry, GLB bin chunk slice,
    /// FBX embedded media handle).
    pub fn from_source(source: Arc<dyn AssetSource>) -> Self {
        Self {
            name: None,
            image: ImageData::Source(source),
            sampler: Sampler::default_sampler(),
        }
    }

    /// Convenience constructor that wraps owned `bytes` + `mime` in
    /// an [`InMemoryAsset`] and exposes it as `ImageData::Source`.
    /// Replaces round-1's `from_encoded(mime, bytes)` constructor —
    /// migration is field-for-field, but the in-memory blob now goes
    /// through the trait so consumers get one uniform code path.
    pub fn from_encoded(mime: impl Into<String>, bytes: Vec<u8>) -> Self {
        let asset = Arc::new(InMemoryAsset {
            mime: Some(mime.into()),
            bytes,
        });
        Self::from_source(asset)
    }
}
