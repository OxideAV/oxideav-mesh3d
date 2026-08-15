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
/// names verbatim. [`mipmap_mode`](Self::mipmap_mode) /
/// [`base_filter`](Self::base_filter) split each variant into its
/// two independent axes.
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

impl MinFilter {
    /// Whether this filter reads mipmap levels at all — `true` for
    /// the four `*Mip*` variants. A texture sampled with a
    /// mipmap-reading filter needs its mip chain generated (or
    /// loaded); the two non-mip variants sample the base level only.
    pub fn uses_mipmaps(&self) -> bool {
        !matches!(self, Self::Nearest | Self::Linear)
    }

    /// How mipmap *levels* are selected and combined — the mip axis
    /// of the filter, independent of the within-level texel filter.
    pub fn mipmap_mode(&self) -> MipmapMode {
        match self {
            Self::Nearest | Self::Linear => MipmapMode::Disabled,
            Self::NearestMipNearest | Self::LinearMipNearest => MipmapMode::Nearest,
            Self::NearestMipLinear | Self::LinearMipLinear => MipmapMode::Linear,
        }
    }

    /// How texels are sampled *within* the selected level — the
    /// texel axis of the filter, independent of the mip behaviour.
    /// Expressed as a [`MagFilter`] because that is exactly the
    /// nearest-vs-linear choice magnification makes.
    pub fn base_filter(&self) -> MagFilter {
        match self {
            Self::Nearest | Self::NearestMipNearest | Self::NearestMipLinear => MagFilter::Nearest,
            Self::Linear | Self::LinearMipNearest | Self::LinearMipLinear => MagFilter::Linear,
        }
    }
}

/// How mipmap levels are selected and combined during minification —
/// the mip axis of a [`MinFilter`], see
/// [`MinFilter::mipmap_mode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MipmapMode {
    /// Only the base level is sampled; no mip chain is needed.
    Disabled,
    /// The single nearest mip level is sampled.
    Nearest,
    /// The two nearest mip levels are sampled and linearly blended.
    Linear,
}

/// UV-coordinate behaviour outside `[0, 1]`.
///
/// The default is [`Repeat`](Self::Repeat), matching glTF's `wrapS` /
/// `wrapT` default. Wrapping is normative — a consumer MUST honour
/// the stored mode (unlike the filters, which are hints when
/// undefined). [`wrap`](Self::wrap) evaluates the mode on a scalar
/// coordinate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WrapMode {
    /// Coordinates are clamped to the `[0, 1]` range.
    ClampToEdge,
    /// Tiling that mirrors every other tile.
    MirroredRepeat,
    /// Standard repeat — fractional part of the UV is sampled.
    #[default]
    Repeat,
}

impl WrapMode {
    /// Map one texture coordinate into the sampled `[0, 1]` range
    /// under this wrap mode:
    ///
    /// * `ClampToEdge` — clamp to `[0, 1]`.
    /// * `Repeat` — take the fractional part (`1.25 → 0.25`,
    ///   `-0.25 → 0.75`; exact integers land on `0.0`).
    /// * `MirroredRepeat` — period-2 triangle wave (`1.25 → 0.75`,
    ///   `-0.25 → 0.25`).
    ///
    /// A non-finite input yields `0.0` rather than poisoning the
    /// sample. This is the CPU-side reference semantics for the wrap
    /// modes (e.g. for software sampling or for deciding whether a
    /// transformed UV range needs wrapping support); GPU samplers
    /// implement the same mapping in hardware.
    pub fn wrap(&self, coord: f32) -> f32 {
        if !coord.is_finite() {
            return 0.0;
        }
        match self {
            Self::ClampToEdge => coord.clamp(0.0, 1.0),
            Self::Repeat => coord - coord.floor(),
            Self::MirroredRepeat => {
                let m = coord - 2.0 * (coord / 2.0).floor();
                if m > 1.0 {
                    2.0 - m
                } else {
                    m
                }
            }
        }
    }
}

/// Sampler state controlling how a texture is fetched.
///
/// The wrap modes are always defined (glTF defaults them to
/// `REPEAT`, and a consumer MUST follow them). The two filters are
/// `Option`-shaped because glTF's `magFilter` / `minFilter` have
/// **no** default: an undefined filter means the runtime MAY apply
/// its own preference (glTF 2.0 §3.8.4.1), and a round-trip-able
/// model has to keep "undefined" distinguishable from any explicit
/// choice. Use [`effective_mag_filter`](Self::effective_mag_filter) /
/// [`effective_min_filter`](Self::effective_min_filter) when a
/// concrete filter is needed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sampler {
    /// Magnification filter; `None` = undefined in the source file
    /// (runtime's choice — spec permits any default).
    pub mag_filter: Option<MagFilter>,
    /// Minification filter; `None` = undefined in the source file
    /// (runtime's choice — spec permits any default).
    pub min_filter: Option<MinFilter>,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
}

impl Sampler {
    /// The glTF default sampler — the state a texture without a
    /// sampler object gets: repeat wrapping on both axes, filters
    /// undefined (left to the runtime). Same as `Sampler::default()`.
    pub fn default_sampler() -> Self {
        Self {
            mag_filter: None,
            min_filter: None,
            wrap_s: WrapMode::Repeat,
            wrap_t: WrapMode::Repeat,
        }
    }

    /// Builder-style magnification-filter setter.
    pub fn with_mag_filter(mut self, filter: MagFilter) -> Self {
        self.mag_filter = Some(filter);
        self
    }

    /// Builder-style minification-filter setter.
    pub fn with_min_filter(mut self, filter: MinFilter) -> Self {
        self.min_filter = Some(filter);
        self
    }

    /// Builder-style wrap-mode setter for both axes at once.
    pub fn with_wrap(mut self, wrap_s: WrapMode, wrap_t: WrapMode) -> Self {
        self.wrap_s = wrap_s;
        self.wrap_t = wrap_t;
        self
    }

    /// The magnification filter a renderer should apply: the stored
    /// filter when defined, else this crate's documented fallback
    /// [`MagFilter::Linear`] (the spec leaves the undefined case to
    /// the implementation; linear is the common high-quality choice).
    pub fn effective_mag_filter(&self) -> MagFilter {
        self.mag_filter.unwrap_or(MagFilter::Linear)
    }

    /// The minification filter a renderer should apply: the stored
    /// filter when defined, else this crate's documented fallback
    /// [`MinFilter::LinearMipLinear`] (trilinear — the spec leaves
    /// the undefined case to the implementation).
    pub fn effective_min_filter(&self) -> MinFilter {
        self.min_filter.unwrap_or(MinFilter::LinearMipLinear)
    }

    /// Whether sampling through this state reads mipmap levels —
    /// [`MinFilter::uses_mipmaps`] of the
    /// [effective](Self::effective_min_filter) minification filter
    /// (an undefined filter falls back to trilinear, which does).
    /// Tells an importer whether the texture needs its mip chain.
    pub fn uses_mipmaps(&self) -> bool {
        self.effective_min_filter().uses_mipmaps()
    }

    /// Apply the per-axis wrap modes to one UV coordinate —
    /// [`wrap_s`](Self::wrap_s) on `u`, [`wrap_t`](Self::wrap_t) on
    /// `v` — mapping it into the sampled `[0, 1]²` square. See
    /// [`WrapMode::wrap`].
    pub fn wrap_uv(&self, uv: [f32; 2]) -> [f32; 2] {
        [self.wrap_s.wrap(uv[0]), self.wrap_t.wrap(uv[1])]
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
