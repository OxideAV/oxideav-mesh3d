//! Lazy-access asset trait for textures, audio, and any other
//! large blob a 3D scene can carry.
//!
//! ## Why a trait instead of `Vec<u8>`
//!
//! Round 1 ran with `ImageData::Encoded { mime, bytes: Vec<u8> }`,
//! which materialises the entire payload up-front. That's fine for a
//! 64 KiB icon but pathological for the actual workloads the model
//! has to carry — USDZ archives are routinely hundreds of megabytes,
//! glTF GLB binary chunks pin tens of MB of textures, FBX embedded
//! media can be larger still. Eagerly copying every blob into a
//! `Vec` triples peak memory (file → mmap → owned `Vec` →
//! decoded), and worse it forecloses on the obvious optimisation:
//! when a converter is asked to write the SAME container scheme it
//! read from (USDZ → USDZ, GLB → GLB), the deflated/raw payload can
//! pass through unchanged with no decode + re-encode round-trip.
//!
//! [`AssetSource`] solves both:
//!
//! * `open()` returns a streaming reader, so large assets can be
//!   chunked into the encoder without ever holding the full payload
//!   in RAM. Small callers `.read_to_end()` and move on.
//! * `raw_storage()` is the optional pass-through hint. A USDZ
//!   reader exposes `RawStorage { scheme: "zip-deflate", bytes: ... }`
//!   for its embedded files; a USDZ writer that sees the same scheme
//!   on input copies the deflated bytes verbatim into its output ZIP
//!   instead of inflating + re-deflating. Crates that don't share a
//!   scheme transparently fall back to `open()`.
//!
//! ## Scheme names
//!
//! `RawStorage::scheme` is a free-form string; format crates should
//! agree on canonical names so reader/writer pairs can recognise
//! each other across crate boundaries. Conventions:
//!
//! * `"zip-deflate"` — bytes are RFC 1951 deflate-compressed,
//!   uncompressed size in `uncompressed_size`.
//! * `"zip-stored"` — bytes are the uncompressed payload as stored
//!   in a ZIP container (no transform).
//! * `"usdc-crate"` — Pixar USD binary crate file payload.
//! * `"tar-stored"` — uncompressed payload of a tar entry.
//!
//! New schemes are added by convention; downstream pairs that don't
//! recognise a scheme just ignore `raw_storage()` and use `open()`.

use std::io::Result as IoResult;

/// Re-export of [`oxideav_core::ReadSeek`] so callers don't have to
/// pull in the framework crate directly to name the [`AssetSource::open`]
/// return type.
#[cfg(feature = "registry")]
pub use oxideav_core::ReadSeek;

/// Streaming reader trait alias used when the `registry` feature is
/// off. Mirrors [`oxideav_core::ReadSeek`] so the [`AssetSource`]
/// signature is identical with or without the feature.
#[cfg(not(feature = "registry"))]
pub trait ReadSeek: std::io::Read + std::io::Seek + Send {}
#[cfg(not(feature = "registry"))]
impl<T: std::io::Read + std::io::Seek + Send> ReadSeek for T {}

/// Lazy reference to a binary asset payload (image, audio, anything
/// the type model carries by reference).
///
/// Implementors are typically thin wrappers around a file handle, a
/// memory-mapped region, or a slice into a larger ZIP archive.
/// [`InMemoryAsset`] is the trivial owning implementation provided
/// for tests and small embedded payloads.
pub trait AssetSource: std::fmt::Debug + Send + Sync {
    /// MIME hint, if known. May be `None` even when the underlying
    /// bytes are well-formed — the loader is allowed to leave format
    /// detection to the consumer.
    fn mime(&self) -> Option<&str>;

    /// Total uncompressed size in bytes, if it can be determined
    /// without consuming the asset. Reader implementations backed by
    /// a `Cursor` over a slice always know this; streaming sources
    /// over an HTTP body may not.
    fn size_hint(&self) -> Option<u64>;

    /// Open a streaming reader positioned at offset 0.
    ///
    /// Callers handling small assets typically call `.read_to_end()`
    /// once; callers handling large assets read in chunks. The
    /// returned reader is owned and `Send`, so it can be moved into a
    /// worker thread.
    fn open(&self) -> IoResult<Box<dyn ReadSeek + Send>>;

    /// Optional pass-through hint exposing the asset's underlying
    /// stored bytes plus the scheme they were stored under.
    ///
    /// Default returns `None` — most implementations only support the
    /// streaming `open()` path and let the consumer decode lazily.
    /// Format crates that ARE backed by a recognisable on-disk
    /// representation (a ZIP entry, a USD crate-file blob) override
    /// this to enable scheme-matched zero-copy passthrough.
    ///
    /// Consumer pseudocode:
    ///
    /// ```ignore
    /// match source.raw_storage() {
    ///     Some(rs) if rs.scheme == "zip-deflate" => {
    ///         // copy the already-deflated bytes straight into the
    ///         // output ZIP, skipping inflate + re-deflate.
    ///         out_zip.write_deflated(rs.bytes, rs.uncompressed_size);
    ///     }
    ///     _ => {
    ///         // fallback: stream through the regular open() path.
    ///         let mut r = source.open()?;
    ///         let mut buf = Vec::new();
    ///         r.read_to_end(&mut buf)?;
    ///         out.write_payload(&buf);
    ///     }
    /// }
    /// ```
    fn raw_storage(&self) -> Option<RawStorage<'_>> {
        None
    }
}

/// Result of [`AssetSource::raw_storage`]: the asset's stored bytes
/// (in their on-disk form) plus the scheme they were stored under.
///
/// `bytes` may be compressed depending on `scheme`; `uncompressed_size`
/// is the post-decode length when the scheme tracks it (e.g. ZIP
/// stores it in the local file header).
#[derive(Debug)]
pub struct RawStorage<'a> {
    /// Canonical scheme identifier — see the module docs for the
    /// recommended set of names.
    pub scheme: &'a str,
    /// Bytes as stored under `scheme` (may be compressed).
    pub bytes: &'a [u8],
    /// Decoded length when the scheme records it.
    pub uncompressed_size: Option<u64>,
}

/// Trivial in-memory [`AssetSource`] backed by an owned `Vec<u8>`.
///
/// Construct one when a caller already has the bytes on hand (a unit
/// test, a small embedded icon, a generated procedural payload) and
/// just needs an `Arc<dyn AssetSource>` to pass into the type model.
/// `raw_storage()` is unimplemented because the bytes are stored
/// uncompressed under no particular container scheme; format crates
/// that want pass-through should expose their own `AssetSource` impl
/// against the original ZIP / USDZ / GLB payload.
#[derive(Clone, Debug)]
pub struct InMemoryAsset {
    pub mime: Option<String>,
    pub bytes: Vec<u8>,
}

impl InMemoryAsset {
    /// Build from MIME + owned byte vector. Both fields stay
    /// directly accessible so callers can mutate after construction.
    pub fn new(mime: impl Into<Option<String>>, bytes: Vec<u8>) -> Self {
        Self {
            mime: mime.into(),
            bytes,
        }
    }
}

impl AssetSource for InMemoryAsset {
    fn mime(&self) -> Option<&str> {
        self.mime.as_deref()
    }

    fn size_hint(&self) -> Option<u64> {
        Some(self.bytes.len() as u64)
    }

    fn open(&self) -> IoResult<Box<dyn ReadSeek + Send>> {
        // Cursor::clone of the bytes is unavoidable here — the trait
        // contract gives every caller their own positionable reader,
        // so two concurrent `open()`s on one InMemoryAsset must each
        // get an independent cursor.
        Ok(Box::new(std::io::Cursor::new(self.bytes.clone())))
    }
    // raw_storage stays at the default `None` — see the type docs.
}
