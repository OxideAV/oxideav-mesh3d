//! [`Mesh3DRegistry`] — runtime lookup table from format-id /
//! file-extension to decoder/encoder factory.
//!
//! Mirrors `oxideav_core::CodecRegistry`: each entry stores a
//! format-id string (canonical name like `"gltf"` / `"stl"` /
//! `"obj"`), the file extensions that map to it, and a `Box<dyn Fn>`
//! factory that constructs a fresh decoder or encoder per call.
//!
//! Lookup is case-insensitive on the extension so callers can pass
//! either `"glb"` or `"GLB"` from a `Path::extension()` directly.

use std::collections::HashMap;
use std::fmt;

use crate::{Mesh3DDecoder, Mesh3DEncoder};

/// Factory closure that constructs a fresh decoder per call.
pub type DecoderFactory = Box<dyn Fn() -> Box<dyn Mesh3DDecoder> + Send + Sync>;

/// Factory closure that constructs a fresh encoder per call.
pub type EncoderFactory = Box<dyn Fn() -> Box<dyn Mesh3DEncoder> + Send + Sync>;

struct DecoderEntry {
    factory: DecoderFactory,
    extensions: Vec<String>,
}

struct EncoderEntry {
    factory: EncoderFactory,
    extensions: Vec<String>,
}

/// Runtime lookup table for 3D-format decoders + encoders.
pub struct Mesh3DRegistry {
    decoders: HashMap<String, DecoderEntry>,
    encoders: HashMap<String, EncoderEntry>,
    decoder_by_ext: HashMap<String, String>,
    encoder_by_ext: HashMap<String, String>,
}

impl fmt::Debug for Mesh3DRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mesh3DRegistry")
            .field("decoder_formats", &self.decoders.keys().collect::<Vec<_>>())
            .field("encoder_formats", &self.encoders.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Mesh3DRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self {
            decoders: HashMap::new(),
            encoders: HashMap::new(),
            decoder_by_ext: HashMap::new(),
            encoder_by_ext: HashMap::new(),
        }
    }

    /// Register a decoder for `format` reachable via any of `extensions`.
    ///
    /// Calling this twice with the same `format` overwrites the prior
    /// entry — last writer wins, mirroring the codec-registry policy.
    /// Extensions are stored case-folded.
    pub fn register_decoder(&mut self, format: &str, extensions: &[&str], factory: DecoderFactory) {
        let lc_format = format.to_ascii_lowercase();
        let stored_exts: Vec<String> = extensions.iter().map(|e| e.to_ascii_lowercase()).collect();
        for ext in &stored_exts {
            self.decoder_by_ext.insert(ext.clone(), lc_format.clone());
        }
        self.decoders.insert(
            lc_format,
            DecoderEntry {
                factory,
                extensions: stored_exts,
            },
        );
    }

    /// Register an encoder for `format` reachable via any of `extensions`.
    pub fn register_encoder(&mut self, format: &str, extensions: &[&str], factory: EncoderFactory) {
        let lc_format = format.to_ascii_lowercase();
        let stored_exts: Vec<String> = extensions.iter().map(|e| e.to_ascii_lowercase()).collect();
        for ext in &stored_exts {
            self.encoder_by_ext.insert(ext.clone(), lc_format.clone());
        }
        self.encoders.insert(
            lc_format,
            EncoderEntry {
                factory,
                extensions: stored_exts,
            },
        );
    }

    /// Resolve a decoder by file extension (case-insensitive).
    pub fn decoder_for_extension(&self, ext: &str) -> Option<Box<dyn Mesh3DDecoder>> {
        let key = ext.to_ascii_lowercase();
        let format = self.decoder_by_ext.get(&key)?;
        self.decoders.get(format).map(|e| (e.factory)())
    }

    /// Resolve an encoder by file extension (case-insensitive).
    pub fn encoder_for_extension(&self, ext: &str) -> Option<Box<dyn Mesh3DEncoder>> {
        let key = ext.to_ascii_lowercase();
        let format = self.encoder_by_ext.get(&key)?;
        self.encoders.get(format).map(|e| (e.factory)())
    }

    /// Resolve a decoder by canonical format id (case-insensitive).
    pub fn decoder_for_format(&self, format: &str) -> Option<Box<dyn Mesh3DDecoder>> {
        let key = format.to_ascii_lowercase();
        self.decoders.get(&key).map(|e| (e.factory)())
    }

    /// Resolve an encoder by canonical format id (case-insensitive).
    pub fn encoder_for_format(&self, format: &str) -> Option<Box<dyn Mesh3DEncoder>> {
        let key = format.to_ascii_lowercase();
        self.encoders.get(&key).map(|e| (e.factory)())
    }

    /// All registered decoder format ids.
    pub fn decoder_formats(&self) -> impl Iterator<Item = &str> {
        self.decoders.keys().map(String::as_str)
    }

    /// All registered encoder format ids.
    pub fn encoder_formats(&self) -> impl Iterator<Item = &str> {
        self.encoders.keys().map(String::as_str)
    }

    /// Extensions mapped to a given decoder format, if registered.
    pub fn decoder_extensions(&self, format: &str) -> Option<&[String]> {
        self.decoders
            .get(&format.to_ascii_lowercase())
            .map(|e| e.extensions.as_slice())
    }

    /// Extensions mapped to a given encoder format, if registered.
    pub fn encoder_extensions(&self, format: &str) -> Option<&[String]> {
        self.encoders
            .get(&format.to_ascii_lowercase())
            .map(|e| e.extensions.as_slice())
    }
}

impl Default for Mesh3DRegistry {
    fn default() -> Self {
        Self::new()
    }
}
