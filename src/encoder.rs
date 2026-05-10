//! [`Mesh3DEncoder`] trait — [`Scene3D`]-in, bytes-out.
//!
//! Symmetric to [`Mesh3DDecoder`](crate::Mesh3DDecoder); same
//! lifecycle expectations.

use crate::{Result, Scene3D};

/// Serialise a [`Scene3D`] into a 3D-format byte buffer.
///
/// `&mut self` lets the encoder hold scratch buffers or output
/// options between calls.
pub trait Mesh3DEncoder {
    /// Serialise `scene` to bytes in the encoder's format.
    fn encode(&mut self, scene: &Scene3D) -> Result<Vec<u8>>;
}
