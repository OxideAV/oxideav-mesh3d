//! [`Mesh3DDecoder`] trait — bytes-in, [`Scene3D`]-out.
//!
//! Sibling format crates (`oxideav-stl`, `oxideav-obj`,
//! `oxideav-gltf`, ...) implement this trait against their own
//! parser state and call [`Mesh3DRegistry::register_decoder`](crate::Mesh3DRegistry::register_decoder)
//! at framework `register()` time.

use crate::{Result, Scene3D};

/// Decode a 3D-format byte buffer into a [`Scene3D`].
///
/// `&mut self` is intentional — many parsers carry working buffers
/// or option state across calls. Implementations should be safe to
/// reuse across consecutive `decode` calls (i.e. internal buffers
/// get cleared at the start of each call).
pub trait Mesh3DDecoder {
    /// Parse `bytes` into a [`Scene3D`].
    fn decode(&mut self, bytes: &[u8]) -> Result<Scene3D>;
}
