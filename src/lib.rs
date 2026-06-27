//! Pure-Rust 3D scene + mesh typed model.
//!
//! This crate is the foundation for the OxideAV 3D format ecosystem
//! (`oxideav-stl`, `oxideav-obj`, `oxideav-gltf`, future
//! `oxideav-fbx` / `oxideav-usd`). It defines a single PBR-aligned
//! data model that every format crate can decode into and encode
//! from, and it exposes [`Mesh3DDecoder`] / [`Mesh3DEncoder`] traits
//! plus a [`Mesh3DRegistry`] that mirrors the codec-registry pattern
//! from `oxideav-core`.
//!
//! The model is aligned with **glTF 2.0** as the spec-stable common
//! denominator (Khronos KHR-public spec, royalty-free). This means:
//!
//! * Right-handed coordinates, Y-up, -Z forward as the canonical
//!   default. Files coming from Z-up formats (STL/OBJ Wavefront) set
//!   [`Scene3D::up_axis`] to [`Axis::PosZ`] without any geometry
//!   re-orientation; downstream renderers honour the metadata.
//! * Quaternions are stored xyzw to match glTF.
//! * Tangents carry handedness in `w` (`±1.0`).
//! * Materials are metallic-roughness PBR with `base_color`,
//!   `metallic`, `roughness`, `normal`, `occlusion`, and `emissive`
//!   slots (see [`Material`]), refined by a typed [`MaterialExt`]
//!   surface for the ratified KHR dielectric extensions (emissive
//!   strength, IOR, specular, unlit).
//!
//! Round 1 ships the type model + the trait surface only. No format
//! support — that lands in dedicated sibling crates.
//!
//! Round 2 adds:
//!
//! * [`AssetSource`] trait — lazy reference to a binary asset
//!   ([`Texture`] or [`AudioSource`]) so massive scenes (USDZ
//!   archives in the hundreds of MB) don't have to be materialised
//!   in RAM. Supports optional `raw_storage()` pass-through so a
//!   USDZ → USDZ converter can copy deflated payloads verbatim
//!   instead of inflate + re-deflate.
//! * `Audio*` types — [`AudioSource`], [`AudioEmitter`],
//!   [`SpatialAudio`], [`AuralMode`], [`DistanceModel`]. Aligned
//!   with USD `UsdMediaSpatialAudio` + glTF `KHR_audio_emitter`.
//! * [`ImageData::Source`] replaces the round-1
//!   `ImageData::Encoded { mime, bytes }` variant. Migration: wrap
//!   bytes in [`InMemoryAsset`], `Arc::new()` it, pass into
//!   `ImageData::Source(...)` (or keep using the
//!   [`Texture::from_encoded`] convenience constructor — the
//!   signature is unchanged).
//!
//! ## Standalone build
//!
//! `oxideav-core` is gated behind the default-on `registry` cargo
//! feature. Drop the framework dependency entirely with:
//!
//! ```toml
//! oxideav-mesh3d = { version = "0.0", default-features = false }
//! ```
//!
//! The typed model and trait definitions remain available — only the
//! [`Mesh3DRegistry`] glue is feature-gated, and the [`Error`] alias
//! resolves to a crate-local enum instead of `oxideav_core::Error`.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod animation;
pub mod asset;
pub mod audio;
pub mod bvh;
pub mod camera;
pub mod combine;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod extrude;
pub mod instance_bvh;
pub mod light;
pub mod material;
pub mod mesh;
pub mod optimize;
pub mod ray;
pub mod registry;
pub mod scene;
pub mod simplify;
pub mod skin;
pub mod smooth;
pub mod texture;

pub use animation::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, Interpolation, SampledValue,
};
pub use asset::{AssetSource, InMemoryAsset, RawStorage};
pub use audio::{
    AudioData, AudioEmitter, AudioEmitterId, AudioSource, AudioSourceId, AuralMode, DistanceModel,
    SpatialAudio,
};
pub use bvh::{Bvh, BvhNode};
pub use camera::Camera;
pub use decoder::Mesh3DDecoder;
pub use encoder::Mesh3DEncoder;
pub use error::{Error, Result};
pub use extrude::Profile2D;
pub use instance_bvh::{Instance, InstanceBvh, InstanceBvhNode};
pub use light::Light;
pub use material::{
    AlphaMode, Anisotropy, Clearcoat, Iridescence, Material, MaterialExt, Sheen, Specular,
    TextureRef, Transmission, Volume,
};
pub use mesh::{
    EdgeManifoldReport, Indices, Mesh, MorphTarget, MorphedAttributes, OrientationReport,
    Primitive, Topology, TopologySummary,
};
pub use optimize::{simulate_cache, CacheStats, DEFAULT_CACHE_SIZE};
pub use ray::{Ray, RayHit};
pub use registry::Mesh3DRegistry;
pub use scene::{
    Axis, BoundingBox, CameraId, LightId, MaterialId, MeshId, Node, NodeId, Scene3D, SceneRayHit,
    SkeletonId, SkinId, TextureId, Transform, Unit, ValidationError,
};
pub use skin::{Skeleton, Skin};
pub use smooth::SmoothOptions;
pub use texture::{ImageData, MagFilter, MinFilter, Sampler, Texture, WrapMode};
