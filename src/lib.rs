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
//!   slots (see [`Material`]).
//!
//! Round 1 ships the type model + the trait surface only. No format
//! support — that lands in dedicated sibling crates.
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
pub mod camera;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod light;
pub mod material;
pub mod mesh;
pub mod registry;
pub mod scene;
pub mod skin;
pub mod texture;

pub use animation::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationTarget,
    AnimationValues, Interpolation,
};
pub use camera::Camera;
pub use decoder::Mesh3DDecoder;
pub use encoder::Mesh3DEncoder;
pub use error::{Error, Result};
pub use light::Light;
pub use material::{AlphaMode, Material, TextureRef};
pub use mesh::{Indices, Mesh, Primitive, Topology};
pub use registry::Mesh3DRegistry;
pub use scene::{
    Axis, CameraId, LightId, MaterialId, MeshId, Node, NodeId, Scene3D, SkeletonId, SkinId,
    TextureId, Transform, Unit,
};
pub use skin::{Skeleton, Skin};
pub use texture::{ImageData, MagFilter, MinFilter, Sampler, Texture, WrapMode};
