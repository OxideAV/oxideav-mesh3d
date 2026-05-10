# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Round 2 (asset trait + audio types)

- `AssetSource` trait — `Send + Sync + Debug` lazy reference to a
  binary asset (texture, audio, anything blob-shaped). Methods:
  - `mime() -> Option<&str>`
  - `size_hint() -> Option<u64>` — total uncompressed length when
    known without consuming the asset
  - `open() -> io::Result<Box<dyn ReadSeek + Send>>` — streaming
    reader, callers `read_to_end()` for small assets or chunk for
    large ones
  - `raw_storage() -> Option<RawStorage>` — optional pass-through
    hint exposing the asset's underlying stored bytes plus a
    scheme identifier (`"zip-deflate"`, `"zip-stored"`,
    `"usdc-crate"`, `"tar-stored"`, ...). When a writer's output
    container scheme matches, it can copy the bytes verbatim and
    skip decode + re-encode (USDZ → USDZ pass-through).
- `RawStorage<'a> { scheme: &str, bytes: &[u8], uncompressed_size: Option<u64> }`
  — payload + scheme identifier returned by `raw_storage()`.
- `InMemoryAsset { mime, bytes }` — trivial owning implementor of
  `AssetSource` for unit tests and small embedded payloads.
- Audio type surface (aligned with USD `UsdMediaSpatialAudio` +
  glTF `KHR_audio_emitter`):
  - `AudioData { Embedded(AudioFrame), Source(Arc<dyn AssetSource>),
    External { uri, mime } }` — analogue of `ImageData` for audio.
    `Embedded` is feature-gated behind `registry`.
  - `AudioSource { name, data, extras }` — owned audio asset.
  - `AudioEmitter { name, source, gain, looping, auto_play, spatial,
    extras }` — in-scene playback instance. Defaults: `gain = 1.0`,
    `looping = false`, `auto_play = false`, `spatial = None`
    (global non-positional source).
  - `SpatialAudio { aural_mode, cone_inner_angle, cone_outer_angle,
    cone_outer_gain, min_distance, max_distance, rolloff_factor,
    distance_model }` — positional rendering parameters. Defaults:
    `aural_mode = SpatialNonAcoustic`, both cone angles `2π`
    (omnidirectional), `cone_outer_gain = 0.0`, `min_distance = 1.0`
    metres, `max_distance = 10000.0` metres, `rolloff_factor = 1.0`,
    `distance_model = Inverse`.
  - `AuralMode { SpatialNonAcoustic, SpatialAcoustic }` — flag
    carried through to the renderer; type model doesn't apply HRTF
    itself.
  - `DistanceModel { Linear, Inverse, Exponential }` — WebAudio /
    OpenAL attenuation curves.
  - `AudioSourceId(u32)` and `AudioEmitterId(u32)` newtypes.
- `Scene3D::audio_sources: Vec<AudioSource>` and
  `Scene3D::audio_emitters: Vec<AudioEmitter>` arenas, plus
  `add_audio_source` / `add_audio_emitter` push-and-id helpers and
  `audio_source(id)` / `audio_emitter(id)` lookups.
- `Node::audio_emitter: Option<AudioEmitterId>` field plus
  `Node::with_audio_emitter` builder method.
- `oxideav_core::ReadSeek` re-exported from `asset::ReadSeek` so
  callers don't need a direct framework dependency to name the
  `AssetSource::open` return type. Standalone build supplies a
  crate-local trait alias with the same shape.

### Changed — BREAKING (round 2, pre-v0.1)

- **`ImageData::Encoded { mime, bytes }` removed**, replaced by
  **`ImageData::Source(Arc<dyn AssetSource>)`**. Migration:
  ```rust
  // Before:
  ImageData::Encoded { mime: "image/png".into(), bytes: payload }
  // After:
  ImageData::Source(Arc::new(InMemoryAsset {
      mime: Some("image/png".into()),
      bytes: payload,
  }))
  ```
  The `Texture::from_encoded(mime, bytes)` constructor signature is
  unchanged — it now wraps in an `InMemoryAsset` internally — so
  most call sites that used the constructor compile unchanged.
- **`Node` gains a non-`Option`-defaulted `audio_emitter` field.**
  Callers constructing `Node` literally (`Node { name, transform,
  ... }` without `..Node::new()`) need to add
  `audio_emitter: None`. Builder / `Node::new()` paths are
  unaffected.
- **`Scene3D` gains `audio_sources` + `audio_emitters` fields.**
  Same caveat: literal struct construction must populate them.

### Design intent

- Lazy access: huge scenes (USDZ archives in the hundreds of MB)
  shouldn't materialise every embedded blob in `Vec<u8>`. Format
  crates expose blobs through `AssetSource::open()` so consumers
  stream chunks on demand.
- Pass-through: a USDZ → USDZ (or GLB → GLB) converter that sees
  matching `raw_storage()` schemes copies the deflated payload
  verbatim instead of inflate + decode + re-encode + deflate.
- Trait sits in `oxideav-mesh3d` for now to keep the type model
  self-contained; can promote to `oxideav-core` in a future round
  if more crates want it.

### Added — Round 1 (initial bootstrap)
  - `Scene3D` top-level container with `nodes`, `roots`, `meshes`,
    `materials`, `textures`, `skeletons`, `skins`, `animations`,
    `cameras`, `lights`, `up_axis`, `front_axis`, `unit`, and a
    free-form `extras: HashMap<String, serde_json::Value>` round-trip
    side-channel. Defaults align with glTF 2.0: Y-up, -Z forward,
    metres.
  - `Node` + `Transform { Matrix, Trs }` with best-effort matrix↔TRS
    decompose (`from_matrix` / `to_matrix`).
  - `Mesh` / `Primitive` / `Topology` (Triangles, TriangleStrip,
    TriangleFan, Lines, LineStrip, LineLoop, Points) /
    `Indices { U16, U32 }` with multi-channel UVs and vertex colours
    plus optional skinning joint indices + weights.
  - `Material` — full glTF 2.0 metallic-roughness PBR slots
    (`base_color`, `metallic`, `roughness`, `normal`, `occlusion`,
    `emissive`) plus `AlphaMode { Opaque, Mask{cutoff}, Blend }` and
    `double_sided`.
  - `Texture` / `ImageData { Embedded(VideoFrame), External, Encoded }`
    / `Sampler` with mag/min filters and wrap modes; the `Embedded`
    variant is feature-gated behind the default-on `registry` feature.
  - `Skeleton` (joint nodes + inverse-bind matrices) + `Skin`.
  - `Animation` / `AnimationChannel` / `AnimationSampler` /
    `AnimationProperty { Translation, Rotation, Scale, MorphWeights }`
    / `Interpolation { Step, Linear, CubicSpline }`.
  - `Camera { Perspective, Orthographic }` and
    `Light { Directional, Point, Spot }`.
  - `Mesh3DDecoder` / `Mesh3DEncoder` traits + `Mesh3DRegistry`
    (case-insensitive extension and format-id lookup).
  - Default-on `registry` cargo feature gates the `oxideav-core`
    dependency. Standalone build (`--no-default-features`) keeps the
    typed model + traits with a crate-local `Error` / `Result` alias.
