# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Round 4 (skinning + multi-primitive + extras audit coverage)

- `tests/extras_and_skinning_coverage.rs` — 15 tests grouped into
  four sections that extend round 3's matrix without overlapping it:
  - **Skinning data primitive-level survival** — gltf round-trips
    `Primitive::joints` + `Primitive::weights` bit-exact (JOINTS_0 +
    WEIGHTS_0 accessor channels); STL + OBJ silently drop them
    (no in-format surface). Two pinning tests assert that the
    `Scene3D::skins` / `Scene3D::skeletons` / `Scene3D::animations`
    arrays are dropped by gltf 0.0.0 — when the encoder gains
    skin-array + animation serialisation the assertions flip in the
    same commit that lifts the gap.
  - **Multi-primitive vertex pool dedup (OBJ)** — six raw vertex
    positions across two primitives sharing four physical corners
    pool to four `v` lines in the OBJ output (intra+inter-primitive
    interning). Triangle count survives the round-trip; gltf keeps
    the two primitives distinct in the round-tripped mesh.
  - **Multi-material binding** — two primitives, two materials emit
    two distinct `usemtl` directives in OBJ output and round-trip
    both names back through `Primitive::extras["obj:usemtl"]`. gltf
    preserves the per-primitive `material` index; resolved names
    survive.
  - **Cross-format extras audit** — pins which `extras` keys a
    downstream conversion pipeline can rely on:
    `stl:source = "binary" | "ascii"` is idempotent across STL → STL
    round-trips; STL decoder stamps `up_axis = PosZ` +
    `unit = Millimetres`; OBJ decoder stamps `up_axis = PosY` +
    `unit = Metres`; gltf-side `Scene3D::extras` are silently
    dropped on a gltf → OBJ pass (OBJ has no scene-level free-form
    key surface); arbitrary JSON values on `Primitive::extras`
    survive a gltf round-trip intact.

### Added — Round 3 (cross-format conversion roundtrip suite)

- `tests/cross_format_roundtrip.rs` — 16 tests exercising the
  `Mesh3DRegistry` decode → re-encode matrix end-to-end across the
  four sibling format crates (`oxideav-stl`, `oxideav-obj`,
  `oxideav-gltf`, `oxideav-usdz`) consumed as `[dev-dependencies]`.
  Coverage:
  - **Typed → format → typed** for all five
    encoder targets: STL binary, STL ASCII, OBJ, glTF GLB,
    glTF JSON-embedded.
  - **Format → format chain**: stl→obj, stl→gltf, obj→stl, obj→gltf,
    gltf→stl, gltf→obj — each authored as a one-triangle (or
    two-triangle quad) `Scene3D`, encoded → decoded → re-encoded →
    re-decoded, with positions resolved through the index buffer
    when present so the assert is codec-layout-agnostic.
  - **Side-channel preservation**: OBJ `usemtl` round-trips through
    `Primitive::extras["obj:usemtl"]`; glTF preserves PBR
    `base_color` literally; STL silently drops the material binding
    while keeping geometry; `ImageData::Source(InMemoryAsset)`
    survives a glTF JSON `data:` URI round-trip.
  - **Encoder rejection**: STL surfaces an `Err` (not a panic) when
    asked to encode a `Lines`-topology primitive.
- `tests/registry_lookup.rs` — 18 tests covering the
  `Mesh3DRegistry` resolution surface after the four sibling crates'
  `register(&mut reg)` helpers run:
  - Every canonical extension (`.stl`, `.obj`, `.mtl`, `.gltf`,
    `.glb`, `.usdz`) and format id resolves through both
    `decoder_for_*` and `encoder_for_*` lookups (USDZ is
    decoder-only, asserted explicitly).
  - Case-insensitivity contract: `STL`, `Stl`, `GLTF`, `Glb`, `OBJ`
    all hit; `decoder_for_format("GlTf")` resolves.
  - `register_decoder` / `register_encoder` overwrite semantics
    when called twice with the same format id (last writer wins),
    plus the extension-route remap when the second call narrows the
    extension list.
  - Reverse lookup via `decoder_extensions("gltf")` returning both
    `[gltf, glb]`, encoder reverse lookup for `gltf`/`glb` as
    separate format ids, USDZ encoder absence.
  - Unknown extension / format id → `None`; empty-registry contract;
    `Default::default()` parity with `new()`; `Debug` impl
    enumerates the registered keysets.
- `[dev-dependencies]` adds the four format crates at version
  `"0.0"` so future patch releases don't ripple. A
  `[patch.crates-io] oxideav-mesh3d = { path = "." }` block keeps
  the test-time dependency graph single-version (the format crates'
  own `oxideav-mesh3d = "0.0"` runtime dep otherwise resolves to
  the published copy alongside our local one, producing two
  incompatible `Scene3D` types).

## [0.0.1](https://github.com/OxideAV/oxideav-mesh3d/compare/v0.0.0...v0.0.1) - 2026-05-10

### Other

- Round 2: AssetSource trait + audio types (BREAKING, pre-publish)

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
