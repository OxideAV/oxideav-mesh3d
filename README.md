# oxideav-mesh3d

Pure-Rust 3D scene + mesh typed model.

The shared data model that every OxideAV 3D-format crate
(`oxideav-stl`, `oxideav-obj`, `oxideav-gltf`, future
`oxideav-fbx` / `oxideav-usd`) decodes into and encodes from. The
type model is aligned with **glTF 2.0** (Khronos KHR-public spec) as
the spec-stable common denominator: right-handed coordinates,
Y-up, -Z forward, metres, metallic-roughness PBR, xyzw quaternions.
Files coming from Z-up formats (STL/OBJ Wavefront) just set
[`Scene3D::up_axis`] to `Axis::PosZ` — the model stores the
orientation, no implicit re-projection happens.

Round 1 ships:

- `Scene3D` — top-level container holding `Vec`s of nodes, meshes,
  materials, textures, skeletons, skins, animations, cameras,
  lights, plus `up_axis` / `front_axis` / `unit` metadata and a
  free-form `extras: HashMap<String, serde_json::Value>` round-trip
  side-channel.
- `Node` + `Transform { Matrix, Trs }` with best-effort matrix↔TRS
  decompose.
- `Mesh` / `Primitive` / `Topology` (Triangles, TriangleStrip,
  TriangleFan, Lines, LineStrip, LineLoop, Points) / `Indices` (U16
  or U32). Multi-channel UVs and vertex colours; optional skinning
  joint indices + weights.
- `Material` — full glTF 2.0 metallic-roughness PBR slots
  (`base_color`, `metallic`, `roughness`, `normal`, `occlusion`,
  `emissive`) plus `AlphaMode { Opaque, Mask{cutoff}, Blend }` and
  `double_sided`.
- `Texture` / `ImageData { Embedded(VideoFrame), Source(Arc<dyn AssetSource>), External }`
  / `Sampler` with the usual mag/min filters and wrap modes. The
  `Source` variant lets format crates pass a lazy reader through
  the type model without materialising a `Vec<u8>` (round 2).
- `Skeleton` (joint nodes + inverse-bind matrices) + `Skin`
  binding to a mesh.
- `Animation` / `AnimationChannel` / `AnimationSampler` /
  `AnimationProperty { Translation, Rotation, Scale, MorphWeights }`
  / `Interpolation { Step, Linear, CubicSpline }`.
- `Camera { Perspective, Orthographic }` and `Light { Directional,
  Point, Spot }`.
- `Mesh3DDecoder` / `Mesh3DEncoder` traits + `Mesh3DRegistry`
  (case-insensitive extension lookup) — mirrors the codec-registry
  pattern from `oxideav-core`.

Round 2 adds (still pre-publish, BREAKING vs round 1):

- `AssetSource` trait — `Send + Sync + Debug` lazy reference to a
  binary asset (texture, audio, anything blob-shaped). `open()`
  returns a streaming reader; optional `raw_storage()` exposes
  the asset's stored bytes + a scheme identifier so a writer
  targeting the same scheme (USDZ → USDZ, GLB → GLB) can pass
  the payload through without re-encoding.
- `RawStorage<'a> { scheme, bytes, uncompressed_size }` and
  `InMemoryAsset` (trivial owning impl).
- Audio surface — `AudioSource`, `AudioEmitter`, `SpatialAudio`,
  `AuralMode { SpatialNonAcoustic, SpatialAcoustic }`,
  `DistanceModel { Linear, Inverse, Exponential }`. Aligned with
  USD `UsdMediaSpatialAudio` + glTF `KHR_audio_emitter`.
  `Scene3D` gains `audio_sources` + `audio_emitters` arenas;
  `Node` gains `audio_emitter: Option<AudioEmitterId>`.
- BREAKING: `ImageData::Encoded { mime, bytes }` removed in favour
  of `ImageData::Source(Arc<dyn AssetSource>)`. Migration: wrap
  bytes in `InMemoryAsset { mime, bytes }`. The
  `Texture::from_encoded(mime, bytes)` helper signature is
  unchanged — it now wraps internally.

No format support yet; sibling crates (`oxideav-stl`,
`oxideav-obj`, `oxideav-gltf`) plug in via `Mesh3DRegistry` once
this crate is published.

## Round 3 candidates

- `oxideav-stl` (binary + ASCII) — STL is the smallest realistic
  consumer of the type model and validates the encoder side
  immediately.
- `oxideav-obj` + Wavefront MTL — exercises the multi-material
  primitive split + the `Material::extras` round-trip path.
- `oxideav-gltf` (JSON + GLB) — proves the type model round-trips
  losslessly against its design source-of-truth, including the
  `KHR_audio_emitter` extension against the new audio types.
- `oxideav-usdz` — first real consumer of the `raw_storage()`
  pass-through path.
- KHR extension surface (`KHR_materials_emissive_strength`,
  `KHR_materials_unlit`, `KHR_lights_punctual`,
  `KHR_audio_emitter`) once `oxideav-gltf` lands.

## Standalone build

`oxideav-core` is gated behind the default-on `registry` cargo
feature. Drop the framework dependency entirely with:

```toml
oxideav-mesh3d = { version = "0.0", default-features = false }
```

The typed model and trait definitions stay available — only the
embedded `VideoFrame` / `AudioFrame` variants
(`ImageData::Embedded` / `AudioData::Embedded`) disappear, and the
`Error` / `Result` aliases resolve to a crate-local enum instead
of `oxideav_core::Error`. `AssetSource::open()` returns a
crate-local `ReadSeek` trait alias with the same shape as
`oxideav_core::ReadSeek`, so the trait surface is identical
either way.

## License

MIT — see `LICENSE`.
