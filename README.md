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

Round 3 lands a cross-format roundtrip suite that exercises the
`Mesh3DRegistry` surface against the four published siblings
(`oxideav-stl 0.0`, `oxideav-obj 0.0`, `oxideav-gltf 0.0`,
`oxideav-usdz 0.0`) consumed as `[dev-dependencies]`:

- `tests/cross_format_roundtrip.rs` — 16 tests covering
  typed-fixture → encoder → decoder fidelity for STL binary, STL
  ASCII, OBJ, glTF GLB, glTF JSON, plus six format-to-format chains
  (stl ↔ obj, stl ↔ gltf, obj ↔ gltf). Side-channel checks confirm
  that OBJ `usemtl` round-trips through `Primitive::extras`, glTF
  preserves `base_color` literally, STL drops materials but keeps
  geometry, and `ImageData::Source(InMemoryAsset)` survives a glTF
  JSON `data:` URI inline. Encoder rejection of `Lines` topology by
  STL is asserted as a typed `Err`, not a panic.
- `tests/registry_lookup.rs` — 18 tests for the
  `Mesh3DRegistry` resolution surface after every sibling crate's
  `register(&mut reg)` helper has run: extension and format-id
  routes for every codec, case-insensitivity contract, overwrite
  semantics on repeated `register_decoder` / `register_encoder`
  calls, unknown-key behaviour, reverse lookup
  (`decoder_extensions(format_id)`), and the `Default::default()` /
  `new()` parity.

Round 4 lands a 15-test extension to the cross-format matrix in
`tests/extras_and_skinning_coverage.rs`:

- **Skinning data primitive-level** — `Primitive::joints` +
  `Primitive::weights` survive a glTF round-trip bit-exact (JOINTS_0 +
  WEIGHTS_0 accessor channels); STL + OBJ silently drop them. The
  scene-level `skins` / `skeletons` / `animations` arrays are *not*
  yet round-tripped by glTF 0.0.0 — two tests pin that gap so a
  future encoder upgrade flips them from "drops" to "survives" in
  the same commit.
- **Multi-primitive vertex pool dedup (OBJ)** — two primitives
  sharing four corners pool to four `v` lines in the OBJ output;
  triangle count survives; glTF keeps the per-primitive partition
  distinct.
- **Multi-material binding** — two primitives × two materials emit
  two `usemtl` directives in OBJ; both names round-trip via
  `Primitive::extras["obj:usemtl"]`. glTF preserves per-primitive
  `material` index.
- **Cross-format extras audit** — pins `stl:source` idempotence,
  STL `up_axis = PosZ` / `unit = Millimetres`, OBJ
  `up_axis = PosY` / `unit = Metres`, gltf → OBJ scene-extras drop,
  and gltf primitive-extras JSON value preservation.

Round 5 lands two test extensions:

- `tests/multi_material_pool_stress.rs` (12 tests) — five-material
  bindings, cross-mesh OBJ vertex-pool dedup, multi-mesh hierarchy,
  material aliasing. Confirms the OBJ encoder's global vertex pool
  collapses across separate `Mesh`es (not just within one), the
  glTF encoder preserves five distinct `material` indices on five
  primitives, and STL flattens hierarchy into a flat triangle list.
- `tests/encoder_options_roundtrip.rs` (18 tests) — pins every
  configuration knob the published 0.0.0 sibling crates expose:
  `StlEncoder` constructor parity (`new_binary` / `new(StlFormat::Binary)`
  / `default`) + `format()` getter + `solid` and 80-byte-header
  byte markers, `ObjEncoder::with_mtl_basename` directive injection
  + `obj:mtllibs` round-trip into `Scene3D::extras`, `MtlEncoder`
  ↔ `parse_mtl` / `serialize_mtl` parity, `GltfEncoder` flavour
  selection (`new` / `with_output(Glb)` / `default()`), `json_encoder()`
  helper parity with `with_output(JsonEmbedded)`. Byte-equality is
  not asserted on the glTF side because the encoder serialises a
  `HashMap` (per-primitive `attributes`) whose iteration order
  varies per process invocation; we test flavour-id parity +
  decode-equivalence instead.

Round 6 lands typed morph-target fields (BREAKING, pre-v0.1):

- `MorphTarget { position, normal, tangent }` — typed delta-buffer
  struct for one named morph pose per glTF 2.0 §3.7.2.2. Each slot
  is `Option<Vec<[f32; 3]>>` (the TANGENT slot is xyz-only — the
  base TANGENT's handedness `w` is not morphed per spec).
- `Primitive::targets: Vec<MorphTarget>` for the per-pose roster.
- `Mesh::weights: Vec<f32>` for the static default morph blend (an
  `AnimationProperty::MorphWeights` channel overrides at runtime).
- New builder helper `Mesh::with_weights`. The forward-compatible
  construction style is `Primitive::new(Topology)` /
  `Mesh::new(name)` + `with_*` builders + per-field assignment;
  literal `Primitive { … }` / `Mesh { … }` construction still
  compiles in this round but must populate the two new fields.
  `#[non_exhaustive]` is deferred to round 7.

Sketch:

```rust
use oxideav_mesh3d::{Mesh, MorphTarget, Primitive, Topology};

let mut prim = Primitive::new(Topology::Triangles);
prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
// One morph target ("smile") with POSITION + NORMAL deltas.
prim.targets.push(MorphTarget {
    position: Some(vec![[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]),
    normal: Some(vec![[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]),
    tangent: None,
});

let mesh = Mesh::new(Some("face".to_owned()))
    .with_primitive(prim)
    .with_weights(vec![0.0_f32]);  // static blend: target 0 disabled
```

## Round 7 candidates

- USDZ row of the cross-format matrix — needs `oxideav-usdz` to
  publish 0.0.1 (which lands its first encoder); 0.0.0 ships a
  decoder only. Once the encoder is on crates.io, add stl→usdz,
  obj→usdz, gltf→usdz pairs to `cross_format_roundtrip.rs`.
- glTF consumer migration off the `__morph_targets` /
  `__mesh_weights` extras sentinels onto the new typed
  `Primitive::targets` / `Mesh::weights` fields (the round-6 typed
  surface lands here; gltf encoder/decoder consumes it once mesh3d
  publishes 0.0.2).
- glTF scene-level `skins` + `skeletons` + `animations` array
  serialisation — the round-4 pinning tests will flip from "drops"
  to "survives" in the same commit. Producer-side change in
  `oxideav-gltf`; consumer-side flip-and-republish here.
- KHR extension surface (`KHR_materials_emissive_strength`,
  `KHR_materials_unlit`, `KHR_lights_punctual`,
  `KHR_audio_emitter`) on top of the existing glTF round-trip.
- Per-face material binding via `UsdGeomSubset` (USDZ decoder
  side) + glTF KHR_materials_variants for per-LOD or per-instance
  swapping.

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
