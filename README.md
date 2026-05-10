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
- `Texture` / `ImageData { Embedded(VideoFrame), External, Encoded }`
  / `Sampler` with the usual mag/min filters and wrap modes.
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

No format support this round; sibling crates (`oxideav-stl`,
`oxideav-obj`, `oxideav-gltf`) plug in via `Mesh3DRegistry` once
this crate is published.

## Round 2 candidates

- `oxideav-stl` (binary + ASCII) — STL is the smallest realistic
  consumer of the type model and validates the encoder side
  immediately.
- `oxideav-obj` + Wavefront MTL — exercises the multi-material
  primitive split + the `Material::extras` round-trip path.
- `oxideav-gltf` (JSON + GLB) — proves the type model round-trips
  losslessly against its design source-of-truth.
- KHR extension surface (`KHR_materials_emissive_strength`,
  `KHR_materials_unlit`, `KHR_lights_punctual`) once
  `oxideav-gltf` lands.

## Standalone build

`oxideav-core` is gated behind the default-on `registry` cargo
feature. Drop the framework dependency entirely with:

```toml
oxideav-mesh3d = { version = "0.0", default-features = false }
```

The typed model and trait definitions stay available — only the
embedded-`VideoFrame` `ImageData::Embedded` variant disappears and
the `Error` / `Result` aliases resolve to a crate-local enum
instead of `oxideav_core::Error`.

## License

MIT — see `LICENSE`.
