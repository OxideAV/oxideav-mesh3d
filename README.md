# oxideav-mesh3d

Pure-Rust 3D scene + mesh typed model.

The shared data model that every OxideAV 3D-format crate
(`oxideav-stl`, `oxideav-obj`, `oxideav-gltf`, `oxideav-usdz`,
`oxideav-fbx`) decodes into and encodes from. The type model is
aligned with **glTF 2.0** (Khronos KHR-public spec) as the
spec-stable common denominator: right-handed coordinates, Y-up,
-Z forward, metres, metallic-roughness PBR, xyzw quaternions. Files
coming from Z-up formats (STL/OBJ Wavefront) set
[`Scene3D::up_axis`] to `Axis::PosZ` — the model stores the
orientation, no implicit re-projection happens.

This crate carries **no format parsers** of its own; sibling crates
plug their decoders/encoders into [`Mesh3DRegistry`] (or use
`oxideav-meta::populate_mesh3d_registry`). Cross-format roundtrip
coverage lives in the workspace umbrella's `oxideav-tests`.

## Type model

- `Scene3D` — top-level container holding arenas of nodes, meshes,
  materials, textures, skeletons, skins, animations, cameras, lights,
  audio sources/emitters, plus `up_axis` / `front_axis` / `unit`
  metadata and a free-form `extras` round-trip side-channel.
- `Node` + `Transform { Matrix, Trs }` with matrix↔TRS decompose.
- `Mesh` / `Primitive` / `Topology` (Triangles, TriangleStrip,
  TriangleFan, Lines, LineStrip, LineLoop, Points) / `Indices`
  (U16 or U32). Multi-channel UVs, vertex colours, optional skinning
  joints + weights, and `MorphTarget` delta buffers.
- `Material` — full glTF 2.0 metallic-roughness PBR slots plus
  `AlphaMode` and `double_sided`.
- `Texture` / `ImageData { Embedded, Source(Arc<dyn AssetSource>),
  External }` / `Sampler`. `AssetSource` lets format crates pass a
  lazy reader through the model without materialising bytes, with
  optional `raw_storage()` pass-through for archive-to-archive copy.
- `Skeleton` + `Skin`; `Animation` / channels / samplers /
  `Interpolation`; `Camera`; `Light`; `Audio*` types
  (`AudioSource`, `AudioEmitter`, `SpatialAudio`, `AuralMode`,
  `DistanceModel`).
- `Mesh3DDecoder` / `Mesh3DEncoder` traits + `Mesh3DRegistry`
  (case-insensitive extension / format-id lookup).

## Geometry reductions

The crate ships a large pure-Rust geometry toolkit on `Primitive`
(lifted to `Mesh` / `Scene3D` where additive). All reductions are
pure (no mutation), robust (out-of-range / degenerate / NaN inputs
are skipped or fall back rather than panic), and topology-aware
(strips/fans feed through `triangle_indices`):

- **Connectivity** — `triangle_indices`, `to_triangle_list`,
  `weld_vertices`, `triangle_adjacency`, `degenerate_triangles`,
  `boundary_edges`, `boundary_loops`, `edge_manifold_report` /
  `is_closed_manifold`, `topology_summary` (V/E/F, Euler
  characteristic, component count, orientable genus),
  `orient_consistent` (flood-fill the face-dual graph to repair
  mixed-winding facet soup into per-component coherent winding,
  returning the re-oriented faces + an `OrientationReport`).
- **Attributes** — `compute_normals` (area-weighted),
  `compute_tangents` (MikkTSpace-style basis), `apply_morph_weights`.
- **Measures** — `surface_area`, `surface_centroid`,
  `signed_volume` / `volume`, `volume_centroid` (centre of mass),
  `inertia_tensor` (unit density). Each has `Mesh` / `Scene3D`
  rollups and transform-folding `world_*` variants
  (`world_surface_area`, `world_surface_centroid`,
  `world_signed_volume`, `world_volume_centroid`,
  `world_inertia_tensor`).
- **Scene graph** — `world_node_transforms`, `world_node_bounds`,
  `bounding_box`.
- **Ray queries** — `Ray` / `RayHit`, `ray::intersect_triangle`
  (Möller-Trumbore) / `intersect_aabb` (slab), `Primitive::
  intersect_ray` / `any_ray_intersection`, an object-median per-
  primitive `Bvh`, a scene-level `InstanceBvh`, and a world-space
  `Scene3D::intersect_ray` returning `SceneRayHit`.
- **Parametric solids** — `extrude::Profile2D` with ear-clip
  `triangulate` and watertight `extrude`.

## Sketch

```rust
use oxideav_mesh3d::{Mesh, Primitive, Topology};

let mut prim = Primitive::new(Topology::Triangles);
prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
prim.normals = Some(prim.compute_normals());
let mesh = Mesh::new(Some("tri".to_owned())).with_primitive(prim);
```

## Standalone build

`oxideav-core` is gated behind the default-on `registry` cargo
feature. Drop the framework dependency entirely with:

```toml
oxideav-mesh3d = { version = "0.0", default-features = false }
```

The typed model and trait definitions stay available — only the
embedded-frame `ImageData` / `AudioData` variants and the
`Mesh3DRegistry` glue are feature-gated, and `Error` / `Result`
resolve to a crate-local enum instead of `oxideav_core::Error`.

## License

MIT — see `LICENSE`.
