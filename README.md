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
  `AlphaMode` and `double_sided`, refined by a typed `MaterialExt`
  surface for the ratified KHR extensions: emissive strength, index of
  refraction, `Specular` (strength + F0-colour factors and textures),
  the unlit flag, and six layered refinements — `Clearcoat`, `Sheen`,
  `Transmission`, `Volume`, `Iridescence`, `Anisotropy`. Each field is
  `Option`/flag-shaped so an absent extension stays distinguishable
  from its spec default; `effective_ior()` /
  `effective_emissive_strength()` /
  `Volume::effective_attenuation_distance()` substitute the default on
  demand.
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
  returning the re-oriented faces + an `OrientationReport`),
  `fill_holes` (cap every boundary loop with a Newell-projected
  ear-clip patch wound consistent with the surrounding surface,
  making a torn surface watertight again).
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
- **Consolidation** — `Primitive::merge` concatenates two primitives
  into one indexed `Triangles` primitive, re-basing the second's
  indices onto the end of the first's vertex pool and reconciling
  optional attributes by *union with spec-neutral fill* (an attribute
  present on either input survives; the side that lacked it gets a
  default row — `[0,0,1]` normal, white colour, origin UV, …). UV/colour
  set counts take the max; morph deltas are dropped.
  `Mesh::merge_primitives_by_material` partitions a mesh's primitives by
  their material reference (with the unmaterialled `None` group kept
  separate) and fuses each group, collapsing an N-primitive mesh to at
  most one draw call per distinct material — the standard batching /
  pre-compression pass for an OBJ-`g`-split or per-accessor glTF import.
- **Parametric solids** — `extrude::Profile2D` with ear-clip
  `triangulate` and watertight `extrude`.
- **Refinement** — `subdivide_loop` (one step of Loop subdivision):
  welds, splits every triangle `1 → 4`, places interior edge vertices
  with the `3/8·(A+B) + 1/8·(C+D)` mask / boundary edge vertices at the
  midpoint, and relaxes original vertices (interior Warren β, boundary
  cubic-B-spline mask). Positions carry the Loop masks; all other
  attributes (normals/tangents/UVs/colours/joints/weights/morph deltas)
  are linearly interpolated. Boundaries stay watertight; iterate for a
  smooth limit surface from a coarse STL/OBJ/CSG cage.
- **Fairing** — `smooth_laplacian(lambda, opts)` relaxes every vertex a
  fraction `λ` of the way toward its one-ring centroid (the uniform
  umbrella Laplacian `vᵢ' = vᵢ + λ·(centroid(N(i)) − vᵢ)`), a Jacobi
  sweep so the result is order-independent; `smooth_taubin(lambda, mu,
  opts)` alternates a positive shrink pass with a negative inflate pass
  (`μ < −λ < 0`) for the same noise removal **without the volume
  shrinkage** of the plain Laplacian. Both weld first, preserve the
  connectivity exactly (only positions move; all other attributes pass
  through), pin boundary vertices by default
  (`SmoothOptions::preserve_boundary`), and skip any vertex whose
  smoothed position would be non-finite.
- **GPU vertex-buffer optimisation** — three storage-only reorderings
  that leave the rendered result exactly invariant (they permute order,
  never geometry) and double as the recommended pre-processing for
  index/vertex stream compression. `simulate_cache` / `cache_stats`
  simulate a FIFO post-transform vertex cache and report **ACMR**
  (`misses / triangles`) + **ATVR** (`misses / vertices`) so every
  reorder is measurable. `optimize_vertex_cache` greedily reorders
  *triangles* to maximise post-transform vertex reuse (cache-recency +
  low-valence scoring, near-linear, pool untouched).
  `optimize_vertex_fetch` relabels the *vertex pool* into index-stream
  first-appearance order so the pre-transform attribute fetch sweeps the
  buffer front-to-back (lossless: unreferenced vertices kept).
  `optimize_vertex_spatial(bits)` sorts the pool along a Morton (Z-order)
  curve for non-indexed / seam-heavy meshes where there is no reused
  index order to follow. Recommended pipeline:
  `optimize_vertex_cache().optimize_vertex_fetch()`.
- **Decimation** — two reducers, both duals of `subdivide_loop`:
  - `simplify_cluster(grid)` lays a cube grid (`grid` cells along the
    longest bounding-box axis, equal cell edge on every axis), collapses
    every vertex sharing a cell to one per-cell averaged representative,
    and re-emits the de-duplicated connectivity over the occupied cells —
    dropping triangles that no longer span three distinct cells and
    pruning orphan cells. All attributes are averaged per cell (normals /
    tangent `xyz` re-normalised, tangent `w` by cell-majority handedness,
    `weights` re-normalised to sum 1, `joints` carried verbatim).
    Unconditionally robust (no legality tests) but position-quantising.
  - `simplify_quadric(target_triangles)` is the **error-optimal**
    counterpart: each vertex accumulates the sum of the fundamental error
    quadrics (`Kp = p·pᵀ`) of its incident triangle planes, and a greedy
    min-heap collapses the edge whose merged residual `v̄ᵀ Q̄ v̄` is
    smallest, re-positioning the survivor at the error-minimising point
    (solved from the `3×3` quadric sub-block, endpoint/midpoint fallback
    when singular). A fold-over guard rejects collapses that would flip or
    sliver a triangle; a boundary lock + per-seam constraint quadric keep
    open silhouettes; non-manifold and link-condition-violating edges are
    left intact. Position is metric-driven; other attributes blend the
    endpoints at the optimal split (same conventions as `subdivide_loop`).
    `~O(E log E)`; far truer to the original shape at a given triangle
    budget than clustering. `simplify_quadric_error(max_error)` is the
    error-bounded companion — it stops when the cheapest remaining collapse
    would exceed a squared-distance budget instead of a triangle count, so
    the surface never deviates past the budget at any step (`0.0` does only
    coplanar collapses).
  Both yield a coarser indexed `Triangles` proxy for level-of-detail or a
  cheap collision hull; robust against degenerate / non-finite /
  non-triangle input; neither mutates `self`. `simplify_quadric` /
  `simplify_quadric_error` additionally lift to `Mesh` roll-ups that
  reduce every primitive independently to the same per-primitive budget.

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
