# oxideav-mesh3d

[![CI](https://github.com/OxideAV/oxideav-mesh3d/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-mesh3d/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-mesh3d.svg)](https://crates.io/crates/oxideav-mesh3d) [![docs.rs](https://docs.rs/oxideav-mesh3d/badge.svg)](https://docs.rs/oxideav-mesh3d) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
  `Scene3D::append` splices another scene's every arena onto the end of
  this one's, rewriting all internal ids (node↔mesh↔material↔texture,
  skin↔skeleton↔joint-node, animation-target nodes, audio
  emitter↔source) so the relocated resources still reference each other
  — the primitive behind multi-file import / instancing — returning an
  `AppendOffsets` recording where each source arena landed.
  `Material::map_texture_ids` remaps every texture slot (core + all
  extensions) in one call.
- `Node` + `Transform { Matrix, Trs }` with matrix↔TRS decompose.
- `Mesh` / `Primitive` / `Topology` (Triangles, TriangleStrip,
  TriangleFan, Lines, LineStrip, LineLoop, Points) / `Indices`
  (U16 or U32). Multi-channel UVs, vertex colours, optional skinning
  joints + weights, and `MorphTarget` delta buffers.
- `Material` — full glTF 2.0 metallic-roughness PBR slots plus
  `AlphaMode` and `double_sided`, refined by a typed `MaterialExt`
  surface for the KHR material extensions: emissive strength, index of
  refraction, dispersion, `Specular` (strength + F0-colour factors and
  textures), the unlit flag, and seven layered refinements —
  `Clearcoat`, `Sheen`, `Transmission`, `DiffuseTransmission`,
  `Volume`, `Iridescence`, `Anisotropy`. Each field is `Option`/flag-
  shaped so an absent extension stays distinguishable from its spec
  default; `effective_ior()` / `effective_emissive_strength()` /
  `effective_dispersion()` /
  `Volume::effective_attenuation_distance()` substitute the default on
  demand, and `dielectric_f0()` / `dielectric_f0_rgb()` / `rgb_iors()`
  evaluate the spec-documented IOR/specular/dispersion combination
  formulas. This surface covers every input of USD's
  UsdPreviewSurface (specular workflow, clearcoat + clearcoat
  roughness, IOR, opacity threshold via `AlphaMode::Mask`) so format
  importers keep `extras` for genuinely exotic parameters only.
  `Material::texture_refs()` enumerates every occupied texture slot
  (core + all extensions) for resource walkers, and is what
  `Scene3D::validate` checks texture ids through.
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
- **Curvature** — `Primitive::curvature` returns a `CurvatureReport`
  carrying per-vertex discrete **Gaussian** curvature (Gauss-Bonnet
  angle defect `2π − Σθ` / `π − Σθ` on boundary, normalised by the mixed
  Voronoi area), **mean** curvature (`½‖Δp‖` from the cotangent
  Laplace-Beltrami operator), and the per-vertex mixed area itself —
  parallel to the welded vertex pool (also returned). The discrete
  Gauss-Bonnet identity `Σ defect = 2π·χ` holds to machine precision on
  closed meshes (`total_angle_defect` exposes it as a cheap topological
  check), and the mixed areas sum to the surface area exactly. The
  signal feeds feature-preserving decimation, adaptive remeshing, and
  decoded-mesh quality metrics.
- **Scene graph** — `world_node_transforms`, `world_node_bounds`,
  `bounding_box`. Navigation + flattening: `Scene3D::parents`
  (first-parent spanning-forest map), `ancestors` (root→parent chain),
  `descendants` (DFS subtree incl. self), `is_ancestor_of`, and
  `Node::local_matrix`. `Scene3D::bake_transforms` collapses the
  hierarchy into a flat root list — every reachable node carries its
  full world matrix as `Transform::Matrix` with cleared children, node
  indices preserved, draws identically — exactly what a
  hierarchy-free target (STL/OBJ) needs. All navigation shares the
  first-arrival DFS semantics of `world_node_transforms`.
- **Ray queries** — `Ray` / `RayHit`, `ray::intersect_triangle`
  (Möller-Trumbore) / `intersect_aabb` (slab), `Primitive::
  intersect_ray` / `any_ray_intersection`, an object-median per-
  primitive `Bvh`, a scene-level `InstanceBvh`, and a world-space
  `Scene3D::intersect_ray` returning `SceneRayHit`.
- **Geometry transform** — `Primitive::transformed(m)` /
  `Mesh::transformed(m)` bake an affine 4×4 into the vertex data:
  positions move by the full affine, normals by the **inverse-transpose**
  of the linear part (so they stay perpendicular to the surface under
  non-uniform scale), tangent directions by the linear part with
  handedness `w` flipped on a mirroring (negative-determinant) transform;
  a singular linear part leaves normals untouched but still moves points.
  `Primitive::reverse_winding` flips triangle orientation (swap two
  corners, negate normals, invert tangent `w`) for importing
  clockwise / left-handed sources. The vertex-data complement to
  `Scene3D::bake_transforms` — together they let an exporter flatten
  into a transform-free, hierarchy-free target.
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
- **Polygon triangulation** — `Primitive::from_polygons(positions,
  faces)` builds an indexed `Triangles` primitive from a shared vertex
  pool plus a list of n-gon faces (each an index list), triangulating
  every >3-corner face with the same Newell-projected ear-clip
  `fill_holes` uses — so concave and non-planar faces are handled
  correctly (a naïve fan would self-overlap). The bridge from the
  polygon-face formats (OBJ `f`, FBX/USD face lists). Vertex indices
  are preserved (no welding/reorder) so the caller can attach its own
  attribute buffers afterward; each n-gon yields `n−2` triangles
  inheriting the face's winding; out-of-range / degenerate faces are
  skipped.
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

## Rigging & animation evaluation

The runtime half of the `Skeleton`/`Skin`/`Animation` type model —
glTF 2.0 §3.7.3 (skins), §3.6 + Appendix C (animation), §3.7.4
(instantiation), all pure and rest-scene-preserving:

- **Joint matrices** — `Scene3D::joint_matrices(node)` builds the
  skinning palette `globalTransform(joint) · inverseBindMatrix` per
  joint of the skin bound to a node; the skinned-mesh node's own
  transform is ignored (§3.7.3.2), so deformed vertices land directly
  in scene world space. Empty IBM list ⇒ identity per joint; extra
  trailing IBMs conforming (§3.7.3.1 count ≥ joints), fewer malformed.
  `joint_matrices_with(node, worlds)` runs against caller-supplied
  (shared or posed) world matrices.
- **Linear-blend skinning** — `Primitive::skinned(palette)` /
  `Mesh::skinned(palette)`: per-vertex `M = Σ wᵢ·palette[jᵢ]`;
  positions by the blended affine, normals by its per-vertex
  inverse-transpose (renormalised, singular ⇒ untouched), tangents by
  its linear part with handedness flipped on mirroring blends.
  Invalid influences contribute nothing; all-zero-weight vertices
  keep rest pose; weights used as stored.
  `normalize_joint_weights()` (both levels) is the explicit repair —
  clamp negatives/NaN to 0, renormalise rows to sum 1, keep all-zero
  rows — making skinning invariant to uniform weight scaling.
  `validate()` reports out-of-range joints (per node→skin→skeleton
  binding), negative/non-finite weights, and IBM shortfalls.
- **Pose evaluation** — `Animation::sample_pose(t, n)` evaluates every
  channel per Appendix C (clamp / Step / Linear-SLERP / cubic Hermite)
  into a sparse `Pose` (per-node T/R/S + morph-weight overrides),
  renormalising rotations per the C.5 note. `Pose::local_transform`
  merges overrides component-wise into rest TRS;
  `Scene3D::posed_node_transforms(&pose)` walks posed world matrices
  with the `world_node_transforms` contract. `Animation::duration()`.
- **Skin root** — `Scene3D::skin_root(skin)`: the explicit
  `Skin::root_node` when stored, else the lowest common ancestor of
  the joint set over the first-parent spanning forest (§3.7.3.2's
  documented fallback); pairs with `descendants` for one-subtree rig
  culling.
- **Instantiation** — `Scene3D::world_mesh(node)` bakes the §3.7.4
  pipeline (default morph weights folded in, then skin-or-transform)
  into one static world-space mesh; `world_mesh_at(anim, t, node)` is
  the animated frame (animated morph weights beat static defaults);
  `Scene3D::posed(anim, t)` bakes a whole frame into a scene copy for
  exporter flattening. Broken palettes yield `None`, never a silent
  rigid fallback.

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
