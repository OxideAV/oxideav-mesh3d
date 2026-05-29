# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Round 175 (surface-area reduction)

- `Primitive::surface_area(&self) -> f64` — total area of the
  primitive's triangle tessellation, in the unit-squared of
  `Primitive::positions` (matching the parent `Scene3D::unit`). Each
  triangle's area is the half cross-product magnitude
  `|E1 × E2| / 2` (Marsden & Tromba, *Vector Calculus* — the
  cross-product magnitude is the parallelogram-area definition; a
  triangle occupies half of that parallelogram). The same `E1 × E2`
  cross product already drives `compute_normals` (its magnitude is
  twice the triangle area, which is why summing the un-normalised
  face normal into each vertex automatically area-weights smooth
  shading); `surface_area` reuses the edge-cross machinery and
  divides by two. Topology integration goes through
  `triangle_indices`, so `Triangles` / `TriangleStrip` (alternating
  winding honoured) / `TriangleFan` all feed in. Non-triangle
  topologies (lines/points) contribute 0.0. Accumulator is `f64` so
  million-triangle meshes don't drift under `f32` summation; per-
  triangle math is also `f64`. Degenerate triangles
  (collinear/coincident corners, same set `degenerate_triangles`
  reports), NaN- or Inf-producing faces, and out-of-range index
  entries all contribute 0.0 — the result is always finite. Pure;
  cost `O(triangle_count)`.
- `Mesh::surface_area(&self) -> f64` — sum across every contained
  primitive in mesh-local space (no transforms / skin pose / morph
  deltas applied).
- `Scene3D::surface_area(&self) -> f64` — sum across every mesh in
  the scene, in the scene's local unit-squared (matching
  `Scene3D::unit`). Walks meshes once, not node instances — a mesh
  instanced by two nodes contributes its area once. For a
  transform-aware total, walk `world_node_transforms` and apply the
  per-node scale's determinant per primitive instance.
- `tests/surface_area.rs` (30 tests): unit right-triangle = 0.5,
  unit square (list + indexed) = 1.0, unit cube = 6.0, equilateral
  side-1 = √3/4, winding-invariance (unsigned), scaling-squares-the-
  area dimensional check, rotation / translation isometry preserves
  area, `to_triangle_list` and `weld_vertices` are area-preserving,
  empty primitive / Mesh / Scene = 0.0, incomplete trailing
  vertices dropped, degenerate triangles contribute 0 (including
  one-among-valid), NaN- / Inf-bearing faces skipped, out-of-range
  index skipped, TriangleStrip + TriangleFan match the equivalent
  list, non-triangle topologies = 0, million-triangle stress
  (`f64` no-drift), morph targets ignored (`surface_area` is base-
  only), `Mesh::surface_area` sum + lines-only / empty zero,
  `Scene3D::surface_area` sum + empty zero + instanced-mesh
  counted-once contract.

### Added — Round 155 (mesh-validity invariants: degenerate-triangle detection + edge-manifold classification)

- `Primitive::degenerate_triangles(&self) -> Vec<usize>` — returns the
  indices into `triangle_indices()` for **degenerate** triangles
  (zero-area: collinear, coincident corners, out-of-range index, or
  NaN-producing face). The detection criterion is the same
  `|E1 × E2| == 0.0` test that `compute_normals` / `compute_tangents`
  already use to silently drop bad faces; this surface is the
  detection-only counterpart so a validator can warn, a repair pass
  can prune, or a fixture-comparison test can pin them. No epsilon
  thresholding — a triangle that is almost-but-not-quite collinear
  within float precision is reported valid (proximity-based pruning is
  a separate lossy operation, intentionally out of scope). Non-triangle
  topologies (lines/points) return an empty `Vec`. Pure; cost
  `O(triangle_count)`.
- `Primitive::edge_manifold_report(&self) -> EdgeManifoldReport` —
  classifies every undirected triangle edge of the primitive by its
  **use count** (how many triangles share it):
  * `1` — boundary edge (hole, crack, open rim)
  * `2` — manifold-interior edge (clean two-manifold seam)
  * `≥ 3` — non-manifold edge (T-junction / book-spine / feather)

  A **closed two-manifold** mesh has zero boundary edges and zero
  non-manifold edges — every edge is used by exactly two triangles.
  This is the STL spec's "vertex-to-vertex rule" (Fabbers / Stratasys
  1989: *"each triangle must share two vertices with each of its
  adjacent triangles"*) and the classical solid-printable condition.
  Triangles with a duplicate corner index or with an out-of-range
  entry are excluded whole — they're degenerate by index and their
  bogus edges would otherwise pollute neighbour counts. Topology
  comparison is by vertex index, not by 3D position; run
  `weld_vertices` first to merge positionally coincident corners
  before counting. Connectivity goes through `triangle_indices`, so
  `Triangles` / `TriangleStrip` (alternating winding) / `TriangleFan`
  all feed in. Pure; cost `O(triangle_count)`; allocates one
  `HashMap` of undirected edges.
- `EdgeManifoldReport { total_edge_count, boundary_edge_count,
  manifold_interior_edge_count, non_manifold_edge_count,
  max_edge_use }` — typed summary returned by
  `edge_manifold_report`. The report does not retain the per-edge map
  (the heavy `HashMap` is freed after counting); callers needing the
  actual edge endpoints can re-derive them from `triangle_indices`.
  `Clone + Copy + Debug + Default + PartialEq + Eq`.
  `EdgeManifoldReport::is_closed_manifold(&self) -> bool` is the
  shortcut for `total_edge_count > 0 && boundary == 0 &&
  non_manifold == 0`.
- `tests/mesh_validity.rs` (34 tests) — degenerate detection across
  empty primitives, single triangles, mixed lists, out-of-range
  indices, NaN faces, strips/fans, and the almost-but-not-quite-
  collinear boundary; edge classification on single triangles
  (3 boundary), quad-split (1 interior + 4 boundary), tetrahedron
  (6 interior, 0 boundary — closed manifold), three-page book-spine
  (1 non-manifold, 6 boundary), four-page spine (`max_edge_use = 4`),
  strip topology, lines/points (empty report), winding-independence,
  index-vs-position semantics + post-weld behaviour, and sum/copy
  invariants on the `EdgeManifoldReport` itself.

### Added — Round 105 (per-vertex MikkTSpace-style tangent-space recomputation)

- `Primitive::compute_tangents(&self, uv_set: usize) -> Option<Vec<[f32; 4]>>` —
  derives per-vertex tangents from positions, the selected UV channel,
  and the existing per-vertex normals. Each `[f32; 4]` is xyz = unit
  tangent T plus w = ±1.0 handedness, matching exactly the shape of
  `Primitive::tangents` and the glTF 2.0 §3.7.2.1 `TANGENT` accessor
  (renderer reconstructs `B = w * (N × T)`).
  - Closed-form derivation per triangle: invert the 2×2 UV-delta
    linear system `[E1; E2] = [Δu1 Δv1; Δu2 Δv2] · [T; B]` to get
    `T = (Δv2·E1 − Δv1·E2) / det` and `B = (−Δu2·E1 + Δu1·E2) / det`
    (Lengyel, "Computing Tangent Space Basis Vectors for an Arbitrary
    Mesh" 2001; same derivation in the Normal Mapping chapter of
    Akenine-Möller, Haines & Hoffman, *Real-Time Rendering*).
  - Per-triangle contributions scaled by `sign(det)` and accumulated
    per vertex — area-weighting by unsigned UV area `|det|/2`,
    symmetric with how `compute_normals` area-weights by
    cross-product magnitude. Mirrored UV charts still pull T in the
    +U surface direction; the mirror-vs-not signal is recovered
    separately as the per-vertex handedness w.
  - Per-vertex Gram-Schmidt orthonormalisation against N
    (`T' = normalise(T_sum − (T_sum·N)/|N|² · N)`); handedness
    `w = sign((N × T') · B_sum)` rounded to ±1.0.
  - Returns `None` when prerequisites are missing (no `normals`, UV
    set absent / out of range, attribute length mismatches);
    otherwise output length always equals `positions.len()`.
  - Robust: vertices not referenced by any triangle, vertices whose
    UV chart is degenerate (all incident triangles have `det ≈ 0`),
    vertices whose accumulated T_sum is parallel to N, and out-of-
    range index entries / NaN UVs all fall back to
    `[1.0, 0.0, 0.0, 1.0]` — a unit tangent with positive handedness
    — so the result is always renderable.
  - Topology-integrated via `triangle_indices`, so `Triangles` /
    `TriangleStrip` / `TriangleFan` all feed in; non-triangle
    topologies (lines/points) produce an all-fallback buffer.
  - Pure: does not mutate `self`. Assign to `Primitive::tangents`
    to store. This is the recompute step a format decoder runs when
    the wire stream omits tangents (OBJ has no native tangent
    channel, glTF without `TANGENT`).
- `tests/compute_tangents.rs` (28 tests) covers the axis-aligned XY-
  and XZ-plane references, V-flip / U-flip handedness inversion,
  Gram-Schmidt orthogonality to N, |w|=1 contract, bitangent
  reconstruction `B = w · (N × T)`, missing-input None returns, length
  mismatches, output-length contract, assignability to the
  `Primitive::tangents` field, strip topology equivalence, non-
  triangle fallback, degenerate UV / unreferenced vertex / out-of-
  range index / NaN UV robustness, T-parallel-to-N fallback, multi-
  UV-set channel selection, U16/U32 index parity, purity (no `self`
  mutation, deterministic repeat).

### Added — Round 101 (area-weighted per-vertex normal recomputation)

- `Primitive::compute_normals(&self) -> Vec<[f32; 3]>` — recomputes
  smooth, area-weighted per-vertex normals from the primitive's
  triangle connectivity (the de-stripped list from `triangle_indices`,
  so `Triangles` / `TriangleStrip` / `TriangleFan` all feed in). For
  each triangle the un-normalised face normal is the edge cross product
  `(P[b]-P[a]) × (P[c]-P[a])`; its magnitude equals twice the triangle
  area, so accumulating it into each incident vertex and normalising the
  sum yields the area-weighted average of neighbouring face normals —
  the textbook smooth-shading recomputation (Gouraud 1971; Foley, van
  Dam et al., *Computer Graphics: Principles and Practice*).
  - Output length always equals `positions.len()`; CCW = front-facing
    (right-handed, glTF-aligned) so the normal points out of the front
    face.
  - Robust: unreferenced vertices, vertices touched only by degenerate
    (collinear/coincident) faces, out-of-range index entries, and
    NaN-producing faces all fall back to `[0, 0, 1]` rather than a zero
    vector or a panic. Non-triangle topologies (lines/points) produce an
    all-fallback buffer.
  - Pure: does not mutate `self`. Assign to `Primitive::normals` to
    store. This is the recompute step a format decoder runs when the
    wire stream omits normals (OBJ without `vn`, glTF without `NORMAL`).
- `tests/compute_normals.rs` — 22 tests: flat-triangle direction in
  XY/XZ/tilted planes, CW winding flip, area weighting (large face
  dominates a shared vertex; coplanar neighbours stay identical),
  strip/fan integration, U16/U32 index parity, degenerate-face / NaN /
  out-of-range robustness, fallback for unreferenced vertices,
  `to_triangle_list` invariance, and a cube-corner diagonal normal.

### Added — Round 97 (strip/fan → triangle-list de-stripping)

- `Primitive::triangle_indices(&self) -> Vec<[u32; 3]>` — de-strips the
  primitive's topology into a flat list of triangle vertex-index
  triples per the OpenGL/glTF primitive-assembly rules:
  - `TriangleStrip` — `v[0],v[1],v[2]`, then `v[1],v[2],v[3]`, … with
    the alternating-winding rule: odd-numbered triangles swap their
    last two vertices so the visible (front-facing) winding stays
    consistent.
  - `TriangleFan` — `v[0],v[1],v[2]`, `v[0],v[2],v[3]`, … sharing the
    anchor vertex `v[0]`; uniform winding.
  - `Triangles` — index triples returned verbatim; a trailing
    incomplete triple is dropped.
  - Non-triangle topologies (`Lines`, `LineStrip`, `LineLoop`,
    `Points`) yield an empty list.
  - When an index buffer is present each entry is dereferenced so the
    result indexes the vertex pool, not the index buffer; `U16` and
    `U32` are both widened to `u32`. Output count equals
    `triangle_count()` for triangle topologies.
- `Primitive::to_triangle_list(&self) -> Primitive` — materialises an
  equivalent `Topology::Triangles` primitive with a fresh `U32` index
  buffer (the row-major flattening of `triangle_indices`). Attribute
  buffers (`positions`, `normals`, `tangents`, `uvs`, `colors`,
  `joints`, `weights`), `material`, morph `targets`, and `extras` are
  carried over verbatim — only connectivity is rewritten. A
  non-triangle source yields a `Triangles` primitive with an empty
  index buffer (attribute pool preserved). For `Triangles` input it is
  a normalising round-trip (output gains an explicit `U32` index
  buffer; re-running is stable).
- `tests/destrip.rs` — 25 tests: Triangles passthrough + incomplete-
  triple drop + indexed-deref (U16/U32), TriangleStrip 4/5/6-vertex
  alternating winding + indexed-deref + too-short cases, TriangleFan
  shared-anchor + indexed-anchor + too-short cases, non-triangle
  topologies yield empty, `to_triangle_list` strip/fan conversion +
  attribute/material/morph-target carry-through + Triangles
  idempotence + lines empty-index, and `triangle_indices().len()` ==
  `triangle_count()` cross-checks for n = 3..=20 plus an
  index-in-range invariant.

### Added — Round 10 (Primitive::apply_morph_weights)

- `Primitive::apply_morph_weights(weights: &[f32]) -> MorphedAttributes`
  — typed evaluator for the glTF 2.0 §3.7.2.2 morph-blend formula
  `morphed[k] = base[k] + Σ weights[i] * targets[i].ATTR[k]` over the
  three typed slots (`POSITION`, `NORMAL`, `TANGENT`). Contract:
  - Output `positions` is always present; `normals` / `tangents` are
    `Some` iff the corresponding base attribute was `Some` (spec line
    3586: morph attributes require a base attribute).
  - Tangent handedness `w` is preserved verbatim — morph TANGENT is
    xyz-only per spec line 3616.
  - Missing or excess `weights` default to zero (spec line 3697); a
    weight of `0.0` short-circuits per-target.
  - Buffer-length mismatch is a soft error: the prefix is applied,
    the remainder is left untouched (callers run `Scene3D::validate`
    first to catch this in their test pipeline).
- `MorphedAttributes { positions, normals, tangents }` — `Clone +
  Debug + PartialEq` output struct re-exported from the crate root.
- `tests/morph_apply.rs` — 20 tests across no-op cases (empty weights,
  no targets, all-zero weights), single-target full / half / negative
  weights, two-target weighted sum, missing-weight default-to-zero,
  excess weights ignored, NORMAL slot blending + base-absence drop,
  TANGENT handedness preservation, combined three-attribute blend,
  output-presence mirrors input-presence (3 cases), soft-error
  prefix/excess length handling, `MorphedAttributes` clone+eq, and
  crate-root re-export parity.

### Added — Round 9 (AnimationSampler::sample + IBM affine validation)

- `AnimationSampler::sample(t: f32) -> Option<SampledValue>` — keyframe
  evaluator implementing every glTF 2.0 Appendix C interpolation mode.
  `SampledValue { Vec3([f32; 3]), Quat([f32; 4]), Scalar(Vec<f32>) }`
  is re-exported from the crate root; `Scalar` carries `Vec<f32>` since
  `MorphWeights` samplers stride the per-frame value table by the
  per-mesh morph-target count (variable arity, not fixed-width).
  - **C.1 clamping.** `t <= keyframes[0]` returns the first centre
    value; `t >= keyframes[n-1]` returns the last; an exact-keyframe
    match short-circuits the interpolation arithmetic.
  - **C.2 STEP** — `v_t = v_k`.
  - **C.3 LINEAR** (non-rotation) — `(1-u)*v_k + u*v_{k+1}`,
    componentwise; same logic re-used for `Scalar` morph weights with
    per-frame stride.
  - **C.4 SLERP** (rotation under LINEAR) — short-arc dot-product sign
    correction + `sin(a*(1-t))/sin(a) * v_k + sign*sin(a*t)/sin(a) * v_{k+1}`,
    with a normalised componentwise-lerp fallback when `|sin(a)| < 1e-6`
    (the spec's "close-to-zero a" note).
  - **C.5 CUBICSPLINE** — Hermite blend
    `(2u³-3u²+1)v_k + t_d(u³-2u²+u)b_k + (-2u³+3u²)v_{k+1} + t_d(u³-u²)a_{k+1}`
    over the `[in, value, out]` storage layout; rotation output is
    **not** auto-normalised (spec note: caller's responsibility).
  - Empty samplers, length-mismatched value tables, and zero per-frame
    strides return `None` — a fuzzer-safe surface that codec authors
    can plug straight into a renderer.
- `tests/animation_sample.rs` — 19 tests across §C.1 (clamping +
  exact-match), §C.2 (Step), §C.3 (Linear Vec3 + Scalar with two morph
  targets), §C.4 (SLERP identity-to-identity, 90° about Z midpoint =
  45° about Z, short-arc dot-negative collapse), §C.5 (centre value at
  keyframe, zero-tangent ↔ Hermite basis collapse, non-zero out-tangent
  shifts midpoint off the linear blend, pre/post clamp ignores
  tangents), plus a multi-segment binary-search walk and Step
  morph-weights stride check.

### Added — Round 9 (IBM affine-row validation)

- `Scene3D::validate()` gains a glTF 2.0 §5.28.1 check: every
  `Skeleton::inverse_bind_matrices` entry must have its fourth row set
  to `[0, 0, 0, 1]`. Non-affine IBMs would silently corrupt the
  skinning math `sum_i weight_i * joint_world_i * IBM_i * pos`. New
  variant `ValidationError::SkeletonBindMatrixNotAffine { location,
  last_row }` with `Display` breadcrumb. Existing
  `full_valid_scene_with_all_resource_kinds_passes` oracle updated to
  use a proper identity IBM. Three new tests in
  `tests/scene_validate.rs`: zero-last-row caught, projective-last-row
  caught, identity passes, plus a Display breadcrumb check.

### Added — Round 8 (extended validate rules + bounding_box)

- `Scene3D::validate()` gains cross-collection checks for the resource
  arenas that round 7 didn't yet cover:
  - **Materials → textures.** Every `Material::*_texture` slot's
    `TextureRef::texture` is range-checked against `Scene3D::textures`.
    All five PBR slots (`base_color`, `metallic_roughness`, `normal`,
    `occlusion`, `emissive`) are reported individually so the
    breadcrumb identifies which slot dangled.
  - **Skeletons → nodes + IBM parity.** Every `Skeleton::joints` entry
    is range-checked against `Scene3D::nodes`; when
    `Skeleton::inverse_bind_matrices` is non-empty its length must
    equal `joints.len()` (an empty vec is a documented escape hatch
    per glTF's `inverseBindMatrices` being optional). New variant
    `ValidationError::SkeletonBindMatrixCountMismatch`.
  - **Skins → skeletons + optional root_node.** Both
    `Skin::skeleton` and `Skin::root_node` are range-checked.
  - **Audio emitters → audio sources.** `AudioEmitter::source` is
    range-checked against `Scene3D::audio_sources`.
  - **Animations.** Every `AnimationChannel::target.node` is
    range-checked. Per-sampler parity is enforced:
    - keyframes must be non-empty and strictly increasing
      (`ValidationError::AnimationSamplerEmpty` /
      `AnimationKeyframesNotStrictlyIncreasing`),
    - the `AnimationValues` variant must match the channel's
      `AnimationProperty` (`Translation`/`Scale` → `Vec3`,
      `Rotation` → `Quat`, `MorphWeights` → `Scalar`) —
      `ValidationError::AnimationValueVariantMismatch`,
    - sample count must equal `keyframes.len() * factor` where
      `factor = 3` for `CubicSpline` and `1` otherwise; for
      `MorphWeights` the check relaxes to divisibility by
      `keyframes.len() * factor` since the per-mesh
      morph-target count needs out-of-band binding context to
      verify exactly — `ValidationError::AnimationSamplerLengthMismatch`.
- `tests/scene_validate.rs` adds 18 new tests covering every new
  variant + a "full valid scene with every resource kind" happy-path
  oracle that exercises every new check at once.

### Added — Round 8 (bounding_box)

- `BoundingBox { min: [f32; 3], max: [f32; 3] }` axis-aligned bounding
  box value type, re-exported from the crate root. Methods:
  `from_point`, `from_points<I: IntoIterator>` (NaN coordinates
  skipped, empty input ⇒ `None`), `expand`, `union`, `center`, `size`,
  `is_valid`, `transform([[f32; 4]; 4])`. `transform` fits a tight
  AABB around the eight transformed corners, so a rotated input box
  produces a tight (not loose) output box.
- `Primitive::bounding_box() -> Option<BoundingBox>` — extent over
  `Primitive::positions` in primitive-local space. Indices are *not*
  consulted (data extent, not drawn extent); callers needing the
  index-aware version go through `BoundingBox::from_points` themselves.
- `Mesh::bounding_box() -> Option<BoundingBox>` — union over every
  contained primitive in mesh-local space.
- `Scene3D::bounding_box() -> Option<BoundingBox>` — depth-first walk
  from `Scene3D::roots`, composing each ancestor node's transform
  into a world matrix and folding the per-mesh AABB through it.
  Skin pose + morph deltas are **not** applied (the typed model
  stays runtime-agnostic). Detached meshes (reachable through
  `Scene3D::meshes` but not from any root) are skipped — use
  `Scene3D::meshes` + `Mesh::bounding_box` directly when that's the
  intent. Self-cycles in the children graph are guarded with a
  visited-set so the walk terminates.
- `ValidationError` is now `PartialEq` only (no longer `Eq`) because
  `AnimationKeyframesNotStrictlyIncreasing` carries `f32`. Existing
  patterns continue to compile; only direct `assert_eq!`-on-errors
  callers need to switch to pattern matching.
- `tests/bounding_box.rs` — 23 new tests across `BoundingBox` basics
  (zero-size point, NaN handling, union, transform under
  translation/rotation/scale), `Primitive`/`Mesh` helpers, and full
  `Scene3D` walks (parent-child compose, multi-root union, detached
  mesh skipping, self-cycle termination).

### Changed — BREAKING (round 7, pre-v0.1)

- **`Primitive` is now `#[non_exhaustive]`.** Future attribute fields
  (a second skinning channel, a wireframe-edge buffer, …) land in
  minor releases without breaking downstream callers — *provided*
  those callers go through `Primitive::new(Topology::…)` and assign
  fields per-attribute. Construct with:

  ```rust
  let mut p = Primitive::new(Topology::Triangles);
  p.positions = positions;
  p.normals = Some(normals);
  // … etc.
  ```

  Literal struct construction `Primitive { topology, positions, … }`
  from outside this crate is now a compile error (`E0639`).

- **`Mesh` is now `#[non_exhaustive]`.** Construct with `Mesh::new(name)`
  and the existing `with_primitive` / `with_weights` builders. Literal
  `Mesh { name, primitives, weights }` is rejected from outside this
  crate.

This is the round-7 follow-through on the deferral noted in round 6
(see "Round-6 candidate" doc-comments removed from `mesh.rs` in this
commit). The same `#[non_exhaustive]` treatment for `oxideav_core::Group`
remains tracked separately in MEMORY.

### Added — Round 7 (Scene3D::validate)

- `Scene3D::validate() -> Result<(), Vec<ValidationError>>` — defensive
  cross-collection consistency check intended for fuzzers and codec
  authors. Walks every typed `IdT(u32)` reference for danglers,
  cross-checks per-primitive attribute buffer lengths against
  `positions.len()`, range-checks `Primitive::indices`, validates
  `MorphTarget` slot lengths, and asserts `Mesh::weights` parity with
  child-primitive morph-target counts. Does not short-circuit — one
  pass returns every issue found, so callers don't have to chase
  cascade failures.
- `ValidationError` enum (also `#[non_exhaustive]`): `DanglingId`,
  `AttributeLengthMismatch`, `IndexOutOfRange`, `MorphWeightCountMismatch`.
  Implements `Display` + `std::error::Error`. Re-exported from the
  crate root.
- `tests/scene_validate.rs` — 11 tests covering empty-scene OK path,
  every error variant, the multi-error single-pass contract, and
  `Display` breadcrumb format.

### Sibling cascade impact

After this commit lands and `oxideav-mesh3d 0.0.2` publishes, sibling
format crates that construct `Primitive` / `Mesh` literally will fail
to compile. Audit at the time of this commit:

- `oxideav-stl 0.0.0` — **breaks**. Literal `Primitive { … }` /
  `Mesh { … }` in `src/binary.rs`, `src/ascii.rs`, `src/encoder.rs` +
  several test files. Fix: switch to `Primitive::new(Topology::…)` +
  per-field assignment + `Mesh::new(name).with_primitive(…)`.
- `oxideav-obj 0.0.0` — clean. Already uses builders.
- `oxideav-gltf 0.0.0` — clean. Operates on its own crate-local
  `gltf::json_model::{Primitive, Mesh}` types.
- `oxideav-usdz 0.0.0` — clean. Already uses builders.
- `crates/oxideav-tests/tests/mesh3d_usdz_apple_oracle.rs` — **breaks**.
  Two literal sites; switch to constructors in the same release-batch
  commit that consumes mesh3d 0.0.2.

Recommended publish + bump order:

1. Publish `oxideav-mesh3d` 0.0.2 (this commit) to crates.io.
2. Patch `oxideav-stl` to migrate its 4 src/ + ~10 tests/ literal
   sites onto the constructor + builders. Publish 0.0.1.
3. Patch `crates/oxideav-tests/tests/mesh3d_usdz_apple_oracle.rs`
   the same way (no separate publish — workspace-internal tests crate).

## [0.0.2](https://github.com/OxideAV/oxideav-mesh3d/compare/v0.0.1...v0.0.2) - 2026-05-10

### Other

- Drop dev-deps + cross-format tests; live in crates/oxideav-tests/ now
- Round 6: typed morph-target fields (Primitive.targets + Mesh.weights)
- Round 5: multi-mesh stress + encoder option round-trip coverage
- Round 4: skinning + multi-primitive + extras audit coverage
- Round 3: cross-format conversion roundtrip suite

### Added — Round 6 (typed morph-target fields)

- `MorphTarget { position, normal, tangent }` — typed delta-buffer
  struct for one named morph pose per glTF 2.0 §3.7.2.2. Each slot
  is `Option<Vec<[f32; 3]>>`; `position` and `normal` deltas add to
  the base attribute, `tangent` carries xyz only (handedness `w` on
  the base TANGENT is not morphed per spec). Re-exported from the
  crate root.
- `Primitive::targets: Vec<MorphTarget>` — per-primitive morph-target
  roster. Empty vec means no morph targets. The `i`th entry across
  every primitive in a [`Mesh`] shares one weight slot per spec
  §3.7.2.2.
- `Mesh::weights: Vec<f32>` — static default morph-blend weights
  (spec §3.7.2.2 `mesh.weights`); empty default. An animation channel
  of `AnimationProperty::MorphWeights` overrides at runtime.
- `Mesh::with_weights(impl Into<Vec<f32>>)` builder helper for
  chained construction.
- `tests/morph_targets.rs` — 13 tests covering: defaults are empty
  (5 sub-tests on `Primitive::new` / `Mesh::new` / `Mesh::default` /
  `MorphTarget::new` / `MorphTarget::default` empty + None invariants),
  `Primitive` with two `MorphTarget`s (POSITION + NORMAL each)
  round-trips bit-exact through `Clone` + `PartialEq`, TANGENT-slot
  dimensionality is `[f32; 3]` per spec §3.7.2.2, four-weight `Mesh`
  round-trips, crate-root re-export parity with the module path,
  `with_weights` overwrite + array-literal acceptance, recommended
  builder construction style for both `Primitive` and `Mesh`.

### Changed — BREAKING (round 6, pre-v0.1)

- **`Primitive` gains a non-`Option`-defaulted `targets` field.**
  Callers constructing `Primitive` literally
  (`Primitive { topology, positions, … extras }`) must add
  `targets: Vec::new()` to the literal, or migrate to the
  `Primitive::new(Topology::…)` constructor + per-field assignment
  (recommended forward-compatible style).
- **`Mesh` gains a non-`Option`-defaulted `weights` field.**
  Callers constructing `Mesh` literally must add
  `weights: Vec::new()`, or migrate to `Mesh::new(name)` +
  the `with_*` builders, or struct-update syntax with
  `..Mesh::default()`.

`Primitive` and `Mesh` are **not** marked `#[non_exhaustive]` in
this round (deferred to round 7 alongside the
`oxideav_core::Group` deferral tracked in MEMORY) so existing
struct-literal construction sites still compile after adding the
two new fields, instead of also having to refactor onto builders.

### Release-order note

Sibling format crates (`oxideav-stl 0.0.0`, `oxideav-obj 0.0.0`,
`oxideav-gltf 0.0.0`, `oxideav-usdz 0.0.0`) at the time of this
commit:

- `oxideav-stl 0.0.0` constructs `Primitive { … }` and `Mesh { … }`
  literally and will fail to compile against new mesh3d until it
  adds the two new fields to the literal.
- `oxideav-obj 0.0.0` and `oxideav-usdz 0.0.0` already use
  `Primitive::new` / `Mesh::new` + builders and are unaffected.
- `oxideav-gltf 0.0.0` operates on its own crate-local
  `gltf::json_model::Primitive` / `Mesh` types and is unaffected;
  the migration of its `__morph_targets` / `__mesh_weights`
  sentinels onto the new typed fields lands in round 7 on the gltf
  side.

Recommended publish order:

1. Publish `oxideav-mesh3d` 0.0.2 (this commit) to crates.io.
2. Update `oxideav-stl` to add `targets: Vec::new()` and
   `weights: Vec::new()` to its two struct-literal construction
   sites in `binary.rs` and `ascii.rs`. Publish 0.0.1.
3. Update `oxideav-gltf` to consume the new typed fields in place
   of the `__morph_targets` / `__mesh_weights` extras sentinels;
   round-4 / round-5 pinning tests in this crate flip from "drops"
   to "survives". Publish 0.0.1.
4. mesh3d's own dev-dep CI (cross-format roundtrip suite) goes
   green again on the next release-plz cycle once the bumped
   sibling versions are published.

### Added — Round 5 (multi-mesh stress + encoder option round-trip)

- `tests/multi_material_pool_stress.rs` — 12 tests across four
  sections that extend round 4's two-primitive coverage:
  - **Cross-mesh OBJ vertex pool dedup** — six raw vertex positions
    spanning two separate `Mesh`es collapse to three `v` lines (the
    `serialize_obj` global pool reaches *across* meshes, not just
    across primitives within one mesh). glTF keeps both meshes
    distinct; STL flattens to one triangle stream and survives the
    triangle count.
  - **Five-material binding survival** — one mesh with five
    primitives bound to five distinct materials emits five `usemtl`
    directives in OBJ output; all five names round-trip via
    `Primitive::extras["obj:usemtl"]`. glTF keeps five distinct
    `materials[]` and five distinct primitive `material` indices.
  - **Multi-mesh hierarchy** — a parent node with two child nodes
    (each carrying its own mesh) survives all three encoders;
    glTF keeps the per-mesh partition; STL flattens.
  - **Material aliasing** — three primitives bound to the *same*
    material id still emit three `usemtl` directives in OBJ output
    (one per primitive boundary) but glTF collapses to one
    `materials[]` entry with three indices pointing at index 0.
- `tests/encoder_options_roundtrip.rs` — 18 tests pinning every
  configuration knob the published 0.0.0 sibling crates expose:
  - **STL** — `new_binary` / `new_ascii` / `new(StlFormat)` /
    `default()` constructor parity (byte-equal output, `format()`
    getter agrees), `solid` token marker on ASCII output, 80-byte
    header + `u32` triangle count layout on binary output, both
    flavours round-trip to the same decoded positions.
  - **OBJ** — `ObjEncoder::new()` ↔ `default()` byte-equal,
    `with_mtl_basename("foo")` injects exactly one
    `mtllib foo.mtl` directive, default emits zero, the directive
    survives a re-encode via `Scene3D::extras["obj:mtllibs"]`.
    `MtlEncoder::new()` output re-parses through `parse_mtl` to
    the same material name + base colour and matches the
    free-function `serialize_mtl` byte-for-byte.
  - **glTF** — `new()` and `with_output(Glb)` both target
    `OutputFlavour::Glb` and decode to identical positions
    (byte-equal not asserted: glTF JSON serialises a `HashMap`
    whose iteration order varies per process). `OutputFlavour::default()`
    is `Glb`. `Glb` bytes start with `b"glTF"` magic;
    `JsonEmbedded` bytes start with `b'{'`. The `json_encoder()`
    helper picks `JsonEmbedded` and decodes equivalently to
    `with_output(JsonEmbedded)`. Both flavours decode the same
    `Scene3D` to identical positions and material name.

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
