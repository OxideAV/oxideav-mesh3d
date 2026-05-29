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

Round 10 lands the typed morph-blend evaluator
(`tests/morph_apply.rs`, 20 tests):

- `Primitive::apply_morph_weights(weights: &[f32]) -> MorphedAttributes`
  — pure-Rust evaluation of the glTF 2.0 §3.7.2.2 morph formula
  `morphed[k] = base[k] + Σ weights[i] * targets[i].ATTR[k]` over the
  three typed slots (`POSITION`, `NORMAL`, `TANGENT`). The base
  attribute presence drives output presence (a target slot named when
  the base is absent is silently dropped per spec line 3586).
  Tangent handedness `w` is preserved verbatim per spec line 3616
  (delta is xyz-only). Missing or excess weights default to zero per
  spec line 3697; buffer-length mismatches apply the prefix and
  leave the remainder untouched. Empty weights / no targets short-
  circuit to a verbatim base clone.
- `MorphedAttributes { positions, normals, tangents }` — `Clone +
  Debug + PartialEq` output struct re-exported from the crate root.

Round 97 lands strip/fan → triangle-list de-stripping
(`tests/destrip.rs`, 25 tests):

- `Primitive::triangle_indices(&self) -> Vec<[u32; 3]>` — expands the
  primitive's topology into a flat list of triangle vertex-index
  triples following the OpenGL/glTF primitive-assembly rules.
  `TriangleStrip` applies the alternating-winding rule (odd-numbered
  triangles swap their last two vertices so front-facing winding stays
  consistent); `TriangleFan` shares the anchor vertex `v[0]`;
  `Triangles` returns its triples verbatim (dropping a trailing
  incomplete triple). When an index buffer is present each entry is
  dereferenced so the result indexes the vertex pool (not the index
  buffer), `U16`/`U32` both widened to `u32`. Non-triangle topologies
  (lines/points) return an empty list. Output count equals
  `triangle_count()` for triangle topologies.
- `Primitive::to_triangle_list(&self) -> Primitive` — materialises an
  equivalent `Topology::Triangles` primitive with a fresh `U32` index
  buffer (the flattened `triangle_indices`). Attribute buffers,
  `material`, morph `targets`, and `extras` are carried over verbatim —
  only connectivity is rewritten. STL (list-only) and OBJ encoders that
  can't emit strips/fans natively consume this to flatten glTF/FBX
  strip primitives.

Round 105 lands per-vertex MikkTSpace-style tangent-space basis
recomputation (`tests/compute_tangents.rs`, 28 tests):

- `Primitive::compute_tangents(&self, uv_set: usize) -> Option<Vec<[f32; 4]>>` —
  derives per-vertex tangents from positions, the selected UV channel
  (`uv_set` indexes `Primitive::uvs`), and the existing per-vertex
  normals. Each `[f32; 4]` is `xyz` = unit tangent T plus `w` = ±1.0
  handedness, so the renderer reconstructs the bitangent as
  `B = w * (N × T)` exactly the way glTF 2.0 §3.7.2.1 / the MikkTSpace
  contract specifies the `TANGENT` accessor. The math is the closed
  form `T = (Δv2·E1 − Δv1·E2) / det`, `B = (−Δu2·E1 + Δu1·E2) / det`
  obtained by inverting the per-triangle 2×2 UV-delta linear system
  (Lengyel, "Computing Tangent Space Basis Vectors for an Arbitrary
  Mesh" 2001; the same derivation appears in the Normal Mapping
  chapter of Akenine-Möller, Haines & Hoffman, *Real-Time
  Rendering*). Per-triangle contributions are accumulated with the
  numerator scaled by `sign(det)` (so the area weighting is by
  unsigned UV area `|det|/2`, while a mirrored UV chart still pulls
  T in the +U surface direction); per-vertex sums are then
  Gram-Schmidt orthonormalised against N, and handedness is recovered
  from `sign((N × T') · B_sum)`. Topology integration goes through
  `triangle_indices`, so `Triangles` / `TriangleStrip` /
  `TriangleFan` all feed in. Returns `None` when prerequisites are
  missing (no normals, UV set absent, length mismatch); otherwise
  output length always equals `positions.len()`, with unreferenced /
  degenerate / T-parallel-to-N vertices falling back to
  `[1.0, 0.0, 0.0, 1.0]` so the result is always renderable. Pure
  (no `self` mutation) — assign to `Primitive::tangents`. This is the
  recompute step a format decoder runs when the wire stream omits
  tangents (OBJ has no native tangent channel, glTF without
  `TANGENT`).

Round 101 lands area-weighted per-vertex normal recomputation
(`tests/compute_normals.rs`, 22 tests):

- `Primitive::compute_normals(&self) -> Vec<[f32; 3]>` — recomputes
  smooth per-vertex normals from the primitive's triangle connectivity
  (the de-stripped list from `triangle_indices`, so `Triangles` /
  `TriangleStrip` / `TriangleFan` all feed in correctly). Each
  triangle's un-normalised face normal is the edge cross product
  `(P[b]-P[a]) × (P[c]-P[a])`; because its magnitude is twice the
  triangle area, accumulating it into each incident vertex and
  normalising the sum gives the **area-weighted** average of the
  neighbouring face normals — the textbook smooth-shading recomputation
  (Gouraud 1971; Foley, van Dam et al., *Computer Graphics: Principles
  and Practice*). CCW = front-facing (right-handed, glTF-aligned).
  Output length always equals `positions.len()`. Unreferenced vertices,
  vertices touched only by degenerate (collinear/coincident) faces,
  out-of-range index entries, and NaN-producing faces fall back to
  `[0, 0, 1]` rather than a zero vector or a panic; non-triangle
  topologies produce an all-fallback buffer. Pure (no `self` mutation) —
  assign the result to `Primitive::normals` to store. This is the
  recompute step a format decoder runs when the wire stream omits
  normals (OBJ without `vn`, glTF without `NORMAL`).

Round 155 lands mesh-validity invariants — degenerate-triangle
detection + edge-manifold classification (`tests/mesh_validity.rs`,
34 tests):

- `Primitive::degenerate_triangles(&self) -> Vec<usize>` — returns the
  indices into `triangle_indices()` for triangles whose three corners
  are collinear or coincident in 3D (i.e. zero-area). Detection is the
  same `|E1 × E2| == 0` test that `compute_normals` / `compute_tangents`
  already use to silently drop bad faces; this is the **detection-only**
  counterpart so a validator can warn or a repair pass can prune them.
  No epsilon thresholding (a triangle that is *almost* collinear within
  float precision but produces a non-zero cross product is reported
  valid — proximity-based pruning is a separate lossy op, out of scope).
  Out-of-range index entries and NaN-producing faces are also reported.
  Non-triangle topologies (lines/points) return an empty `Vec`. Pure;
  `O(triangle_count)`.
- `Primitive::edge_manifold_report(&self) -> EdgeManifoldReport` +
  `EdgeManifoldReport { total_edge_count, boundary_edge_count,
  manifold_interior_edge_count, non_manifold_edge_count, max_edge_use }`
  + `is_closed_manifold(&self) -> bool` — classifies every undirected
  triangle edge by use count: `1` (boundary — hole/crack/open rim),
  `2` (manifold-interior — clean two-manifold seam), `≥ 3` (non-manifold —
  T-junction / book-spine / feather). A closed two-manifold mesh has
  `boundary_edge_count == 0 && non_manifold_edge_count == 0`, which is
  exactly the STL spec's **vertex-to-vertex rule** ("each triangle must
  share two vertices with each of its adjacent triangles" — Fabbers /
  Stratasys 1989) and the classical solid-printable condition. Triangles
  with a duplicate corner index or an out-of-range entry are excluded
  whole (their bogus edges don't pollute neighbour counts). Topology
  comparison is by **vertex index**, not by 3D position — run
  `weld_vertices` first if positional duplicates should merge before
  counting. Pure; `O(triangle_count)`.

Round 118 lands coincident-vertex welding / index de-duplication
(`tests/weld_vertices.rs`, 29 tests):

- `Primitive::weld_vertices(&self) -> Primitive` — merges bit-identical
  rendering vertices into a shared pool and returns an equivalent
  **indexed** primitive whose attribute buffers hold only the distinct
  vertices, with the index buffer rewritten to reference the
  deduplicated pool. This is the inverse of attribute explosion: a
  decoder for a non-shared format (binary STL stores three fresh
  vertices per facet; an OBJ `f` corner is a distinct rendering vertex)
  produces a vertex *soup* where coincident corners are duplicated;
  welding collapses them so the GPU's post-transform vertex cache can
  reuse a shared vertex. Two source vertices merge iff **every present
  attribute slot is bit-identical** — `positions`, each `NORMAL` /
  `TANGENT`, every UV and colour set, the `joints` / `weights` quads,
  *and* every [`MorphTarget`] delta — because one index in an indexed
  draw selects one tuple across all attribute streams at once (so a UV
  seam or a hard-edge normal correctly stays split). Float keys are
  exact (bit pattern), with `-0.0` folded to `+0.0` and every `NaN`
  canonicalised so dedup stays deterministic; no epsilon tolerance
  (proximity welding is a separate lossy op, out of scope). `topology`
  is preserved verbatim (valid for triangles/strips/fans/lines/points,
  not just triangle lists); an existing index buffer is remapped
  through the dedup table (out-of-range entries dropped, not panicked),
  a non-indexed input materialises its implicit order. Index width
  follows glTF promotion: `U16` while the pool is `≤ 65 536` entries,
  else `U32`. The pool is gathered in first-seen order so the result is
  reproducible; `material` / `targets` shape / `extras` carry over.
  Pure (no `self` mutation). A position-only cube soup welds 36 → 8
  corners; a flat-shaded cube (normal in the identity) welds 36 → 24.

Round 182 lands the signed-volume reduction (`tests/volume.rs`,
30 tests):

- `Primitive::signed_volume(&self) -> f64` — divergence-theorem
  reduction `V = (1/6) Σ (P_a · (P_b × P_c))` over the primitive's
  triangle tessellation, in the unit-cubed of `Primitive::positions`.
  The derivation comes from substituting the radial field `F = x/3`
  (`∇ · F = 1`) into Gauss's theorem `∫∫∫_V (∇·F) dV = ∫∫_S F · dS`,
  which collapses each triangle's contribution to
  `(P_a · (P_b × P_c)) / 6` — geometrically, each triangle plus the
  origin forms a tetrahedron whose signed volume is that scalar
  triple product, and the origin-coincident faces cancel pairwise
  for a closed mesh, leaving only the boundary shells (Cha & Chen,
  "Efficient feature extraction for 2D/3D objects in mesh
  representation", ICIP 2001; the closed form also appears in any
  introductory divergence-theorem treatment, e.g. Marsden & Tromba,
  *Vector Calculus*). The cross-product machinery is the same one
  `compute_normals` / `surface_area` already share — `signed_volume`
  adds one scalar dot per triangle. Sign follows the winding
  convention: CCW-from-outside (right-handed, glTF-aligned) is
  positive; an inside-out mesh produces the same magnitude with the
  opposite sign. Topology integration goes through
  `triangle_indices`, so `Triangles` / `TriangleStrip` (alternating
  winding) / `TriangleFan` all feed in correctly; non-triangle
  topologies (lines/points) contribute 0.0. Accumulator is `f64` so
  million-triangle meshes don't drift under `f32` summation.
  Degenerate (collinear/coincident corners), NaN- or Inf-producing
  faces, and out-of-range index entries all contribute 0.0 — the
  result is always finite. **Translation-invariant for a closed
  surface** (origin-coincident tetra contributions cancel). Only
  physically meaningful for a closed two-manifold (see
  `is_closed_manifold`); arithmetically well-defined regardless.
  Pure; cost `O(triangle_count)`.
- `Primitive::volume(&self) -> f64` — unsigned `|signed_volume()|`,
  robust to inside-out winding.
- `Mesh::signed_volume(&self) -> f64` / `Mesh::volume(&self) -> f64`
  — sum across every contained primitive (mesh-local, no transforms
  / skin pose / morph deltas). `Mesh::volume` is `|Σ signed|`, not
  `Σ |signed|` (single-shell assumption); for a multi-shell mesh,
  sum each primitive's `volume()` separately.
- `Scene3D::signed_volume(&self) -> f64` /
  `Scene3D::volume(&self) -> f64` — sum across every mesh in the
  scene. Walks meshes once, not node instances — a mesh instanced
  by two nodes contributes its volume once. For a transform-aware
  total, walk `world_node_transforms` and apply each node's scale's
  signed determinant per primitive instance (a negative scale flips
  winding and thus flips the sign).

Round 175 lands the surface-area reduction (`tests/surface_area.rs`,
30 tests):

- `Primitive::surface_area(&self) -> f64` — total area of the
  primitive's triangle tessellation in the unit-squared of
  `Primitive::positions` (matching the parent `Scene3D::unit`). Each
  triangle's area is the half cross-product magnitude `|E1 × E2| / 2`
  (Marsden & Tromba, *Vector Calculus* — the cross-product magnitude
  is the parallelogram-area definition; a triangle occupies half of
  that parallelogram). The same `E1 × E2` already drives
  `compute_normals` (its magnitude is twice the triangle area, which
  is exactly why summing the un-normalised face normal into each
  vertex automatically area-weights smooth shading);
  `surface_area` reuses the edge-cross machinery and divides by two.
  Topology integration goes through `triangle_indices`, so
  `Triangles` / `TriangleStrip` (alternating winding) / `TriangleFan`
  all feed in correctly; non-triangle topologies (lines/points)
  contribute 0.0. Accumulator is `f64` so million-triangle meshes
  don't drift under `f32` summation. Degenerate (collinear/coincident
  corners), NaN- or Inf-producing faces, and out-of-range index
  entries all contribute 0.0 — the result is always finite. Pure;
  cost `O(triangle_count)`.
- `Mesh::surface_area(&self) -> f64` — sum across every contained
  primitive (mesh-local, no transforms / skin pose / morph deltas).
- `Scene3D::surface_area(&self) -> f64` — sum across every mesh in the
  scene. Walks meshes once, not node instances — a mesh instanced by
  two nodes contributes its area once. For a transform-aware total,
  walk `world_node_transforms` and apply the per-node scale
  determinant per primitive instance.

## Round 11 candidates

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
