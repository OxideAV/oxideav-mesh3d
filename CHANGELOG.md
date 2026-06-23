# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Round 362 (quadric-error-metric edge-collapse simplification)

- New `simplify` module — **error-optimal** mesh decimation, the
  high-quality counterpart of the grid-snapping
  `Primitive::simplify_cluster` and the reduction dual of
  `Primitive::subdivide_loop`.
- `Primitive::simplify_quadric(target_triangles)` — greedy
  quadric-error-metric edge collapse. Each vertex accumulates the sum of
  the fundamental error quadrics (`Kp = p·pᵀ`) of its incident triangle
  planes; the reducer repeatedly collapses the edge whose merged residual
  `v̄ᵀ Q̄ v̄` is smallest, placing the survivor at the error-minimising
  point (solved from the `3×3` quadric sub-block, with an endpoint /
  midpoint fallback when singular). A min-heap with per-vertex
  version-stamp lazy deletion drives the schedule (`~O(E log E)`).
- Legality guards keep the result a clean, crack-free, same-silhouette
  approximation: a **fold-over guard** rejects any collapse that would
  flip or sliver a surviving incident triangle; a **boundary lock**
  forbids boundary→interior collapse and adds a perpendicular constraint
  quadric per boundary edge so open seams keep their shape; non-manifold
  edges (≠ 2 owning faces) and link-condition-violating collapses are
  left intact.
- Attributes follow the `subdivide_loop` conventions: position is driven
  by the metric, every other attribute (`normals`, `tangents`, `uvs`,
  `colors`, `weights`, morph deltas) is the linear blend of the two
  endpoints at the optimal split fraction — normals / tangent-`xyz`
  renormalised, tangent `w` and categorical `joints` from the surviving
  endpoint, `weights` renormalised to sum 1. Output is an indexed
  `Triangles` primitive with unreferenced vertices pruned; index width
  follows the crate convention. Robust against degenerate / non-finite /
  non-triangle / empty input; does not mutate `self`.
- `Primitive::simplify_quadric_error(max_error)` — the **error-bounded**
  companion: collapses until the cheapest remaining legal collapse would
  add more than `max_error` squared-distance error to the surface, so the
  surface never deviates past the budget at any single step. `0.0`
  performs only zero-cost (coplanar) collapses; a non-finite / negative
  bound means "no bound" (reduces to the legality floor, like
  `simplify_quadric(0)`). The stop test reads the monotone heap key, and
  a collapse whose re-validated cost rose above its key is re-queued
  rather than applied out of order, keeping the bound sound.
- 21 integration tests (`tests/simplify_quadric.rs`): empty/degenerate →
  empty, target-above-input no-op, flat-plane hard reduction staying
  planar, closed-octahedron lower bound, in-range / non-degenerate
  output faces, no unreferenced vertices, `self` immutability, attribute
  carry-through + weight renormalisation, strip-input flattening,
  NaN-vertex exclusion, a raised-apex grid whose high-curvature peak
  survives, plus error-bound coverage (zero-bound planar-only reduction,
  zero-bound apex preservation, monotonicity in the budget, infinite ≡
  `simplify_quadric(0)`, negative/NaN ≡ no-bound, empty input, well-formed
  output).

### Added — Round 354 (GPU vertex-buffer optimisation, part 1: post-transform cache)

- New `optimize` module — triangle/vertex reordering for locality of
  reference. The rendered result is invariant; only memory order changes.
- `simulate_cache(indices, cache_size)` + `CacheStats` — a FIFO
  post-transform vertex-cache simulator yielding `misses`, **ACMR**
  (`misses / triangles`) and **ATVR** (`misses / vertices`). The
  measurement primitive that makes the reordering verifiable.
- `Primitive::cache_stats(cache_size)` — runs the simulator over the
  de-stripped triangle list (strips / fans expanded); `DEFAULT_CACHE_SIZE
  = 32` is the optimiser's reference size.
- `Primitive::optimize_vertex_cache()` (+ `_sized(cache_size)`) — a
  greedy, near-linear score-driven triangle reordering that maximises
  post-transform vertex reuse. Each candidate triangle is scored by its
  vertices' cache residency (recency) plus a low-remaining-valence bias;
  the best triangle is emitted, its vertices pushed to the cache front,
  and only incident triangles re-scored. Disconnected components seed
  from input order. Returns an indexed `Triangles` primitive with the
  **vertex pool unchanged** — only triangle order is permuted.
  Out-of-range corners are dropped; non-triangle topologies pass through.
- 19 integration tests (`tests/optimize_cache.rs`): simulator edge cases,
  triangle-set + pool invariance, determinism, disconnected components,
  and a measured ACMR drop (a 24×24 scrambled grid falls below 1.0 ACMR).

### Added — Round 354 (GPU vertex-buffer optimisation, part 2: vertex fetch)

- `Primitive::optimize_vertex_fetch()` — linearises the vertex pool into
  the order the index buffer first references each vertex, so the
  pre-transform attribute fetch sweeps the buffer front-to-back. Every
  attribute stream (`positions`, `normals`, `tangents`, all `uvs` /
  `colors` sets, `joints`, `weights`) and every `MorphTarget` delta is
  gathered into the new slot order; the index buffer is relabelled
  in place (an implicit `0..n` order is materialised). Vertices the index
  buffer never touches are appended in original order, so a re-index
  round-trip is lossless. Out-of-range references are dropped. Valid for
  every topology — it relabels the pool, never the stitching rule, so
  strips / fans keep their topology.
- Recommended pipeline: `optimize_vertex_cache().optimize_vertex_fetch()`
  — fix the triangle order first, then relabel the pool to match that
  final draw order.
- 11 integration tests (`tests/optimize_fetch.rs`): first-use relabel,
  full attribute carry, unreferenced-vertex preservation, fetch-scatter
  reduction, topology preservation, and a lossless cache→fetch pipeline
  check asserting contiguous first-use order on the output.

### Added — Round 354 (GPU vertex-buffer optimisation, part 3: spatial Z-order)

- `Primitive::optimize_vertex_spatial(bits)` — sorts the vertex pool
  along a Morton (Z-order) space-filling curve so spatially-adjacent
  vertices land at adjacent storage slots. Each position is quantised to
  a `2^bits`-per-axis lattice over the bounding box (`bits` clamped
  `1..=10`, a 30-bit code) and bit-interleaved; the stable sort on that
  code is the permutation. The spatial-locality companion to the two
  index-driven passes — the right heuristic for **non-indexed** draws
  (point clouds, triangle soup) and **seam-heavy** meshes where the index
  buffer reuses almost nothing.
- All attribute streams + morph deltas are gathered into Z-order; any
  index buffer is remapped through the permutation so the draw is
  unchanged. Non-finite positions sort to the end; a degenerate axis
  collapses to lattice 0; empty primitives round-trip.
- 11 integration tests (`tests/optimize_spatial.rs`): pool-multiset +
  geometry invariance, index remap, cluster contiguity, a measured
  neighbour-distance cut (>40% on a scrambled 16×16 grid), attribute
  carry, non-finite ordering, bit clamping, and a spatial→cache pipeline.

### Added — Round 336 (uniform-grid clustering decimation: `Primitive::simplify_cluster`)

- `Primitive::simplify_cluster(grid)` — the reduction dual of
  `subdivide_loop`. Lays a cube grid (`grid` cells along the longest
  bounding-box axis, equal cell edge on every axis) over the primitive,
  collapses every vertex sharing a cell to one representative, and
  re-emits the connectivity over the occupied cells. Produces a coarser
  watertight-by-construction indexed `Triangles` proxy suitable as a
  level-of-detail / collision hull.
  - **Robust, no panic path**: out-of-range / duplicate-corner triangles,
    non-finite vertex positions, non-triangle topologies, and empty
    primitives all yield an empty indexed `Triangles` primitive; a
    non-finite vertex is dropped from the grid frame and clamped to
    cell 0. `grid` is clamped to `>= 1` (`1` → empty; very fine grid →
    welded input).
  - **Attribute-aware**: positions and every present attribute
    (`normals`, `tangents`, all `uvs` / `colors` sets, `weights`, each
    `MorphTarget` delta) are averaged per cell; normals / tangent `xyz`
    are re-normalised, tangent handedness `w` takes the cell-majority
    sign, `weights` re-normalise to sum 1, and `joints` are carried from
    a member verbatim (categorical, never averaged). `material`,
    `targets` roster, and `extras` carry over.
  - **De-duplicated faces**: two input triangles collapsing onto the
    same unordered cell triple emit one output face; faces spanning
    fewer than three distinct cells are dropped; orphan cells are
    pruned so the output pool has no unreferenced slots.
  - 21 integration tests (`tests/simplify_cluster.rs`).

### Added — Round 329 (typed ratified-KHR material extensions: `Material::ext`)

- `MaterialExt` — a typed surface for the simply-shaped ratified
  Khronos KHR dielectric material extensions, hung off `Material::ext`
  (all absent / `false` by default = plain core metallic-roughness):
  - **`emissive_strength: Option<f32>`** — the unitless multiplier on
    the core emissive term (`KHR_materials_emissive_strength`), lifting
    emission out of the core `[0,1]` clamp for HDR bloom. Spec default
    `1.0`.
  - **`ior: Option<f32>`** — index of refraction of the dielectric
    BRDF (`KHR_materials_ior`), replacing the fixed `1.5`. Valid `>= 1`,
    with the special case `0.0` permanently selecting the legacy
    specular-glossiness backwards-compatibility mode (effective IOR →
    +∞). The sentinel `0.0` round-trips verbatim, never normalised.
  - **`specular: Option<Specular>`** — `KHR_materials_specular`
    strength + F0 colour; each a constant factor (`factor` default
    `1.0`, `color_factor` default `[1,1,1]`) optionally modulated by a
    `TextureRef` (alpha channel for strength, RGB for colour).
  - **`unlit: bool`** — the `KHR_materials_unlit` flag selecting a
    constant-shaded, lighting-independent model.
- Builders `Material::with_emissive_strength` / `with_ior` /
  `with_specular` / `with_unlit`, and `effective_emissive_strength()` /
  `effective_ior()` accessors that substitute the normative default
  when the extension is absent.
- `Scene3D::validate` now resolves the two `Specular` texture
  references (`ext.specular.factor_texture` / `color_texture`) against
  the texture arena, reporting dangling ids alongside the core slots.
- Re-exports: `MaterialExt`, `Specular`.

### Added — Round 324 (Loop subdivision refinement: `Primitive::subdivide_loop`)

- `Primitive::subdivide_loop(&self) -> Primitive` — one step of **Loop
  subdivision** (Charles Loop, *Smooth Subdivision Surfaces Based on
  Triangles*, master's thesis, University of Utah, 1987). Welds the
  input (so the neighbourhood masks see shared vertices, not a per-face
  soup), then replaces every triangle with the classic `1 → 4` fan: the
  three corner sub-triangles plus a central one wound CCW-consistent
  with the parent (glTF 2.0 §3.7.2.1).
- **Edge vertices** (one new vertex per undirected edge `A–B`):
  interior edge with opposite apexes `C`, `D` →
  `3/8·(A+B) + 1/8·(C+D)`; boundary / non-manifold edge → the midpoint
  `1/2·(A+B)`, so a seam subdivides to the same limit curve regardless
  of the interior and stays crack-free.
- **Repositioned original vertices**: interior valence-`n` vertex with
  one-ring sum `S` → `(1 − n·β)·V + β·S` using Warren's `β = 3/16`
  (`n == 3`) / `β = 3/(8n)` (`n > 3`); boundary vertex with two
  boundary neighbours `B0`, `B1` → `3/4·V + 1/8·(B0+B1)` (cubic-B-spline
  curve mask); corner / pinch boundary vertices stay put.
- Positions carry the Loop masks; **all other attributes** (`normals`
  — re-normalised after interpolation, `tangents` — xyz interpolated +
  re-normalised with the endpoint-`a` handedness `w`, every `uvs` /
  `colors` set, `weights`, and each `MorphTarget` delta) are **linearly
  interpolated** at edge midpoints, originals unchanged. `joints` copy
  the lower-index endpoint's quad (indices aren't interpolable; the
  blended weights carry the influence split).
- Robust: malformed triangles (out-of-range / duplicate-corner) are
  dropped against `self` *before* welding (so the weld never sees a
  mis-grouped index stream); non-finite mask results fall back to the
  welded position / midpoint; non-triangle / empty inputs return a
  de-stripped empty `Triangles` primitive. Output index width promotes
  `U16 → U32` past 65 536 vertices like `weld_vertices`. Boundaries stay
  watertight (`boundary_loops` count preserved); closed two-manifolds
  stay closed. Pure; `material` / `targets` roster / `extras` carried
  through. (`tests/subdivide_loop.rs`, 25 tests.)

### Added — Round 316 (Hole-filling cap pass: `Primitive::fill_holes`)

- `Primitive::fill_holes(&self) -> Primitive` — caps every boundary loop
  of the surface with an ear-clip patch, the headline use the round-267
  `boundary_edges` / round-275 `boundary_loops` docs name ("each closed
  loop is a polygon a fan / ear-clip triangulator can cap to make the
  surface watertight"). Traces each hole / crack / open rim with
  `boundary_loops`, projects the (generally non-planar) loop onto its
  best-fit plane via **Newell's** area-vector normal
  `N = ½ Σ (Pᵢ × Pᵢ₊₁)`, expresses it in an orthonormal in-plane basis
  `(u, v)` with `u × v = N̂`, and **ear-clips** in 2D by the two-ears
  theorem (Meisters, "Polygons Have Ears", *American Mathematical
  Monthly* 82(6), 1975). The result is a de-stripped
  [`Topology::Triangles`] copy (via `to_triangle_list`) with the cap
  triangles appended; caps reference existing pool indices only, so no
  new vertices are added and every attribute buffer (`normals`,
  `tangents`, `uvs`, `colors`, `joints`, `weights`, morph `targets`)
  stays index-aligned and is carried over unchanged.
- Each cap triangle is wound to **cross every boundary edge opposite to
  the loop traversal**: a boundary half-edge keeps the orientation of the
  single triangle that owns it (glTF 2.0 §3.7.2.1: CCW = front-facing),
  so crossing it the other way makes each filled interior edge traversed
  once each way — the same manifold-consistency condition
  `orient_consistent` enforces — and the patch's front-face normal agrees
  with the surrounding surface (so `signed_volume`'s sign is preserved; a
  cube / tetrahedron with a face removed refills to its original positive
  volume).
- **Every** loop is capped: a free-floating patch has no intrinsic
  "outer" rim distinct from an interior hole, so a flat patch with a hole
  is closed into a zero-thickness double-sided shell with both filled.
  Loops with fewer than three distinct projected vertices, a non-finite
  Newell normal, a zero-length area vector (collinear), or any
  non-finite / out-of-range vertex are skipped without panic. Topology
  feed matches `boundary_loops` (`Triangles` / `TriangleStrip` /
  `TriangleFan`); a closed two-manifold or non-triangle topology returns
  the de-stripped surface unchanged. Intended pipeline
  `weld_vertices().fill_holes()` (boundary detection is by vertex index).
  Pure; `O(triangle_count + Σ kᵢ²)` over the loop lengths.
  (`tests/fill_holes.rs`, 15 tests.)

### Added — Round 313 (Winding-consistency repair: `orient_consistent`)

- `Primitive::orient_consistent(&self) -> (Vec<[u32; 3]>,
  OrientationReport)` — flood-fills the face-dual adjacency graph (the
  same shared-edge relation `triangle_adjacency` exposes) to bring every
  triangle of each edge-connected component into a consistent winding,
  fixing the mixed-winding vertex soup that binary STL / stitched-OBJ /
  CSG output produces. Two edge-adjacent triangles are consistent iff
  they traverse their shared undirected edge in **opposite** directions
  (glTF 2.0 §3.7.2.1: CCW = front-facing under a positive-determinant
  transform, so each interior edge of a coherent manifold is crossed
  once each way); a neighbour sharing the edge the **same** way is
  flipped (`[a, b, c] → [a, c, b]`, reversing the per-face normal). The
  reference is the **lowest-indexed valid triangle** per component (kept
  verbatim); winding is only defined relative to a seed, so this does not
  decide global "outward" orientation (use `signed_volume`'s sign or a
  known seed face for that), and each connected component is seeded
  independently. The returned faces are parallel to `triangle_indices`
  (same length / order); excluded triangles (out-of-range /
  duplicate-corner — same rules as `triangle_adjacency`) keep their slot
  verbatim and never link. Only clean manifold-interior edges (exactly
  two sharers) constrain; boundary (one) and non-manifold (≥ 3) edges are
  unconstrained, so the components they would have joined orient
  independently. `Triangles` / `TriangleStrip` (alternating winding
  pre-resolved) / `TriangleFan` feed in; non-triangle / empty primitives
  return `(vec![], OrientationReport::default())`. Deterministic
  (lowest-index seeds, FIFO walk) and pure; `O(triangle_count · α)`.
  (`tests/orient_consistent.rs`, 13 tests.)
- `OrientationReport { flipped_count, component_count, non_orientable }`
  — `Clone + Copy + Debug + Default + PartialEq + Eq` tally re-exported
  from the crate root. `non_orientable` is set when a component's
  face-dual walk reaches an already-decided triangle the current face
  contradicts (a Möbius-style loop the flood-fill cannot satisfy); the
  earlier decision is kept and the result is flagged best-effort.

### Added — Round 305 (Face-dual adjacency graph: `triangle_adjacency`)

- `Primitive::triangle_adjacency(&self) -> Vec<[Option<u32>; 3]>` — the
  explicit facet-adjacency (face-dual) graph that
  `topology_summary`'s union-find walks implicitly. Indexed by
  `triangle_indices` enumeration order; entry `i` is `[n01, n12, n20]`,
  the index of the triangle sharing triangle `i`'s edge `0→1` / `1→2`
  / `2→0`, or `None`. `None` for a boundary edge (one user) or a
  non-manifold edge (≥ 3 users — no single well-defined neighbour);
  only a clean manifold-interior edge (exactly two users) yields a
  symmetric `Some(other)` link, so total links =
  `2 · EdgeManifoldReport::manifold_interior_edge_count`. Edge bucketing
  and the out-of-range / duplicate-corner whole-triangle exclusion match
  `edge_manifold_report` / `topology_summary`; an excluded triangle
  keeps its (all-`None`, inert) slot so indices line up. Adjacency is by
  vertex index (run `weld_vertices` first to link a coincident seam);
  the `triangle_indices` feed flows `Triangles` / `TriangleStrip` /
  `TriangleFan` in; non-triangle and empty primitives return an empty
  `Vec`. Use cases: winding/normal-consistency repair, region-growing
  segmentation, triangle-strip generation, connected-component
  labelling. Pure; `O(triangle_count)`. (`tests/triangle_adjacency.rs`,
  26 tests.)

### Added — Round 299 (Combinatorial-topology summary: `topology_summary`)

- `Primitive::topology_summary(&self) -> TopologySummary` — rolls the
  triangle tessellation's connectivity graph up into the classical
  topological invariants: the **V / E / F** counts, the **Euler
  characteristic** `χ = V − E + F`, the connected-component count
  (facet adjacency via a path-halving union-find over the edge buckets),
  the boundary-loop count, and — for a single connected closed
  orientable two-manifold — the **genus** recovered from `χ = 2 − 2g`
  (`Some(0)` sphere/cube, `Some(1)` torus, `Some(2)` double-torus). All
  counts are by vertex index (run `weld_vertices` first); **V** counts
  only referenced vertices so unreferenced pool slots don't inflate the
  Euler identity. Face/edge bucketing and the out-of-range /
  duplicate-corner whole-triangle exclusion match
  `edge_manifold_report`; the `triangle_indices` feed flows `Triangles`
  / `TriangleStrip` / `TriangleFan` in. Genus is `None` for any open,
  multi-component, non-manifold, or non-orientable surface. Purely
  combinatorial (a NaN position does not change index-level
  connectivity). Pure; `O(triangle_count · α(V))` plus the boundary-loop
  walk it calls.
- `TopologySummary { vertex_count, edge_count, face_count,
  euler_characteristic, component_count, boundary_loop_count, genus }`
  — `Clone + Copy + Debug + Default + PartialEq + Eq` report struct,
  re-exported from the crate root.

### Added — Round 292 (Transform-aware inertia tensor: `world_inertia_tensor`)

- `Primitive::world_inertia_tensor(&self, world: [[f32; 4]; 4]) -> Option<[[f64; 3]; 3]>`
  — the world-frame sibling of `Primitive::inertia_tensor`, closing
  the per-instance gap round 259's prose flagged as the next-round
  candidate. Maps every corner through the row-major column-vector
  affine 4x4 `world` matrix into world coordinates, then evaluates the
  same origin-anchored per-tetrahedron second-moment integrals
  (Mirtich, JGT 1996; same derivation as `inertia_tensor` /
  `volume_centroid` / `signed_volume`) in the world frame. Folding the
  transform in at the corner level (rather than `M_3 · I_local · M_3ᵀ`
  plus a separate parallel-axis correction) handles rotation,
  non-uniform scale, skew, *and* translation in one pass — mirroring
  the `world_volume_centroid` / `world_signed_volume` corner-mapping
  style. Topology / degenerate / NaN / out-of-range skipping match
  `inertia_tensor`; non-triangle / empty / all-skipped return `None`.
  `f64` throughout. Pure; `O(triangle_count)`.
- `Mesh::world_inertia_tensor(&self, world) -> Option<[[f64; 3]; 3]>`
  — element-wise sum across every contained primitive's
  `world_inertia_tensor` (additivity of the second-moment integral
  over disjoint volumes; sign-aware so an inside-out subshell
  subtracts).
- `Scene3D::world_inertia_tensor(&self) -> Option<[[f64; 3]; 3]>` —
  per-instance world-frame total: depth-first walk over the
  `Scene3D::roots` forest (same cycle-guarded, leftmost-first,
  first-parent-resolution shape as `world_volume_centroid` /
  `world_node_transforms`), folding each reachable node's
  ancestor-chain world matrix into its mesh's tensor and summing
  element-wise. A mesh instanced under N nodes contributes N times.
  Result is about the world origin; shift to the centre of mass via
  the parallel-axis theorem with `world_volume_centroid`.
- `tests/world_inertia_tensor.rs` (24 tests) — identity passthrough,
  parallel-axis theorem under translation, `R · I · Rᵀ` similarity +
  trace-invariance under rotation, `s⁵` scaling under uniform scale,
  mirror sign-flip, symmetry, NaN-matrix / out-of-range / non-triangle
  / empty handling, strip↔list parity, mesh additivity + non-triangle
  skip, and scene-level identity-root / per-instance-doubling /
  ancestor-chain composition / cycle-guard / detached-mesh / two-
  instance-sum cross-checks.

### Added — Round 285 (Parametric extruded solids: `Profile2D` + ear-clip triangulation)

- `extrude::Profile2D { outer, holes }` (re-exported at the crate
  root) — planar profile in the *xy* plane: one outer boundary loop
  plus zero or more hole loops, the shape of the IFC 4.3
  extruded-area-solid `SweptArea` (clean-room source:
  `docs/3d/ifc/ifc43-entity-IfcExtrudedAreaSolid.html`, §8.8.3.15).
  Builders `new` / `with_hole`; accessors `vertex_count` (flattened
  list length) and `area` (shoelace, outer minus holes,
  winding-agnostic).
- `Profile2D::triangulate(&self) -> Option<Vec<[u32; 3]>>` —
  ear-clip triangulation of the enclosed area into counter-clockwise
  triangles indexing the flattened `outer ++ holes[…]` vertex list.
  Correctness rests on the two-ears theorem (Meisters, "Polygons
  Have Ears", *American Mathematical Monthly* 82(6), 1975); holes
  are reduced to the simple-polygon case by splicing each hole into
  the outer ring through a zero-width bridge found by a `+x` ray
  from the hole's rightmost vertex (refined against reflex occluders
  by smallest angle, then distance; holes merged largest-`x` first).
  Triangle count is `n + 2·h − 2` minus zero-area clips. Input
  windings are normalised (outer → CCW, holes → CW); closing /
  consecutive duplicate points are ignored; reflex-only occlusion
  scanning makes convex profiles clip in near-linear time. `None`
  on fewer than 3 distinct outer vertices, zero-area or non-finite
  loops, degenerate holes, or a hole outside the boundary.
- `Profile2D::extrude(&self, direction: [f32; 3], depth: f32) -> Option<Primitive>`
  — sweeps the profile into a closed, watertight indexed
  `Topology::Triangles` primitive: bottom ring at `z = 0`, top ring
  offset by `depth · direction / |direction|`, both caps from
  `triangulate`, two outward-facing wall triangles per boundary
  edge of every loop. Any direction with a non-zero z component is
  accepted (oblique prisms; the IFC validity rule); downward
  extrusion flips every winding so faces stay outward and
  `signed_volume` stays positive. Vertices are shared between caps
  and walls (`is_closed_manifold` holds, `boundary_edges` is empty);
  index width follows the crate's `U16`/`U32` promotion rule;
  `normals`/`uvs` are left for the `compute_normals` /
  `compute_tangents` post-passes. `None` on a non-triangulatable
  profile, non-positive/non-finite depth, or a zero / in-plane /
  non-finite direction.
- Covered by `tests/extrude_profile.rs` (33 tests): cap counts +
  area sums for convex/concave/holed/two-hole profiles, winding
  normalisation at both levels, duplicate-point tolerance, collinear
  handling, hollow-section + L-shape + two-hole watertightness,
  volume / surface-area / centroid cross-checks against the existing
  reductions, oblique / downward / non-unit directions, every `None`
  contract, and the 33 000-gon `U32` index-promotion path.

### Added — Round 275 (Boundary-loop chaining: `Primitive::boundary_loops`)

- `Primitive::boundary_loops(&self) -> Vec<Vec<u32>>` — chains the loose
  boundary edges of `Primitive::boundary_edges` end-to-end into ordered,
  winding-consistent vertex loops, one per hole / crack / open rim. Each
  loop is the ordered sequence of vertex-pool indices walked along the
  boundary in the surface's winding-consistent direction (a boundary
  half-edge keeps the orientation of the single triangle that owns it),
  rotated to start at the loop's smallest vertex index so the
  representation is seed-independent; the start vertex is not repeated at
  the end. The directed-half-edge chaining consumes every boundary
  half-edge exactly once, so the loops partition the boundary-edge set; a
  pinch / non-manifold vertex with multiple outgoing boundary half-edges
  picks the smallest-target continuation deterministically and seeds the
  remaining half-edges as their own loops. Same edge bucketing,
  out-of-range / duplicate-corner whole-triangle exclusion, and
  `triangle_indices` topology feed (`Triangles` / `TriangleStrip` /
  `TriangleFan`) as `boundary_edges`. Non-triangle topologies, empty
  primitives, and closed two-manifolds
  (`EdgeManifoldReport::is_closed_manifold`) return an empty `Vec`. The
  loop list is sorted ascending for determinism. Pure (no `self`
  mutation); cost
  `O(triangle_count + boundary_edge_count · log boundary_edge_count)`.
  This is the hole-detection / hole-filling pre-pass the round-267
  `boundary_edges` docs gesture toward — each closed loop is a polygon a
  fan / ear-clip triangulator can cap to make the surface watertight.
  Covered by `tests/boundary_loops.rs` (32 tests).

### Added — Round 267 (Boundary-edge extractor: `Primitive::boundary_edges`)

- `Primitive::boundary_edges(&self) -> Vec<[u32; 2]>` — returns every
  undirected triangle edge used by exactly one triangle: the holes,
  cracks, and open rims of a non-closed surface. This is the
  detection-only *extractor* counterpart to
  `EdgeManifoldReport::boundary_edge_count` (which only counts them),
  the same way `Primitive::degenerate_triangles` is the extractor
  counterpart to a degenerate-triangle count. Each `[u32; 2]` is the
  edge's two vertex-pool indices in ascending `[min, max]` order so the
  pair is canonical regardless of triangle winding; the output is
  sorted ascending so it is deterministic across runs (the underlying
  `HashMap` walk order is not).
- Edge bucketing matches `Primitive::edge_manifold_report` exactly:
  undirected edges keyed by `(min, max)` vertex index, counted over
  `Primitive::triangle_indices` (so `Triangles` / `TriangleStrip`
  alternating-winding / `TriangleFan` all feed in). Triangles with an
  out-of-range corner index or a duplicate corner index (a zero-length
  edge) are excluded whole before counting, so a malformed neighbour
  cannot wrongly close a valid triangle's boundary seam. Non-triangle
  topologies (lines/points) and empty primitives return an empty `Vec`.
  Topology comparison is by vertex index, not 3D position — run
  `weld_vertices` first to merge positionally-coincident corners.
- A closed two-manifold (`is_closed_manifold()`) returns an empty list;
  a non-empty result on a should-be-solid mesh names exactly where it is
  torn, complementing the aggregate `EdgeManifoldReport`. Headline uses:
  hole-detection / hole-filling pre-pass (chain the edges into boundary
  loops), open-rim wireframe outlining, watertightness diagnostics.
  Pure; `O(triangle_count + boundary_edges · log boundary_edges)`.
- `tests/boundary_edges.rs` — 21 tests: closed-tetrahedron emptiness,
  single open triangle, open quad (rim-only, count-matches-report),
  sorted/canonical/min-max-ordered output, winding independence,
  tetrahedron-with-one-face-removed rim exposure, non-manifold
  book-spine exclusion, out-of-range and duplicate-corner whole-triangle
  exclusion, excluded-neighbour seam reversion, lines/points gating,
  strip/fan topology feed-in, U16/U32 parity, idempotence, and the
  unwelded-seam weld-interaction caveat.

### Added — Round 259 (Unit-density inertia tensor: `Primitive` / `Mesh` / `Scene3D`)

- `Primitive::inertia_tensor(&self) -> Option<[[f64; 3]; 3]>` —
  unit-density inertia tensor of the solid enclosed by this
  primitive's closed triangle tessellation, taken about the
  **origin** of `Primitive::positions`, returned as a row-major
  symmetric `[[f64; 3]; 3]` matrix. Diagonal entries are the
  *moments* `I_αα = ∫_V (β² + γ²) dV` (the two coordinates not
  equal to α); off-diagonals carry the standard rigid-body minus
  sign, `I_αβ = -∫_V x_α · x_β dV`. Multiply by the body's
  density `ρ` to recover the physical tensor; apply the parallel-
  axis theorem (`I_about_C = I_about_O - M · D` with
  `D_αβ = c_α·c_β - δ_αβ·|c|²`) to shift to any other reference
  point — e.g. the centre of mass returned by
  `Primitive::volume_centroid`.
- Derivation reuses the same divergence-theorem decomposition
  `signed_volume` and `volume_centroid` already use: fan the closed
  surface into origin-anchored tetrahedra `(0, P_a, P_b, P_c)` and
  apply the closed-form per-tetrahedron second-moment integrals.
  For an origin-anchored tet (`P_0 = 0`, `P_1 = A`, `P_2 = B`,
  `P_3 = C`):
  `∫_T x_α² dV = (V/10)·(Σ_k p_α[k]² + Σ_{j<k} p_α[j]·p_α[k])` and
  `∫_T x_α·x_β dV = (V/20)·(2·Σ_k p_α[k]·p_β[k] + Σ_{j≠k} p_α[j]·p_β[k] / 2)`,
  the textbook closed form. Same lineage as Mirtich, "Fast and
  Accurate Computation of Polyhedral Mass Properties", *Journal
  of Graphics Tools* 1(2), 1996 (already cited for
  `Primitive::volume_centroid`), specialised here to the
  second-moment kernel.
- Topology integration goes through `triangle_indices`, so
  `Triangles` / `TriangleStrip` / `TriangleFan` all feed in
  correctly; non-triangle topologies (lines/points) return `None`.
  `f64` accumulators; per-triangle math is `f64`. Degenerate,
  NaN-/Inf-producing, and out-of-range-index faces are silently
  skipped — same robustness contract as `signed_volume` /
  `volume_centroid`. Returns `None` only when every face has been
  skipped or the primitive is empty / non-triangle. Pure;
  `O(triangle_count)`.
- Translation-equivariant for a closed surface in the sense that
  the tensor about the centre of mass is invariant under coordinate
  translation (the parallel-axis shift recovers it), even though
  `I_about_origin` shifts. Sign-equivariant under winding flip:
  an inside-out closed mesh produces the negated tensor (the
  same way `signed_volume` flips sign).
- `Mesh::inertia_tensor(&self) -> Option<[[f64; 3]; 3]>` —
  element-wise sum across every contained primitive. The
  second-moment integral is additive over disjoint volumes (same
  argument `Mesh::signed_volume` / `Mesh::volume_centroid` rest
  on), so the per-primitive tensors add component-wise.
  Sign-aware: an inside-out subshell subtracts component-wise.
  Skips primitives whose `inertia_tensor` returns `None`; returns
  `None` only when every primitive returned `None`.
- `Scene3D::inertia_tensor(&self) -> Option<[[f64; 3]; 3]>` —
  element-wise sum across every mesh in the scene. Walks meshes
  once, not node instances: a mesh instantiated by multiple
  reachable nodes contributes once, not once per node (matches
  `Scene3D::volume_centroid` / `Scene3D::signed_volume` resource-
  level counting). Result is in the scene's local frame; for a
  per-instance world-frame total, walk `world_node_transforms`
  alongside `Primitive::inertia_tensor` and apply the rigid-body
  transform rule (`I_world = M_3 · I_local · M_3ᵀ` + parallel-axis
  correction).
- Test coverage in `tests/inertia_tensor.rs` (18 cases):
  - **Unit cube about its corner** — `[0, 1]³` CCW; expected
    `diag(2/3, 2/3, 2/3)` with `-1/4` off-diagonals (closed-form
    `∫ x² dV = 1/3`, `∫ x·y dV = 1/4`).
  - **Centred unit cube** — `[-1/2, 1/2]³`; expected
    `diag(1/6, 1/6, 1/6)` with off-diagonals at numerical zero
    (reflection symmetry collapses the antisymmetric integrand).
  - **Inside-out cube negates the tensor** — flipping every
    triangle's winding flips every component.
  - **Axis-aligned tetrahedron** — corners at origin + unit axes;
    closed-form `∫ x² dV = 1/60`, `∫ x·y dV = 1/120`.
  - **Symmetry** — off-diagonals match across the diagonal for
    cube, centred-cube, tetrahedron.
  - **Parallel-axis theorem** — `I_corner = I_centred + M·D` with
    `M = 1`, `D_αβ = c_α·c_β - δ_αβ·|c|²`, `c = (1/2, 1/2, 1/2)`.
  - **Empty / non-triangle topologies return `None`**.
  - **Uniform-scale law** — scaling all corners by `s` scales every
    component by `s⁵` (one `s²` from each `x_α` factor, one `s³`
    from the volume element).
  - **Out-of-range indices skipped without panic.**
  - **TriangleFan accepted** — a degenerate flat fan yields a
    near-zero tensor without NaN.
  - **`Mesh::inertia_tensor` additive across primitives.**
  - **Mesh skips non-triangle primitives; empty mesh returns `None`.**
  - **`Scene3D::inertia_tensor` additive across meshes**, empty
    scene returns `None`.
  - **Instance count is per-mesh, not per-node** — adding the same
    mesh to two nodes still yields one contribution.
  - **General axis-aligned box** — sides `(a, b, c) = (2, 3, 5)`
    centred at origin; expected diagonal
    `(M(b²+c²)/12, M(a²+c²)/12, M(a²+b²)/12)` with off-diagonals
    vanishing — the classical solid-cuboid formula.

### Added — Round 256 (Transform-aware volume-weighted centroid: `Primitive` / `Mesh` / `Scene3D`)

- `Primitive::world_volume_centroid(&self, world: [[f32; 4]; 4]) -> Option<[f64; 3]>`
  — centre of mass of the uniform-density solid enclosed by this
  primitive's closed triangle tessellation after every corner is
  mapped through the row-major column-vector affine 4x4 `world`
  matrix (same convention as `Transform::Matrix` /
  `BoundingBox::transform` / `Primitive::world_surface_area` /
  `Primitive::world_surface_centroid`). Derivation: substitute
  `M·P_*` for each `P_*` in `Primitive::volume_centroid`'s per-tet
  formula — per-tet signed volume becomes
  `V_i_world = ((M·P_a) · ((M·P_b) × (M·P_c))) / 6`, per-tet centroid
  becomes `C_i_world = (M·P_a + M·P_b + M·P_c) / 4`, and the
  textbook ratio `Σ V_i · C_i / Σ V_i` is accumulated in `f64`. For
  a closed two-manifold under invertible affine `M = [M_3 | t]` the
  result reduces analytically to `M · C_local = M_3 · C_local + t`
  (boundary terms cancel pairwise); for an open patch the
  origin-anchored tet sum picks up translation-dependent boundary
  terms — same caveat as the local helper. Topology integration
  goes through `triangle_indices`, so `Triangles` / `TriangleStrip`
  / `TriangleFan` all feed in; non-triangle topologies return
  `None`. `f64` accumulators; degenerate / NaN-/Inf- / out-of-range
  faces silently skipped. Returns `None` when the world-frame
  signed volume sum is `0.0` (every triangle degenerate, non-
  triangle topology, or transform flattens every tet — e.g.
  `[1, 1, 0]` scale) or non-finite. Pure; `O(triangle_count)`.
- `Primitive::world_signed_volume(&self, world: [[f32; 4]; 4]) -> f64`
  — companion helper computing the post-transform signed volume of
  the origin-anchored tetrahedron sum in the world frame (the
  denominator the centroid helper accumulates internally). For a
  closed mesh under affine `M` the result reduces to `det(M_3) ·
  signed_volume()`; for an open patch the translation column of
  `M` enters the result. Mirrors `Primitive::signed_volume`'s
  silent-skip policy for partly-corrupt buffers. `f64` accumulator;
  pure; `O(triangle_count)`.
- `Mesh::world_volume_centroid(&self, world: [[f32; 4]; 4]) -> Option<[f64; 3]>`
  — signed-volume-weighted recombination across every contained
  primitive: each primitive's `world_volume_centroid` is recovered
  with `world_signed_volume` as its weight, then divided once at
  the end. Additivity of the volume integral guarantees correctness
  across primitives; an inside-out subshell subtracts correctly.
  Skips primitives whose centroid is `None` or whose weight is
  `0.0` / non-finite. Pure; `O(Σ triangle_count_per_primitive)`.
- `Scene3D::world_volume_centroid(&self) -> Option<[f64; 3]>` —
  walks the `Scene3D::roots` forest with the same DFS shape as
  `Scene3D::world_signed_volume` / `Scene3D::world_surface_centroid`,
  applies each reachable node's full ancestor-chain world matrix to
  its primitive's triangle vertices, and recombines the per-instance
  centroids weighted by their per-instance post-transform signed
  volumes. A mesh instanced under two nodes contributes twice (once
  per instance), and each instance carries the world-space scale,
  skew, and translation on the path to that node. Cycles guarded
  the same way as `world_surface_centroid`. Returns `None` for
  empty / no-mesh / all-degenerate-instance / all-collapsed-transform
  scenes. Skin pose, morph targets, and unit-axis conversion not
  applied. Cost
  `O(reachable_nodes + Σ triangle_count_per_reachable_mesh)`.
- `tests/world_volume_centroid.rs` — 40 tests covering primitive-
  level identity / pure-translation / uniform-scale / non-uniform-
  scale / mirror / scale-then-translate / 180-degree-Z-rotation /
  `[0,0,0]`-scale-None / partial-collapse-None / non-triangle-None
  / empty-None / NaN-coord-skip / OOB-index-skip / post-divide-
  formula parity, signed-volume identity / translation-unchanged-
  for-closed / uniform-scale-cubes / non-uniform-det /
  mirror-flip-sign / rotation-unchanged, mesh-level passthrough /
  empty-None / all-degenerate-None / two-cubes-midpoint /
  translation-shift / cancelling-shells-None, and scene-level
  empty-None / no-roots-None / single-identity-matches-local /
  translation-shift / uniform-scale / two-instances-midpoint /
  per-instance-not-per-resource / unreachable-mesh-skipped /
  two-meshes / nested-transforms / mirrored-instance-cancels /
  intermediate-pure-xform-node / unreachable-node-skipped.

### Added — Round 247 (Volume-weighted centroid / centre of mass: `Primitive` / `Mesh` / `Scene3D`)

- `Primitive::volume_centroid(&self) -> Option<[f64; 3]>` — centre of
  mass of the uniform-density solid enclosed by this primitive's
  closed triangle tessellation. Derivation: fan the closed surface
  into origin-anchored tetrahedra (the same decomposition
  `Primitive::signed_volume` already uses for `∫∫∫ dV` via the
  divergence theorem); each tetrahedron `(0, P_a, P_b, P_c)` has
  signed volume `V_i = (P_a · (P_b × P_c)) / 6` and centroid
  `C_i = (P_a + P_b + P_c) / 4`. The whole-solid centre of mass is
  `(Σ V_i · C_i) / Σ V_i` — the textbook "Volume Integration"
  reduction (Mirtich, *Journal of Graphics Tools* 1(2), 1996,
  equation (1.16); Cha & Chen, ICIP 2001). Cross-product machinery
  is shared with `signed_volume`; this helper adds one corner-sum
  plus one scalar multiply per axis per triangle. Topology
  integration goes through `triangle_indices`, so `Triangles` /
  `TriangleStrip` / `TriangleFan` all feed in; non-triangle
  topologies (lines/points) return `None`. `f64` accumulators;
  degenerate / NaN-/Inf-/out-of-range faces silently skipped.
  Translation-equivariant for closed surfaces; sign-invariant
  (CW-from-outside winding produces the same centroid as
  CCW-from-outside). Returns `None` when `Σ V_i` is `0.0` or
  non-finite (flat sheet, perfectly cancelling shells). Only
  physically meaningful for a closed two-manifold; arithmetically
  well-defined regardless. Pure; `O(triangle_count)`.
- `Mesh::volume_centroid(&self) -> Option<[f64; 3]>` —
  signed-volume-weighted recombination of every contained
  primitive's centroid; the signed weights correctly handle
  inside-out subshells. Mesh-local; non-triangle / zero-signed-
  volume primitives skipped.
- `Scene3D::volume_centroid(&self) -> Option<[f64; 3]>` —
  signed-volume-weighted recombination of every mesh's centroid
  in the scene's local frame. Walks meshes once, not node
  instances (per-instance centroid needs an explicit walk of
  `world_node_transforms`).
- `tests/volume_centroid.rs` — 35 tests covering unit-cube centre
  of mass, CW-vs-CCW sign invariance, translation equivariance,
  non-uniform scale, axis-tetrahedron corner-mean, indexed vs
  unindexed parity (with signed-volume cross-check), strip /
  fan / lines / points / empty topology `None` paths, all-
  degenerate / collinear-triangle `None` paths, out-of-range
  index skipping, NaN-coord skipping, centred-cube → origin,
  distant-cube no-origin-dependence, centroid-inside-bounding-box,
  cube surface↔volume centroid agreement (uniform-density
  symmetric), tetrahedron surface↔volume divergence, mesh-level
  passthrough / empty / degenerate-skip / equal-weight midpoint /
  unequal-weight 1:8 / cancelling-shells / lines-skip, scene-level
  empty / passthrough / two-mesh midpoint / mesh-once-not-per-node
  / degenerate-skip / finite-components / three-mesh average /
  cube cross-check vs `surface_centroid`.

### Added — Round 234 (Transform-aware world surface centroid: `Primitive` / `Mesh` / `Scene3D`)

- `Primitive::world_surface_centroid(&self, world: [[f32; 4]; 4]) -> Option<[f64; 3]>`
  — area-weighted surface centroid after every corner is mapped
  through the row-major column-vector affine 4x4 `world` matrix
  (same convention as `Transform::Matrix` / `BoundingBox::transform`
  / `Primitive::world_surface_area`). The post-transform centroid is
  `(M·P_a + M·P_b + M·P_c) / 3` and the per-triangle area weight is
  `|(M_3·E1) × (M_3·E2)| / 2`; both bend with the transform in ways
  that don't factor through the local centroid alone, so a per-
  triangle accumulator (not `world * local_centroid` scaled by a
  single area factor) is the only faithful answer under non-uniform
  scale. Translation equivariant (under a pure translation `t` every
  centroid gains `t`); uniform scale `s` scales the result from the
  origin by `s`; mirror scales flip the mirrored axis. Topology
  handling, NaN/Inf guards, degenerate-triangle skipping, and out-of-
  range-index skipping mirror `Primitive::surface_centroid`. Returns
  `None` when the post-transform area accumulator stays at `0.0`
  (every triangle degenerate / non-triangle topology / transform
  flattens every triangle to zero area). `f64` accumulators; pure;
  cost `O(triangle_count)`.
- `Mesh::world_surface_centroid(&self, world: [[f32; 4]; 4]) -> Option<[f64; 3]>`
  — area-weighted recombination of every contained primitive's
  per-primitive world centroid weighted by the per-primitive world
  surface area (recovered via `Primitive::world_surface_area` so the
  per-primitive helper can keep its natural ratio shape rather than
  forcing callers through a numerator-denominator contract). Skips
  primitives whose `world_surface_area` is `0.0` / non-finite or
  whose `world_surface_centroid` returns `None`. Returns `None` when
  every primitive contributes nothing under `world`.
- `Scene3D::world_surface_centroid(&self) -> Option<[f64; 3]>` —
  transform-aware per-instance world centroid across every reachable
  node-mesh instance in the scene. Whereas `Scene3D::surface_centroid`
  walks meshes once regardless of how many nodes carry them,
  `world_surface_centroid` walks the `Scene3D::roots` forest the
  same way as `world_surface_area` / `world_signed_volume` /
  `world_node_transforms`, applies each reachable node's full
  ancestor-chain world matrix to its primitive's triangle vertices,
  and recombines the post-transform per-instance centroids weighted
  by their per-instance post-transform surface area. A mesh
  instanced under two nodes contributes twice (once per instance);
  cycles are guarded the same way as the sibling world helpers
  (each node is visited at most once; shared instances resolve via
  the first DFS arrival). Returns `None` when no reachable triangle
  survives.
- `tests/world_surface_centroid.rs` — 33 tests covering
  identity-passthrough (= local centroid bit-for-bit), pure-
  translation equivariance, uniform scale, non-uniform single-
  triangle scale, single-axis mirror, 90° rotation around +X,
  non-triangle topology → `None`, empty positions → `None`,
  degenerate scale collapse → `None`, out-of-range indices skip,
  NaN matrix → `None`, indexed vs unindexed parity, strip vs list
  parity, mesh-level single-primitive passthrough, mesh-level two-
  equal-area under scale, mesh-level all-degenerate → `None`, mesh-
  level degenerate primitive skip, mesh-level lines-primitive skip,
  scene-level empty / no-nodes / detached-mesh skip, identity-root
  passthrough, two instances translated apart, per-node scale
  weighting (small mesh + scaled big mesh closed-form sum),
  ancestor-chain composition (parent × child translation), cycle-
  guard (mutual `children` reference resolves once), per-instance
  degenerate-transform skip, out-of-range mesh-id skip, three
  identity-instanced meshes equal-weight average, finite-output
  guarantee, cross-check against `Scene3D::surface_centroid` for
  the identity-rooted case, and two identity-coincident instances.

### Added — Round 227 (Area-weighted surface centroid: `Primitive` / `Mesh` / `Scene3D`)

- `Primitive::surface_centroid(&self) -> Option<[f64; 3]>` —
  area-weighted geometric centroid of the primitive's triangle
  tessellation. The closed form is
  `(Σ area_i · centroid_i) / Σ area_i` from the textbook continuous
  identity `C = ∫∫_S x dS / ∫∫_S dS` (Marsden & Tromba, *Vector
  Calculus*); the per-triangle centroid is the corner average
  `(P_a + P_b + P_c) / 3` and per-triangle area is the same
  `|E1 × E2| / 2` already shared with `surface_area` and
  `compute_normals`. Topology integration via `triangle_indices`,
  so `Triangles` / `TriangleStrip` (alternating winding) /
  `TriangleFan` all feed in correctly; non-triangle topologies
  return `None`. Accumulators are `f64`; degenerate / NaN-/Inf-
  producing / out-of-range triangles are skipped (matching the
  silent-skip contract of every other reduction). Returns `None`
  only when no positive-area triangle survives. Translation-
  equivariant by construction; invariant under retriangulation of
  the same patch. Pure; `O(triangle_count)`.
- `Mesh::surface_centroid(&self) -> Option<[f64; 3]>` —
  area-weighted recombination of every contained primitive's
  centroid; additivity of the surface integral over a union of
  patches. Mesh-local; non-triangle / degenerate primitives
  contribute nothing.
- `Scene3D::surface_centroid(&self) -> Option<[f64; 3]>` —
  area-weighted recombination of every mesh's centroid in the
  scene's local frame. Walks meshes once, not node instances
  (same convention as `Scene3D::surface_area` / `signed_volume`);
  for a transform-aware per-instance total, walk
  `world_node_transforms` alongside `Primitive::surface_centroid`
  and combine per-instance centroids with their post-transform
  areas as weights.
- `tests/surface_centroid.rs` — 34 tests covering single-triangle
  corner-mean, translation equivariance, unit square / rectangle
  / cube centroid, indexed vs unindexed parity, barycentric-
  subdivision invariance, strip / fan / triangle-list parity,
  out-of-range-index skipping, NaN / Inf skipping, all-degenerate
  → `None`, empty / non-triangle topology → `None`, mesh-level
  passthrough + equal/unequal area weighting + degenerate-mesh
  skipping, scene-level passthrough + multi-mesh combinations +
  the "walks meshes once, not node instances" contract.

### Added — Round 220 (Scene-level BVH-of-instances on top of `world_node_bounds`)

- `instance_bvh` module — flat-array binary AABB tree over the
  per-instance world AABBs of a `Scene3D`. The seed previously
  flagged in round 216's docs as "the next acceleration layer above
  `Bvh::intersect_ray`" now ships.
- `InstanceBvh::build(&Scene3D) -> Option<Self>` and the
  `Scene3D::build_instance_bvh()` convenience wrapper. The build
  gathers every reachable node-mesh instance with a finite local
  AABB and a non-singular world matrix (same skip conditions as
  `Scene3D::intersect_ray`'s per-instance affine-inverse guard),
  then median-splits on the largest centroid-extent axis with
  `InstanceBvh::LEAF_THRESHOLD = 4` (matches `Bvh::LEAF_THRESHOLD`).
  Returns `None` for a scene whose every reachable instance is
  skipped.
- `Instance { node, mesh, bounds, world, world_inv }` — per-leaf
  descriptor pairing the world AABB with the `(NodeId, MeshId)`
  re-dispatch keys + the cached affine inverse so ray queries don't
  re-invert the same matrix per ray per instance.
- `InstanceBvhNode { bounds, left_or_first, right_child, instance_count }`
  + `is_leaf()` — same flat-array shape as `BvhNode`.
- `InstanceBvh::intersect_ray(&Scene3D, Ray, t_max) -> Option<SceneRayHit>`
  — closest-hit walk with explicit LIFO stack + near-child-first
  ordering (slab-method entry parameter from Kay & Kajiya 1986).
  Cross-validated against `Scene3D::intersect_ray` on a 16x16 ray
  grid through a 16-cube scene; `t` + `front_face` agree exactly
  (the `NodeId` can tie-break differently on strict ties since the
  walk orders by AABB distance rather than DFS order).
- `InstanceBvh::any_ray_intersection(&Scene3D, Ray, t_max) -> bool`
  — shadow-ray short-circuit with the same answer as
  `Scene3D::any_ray_intersection`.
- `InstanceBvh::{node_count, leaf_count, instance_count, bounds}`
  — inspection accessors symmetric with `Bvh`'s.
- Robustness: empty scenes, scenes whose every reachable node lacks
  a mesh, every reachable mesh has no AABB, every world matrix is
  singular, and shared-children / cycle nodes all map to the
  documented `None` / single-instance behaviour. Determinism: same
  leftmost-first DFS gather as `world_node_bounds`; the build's
  median-split partition is deterministic from the gather order +
  centroid bound.
- 18 `instance_bvh::tests` in-tree + 29 integration tests in
  `tests/instance_bvh.rs` covering single-instance, 4-instance
  single-leaf, 16/20/64-instance balanced trees, the rotated /
  translated / detached / singular / cycle / shared-child cases,
  cross-validation against `Scene3D::intersect_ray` on x/z/oblique
  ray sweeps, `t_max` culling, `cached_world_inv * world ≈ I`,
  root-bounds containment, and the strict-binary-tree node-count
  invariant.

### Added — Round 216 (Scene3D per-instance world AABB snapshot, complement to `world_node_transforms`)

- `Scene3D::world_node_bounds(&self) -> Vec<Option<BoundingBox>>` —
  per-node world-space axis-aligned bounding box, indexed by
  `NodeId.0`. Walks the [`Scene3D::roots`] forest with the same
  depth-first shape as `Scene3D::world_node_transforms` /
  `Scene3D::bounding_box`; for each reachable node carrying a mesh,
  the mesh's local AABB is transformed through the node's full
  ancestor-chain world matrix via `BoundingBox::transform` (eight-corner
  refit, orientation-aware). Unreachable nodes, nodes without a mesh,
  empty meshes, and out-of-range mesh references resolve to `None`.
  The output complements `Scene3D::bounding_box` (which reduces the
  same reachable instances into a single scene-wide union) by exposing
  the per-instance bound — feeding per-instance frustum culling, a
  ray-AABB pre-pass before the existing per-instance triangle walk
  inside `Scene3D::intersect_ray`, and the future scene-level
  BVH-of-instances seed that round 210's docs gestured toward as the
  next acceleration layer above `Bvh::intersect_ray`.
- Cycle, shared-instance (first-parent), determinism, and
  out-of-range-NodeId handling match the existing
  `world_node_transforms` contract exactly. Cost
  `O(nodes + total_children + Σ mesh_vertex_count_for_reachable_nodes)`.

### Added — Round 210 (Scene3D world-space ray queries on top of rounds 199 + 204)

- `Scene3D::intersect_ray(&self, Ray, t_max) -> Option<SceneRayHit>` —
  closest-hit world-space ray query across every reachable
  node-mesh instance. The walk reuses the DFS shape of
  `Scene3D::world_node_transforms` /
  `Scene3D::world_surface_area` (iterative LIFO with
  leftmost-first ordering); at each reachable node carrying a
  mesh the world ray is transformed into mesh-local space via the
  inverse of the node's world matrix, `Mesh::intersect_ray` runs
  in that frame, and the ray-parameter `t` is reported back
  verbatim — affine change-of-frame leaves the scalar `t`
  invariant (`P_world = M · P_local` ⇒ `O_world + t · D_world =
  M · (O_local + t · D_local)`). Per-instance hits shrink the
  `t_max` bound deterministically (earlier-visited instances
  win ties), so a scene with many instances behind the closest
  hit pays only the per-instance test cost.
- `Scene3D::any_ray_intersection(&self, Ray, t_max) -> bool` —
  any-hit (shadow-ray) world-space query with first-blocker
  short-circuit. Reachability, cycle-guarding, singular-transform
  skipping, and degenerate-ray handling match
  `Scene3D::intersect_ray`.
- `SceneRayHit { node: NodeId, primitive_index: usize, hit:
  RayHit }` value type re-exported from the crate root —
  pairs a world-space ray hit with the scene-graph location
  that produced it.
- Internal `mat4_affine_inverse` helper — inverts a row-major
  column-vector affine 4x4 (TRS-derived matrices) via the
  3x3-adjugate-and-determinant identity. Computed in `f64` to
  survive very-different-scale child-of-parent transforms
  (e.g. `1e-3` child of `1e3` parent) before casting back to
  `f32`. Returns `None` for non-affine bottom row, non-finite
  entries, or singular linear parts; such instances are
  silently skipped in the ray walk so the surrounding scene
  still produces correct hits.
- `tests/scene_ray.rs` — 27 integration tests covering: empty
  scene; detached / unrooted nodes; identity / translated /
  uniform-scaled / non-uniform-scaled / rotated instances;
  closest-hit selection among multiple instances; deterministic
  tie-breaking on coincident-instance hits; insertion-order
  independence; `t_max` boundary; rays missing in `xy` /
  pointing-away cases; any-hit short-circuit; singular
  (zero-scale axis) instances skipped; nested parent-child
  transform composition; multi-primitive-mesh primitive index
  reporting; world hit-point round-trip through
  `Ray::point_at(scene_hit.hit.t)`; deep instance stack
  (eight instances spaced along +Z); zero-direction + NaN-ray
  no-panic.

### Added — Round 204 (Bvh ray-acceleration structure on top of round 199)

- `bvh` module + `Bvh { nodes, triangles }` + `BvhNode { bounds,
  left_or_first, right_child, tri_count }` value types. Flat-array
  binary AABB tree built top-down by **object-median split on the
  largest-extent axis of the centroid bound** (Goldsmith & Salmon,
  "Automatic Creation of Object Hierarchies for Ray Tracing", IEEE
  CG&A 7(5), 1987). Leaves stop at `Bvh::LEAF_THRESHOLD = 4`
  triangles (Wald, Boulos & Shirley, "Ray Tracing Deformable Scenes
  Using Dynamic Bounding Volume Hierarchies", ACM TOG 26(1), 2007).
  Re-exported from the crate root.
- `Bvh::build(&Primitive) -> Option<Self>` — pure, non-mutating
  build. Returns `None` for non-triangle topologies / all-NaN /
  all-out-of-range primitives (same robustness contract as
  `Primitive::compute_normals` / `surface_area` /
  `intersect_ray`).
- `Primitive::build_bvh(&self) -> Option<Bvh>` — convenience
  wrapper.
- `Bvh::intersect_ray(&self, &Primitive, ray, t_max) -> Option<RayHit>`
  — closest-hit walk with an **explicit LIFO stack** + near-child-
  first traversal. The slab test (Kay & Kajiya 1986) on every
  interior child gives the entry parameter that drives the
  ordering, so the closest-hit query can shrink `best_t` before
  descending into the farther subtree. Cross-validates with
  `Primitive::intersect_ray` on `t` + barycentrics + `front_face`
  across a 16×16 ray grid in `tests/bvh.rs`.
- `Bvh::any_ray_intersection(&self, &Primitive, ray, t_max) -> bool`
  — shadow-ray early-exit; matches `Primitive::any_ray_intersection`
  on the boolean answer.
- `BvhNode::is_leaf()` + `Bvh::{node_count, leaf_count,
  triangle_count, bounds}` inspection accessors.
- `tests/bvh.rs` (12 integration tests) — `Primitive::build_bvh`
  Some/None branches; per-ray cross-validation against
  `Primitive::intersect_ray` and `any_ray_intersection` on an 8×8
  grid (128 triangles, 256 rays); `t_max` culling; miss-outside-
  extent; root bounds == primitive bounds; triangle-strip
  topology; binary-tree node-count invariant
  (`total == 2*leaves - 1`); deterministic repeat queries;
  `LEAF_THRESHOLD`-respecting leaf compression.
- 16 in-tree unit tests — build of single triangle / empty / lines
  topology / all-NaN / out-of-range index; intersect cross-checks
  with brute-force on two parallel triangles + cube + 7×7 fuzz
  grid; `any_ray_intersection` true/false/`t_max`; coincident-
  centroid degenerate-split termination; tight root bounds.

### Added — Round 199 (ray-mesh / ray-AABB intersection primitives)

- `Ray { origin, direction }` value type with `Ray::point_at(t)` helper
  — directed half-line; `direction` is **not required to be unit
  length** (the returned `t` is measured along `direction`). Re-exported
  from the crate root.
- `RayHit { t, triangle_index, barycentric, front_face }` closest-hit
  record. `triangle_index` indexes `Primitive::triangle_indices()`;
  `barycentric` is `[w, u, v]` with `w = 1 − u − v` so the hit point
  reconstructs as `w * P0 + u * P1 + v * P2`; `front_face` is the
  CCW-from-outside front (right-handed, glTF-aligned) — `true` when
  the ray opposes the outward normal `N = E1 × E2` (`D · N < 0`,
  equivalent to `det > 0` since `det = -D · N`).
- `ray::intersect_triangle(ray, p0, p1, p2, t_max) -> Option<(t, u, v, front_face)>`
  — the **Möller-Trumbore** closed form (Möller & Trumbore, "Fast,
  Minimum Storage Ray-Triangle Intersection", Journal of Graphics
  Tools 2(1), 1997). Cramer's-rule denominator `det = E1 · (D × E2)`;
  barycentric `(u, v)` from `(S · P) / det`, `(D · Q) / det` with
  `S = O − P0`, `P = D × E2`, `Q = S × E1`; `t = (E2 · Q) / det`.
  The same cross-product machinery already drives `compute_normals` /
  `surface_area` / `signed_volume`. Degenerate triangles
  (`|det| < 1e-8`, ray parallel to plane or zero-area), NaN/Inf math,
  behind-origin hits, and out-of-triangle barycentrics return `None`
  — matching the silent-skip robustness contract of the existing
  reductions.
- `ray::intersect_aabb(ray, min, max, t_max) -> Option<(t_enter, t_exit)>`
  — slab method (Kay & Kajiya, "Ray Tracing Complex Scenes",
  SIGGRAPH 1986). Per-axis entry/exit `(min − O) / D`, `(max − O) / D`
  intersected across all three axes; an axis-parallel ray
  (`|D[axis]| < 1e-30`) passes through that axis's test when origin
  is inside the slab and immediately misses when outside. Interval
  clamped to `[0, t_max]`; origin-inside-box reports `t_enter = 0`.
  NaN/Inf inputs return `None`.
- `BoundingBox::intersect_ray(self, ray, t_max) -> Option<(f32, f32)>`
  — thin wrapper around `ray::intersect_aabb` so a BVH traverser
  calls one method per box. Cheap O(1) early-out before recursing
  into per-primitive `intersect_ray`.
- `Primitive::intersect_ray(&self, ray, t_max) -> Option<RayHit>` —
  brute-force walk over `triangle_indices()` calling
  `intersect_triangle`, keeping the closest hit (`t_max` shrunk on
  each successful hit). Topology integration goes through the
  existing de-stripping helper, so `Triangles` /
  `TriangleStrip` (alternating winding honoured) / `TriangleFan` all
  feed in; non-triangle topologies (`Lines`/`Points`) return `None`.
  Out-of-range index entries / degenerate faces / NaN positions are
  silently skipped (same contract as `compute_normals` /
  `surface_area` / `signed_volume`). Pure; `O(triangle_count)`.
  Designed as the BVH-leaf inner loop — spatial acceleration is
  the caller's concern, layered by checking
  `BoundingBox::intersect_ray` first.
- `Primitive::any_ray_intersection(&self, ray, t_max) -> bool` —
  shadow-ray early-exit; returns on the first hit found without
  tracking the closest one. Same topology / robustness contract.
- `Mesh::intersect_ray(&self, ray, t_max) -> Option<(usize, RayHit)>`
  — closest-hit across every contained primitive, shrinking the
  search bound as hits land. Returns the primitive index alongside
  the hit record. Mesh-local space — node-graph world transforms
  are not folded in (transform the ray into mesh-local space by
  inverse-multiplying via `Scene3D::world_node_transforms` before
  calling, or iterate one mesh instance at a time).
- `tests/ray_intersect.rs` (46 tests): `Ray::point_at` at t=0/1/-1;
  triangle centre hit (back-face from below), front-face hit (from
  above), parallel-miss, outside-simplex miss, behind-origin miss,
  `t_max` cull; closest-hit picking among two parallel triangles
  (both orderings); 12-triangle cube hit through -X face (front
  face) + diagonal corner hit; indexed U16 + U32 deref; out-of-range
  index silently skipped; degenerate (collinear) triangle skipped;
  NaN-position face skipped; zero-length ray misses; `Lines` /
  `Points` topology returns `None`; `TriangleStrip` + `TriangleFan`
  match the equivalent list; barycentric `[w, u, v]` sums to ~1.0
  and reconstructs the hit point; empty primitive returns `None`;
  `any_ray_intersection` true on hit / false on miss /
  `t_max`-respecting / lines-false; `Mesh::intersect_ray` routes to
  the primitive, picks the closest across primitives, returns `None`
  on empty / all-miss; `BoundingBox::intersect_ray` through-centre
  axis-aligned + diagonal + origin-inside-box (`t_enter = 0`) +
  miss-to-the-side + `t_max`-cull + axis-parallel inside-slab pass +
  axis-parallel outside-slab miss; combined pattern (AABB-cull
  before primitive walk + AABB-pass-then-primitive-hits ordering);
  `Ray` / `RayHit` `Clone + Copy + PartialEq`; deterministic
  repeat-call invariance.

## [0.0.3](https://github.com/OxideAV/oxideav-mesh3d/compare/v0.0.2...v0.0.3) - 2026-05-30

### Added

- *(mesh)* add Primitive::weld_vertices coincident-vertex de-duplication

### Other

- Round 192: Scene3D::world_surface_area + world_signed_volume + world_volume — transform-aware reductions
- Round 189: Scene3D::world_node_transforms — per-node world matrices
- Round 182: Primitive/Mesh/Scene3D::signed_volume + volume reduction
- Round 175: Primitive/Mesh/Scene3D::surface_area reduction
- Round 155: mesh-validity invariants — degenerate-triangle + edge-manifold
- Round 105: Primitive::compute_tangents — per-vertex MikkTSpace tangent basis from UVs
- Round 101: Primitive::compute_normals — area-weighted smooth normals
- Round 97: Primitive de-stripping (triangle_indices + to_triangle_list)
- Round 10: Primitive::apply_morph_weights — typed morph-blend evaluator
- Round 9: AnimationSampler::sample + IBM affine-row validation
- Round 8: extend Scene3D::validate; add BoundingBox + scene/mesh/primitive bounding_box
- Round 7: mark Primitive + Mesh #[non_exhaustive], add Scene3D::validate

### Added — Round 192 (transform-aware scene-level area/volume)

- `Scene3D::world_surface_area(&self) -> f64` — depth-first walk over
  `Scene3D::roots` (same reachability + cycle-guard + first-parent
  shared-instance resolution as `Scene3D::world_node_transforms`)
  applying each reachable node's full ancestor-chain world matrix to
  its referenced primitive's triangle vertices, summing the
  per-instance post-transform area. Whereas `Scene3D::surface_area`
  reports the resource-level total (each mesh once regardless of
  instance count), this helper is per-instance and folds in
  world-space scaling — including non-uniform diagonal scales where
  the area factor depends on triangle orientation and cannot be
  captured by a single determinant.
- `Scene3D::world_signed_volume(&self) -> f64` — same DFS walk,
  using the affine-volume identity `V_world = det(M_3x3) · V_local`
  for closed two-manifold meshes (the open-mesh boundary terms
  vanish under the same origin-cancellation argument that makes the
  local signed volume translation-invariant). Single-axis mirror
  scales flip the sign correctly; uniform scale `s` gives factor
  `s³`; pure translation leaves a closed-mesh volume unchanged.
- `Scene3D::world_volume(&self) -> f64` — unsigned
  `|world_signed_volume()|`; same `|Σ signed|` (not `Σ |signed|`)
  caveat as `Scene3D::volume` / `Mesh::volume`, so a scene mixing
  mirrored and unmirrored instances of the same mesh can cancel.
- `Primitive::world_surface_area(&self, world: [[f32; 4]; 4]) -> f64`
  — per-primitive helper used by the scene-level walk: edge
  differences cancel the translation column, so only the upper-left
  3x3 enters the per-triangle math. Identical topology / degenerate-
  triangle / out-of-range-index / NaN-skipping semantics as
  `Primitive::surface_area`. Pure; `O(triangle_count)`.
- `tests/world_metrics.rs` — 35 tests covering the new helpers:
  identity-passthrough, pure-translation invariance (area and
  closed-mesh volume), uniform scale `s²`/`s³`, non-uniform scale
  (`2·(ab+bc+ac)` for the 2×3×4 box, det product for volume),
  mirror sign flip + double-mirror cancellation, two-instance
  doubling vs the resource-level baseline, ancestor scale chain
  multiplication, cycle and self-cycle single-visit, detached node
  skip, out-of-range mesh and root skip, shared-instance
  first-parent resolution, `Transform::Matrix` variant passthrough,
  NaN-guard finiteness, and bit-exact determinism across repeated
  calls.

### Added — Round 189 (scene-graph world-transform snapshot)

- `Scene3D::world_node_transforms(&self) -> Vec<Option<[[f32; 4]; 4]>>`
  — depth-first walk over the `Scene3D::roots` forest returning the
  composed world-space 4x4 matrix for every reachable node, with `None`
  in the slot for detached nodes. The output vector is indexed by
  `NodeId.0` so a caller can look up a node's world matrix in O(1)
  without re-walking the ancestor chain — the same per-node side-table
  the existing `Scene3D::{surface_area, signed_volume, volume}` doc
  comments already pointed at as the entry point for transform-aware
  aggregate metrics. Each matrix is row-major column-vector, taking a
  position in the node's local frame to world space — matching
  `Transform::to_matrix`, `BoundingBox::transform`, and the rest of
  the crate's conventions. The traversal mirrors the iterative DFS in
  `Scene3D::bounding_box`: roots are visited in `roots`-order,
  children in source order, cycles are guarded (a node revisited via a
  back-edge keeps its first-encountered matrix), and a shared child
  node (listed under two parents) resolves to the first parent's
  chain — a deterministic single-resolution policy; per-instance
  world matrices need a separate instance-list side-channel. Out-of-
  range `NodeId` entries in `roots` / `children` are silently skipped,
  so the walk is total even on partially-built / fuzzer-generated
  scenes. Cost `O(nodes.len() + total_children)`; allocates one
  `Vec<Option<...>>` of length `nodes.len()` plus the DFS stack.
  Static scene-graph only — skin pose deformation, animation
  channels, camera matrices, and unit-axis conversion are layered
  above this primitive.
- `tests/world_transforms.rs` — 21 tests: empty / no-roots / single-
  root identity-or-translation, ancestor-chain composition
  (parent-child, grandchild, nested scale, parent-scale × child-
  translation), `Transform::Matrix` variant passthrough, multi-root
  forest with detached subtree (`None` slot), cycle / shared-instance
  / self-cycle guard, out-of-range root + child, vec-length matches
  `nodes.len()`, deterministic across repeated calls, depth-3 binary
  tree end-to-end, mixed `Matrix` parent + `Trs` child composition,
  plus the cross-check that `world_node_transforms()[node.0]` applied
  to a mesh's local bounding box matches what `Scene3D::bounding_box`
  returns.

### Added — Round 182 (signed-volume reduction)

- `Primitive::signed_volume(&self) -> f64` — divergence-theorem
  reduction `V = (1/6) Σ (P_a · (P_b × P_c))` over the primitive's
  triangle tessellation, in the unit-cubed of `Primitive::positions`
  (matching the parent `Scene3D::unit`). The derivation: substituting
  the radial field `F = x/3` (for which `∇ · F = 1`) into the
  divergence theorem `∫∫∫_V (∇·F) dV = ∫∫_S F · dS` collapses the
  per-triangle contribution to `(P_a · (P_b × P_c)) / 6` — each
  triangle plus the origin forms a tetrahedron whose signed volume
  is that scalar triple product, and for a closed mesh the
  origin-coincident faces cancel pairwise, leaving only the boundary
  shells (Cha & Chen, "Efficient feature extraction for 2D/3D
  objects in mesh representation", ICIP 2001). The cross-product
  machinery is identical to the one `compute_normals` /
  `surface_area` already share — `signed_volume` adds one scalar
  dot per triangle. Sign follows the winding convention:
  CCW-from-outside (right-handed, glTF-aligned) is positive; an
  inside-out (CW-from-outside) mesh produces the same magnitude
  with opposite sign. Topology integration goes through
  `triangle_indices`, so `Triangles` / `TriangleStrip` (alternating
  winding honoured) / `TriangleFan` all feed in. Non-triangle
  topologies (lines/points) contribute 0.0. Accumulator is `f64`;
  per-triangle math is also `f64`. Degenerate triangles
  (collinear/coincident corners), NaN- or Inf-producing faces, and
  out-of-range index entries all contribute 0.0 — the result is
  always finite. **Translation-invariant for a closed surface**
  (origin-coincident tetra contributions cancel). Only physically
  meaningful for a closed two-manifold (see `is_closed_manifold`);
  arithmetically well-defined regardless. Pure; cost
  `O(triangle_count)`.
- `Primitive::volume(&self) -> f64` — unsigned `|signed_volume()|`,
  robust to inside-out winding.
- `Mesh::signed_volume(&self) -> f64` + `Mesh::volume(&self) -> f64`
  — sum across every contained primitive (mesh-local, no transforms
  / skin pose / morph deltas applied). `Mesh::volume` returns
  `|Σ signed|`, not `Σ |signed|` (single-shell assumption); for a
  multi-shell mesh whose primitives differ in sign, sum each
  primitive's `volume()` separately.
- `Scene3D::signed_volume(&self) -> f64` +
  `Scene3D::volume(&self) -> f64` — sum across every mesh in the
  scene. Walks meshes once, not node instances (instanced meshes
  contribute once per mesh, not once per node). For a transform-
  aware total, walk `world_node_transforms` and apply each node's
  scale's signed determinant per primitive instance (a negative
  scale flips winding and thus flips the sign).
- `tests/volume.rs` (30 tests): unit cube = 1.0 (CCW) / -1.0 (CW),
  unit tetrahedron = 1/6, axis-aligned 2×3×4 box = 24.0,
  inside-out-winding sign flip, translation-invariance for closed
  surface, scaling-cubes-the-volume dimensional check,
  to_triangle_list and weld_vertices are volume-preserving,
  indexed-cube matches soup, empty primitive / Mesh / Scene = 0.0,
  incomplete trailing vertices dropped, degenerate triangles
  contribute 0 (including one-among-valid), NaN- / Inf-bearing
  faces skipped, out-of-range index skipped, TriangleStrip +
  TriangleFan match the equivalent list, non-triangle topologies
  = 0, single-triangle arithmetic pin (V_tri = 1/6), million-
  triangle stress (`f64` no-drift), morph targets ignored
  (base-only), `Mesh::volume` sum + lines-only / empty zero +
  opposite-winding-cancels documented contract,
  `Scene3D::volume` sum + empty zero + instanced-mesh
  counted-once contract.

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
