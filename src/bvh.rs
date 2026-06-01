//! Bounding-Volume Hierarchy (BVH) ray acceleration structure.
//!
//! A BVH is a binary tree of axis-aligned bounding boxes. Every leaf
//! holds a (small) set of triangle indices into the source primitive;
//! every interior node holds an AABB that contains both of its
//! children's geometry. A ray query starts at the root, and recurses
//! into a child only when the ray actually enters that child's box.
//! A tree of depth `O(log T)` over `T` triangles converts the
//! brute-force [`crate::Primitive::intersect_ray`] cost from
//! `O(T)` per query to `O(log T)` expected on a well-balanced
//! primitive, which is the standard acceleration any production
//! ray-traced renderer or picker / collision-probe relies on.
//!
//! ## Algorithm
//!
//! Construction is **median-split on the longest axis** of each
//! node's centroid bound — a long-established deterministic
//! partition scheme described in:
//!
//! * J. MacDonald & K. Booth, "Heuristics for ray tracing using
//!   space subdivision," *The Visual Computer* 6 (1990), 153-166,
//!   §3.2 ("Sweep heuristics" / median split as the simplest
//!   baseline).
//! * J. Goldsmith & J. Salmon, "Automatic creation of object
//!   hierarchies for ray tracing," *IEEE Computer Graphics &
//!   Applications* 7(5) (1987), 14-20 — establishes the
//!   incremental-construction baseline that median-split is the
//!   degenerate static case of.
//!
//! Each internal node sorts its triangle slice by centroid
//! coordinate on the chosen axis (the AABB's longest dimension),
//! splits the sorted slice at its midpoint, and recurses on the two
//! halves. Recursion bottoms out at a configurable leaf threshold
//! (default 4 triangles) — any node with `<= leaf_threshold`
//! triangles becomes a leaf. The result is a tree whose depth is
//! exactly `ceil(log2(T / leaf_threshold))` for `T` triangles
//! (modulo the floor-vs-ceiling integer effects of an odd split).
//! Cost is `O(T log T)` build time (one sort per recursion level
//! across all triangles) and `O(T)` storage.
//!
//! Traversal is the standard iterative **stack-based recursion**:
//!
//! 1. Push the root.
//! 2. Pop a node. Run [`crate::ray::intersect_aabb`] against its
//!    box; skip if the ray misses or `t_enter > best_t` (the box is
//!    behind the current closest hit and cannot improve it).
//! 3. If the node is a leaf, run
//!    [`crate::ray::intersect_triangle`] against each triangle in
//!    the leaf and update `best_t` on a successful hit.
//! 4. Otherwise push both children (near child last so the stack
//!    pops it first — a basic optimisation that lets the closer
//!    child shrink `best_t` before the farther child is even
//!    visited).
//!
//! Iterative-stack traversal (rather than recursive function calls)
//! is the standard pattern from Wald, Boulos & Shirley, "Ray
//! Tracing Deformable Scenes Using Dynamic Bounding Volume
//! Hierarchies," *ACM TOG* 26(1) (2007), §3 — a published
//! open-access algorithm. No external implementation is consulted.
//!
//! ## What this module does not (yet) do
//!
//! * **SAH (Surface Area Heuristic)** binned construction (Wald
//!   2007; PBRT 3rd ed. ch. 4). Median-split is the round-1
//!   baseline; SAH would land in a future round as a quality
//!   optimisation. A median-split BVH is already orders of
//!   magnitude faster than the brute-force walk for any non-tiny
//!   mesh.
//! * **Spatial split / SBVH** (Stich, Friedrich & Dietrich 2009).
//!   Same — quality optimisation for future work.
//! * **Stackless / wide-BVH traversal** (Áfra & Szirmay-Kalos 2014;
//!   Aila & Karras 2009 MBVH). Optimisations that pay off on
//!   millions of triangles; not the baseline.
//! * **Top-level BVH (TLAS) over a [`crate::Scene3D`]** with
//!   per-instance transforms. A future round can add a `SceneBvh`
//!   that wraps per-mesh BVHs and applies the instance transform
//!   to the ray on traversal.
//!
//! ## Usage
//!
//! ```no_run
//! use oxideav_mesh3d::{Bvh, Primitive, Ray, Topology};
//!
//! let mut primitive = Primitive::new(Topology::Triangles);
//! // … populate `primitive.positions` + indices …
//!
//! // Build once.
//! let bvh = Bvh::build(&primitive);
//!
//! // Query many.
//! let ray = Ray::new([0.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
//! if let Some(hit) = bvh.intersect_ray(&primitive, ray, f32::INFINITY) {
//!     println!("hit at t = {}, triangle = {}", hit.t, hit.triangle_index);
//! }
//! ```
//!
//! The [`Bvh`] is decoupled from the [`crate::Primitive`] it indexes
//! — the primitive's triangle list (positions + indices) is the
//! source of truth; the BVH only stores AABBs + triangle index
//! permutations. If the primitive's positions / indices change, the
//! BVH must be rebuilt. The BVH is `Clone + Debug` and holds no
//! borrow on the primitive, so it can be cached alongside the
//! primitive or serialised separately.

use crate::mesh::Primitive;
use crate::ray::{self, Ray, RayHit};
use crate::scene::BoundingBox;

/// Default maximum number of triangles a leaf node may hold.
///
/// Smaller values produce a deeper, finer-grained tree (more box
/// rejections but more memory + node-visit overhead); larger values
/// produce a shallower tree (fewer box rejections but more
/// triangles to test on a hit leaf). `4` is the widely-quoted
/// sweet spot for median-split BVHs on triangle-soup geometry
/// (Wald 2007 §4.1, citing experiments on the standard ray tracing
/// benchmark scenes). For very small primitives (`<= 4` triangles)
/// the BVH collapses to a single leaf and adds negligible overhead
/// over [`Primitive::intersect_ray`].
pub const DEFAULT_LEAF_THRESHOLD: usize = 4;

/// One node of a flattened [`Bvh`].
///
/// Internal layout — exposed as `pub` so callers can inspect the
/// tree (e.g. for visualisation or profiling) but not constructed
/// directly. Use [`Bvh::build`] / [`Bvh::build_with_leaf_threshold`].
///
/// Internal nodes carry both children's flat-array indices; leaf
/// nodes carry a half-open slice `[first_triangle, first_triangle + triangle_count)`
/// into the BVH's `triangle_indices` permutation vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNode {
    /// Tight AABB enclosing every triangle in this node's subtree.
    pub bounds: BoundingBox,
    /// First triangle index in the leaf range (leaves only). For
    /// internal nodes this is `u32::MAX` — see [`BvhNode::is_leaf`].
    pub first_triangle: u32,
    /// Number of triangles in the leaf range. `0` for internal
    /// nodes.
    pub triangle_count: u32,
    /// Flat-array index of the left child (internal nodes only).
    /// `u32::MAX` for leaves.
    pub left_child: u32,
    /// Flat-array index of the right child (internal nodes only).
    /// `u32::MAX` for leaves.
    pub right_child: u32,
}

impl BvhNode {
    /// `true` iff this node is a leaf (holds triangles directly).
    pub fn is_leaf(&self) -> bool {
        self.triangle_count > 0
    }
}

/// Bounding-Volume Hierarchy built once over a [`Primitive`]'s
/// triangle tessellation, queried many times for ray intersections.
///
/// The BVH stores no reference to its source primitive — callers
/// must pass the same primitive to [`Bvh::intersect_ray`] /
/// [`Bvh::any_intersection`] that was used in [`Bvh::build`]. A
/// mismatched primitive produces incorrect results (the BVH's
/// triangle indices reference vertex positions in the source); the
/// API does not enforce this invariant at runtime.
///
/// Construction cost is `O(T log T)` for `T` triangles; query cost
/// is `O(log T)` expected on a well-balanced mesh (`O(T)` worst case
/// for a degenerate distribution where every triangle lands on the
/// same partition side). Memory cost is `O(T)`: one `u32` per
/// triangle in [`Bvh::triangle_indices`] plus one [`BvhNode`] per
/// internal + leaf node (between `T / leaf_threshold` and `2T`
/// nodes depending on partition balance).
///
/// **Topology integration.** Like [`Primitive::intersect_ray`], the
/// BVH covers the triangle tessellation produced by
/// [`Primitive::triangle_indices`] — so `Triangles`,
/// `TriangleStrip` (alternating winding honoured), and
/// `TriangleFan` are all supported. Non-triangle topologies
/// (`Lines` / `Points`) produce an empty BVH whose queries always
/// return `None` / `false`.
///
/// **Robustness.** Degenerate triangles (collinear / coincident
/// corners, NaN positions, out-of-range index entries) are
/// silently skipped at build time — they are not added to the
/// triangle-index permutation, so traversal never visits them.
/// This matches the silent-skip contract on
/// [`Primitive::intersect_ray`] / [`Primitive::surface_area`] /
/// [`Primitive::signed_volume`].
#[derive(Clone, Debug)]
pub struct Bvh {
    /// Flattened depth-first node array. Node `0` is the root
    /// when [`Bvh::is_empty`] is `false`.
    pub nodes: Vec<BvhNode>,
    /// Permutation of triangle indices into
    /// [`Primitive::triangle_indices`]. A leaf node's
    /// `[first_triangle, first_triangle + triangle_count)` slice of
    /// this vec lists the triangles it owns.
    pub triangle_indices: Vec<u32>,
}

impl Bvh {
    /// Build a BVH over a primitive's triangle tessellation with
    /// the default leaf threshold ([`DEFAULT_LEAF_THRESHOLD`]).
    ///
    /// See [`Bvh::build_with_leaf_threshold`] for the configurable
    /// version. Degenerate triangles are dropped during build —
    /// see the type-level docs for the robustness contract.
    pub fn build(primitive: &Primitive) -> Self {
        Self::build_with_leaf_threshold(primitive, DEFAULT_LEAF_THRESHOLD)
    }

    /// Build a BVH over the primitive's triangle tessellation with
    /// a custom leaf-size threshold.
    ///
    /// `leaf_threshold` is clamped to `>= 1` — a threshold of `0`
    /// is meaningless (would never bottom out the recursion) and
    /// is silently raised to `1`. Larger thresholds produce
    /// shallower trees with more triangles per leaf; smaller
    /// thresholds produce deeper trees with fewer triangles per
    /// leaf but more box-test overhead.
    pub fn build_with_leaf_threshold(primitive: &Primitive, leaf_threshold: usize) -> Self {
        let leaf_threshold = leaf_threshold.max(1);
        let n = primitive.positions.len();
        let raw = primitive.triangle_indices();

        // Pre-compute the AABB + centroid of every valid triangle.
        // Degenerate / NaN / out-of-range triangles are dropped here
        // so they don't enter the tree at all.
        let mut entries: Vec<TriangleEntry> = Vec::with_capacity(raw.len());
        for (tri_idx, [ia, ib, ic]) in raw.into_iter().enumerate() {
            if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                continue;
            }
            let p0 = primitive.positions[ia as usize];
            let p1 = primitive.positions[ib as usize];
            let p2 = primitive.positions[ic as usize];
            if !point_finite(p0) || !point_finite(p1) || !point_finite(p2) {
                continue;
            }
            // Mirror the degenerate-triangle test used by
            // `Primitive::compute_normals` / `degenerate_triangles`:
            // |E1 × E2| == 0 ⇒ zero-area, drop.
            let e1 = sub(p1, p0);
            let e2 = sub(p2, p0);
            let c = cross(e1, e2);
            if c[0] == 0.0 && c[1] == 0.0 && c[2] == 0.0 {
                continue;
            }
            if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() {
                continue;
            }
            let min = [
                min3(p0[0], p1[0], p2[0]),
                min3(p0[1], p1[1], p2[1]),
                min3(p0[2], p1[2], p2[2]),
            ];
            let max = [
                max3(p0[0], p1[0], p2[0]),
                max3(p0[1], p1[1], p2[1]),
                max3(p0[2], p1[2], p2[2]),
            ];
            let centroid = [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ];
            entries.push(TriangleEntry {
                triangle_index: tri_idx as u32,
                bounds: BoundingBox { min, max },
                centroid,
            });
        }

        if entries.is_empty() {
            return Self {
                nodes: Vec::new(),
                triangle_indices: Vec::new(),
            };
        }

        // Reserve roughly 2N nodes worth of capacity (upper bound
        // for a binary tree with N leaves of size 1; smaller for
        // larger leaf_threshold).
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(entries.len() * 2);
        // Permutation: the order leaves see triangles in, after
        // recursive sort + partition.
        let mut triangle_indices: Vec<u32> = Vec::with_capacity(entries.len());

        build_recursive(
            &mut entries,
            &mut nodes,
            &mut triangle_indices,
            leaf_threshold,
        );

        Self {
            nodes,
            triangle_indices,
        }
    }

    /// `true` if the BVH contains no triangles (the source
    /// primitive was empty, all-degenerate, or a non-triangle
    /// topology).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of nodes in the flattened tree (leaves + internal).
    /// `0` when [`Bvh::is_empty`].
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of valid (non-degenerate, in-range) triangles
    /// indexed by this BVH.
    pub fn triangle_count(&self) -> usize {
        self.triangle_indices.len()
    }

    /// Root [`BoundingBox`] — the AABB enclosing every indexed
    /// triangle. `None` if the BVH is empty.
    pub fn bounds(&self) -> Option<BoundingBox> {
        self.nodes.first().map(|n| n.bounds)
    }

    /// Closest-hit ray query using the BVH for acceleration.
    ///
    /// `primitive` **must** be the same primitive (or one with the
    /// same triangle tessellation) that the BVH was built against;
    /// passing a different primitive produces incorrect results.
    /// `t_max` upper-bounds the ray parameter — pass
    /// `f32::INFINITY` for unbounded queries.
    ///
    /// Returns the closest [`RayHit`] within `t_max`, or `None` if
    /// the ray misses every indexed triangle. The semantics match
    /// [`Primitive::intersect_ray`] bit-for-bit (a property
    /// exercised in the test suite by querying the same ray against
    /// both APIs and comparing the resulting `RayHit`).
    pub fn intersect_ray(&self, primitive: &Primitive, ray: Ray, t_max: f32) -> Option<RayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let raw_triangles = primitive.triangle_indices();
        let n = primitive.positions.len();
        let mut closest: Option<RayHit> = None;
        let mut best_t = t_max;
        // Iterative DFS with an explicit stack.
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx as usize];
            // Box-cull: skip nodes the ray misses or whose entry
            // distance is already worse than the current best.
            match ray::intersect_aabb(ray, node.bounds.min, node.bounds.max, best_t) {
                None => continue,
                Some((t_enter, _)) if t_enter > best_t => continue,
                Some(_) => {}
            }
            if node.is_leaf() {
                let begin = node.first_triangle as usize;
                let end = begin + node.triangle_count as usize;
                for &tri_perm in &self.triangle_indices[begin..end] {
                    let tri_idx = tri_perm as usize;
                    if tri_idx >= raw_triangles.len() {
                        continue;
                    }
                    let [ia, ib, ic] = raw_triangles[tri_idx];
                    if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                        continue;
                    }
                    let p0 = primitive.positions[ia as usize];
                    let p1 = primitive.positions[ib as usize];
                    let p2 = primitive.positions[ic as usize];
                    if let Some((t, u, v, front)) = ray::intersect_triangle(ray, p0, p1, p2, best_t)
                    {
                        let w = 1.0 - u - v;
                        closest = Some(RayHit {
                            t,
                            triangle_index: tri_idx,
                            barycentric: [w, u, v],
                            front_face: front,
                        });
                        best_t = t;
                    }
                }
            } else {
                // Push both children. To exploit `best_t`-shrinking
                // we'd like to visit the closer child first — so it
                // must be pushed LAST (LIFO). Determine "closer" by
                // entry-parameter comparison.
                let left = &self.nodes[node.left_child as usize];
                let right = &self.nodes[node.right_child as usize];
                let t_left = ray::intersect_aabb(ray, left.bounds.min, left.bounds.max, best_t)
                    .map(|(t_e, _)| t_e);
                let t_right = ray::intersect_aabb(ray, right.bounds.min, right.bounds.max, best_t)
                    .map(|(t_e, _)| t_e);
                match (t_left, t_right) {
                    (None, None) => {}
                    (Some(_), None) => stack.push(node.left_child),
                    (None, Some(_)) => stack.push(node.right_child),
                    (Some(tl), Some(tr)) => {
                        if tl <= tr {
                            // Left closer → visit first → push last.
                            stack.push(node.right_child);
                            stack.push(node.left_child);
                        } else {
                            stack.push(node.left_child);
                            stack.push(node.right_child);
                        }
                    }
                }
            }
        }
        closest
    }

    /// Shadow-ray early-exit query: `true` if any indexed triangle
    /// is hit within `(0, t_max]`.
    ///
    /// Returns on the first hit found — does **not** track the
    /// closest. Matches the semantics of
    /// [`Primitive::any_ray_intersection`] (less the implicit
    /// `1e-4` epsilon — the BVH version uses the raw `t_max` so
    /// callers in tight numerical regimes can choose their own
    /// offset).
    pub fn any_intersection(&self, primitive: &Primitive, ray: Ray, t_max: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let raw_triangles = primitive.triangle_indices();
        let n = primitive.positions.len();
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx as usize];
            if ray::intersect_aabb(ray, node.bounds.min, node.bounds.max, t_max).is_none() {
                continue;
            }
            if node.is_leaf() {
                let begin = node.first_triangle as usize;
                let end = begin + node.triangle_count as usize;
                for &tri_perm in &self.triangle_indices[begin..end] {
                    let tri_idx = tri_perm as usize;
                    if tri_idx >= raw_triangles.len() {
                        continue;
                    }
                    let [ia, ib, ic] = raw_triangles[tri_idx];
                    if (ia as usize) >= n || (ib as usize) >= n || (ic as usize) >= n {
                        continue;
                    }
                    let p0 = primitive.positions[ia as usize];
                    let p1 = primitive.positions[ib as usize];
                    let p2 = primitive.positions[ic as usize];
                    if ray::intersect_triangle(ray, p0, p1, p2, t_max).is_some() {
                        return true;
                    }
                }
            } else {
                stack.push(node.left_child);
                stack.push(node.right_child);
            }
        }
        false
    }
}

// ----------------------------------------------------------------------
// Construction helpers
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct TriangleEntry {
    triangle_index: u32,
    bounds: BoundingBox,
    centroid: [f32; 3],
}

fn build_recursive(
    entries: &mut [TriangleEntry],
    nodes: &mut Vec<BvhNode>,
    triangle_indices: &mut Vec<u32>,
    leaf_threshold: usize,
) -> u32 {
    // Pre-allocate this node's slot so children know its parent
    // index; we'll fill it in once we know if it's a leaf.
    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode {
        bounds: BoundingBox {
            min: [0.0; 3],
            max: [0.0; 3],
        },
        first_triangle: u32::MAX,
        triangle_count: 0,
        left_child: u32::MAX,
        right_child: u32::MAX,
    });

    // Compute this node's tight bounds (over triangle AABBs) and
    // centroid bounds (over triangle centroids — used for split
    // axis selection).
    let mut bounds = entries[0].bounds;
    let mut cmin = entries[0].centroid;
    let mut cmax = entries[0].centroid;
    for e in &entries[1..] {
        bounds = bounds.union(e.bounds);
        for axis in 0..3 {
            if e.centroid[axis] < cmin[axis] {
                cmin[axis] = e.centroid[axis];
            }
            if e.centroid[axis] > cmax[axis] {
                cmax[axis] = e.centroid[axis];
            }
        }
    }
    nodes[node_idx as usize].bounds = bounds;

    // Leaf check: small enough or centroids coincident (no
    // meaningful axis to split on).
    let extent = [cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]];
    let degenerate_centroids = extent[0] == 0.0 && extent[1] == 0.0 && extent[2] == 0.0;
    if entries.len() <= leaf_threshold || degenerate_centroids {
        let first = triangle_indices.len() as u32;
        for e in entries.iter() {
            triangle_indices.push(e.triangle_index);
        }
        let n = entries.len() as u32;
        let node = &mut nodes[node_idx as usize];
        node.first_triangle = first;
        node.triangle_count = n;
        return node_idx;
    }

    // Split axis: longest centroid extent.
    let mut axis = 0;
    if extent[1] > extent[axis] {
        axis = 1;
    }
    if extent[2] > extent[axis] {
        axis = 2;
    }

    // Median split: sort by centroid on the chosen axis, partition
    // at the midpoint. Total ordering is enforced via a key cast
    // (NaN centroids are unreachable — we dropped NaN triangles at
    // entry time).
    entries.sort_by(|a, b| {
        a.centroid[axis]
            .partial_cmp(&b.centroid[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = entries.len() / 2;
    let (left, right) = entries.split_at_mut(mid);

    let left_idx = build_recursive(left, nodes, triangle_indices, leaf_threshold);
    let right_idx = build_recursive(right, nodes, triangle_indices, leaf_threshold);

    let node = &mut nodes[node_idx as usize];
    node.left_child = left_idx;
    node.right_child = right_idx;
    node_idx
}

// ----------------------------------------------------------------------
// Vec helpers (kept private; the public crate API doesn't expose vec ops)
// ----------------------------------------------------------------------

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn point_finite(p: [f32; 3]) -> bool {
    p[0].is_finite() && p[1].is_finite() && p[2].is_finite()
}

#[inline]
fn min3(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).min(c)
}

#[inline]
fn max3(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).max(c)
}
