//! Bounding-volume hierarchy over a [`crate::Primitive`]'s triangle
//! tessellation — accelerates many-ray workloads from
//! `O(triangle_count)` per ray to roughly `O(log triangle_count)` once
//! the tree is built.
//!
//! # Construction
//!
//! Two top-down builders share one driver ([`BvhBuildStrategy`]):
//!
//! * **Binned surface-area heuristic** (default,
//!   [`BvhBuildStrategy::BinnedSah`]). The SAH (J. D. MacDonald &
//!   K. S. Booth, "Heuristics for Ray Tracing Using Space
//!   Subdivision", The Visual Computer 6(3), 1990) estimates the
//!   expected cost of a split as
//!
//!   ```text
//!   C = C_trav + C_isect · (A_L · N_L + A_R · N_R) / A_parent
//!   ```
//!
//!   where `A` is box surface area and `N` the primitive count — the
//!   probability that a random ray hitting the parent also hits a
//!   child is proportional to the child's surface area. Following
//!   I. Wald, "On fast Construction of SAH-based Bounding Volume
//!   Hierarchies", IEEE Symposium on Interactive Ray Tracing 2007,
//!   candidate planes are not taken at every primitive but at the
//!   `K` equal-width bin boundaries of the node's **centroid** bound
//!   on each axis ([`BvhBuildOptions::sah_bins`], default 16): one
//!   linear pass bins every primitive (count + bounds per bin), a
//!   left sweep and a right sweep accumulate the prefix / suffix
//!   areas and counts, and the cheapest of the `3 · (K - 1)` planes
//!   wins. The node becomes a leaf when it holds at most
//!   [`BvhBuildOptions::max_leaf_size`] primitives **and** the leaf
//!   cost `C_isect · N` does not exceed the best split cost (Wald's
//!   termination criterion); larger nodes are always split.
//! * **Object median** ([`BvhBuildStrategy::ObjectMedian`]) — the
//!   classical construction reviewed by Goldsmith & Salmon
//!   ("Automatic Creation of Object Hierarchies for Ray Tracing", IEEE
//!   CG&A 7(5), 1987): split at the midpoint of the centroid bound's
//!   largest axis, leaves at `max_leaf_size`. Kept for comparison
//!   benchmarks and tests.
//!
//! Both builders are deterministic from the input order, run with an
//! explicit work stack (no recursion), and fall back to an
//! index-median split when every centroid coincides. Past a depth of
//! 32 the builder switches to a **count**-median split
//! (`select_nth_unstable` on the largest centroid axis) so the tree
//! depth is bounded by `32 + log2(n) ≤ 64` even on adversarial input
//! (e.g. exponentially spaced triangles that would otherwise peel one
//! primitive per level). That bound sizes the traversal stack.
//!
//! # Node layout (CPU and GPU)
//!
//! The tree is flattened into one linear array of 32-byte
//! [`BvhNode`]s (`#[repr(C)]`):
//!
//! ```text
//! offset  size  field
//!      0    12  min: [f32; 3]        — AABB lower corner
//!     12     4  left_or_first: u32   — interior: index of the left child
//!                                      leaf: first slot in `triangles`
//!     16    12  max: [f32; 3]        — AABB upper corner
//!     28     4  tri_count: u32       — 0 for an interior node,
//!                                      primitive count for a leaf
//! ```
//!
//! This matches the natural WGSL / GLSL-std430 struct
//! `struct Node { min: vec3<f32>, left_or_first: u32, max: vec3<f32>, count: u32 }`
//! byte for byte (each `vec3` + scalar pair packs into one 16-byte
//! slot), so the array can be uploaded verbatim as a storage buffer
//! ([`Bvh::node_words`] produces the `u32` word stream without any
//! `unsafe`). Node `0` is the root. **Children are stored as an
//! adjacent pair**: the right child of an interior node is always
//! `left_or_first + 1` ([`BvhNode::right_child`]), so the two boxes a
//! traversal step tests share one 64-byte cache line when the array is
//! 64-byte aligned. Pairs are allocated in depth-first order (the
//! left subtree's pairs precede the right subtree's), so a child
//! index is always greater than its parent's — refitting is a single
//! reverse sweep ([`Bvh::refit`]).
//!
//! Leaf payloads live in two parallel arrays in leaf order:
//! [`Bvh::triangles`] (the triangle's index in
//! [`crate::Primitive::triangle_indices`] order) and
//! [`Bvh::triangle_vertices`] (its three vertex indices, so traversal
//! never re-derives the de-stripped index list per ray).
//!
//! # Traversal
//!
//! Queries take a [`PreparedRay`] (reciprocal direction, sign masks
//! and the watertight shear, computed once per ray) and walk the tree
//! with a fixed-size on-stack stack — no per-ray heap allocation. At
//! each interior node both children are slab-tested (the robust test
//! of Williams et al. 2005 with Ize 2013's conservative far-distance
//! scaling, see [`PreparedRay`]); the traversal descends into the
//! nearer child and pushes the farther one with its entry distance,
//! which is re-checked against the shrinking closest-hit distance when
//! popped (ordered front-to-back traversal with early-out). Leaves use
//! either Möller-Trumbore or the Woop-Benthin-Wald watertight test
//! ([`TriangleTest`]). Any-hit queries stop at the first accepted
//! intersection. Both query kinds accept an optional filter closure
//! that can reject a candidate hit (alpha-masked geometry).
//!
//! # Topology coverage
//!
//! Triangle enumeration goes through
//! [`crate::Primitive::triangle_indices`], so `Triangles`,
//! `TriangleStrip` (with alternating winding honoured), and
//! `TriangleFan` all build a non-trivial BVH. Non-triangle topologies
//! (`Lines`, `LineStrip`, `LineLoop`, `Points`) yield an empty
//! triangle enumeration — [`Bvh::build`] returns `None` for those.
//!
//! # Robustness contract
//!
//! Out-of-range index entries (any vertex index `>= positions.len()`)
//! and triangles whose vertices contain a non-finite coordinate are
//! silently skipped during build — same contract as
//! [`crate::Primitive::compute_normals`] /
//! [`crate::Primitive::surface_area`]. A primitive whose every
//! triangle is in one of those classes builds to `None`. Queries
//! re-check vertex indices against the primitive passed at query time
//! and never panic on a mismatched primitive.

use crate::ray::{PreparedRay, Ray, RayHit, RayQuery, TriangleTest};
use crate::scene::BoundingBox;
use crate::Primitive;

/// Which top-down builder [`Bvh::build_with`] /
/// [`crate::InstanceBvh::build_with`] use. See the module docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BvhBuildStrategy {
    /// Midpoint of the centroid bound on its largest axis
    /// (Goldsmith & Salmon 1987). Cheap, but blind to primitive size
    /// and distribution.
    ObjectMedian,
    /// Binned surface-area heuristic (MacDonald & Booth 1990; Wald
    /// 2007). The default.
    #[default]
    BinnedSah,
}

/// Tuning knobs for BVH construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhBuildOptions {
    /// Builder selection.
    pub strategy: BvhBuildStrategy,
    /// Largest leaf the builder may emit. Nodes with more primitives
    /// are always split; with [`BvhBuildStrategy::BinnedSah`] smaller
    /// nodes may still be split when the SAH says it pays off.
    /// Clamped to `>= 1`.
    pub max_leaf_size: usize,
    /// Number of SAH bins per axis (Wald 2007 uses 16–32). Clamped to
    /// `2..=64`.
    pub sah_bins: usize,
    /// SAH cost of one interior-node visit (two slab tests).
    pub traversal_cost: f32,
    /// SAH cost of one primitive intersection.
    pub intersection_cost: f32,
}

impl BvhBuildOptions {
    /// Binned SAH with the default constants.
    pub fn sah() -> Self {
        Self {
            strategy: BvhBuildStrategy::BinnedSah,
            max_leaf_size: Bvh::LEAF_THRESHOLD,
            sah_bins: 16,
            traversal_cost: 1.0,
            intersection_cost: 1.0,
        }
    }

    /// The legacy object-median builder (leaves of up to
    /// [`Bvh::LEAF_THRESHOLD`] primitives).
    pub fn object_median() -> Self {
        Self {
            strategy: BvhBuildStrategy::ObjectMedian,
            ..Self::sah()
        }
    }
}

impl Default for BvhBuildOptions {
    fn default() -> Self {
        Self::sah()
    }
}

/// One 32-byte node in a [`Bvh`]. See the module docs for the exact
/// memory layout (it doubles as the GPU upload format).
///
/// A leaf is signalled by `tri_count > 0`; an interior node by
/// `tri_count == 0`, with its children at `left_or_first` and
/// `left_or_first + 1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNode {
    /// Lower corner of the tight AABB over every primitive in the
    /// subtree.
    pub min: [f32; 3],
    /// Interior: index of the left child in [`Bvh::nodes`] (the right
    /// child is the next node). Leaf: first slot of the leaf's range
    /// in [`Bvh::triangles`] / [`Bvh::triangle_vertices`].
    pub left_or_first: u32,
    /// Upper corner of the subtree AABB.
    pub max: [f32; 3],
    /// `0` for an interior node; the number of triangles in the leaf
    /// otherwise.
    pub tri_count: u32,
}

impl BvhNode {
    /// `true` if this node is a leaf.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.tri_count > 0
    }

    /// The node's AABB.
    #[inline]
    pub fn bounds(&self) -> BoundingBox {
        BoundingBox {
            min: self.min,
            max: self.max,
        }
    }

    /// Index of the left child (interior nodes only).
    #[inline]
    pub fn left_child(&self) -> u32 {
        self.left_or_first
    }

    /// Index of the right child (interior nodes only) — always
    /// `left_or_first + 1` (sibling pairs are adjacent).
    #[inline]
    pub fn right_child(&self) -> u32 {
        self.left_or_first + 1
    }

    /// The eight little-endian-agnostic `u32` words of this node in
    /// GPU layout order (`f32` fields via `to_bits`).
    #[inline]
    pub fn to_words(&self) -> [u32; 8] {
        [
            self.min[0].to_bits(),
            self.min[1].to_bits(),
            self.min[2].to_bits(),
            self.left_or_first,
            self.max[0].to_bits(),
            self.max[1].to_bits(),
            self.max[2].to_bits(),
            self.tri_count,
        ]
    }
}

/// Bounding-volume hierarchy built from one [`crate::Primitive`].
///
/// See the module-level docs for the construction, layout and
/// traversal algorithms. The struct is a value type with no shared
/// state with the source primitive; queries take the primitive by
/// reference to read vertex positions, so after moving vertices call
/// [`Bvh::refit`] (or rebuild) before querying again.
#[derive(Clone, Debug)]
pub struct Bvh {
    /// Flat array of nodes; node `0` is the root.
    pub nodes: Vec<BvhNode>,
    /// Leaf-order triangle ids: slot `i` holds the index into
    /// [`crate::Primitive::triangle_indices`] of the triangle stored
    /// at slot `i` (a leaf covers slots
    /// `[left_or_first, left_or_first + tri_count)`).
    pub triangles: Vec<u32>,
    /// Leaf-order vertex indices, parallel to [`Bvh::triangles`]:
    /// `triangle_vertices[i] == primitive.triangle_indices()[triangles[i]]`.
    pub triangle_vertices: Vec<[u32; 3]>,
}

/// Internal leaf-slot hit record shared by the per-primitive and
/// per-instance traversals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SlotHit {
    pub slot: u32,
    pub t: f32,
    pub u: f32,
    pub v: f32,
    pub front_face: bool,
}

/// Depth after which the builder abandons SAH / midpoint splits for a
/// balanced count-median split (bounds tree depth to ≤ 64).
const BALANCE_DEPTH: u32 = 32;
/// In-place traversal stack capacity. Trees from the builders never
/// exceed depth 64; a hand-assembled deeper tree spills into a heap
/// vector instead of dropping nodes.
const STACK_CAPACITY: usize = 64;
/// Upper bound on the configurable SAH bin count.
const MAX_BINS: usize = 64;

/// Fixed-capacity traversal stack with a (normally unused) heap spill.
struct TraversalStack {
    items: [(u32, f32); STACK_CAPACITY],
    len: usize,
    spill: Vec<(u32, f32)>,
}

impl TraversalStack {
    #[inline(always)]
    fn new() -> Self {
        Self {
            items: [(0, 0.0); STACK_CAPACITY],
            len: 0,
            spill: Vec::new(),
        }
    }

    #[inline(always)]
    fn push(&mut self, node: u32, t: f32) {
        if self.len < STACK_CAPACITY {
            self.items[self.len] = (node, t);
            self.len += 1;
        } else {
            self.spill.push((node, t));
        }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<(u32, f32)> {
        if let Some(x) = self.spill.pop() {
            return Some(x);
        }
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(self.items[self.len])
        }
    }
}

impl Bvh {
    /// Default maximum number of triangles per leaf
    /// ([`BvhBuildOptions::max_leaf_size`]).
    ///
    /// The value (`4`) is a long-standing balance from the rendering
    /// literature (e.g. Wald, Boulos & Shirley, "Ray Tracing
    /// Deformable Scenes Using Dynamic Bounding Volume Hierarchies",
    /// ACM TOG 26(1), 2007).
    pub const LEAF_THRESHOLD: usize = 4;

    /// Build a binned-SAH BVH over every triangle of `primitive`.
    ///
    /// Returns `None` when the primitive enumerates zero usable
    /// triangles — non-triangle topology, or every triangle skipped by
    /// the robustness contract (out-of-range vertex index or
    /// non-finite coordinate). Pure: the primitive is not mutated.
    pub fn build(primitive: &Primitive) -> Option<Self> {
        Self::build_with(primitive, &BvhBuildOptions::default())
    }

    /// [`Bvh::build`] with explicit [`BvhBuildOptions`] (e.g.
    /// [`BvhBuildOptions::object_median`]).
    pub fn build_with(primitive: &Primitive, options: &BvhBuildOptions) -> Option<Self> {
        let n_pos = primitive.positions.len();
        let all_tris = primitive.triangle_indices();
        if all_tris.is_empty() {
            return None;
        }
        let mut bounds: Vec<BoundingBox> = Vec::with_capacity(all_tris.len());
        let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(all_tris.len());
        let mut ids: Vec<u32> = Vec::with_capacity(all_tris.len());
        for (idx, tri) in all_tris.iter().enumerate() {
            let Some([p0, p1, p2]) = fetch(&primitive.positions, *tri, n_pos) else {
                continue;
            };
            if !finite_point(p0) || !finite_point(p1) || !finite_point(p2) {
                continue;
            }
            let b = BoundingBox::from_point(p0).expand(p1).expand(p2);
            bounds.push(b);
            centroids.push(b.center());
            ids.push(idx as u32);
        }
        if ids.is_empty() {
            return None;
        }
        let (nodes, order) = build_tree(&bounds, &centroids, options);
        let triangles: Vec<u32> = order.iter().map(|&o| ids[o as usize]).collect();
        let triangle_vertices = triangles.iter().map(|&t| all_tris[t as usize]).collect();
        Some(Bvh {
            nodes,
            triangles,
            triangle_vertices,
        })
    }

    /// Closest-hit ray query in `[0, t_max]` with Möller-Trumbore.
    ///
    /// Returns the smallest-`t` hit; `triangle_index` is the index in
    /// [`crate::Primitive::triangle_indices`], identical to what
    /// [`crate::Primitive::intersect_ray`] reports (a strict tie at a
    /// shared edge can resolve to either adjacent triangle; the hit
    /// point is the same). Convenience wrapper over
    /// [`Bvh::closest_hit`].
    pub fn intersect_ray(&self, primitive: &Primitive, ray: Ray, t_max: f32) -> Option<RayHit> {
        self.closest_hit(primitive, &PreparedRay::new(ray), &RayQuery::new(t_max))
    }

    /// Shadow-ray query in `[0, t_max]`: `true` as soon as any
    /// triangle is struck. Same boolean answer as
    /// [`crate::Primitive::any_ray_intersection`]. Wrapper over
    /// [`Bvh::occluded`].
    pub fn any_ray_intersection(&self, primitive: &Primitive, ray: Ray, t_max: f32) -> bool {
        self.occluded(primitive, &PreparedRay::new(ray), &RayQuery::new(t_max))
    }

    /// Closest hit for a prepared ray within `query`'s interval.
    pub fn closest_hit(
        &self,
        primitive: &Primitive,
        ray: &PreparedRay,
        query: &RayQuery,
    ) -> Option<RayHit> {
        self.closest_hit_filtered(primitive, ray, query, |_| true)
    }

    /// Closest hit, consulting `filter` for every candidate
    /// intersection: a candidate counts only when `filter` returns
    /// `true` (e.g. an alpha-mask lookup at the candidate's
    /// barycentrics). The filter may be called for candidates that are
    /// later superseded by a closer hit, and in no particular order.
    pub fn closest_hit_filtered<F: FnMut(&RayHit) -> bool>(
        &self,
        primitive: &Primitive,
        ray: &PreparedRay,
        query: &RayQuery,
        mut filter: F,
    ) -> Option<RayHit> {
        let mut t_max = query.t_max;
        let hit = self.closest_slot(
            &primitive.positions,
            ray,
            query.t_min,
            &mut t_max,
            query.triangle_test,
            &mut |h: &SlotHit| filter(&self.ray_hit(h)),
        )?;
        Some(self.ray_hit(&hit))
    }

    /// Any-hit (occlusion) query: `true` as soon as one triangle is
    /// hit inside `query`'s interval.
    pub fn occluded(&self, primitive: &Primitive, ray: &PreparedRay, query: &RayQuery) -> bool {
        self.occluded_filtered(primitive, ray, query, |_| true)
    }

    /// Any-hit query with a candidate filter (see
    /// [`Bvh::closest_hit_filtered`]): rejected candidates do not
    /// occlude — this is how alpha-masked geometry lets light through
    /// in shadow rays.
    pub fn occluded_filtered<F: FnMut(&RayHit) -> bool>(
        &self,
        primitive: &Primitive,
        ray: &PreparedRay,
        query: &RayQuery,
        mut filter: F,
    ) -> bool {
        self.any_slot(
            &primitive.positions,
            ray,
            query.t_min,
            query.t_max,
            query.triangle_test,
            &mut |h: &SlotHit| filter(&self.ray_hit(h)),
        )
    }

    #[inline]
    fn ray_hit(&self, h: &SlotHit) -> RayHit {
        RayHit {
            t: h.t,
            triangle_index: self.triangles[h.slot as usize] as usize,
            barycentric: [1.0 - h.u - h.v, h.u, h.v],
            front_face: h.front_face,
        }
    }

    /// Core ordered closest-hit traversal. `t_max` is shrunk in place
    /// to the accepted hit distance so several BVHs (one per
    /// primitive of an instance) can share one running bound.
    #[inline]
    pub(crate) fn closest_slot<F: FnMut(&SlotHit) -> bool>(
        &self,
        positions: &[[f32; 3]],
        ray: &PreparedRay,
        t_min: f32,
        t_max: &mut f32,
        test: TriangleTest,
        filter: &mut F,
    ) -> Option<SlotHit> {
        let nodes = &self.nodes[..];
        let root = nodes.first()?;
        if !ray.is_valid() || ray.slab(&root.min, &root.max, t_min, *t_max).is_none() {
            return None;
        }
        let n_pos = positions.len();
        let mut best: Option<SlotHit> = None;
        let mut stack = TraversalStack::new();
        let mut idx = 0usize;
        loop {
            let node = &nodes[idx];
            if node.tri_count > 0 {
                let first = node.left_or_first as usize;
                let end = first + node.tri_count as usize;
                for slot in first..end {
                    let Some(&tri) = self.triangle_vertices.get(slot) else {
                        break;
                    };
                    let Some([p0, p1, p2]) = fetch(positions, tri, n_pos) else {
                        continue;
                    };
                    if let Some((t, u, v, front_face)) =
                        ray.intersect_triangle(test, p0, p1, p2, t_min, *t_max)
                    {
                        let h = SlotHit {
                            slot: slot as u32,
                            t,
                            u,
                            v,
                            front_face,
                        };
                        if filter(&h) {
                            *t_max = t;
                            best = Some(h);
                        }
                    }
                }
            } else {
                let l = node.left_or_first as usize;
                if let (Some(ln), Some(rn)) = (nodes.get(l), nodes.get(l + 1)) {
                    let tl = ray.slab(&ln.min, &ln.max, t_min, *t_max);
                    let tr = ray.slab(&rn.min, &rn.max, t_min, *t_max);
                    match (tl, tr) {
                        (Some(a), Some(b)) => {
                            if a <= b {
                                stack.push((l + 1) as u32, b);
                                idx = l;
                            } else {
                                stack.push(l as u32, a);
                                idx = l + 1;
                            }
                            continue;
                        }
                        (Some(_), None) => {
                            idx = l;
                            continue;
                        }
                        (None, Some(_)) => {
                            idx = l + 1;
                            continue;
                        }
                        (None, None) => {}
                    }
                }
            }
            // Pop the next still-relevant far child.
            loop {
                match stack.pop() {
                    None => return best,
                    Some((n, t_enter)) => {
                        if t_enter <= *t_max {
                            idx = n as usize;
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Core any-hit traversal.
    #[inline]
    pub(crate) fn any_slot<F: FnMut(&SlotHit) -> bool>(
        &self,
        positions: &[[f32; 3]],
        ray: &PreparedRay,
        t_min: f32,
        t_max: f32,
        test: TriangleTest,
        filter: &mut F,
    ) -> bool {
        let nodes = &self.nodes[..];
        let Some(root) = nodes.first() else {
            return false;
        };
        if !ray.is_valid() || ray.slab(&root.min, &root.max, t_min, t_max).is_none() {
            return false;
        }
        let n_pos = positions.len();
        let mut stack = TraversalStack::new();
        let mut idx = 0usize;
        loop {
            let node = &nodes[idx];
            if node.tri_count > 0 {
                let first = node.left_or_first as usize;
                let end = first + node.tri_count as usize;
                for slot in first..end {
                    let Some(&tri) = self.triangle_vertices.get(slot) else {
                        break;
                    };
                    let Some([p0, p1, p2]) = fetch(positions, tri, n_pos) else {
                        continue;
                    };
                    if let Some((t, u, v, front_face)) =
                        ray.intersect_triangle(test, p0, p1, p2, t_min, t_max)
                    {
                        let h = SlotHit {
                            slot: slot as u32,
                            t,
                            u,
                            v,
                            front_face,
                        };
                        if filter(&h) {
                            return true;
                        }
                    }
                }
            } else {
                let l = node.left_or_first as usize;
                if let (Some(ln), Some(rn)) = (nodes.get(l), nodes.get(l + 1)) {
                    let hl = ray.slab(&ln.min, &ln.max, t_min, t_max).is_some();
                    let hr = ray.slab(&rn.min, &rn.max, t_min, t_max).is_some();
                    match (hl, hr) {
                        (true, true) => {
                            stack.push((l + 1) as u32, 0.0);
                            idx = l;
                            continue;
                        }
                        (true, false) => {
                            idx = l;
                            continue;
                        }
                        (false, true) => {
                            idx = l + 1;
                            continue;
                        }
                        (false, false) => {}
                    }
                }
            }
            match stack.pop() {
                Some((n, _)) => idx = n as usize,
                None => return false,
            }
        }
    }

    /// Recompute every node's bounds from `primitive`'s current vertex
    /// positions without changing the tree topology — the cheap update
    /// path for animated (skinned / morphed / edited) geometry whose
    /// connectivity is unchanged.
    ///
    /// Bottom-up in one reverse sweep (children always follow their
    /// parent in the array). Tree quality degrades as vertices move far
    /// from their build-time arrangement; rebuild when it matters.
    ///
    /// Returns `false` (leaving the affected leaves' old bounds in
    /// place) when a stored triangle now references an out-of-range
    /// vertex or a non-finite position; the tree stays safe to query.
    pub fn refit(&mut self, primitive: &Primitive) -> bool {
        let positions = &primitive.positions;
        let n_pos = positions.len();
        let mut ok = true;
        for i in (0..self.nodes.len()).rev() {
            let node = self.nodes[i];
            let mut acc: Option<BoundingBox> = None;
            if node.is_leaf() {
                let first = node.left_or_first as usize;
                for slot in first..first + node.tri_count as usize {
                    let pts = self
                        .triangle_vertices
                        .get(slot)
                        .and_then(|t| fetch(positions, *t, n_pos))
                        .filter(|p| p.iter().all(|q| finite_point(*q)));
                    match pts {
                        Some([a, b, c]) => {
                            let tb = BoundingBox::from_point(a).expand(b).expand(c);
                            acc = Some(acc.map_or(tb, |x| x.union(tb)));
                        }
                        None => ok = false,
                    }
                }
            } else {
                let l = node.left_or_first as usize;
                for c in [l, l + 1] {
                    if let Some(ch) = self.nodes.get(c) {
                        let cb = ch.bounds();
                        acc = Some(acc.map_or(cb, |x| x.union(cb)));
                    }
                }
            }
            if let Some(b) = acc {
                self.nodes[i].min = b.min;
                self.nodes[i].max = b.max;
            }
        }
        ok
    }

    /// Total number of nodes — interior plus leaf.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaf nodes.
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Number of triangles indexed by the tree.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Tight AABB of the whole tree (the root's bounds), `None` if
    /// the node array is empty.
    pub fn bounds(&self) -> Option<BoundingBox> {
        self.nodes.first().map(|n| n.bounds())
    }

    /// Maximum root-to-leaf depth (a single-leaf tree has depth 1).
    pub fn depth(&self) -> usize {
        tree_depth(&self.nodes, |n| n.is_leaf(), |n| n.left_or_first)
    }

    /// The tree's SAH cost with unit traversal / intersection costs,
    /// normalised by the root surface area — a build-quality metric
    /// (lower is better) for comparing builders.
    pub fn sah_cost(&self) -> f32 {
        sah_cost(&self.nodes, |n| (n.bounds(), n.tri_count))
    }

    /// The node array as a flat `u32` word stream in the GPU layout
    /// documented at module level (8 words = 32 bytes per node), ready
    /// for a storage-buffer upload.
    pub fn node_words(&self) -> Vec<u32> {
        self.nodes.iter().flat_map(|n| n.to_words()).collect()
    }
}

/// Fetch a triangle's three positions, `None` on an out-of-range index.
#[inline(always)]
pub(crate) fn fetch(positions: &[[f32; 3]], tri: [u32; 3], n_pos: usize) -> Option<[[f32; 3]; 3]> {
    let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
    if a >= n_pos || b >= n_pos || c >= n_pos {
        return None;
    }
    Some([positions[a], positions[b], positions[c]])
}

#[inline]
fn finite_point(p: [f32; 3]) -> bool {
    p[0].is_finite() && p[1].is_finite() && p[2].is_finite()
}

/// Half surface area of a box (the SAH only needs ratios).
#[inline]
fn half_area(b: &BoundingBox) -> f32 {
    let dx = (b.max[0] - b.min[0]).max(0.0);
    let dy = (b.max[1] - b.min[1]).max(0.0);
    let dz = (b.max[2] - b.min[2]).max(0.0);
    dx * dy + dy * dz + dz * dx
}

/// Maximum root-to-leaf depth of a pair-layout node array.
pub(crate) fn tree_depth<N>(
    nodes: &[N],
    is_leaf: impl Fn(&N) -> bool,
    left: impl Fn(&N) -> u32,
) -> usize {
    if nodes.is_empty() {
        return 0;
    }
    let mut max_depth = 0;
    let mut stack = vec![(0usize, 1usize)];
    while let Some((i, d)) = stack.pop() {
        max_depth = max_depth.max(d);
        let Some(n) = nodes.get(i) else { continue };
        if !is_leaf(n) {
            let l = left(n) as usize;
            if l > i {
                stack.push((l, d + 1));
                stack.push((l + 1, d + 1));
            }
        }
    }
    max_depth
}

/// Root-normalised SAH cost of a pair-layout node array with unit
/// traversal / intersection costs.
pub(crate) fn sah_cost<N>(nodes: &[N], info: impl Fn(&N) -> (BoundingBox, u32)) -> f32 {
    let Some(root) = nodes.first() else {
        return 0.0;
    };
    let root_area = half_area(&info(root).0);
    if root_area <= 0.0 {
        return 0.0;
    }
    let mut cost = 0.0f64;
    for n in nodes {
        let (b, count) = info(n);
        let a = (half_area(&b) / root_area) as f64;
        cost += if count == 0 { a } else { a * count as f64 };
    }
    cost as f32
}

/// One SAH bin: primitive count + bounds of the primitives whose
/// centroid falls in it.
#[derive(Clone, Copy)]
struct Bin {
    count: u32,
    min: [f32; 3],
    max: [f32; 3],
}

impl Bin {
    const EMPTY: Bin = Bin {
        count: 0,
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    #[inline]
    fn grow(&mut self, b: &BoundingBox) {
        for k in 0..3 {
            self.min[k] = self.min[k].min(b.min[k]);
            self.max[k] = self.max[k].max(b.max[k]);
        }
    }

    #[inline]
    fn merge(&mut self, o: &Bin) {
        self.count += o.count;
        for k in 0..3 {
            self.min[k] = self.min[k].min(o.min[k]);
            self.max[k] = self.max[k].max(o.max[k]);
        }
    }

    #[inline]
    fn half_area(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        half_area(&BoundingBox {
            min: self.min,
            max: self.max,
        })
    }
}

/// Builder work item: node slot + primitive range + depth.
struct Task {
    node: usize,
    start: usize,
    end: usize,
    depth: u32,
}

/// Generic top-down builder shared by [`Bvh`] and
/// [`crate::InstanceBvh`]. Returns the pair-layout node array and the
/// leaf-order permutation of the input item indices.
pub(crate) fn build_tree(
    bounds: &[BoundingBox],
    centroids: &[[f32; 3]],
    options: &BvhBuildOptions,
) -> (Vec<BvhNode>, Vec<u32>) {
    let n = bounds.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut nodes: Vec<BvhNode> = Vec::with_capacity(if n == 0 { 0 } else { 2 * n - 1 });
    if n == 0 {
        return (nodes, order);
    }
    let max_leaf = options.max_leaf_size.max(1);
    let bins = options.sah_bins.clamp(2, MAX_BINS);
    let c_trav = options.traversal_cost.max(0.0);
    let c_isect = options.intersection_cost.max(f32::MIN_POSITIVE);
    let placeholder = BvhNode {
        min: [0.0; 3],
        left_or_first: 0,
        max: [0.0; 3],
        tri_count: 0,
    };
    nodes.push(placeholder);
    let mut tasks = vec![Task {
        node: 0,
        start: 0,
        end: n,
        depth: 0,
    }];
    while let Some(Task {
        node,
        start,
        end,
        depth,
    }) = tasks.pop()
    {
        let count = end - start;
        // Node bounds + centroid bounds.
        let mut nb = bounds[order[start] as usize];
        let c0 = centroids[order[start] as usize];
        let mut cmin = c0;
        let mut cmax = c0;
        for &o in &order[start + 1..end] {
            nb = nb.union(bounds[o as usize]);
            let c = centroids[o as usize];
            for k in 0..3 {
                cmin[k] = cmin[k].min(c[k]);
                cmax[k] = cmax[k].max(c[k]);
            }
        }
        nodes[node].min = nb.min;
        nodes[node].max = nb.max;

        let extent = [cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]];
        let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
            0
        } else if extent[1] >= extent[2] {
            1
        } else {
            2
        };

        let mid: Option<usize> = if count == 1 {
            None
        } else if extent[axis] <= 0.0 {
            // Every centroid coincides: index split when forced.
            (count > max_leaf).then_some(start + count / 2)
        } else if depth >= BALANCE_DEPTH {
            if count <= max_leaf {
                None
            } else {
                let half = count / 2;
                order[start..end].select_nth_unstable_by(half, |a, b| {
                    centroids[*a as usize][axis].total_cmp(&centroids[*b as usize][axis])
                });
                Some(start + half)
            }
        } else {
            match options.strategy {
                BvhBuildStrategy::ObjectMedian => {
                    if count <= max_leaf {
                        None
                    } else {
                        let mid_coord = 0.5 * (cmin[axis] + cmax[axis]);
                        let m = partition(&mut order[start..end], |o| {
                            centroids[o as usize][axis] < mid_coord
                        });
                        Some(if m == 0 || m == count {
                            start + count / 2
                        } else {
                            start + m
                        })
                    }
                }
                BvhBuildStrategy::BinnedSah => sah_split(
                    &mut order[start..end],
                    bounds,
                    centroids,
                    &nb,
                    cmin,
                    extent,
                    bins,
                    c_trav,
                    c_isect,
                    max_leaf,
                )
                .map(|m| start + m),
            }
        };

        match mid {
            None => {
                nodes[node].left_or_first = start as u32;
                nodes[node].tri_count = count as u32;
            }
            Some(mid) => {
                let left = nodes.len();
                nodes.push(placeholder);
                nodes.push(placeholder);
                nodes[node].left_or_first = left as u32;
                nodes[node].tri_count = 0;
                // Right pushed first so the left subtree is built
                // (and its child pairs allocated) first: depth-first
                // pair order.
                tasks.push(Task {
                    node: left + 1,
                    start: mid,
                    end,
                    depth: depth + 1,
                });
                tasks.push(Task {
                    node: left,
                    start,
                    end: mid,
                    depth: depth + 1,
                });
            }
        }
    }
    (nodes, order)
}

/// In-place partition: items satisfying `pred` first. Returns the
/// count of satisfying items.
fn partition(items: &mut [u32], pred: impl Fn(u32) -> bool) -> usize {
    let mut left = 0;
    let mut right = items.len();
    while left < right {
        if pred(items[left]) {
            left += 1;
        } else {
            right -= 1;
            items.swap(left, right);
        }
    }
    left
}

/// Binned SAH split decision (Wald 2007). Returns the partition point
/// within `items`, or `None` for "make a leaf".
#[allow(clippy::too_many_arguments)]
fn sah_split(
    items: &mut [u32],
    bounds: &[BoundingBox],
    centroids: &[[f32; 3]],
    node_bounds: &BoundingBox,
    cmin: [f32; 3],
    extent: [f32; 3],
    bins: usize,
    c_trav: f32,
    c_isect: f32,
    max_leaf: usize,
) -> Option<usize> {
    let count = items.len();
    // (cost, axis, plane) — plane p separates bins [0, p] | [p+1, ..).
    let mut best: Option<(f32, usize, usize)> = None;
    let mut scale = [0.0f32; 3];
    for axis in 0..3 {
        if extent[axis] <= 0.0 {
            continue;
        }
        // Slightly under-scaled so the max centroid lands in the last
        // bin rather than one past it.
        scale[axis] = bins as f32 * (1.0 - 1e-5) / extent[axis];
        let mut bin = [Bin::EMPTY; MAX_BINS];
        for &o in items.iter() {
            let b = bin_index(centroids[o as usize][axis], cmin[axis], scale[axis], bins);
            bin[b].count += 1;
            bin[b].grow(&bounds[o as usize]);
        }
        // Right sweep: suffix areas / counts.
        let mut right_area = [0.0f32; MAX_BINS];
        let mut right_count = [0u32; MAX_BINS];
        let mut acc = Bin::EMPTY;
        for i in (1..bins).rev() {
            acc.merge(&bin[i]);
            right_area[i] = acc.half_area();
            right_count[i] = acc.count;
        }
        // Left sweep evaluating each plane.
        let mut acc = Bin::EMPTY;
        for p in 0..bins - 1 {
            acc.merge(&bin[p]);
            let (nl, nr) = (acc.count, right_count[p + 1]);
            if nl == 0 || nr == 0 {
                continue;
            }
            let cost = acc.half_area() * nl as f32 + right_area[p + 1] * nr as f32;
            if best.map_or(true, |(c, _, _)| cost < c) {
                best = Some((cost, axis, p));
            }
        }
    }
    let (cost, axis, plane) = best?;
    // Compare un-normalised costs (multiply through by the parent's
    // area, so a zero-area parent stays well-defined).
    let parent_area = half_area(node_bounds);
    let split_cost = c_trav * parent_area + c_isect * cost;
    let leaf_cost = c_isect * count as f32 * parent_area;
    if count <= max_leaf && leaf_cost <= split_cost {
        return None;
    }
    let m = partition(items, |o| {
        bin_index(centroids[o as usize][axis], cmin[axis], scale[axis], bins) <= plane
    });
    Some(if m == 0 || m == count { count / 2 } else { m })
}

#[inline(always)]
fn bin_index(c: f32, cmin: f32, scale: f32, bins: usize) -> usize {
    let f = (c - cmin) * scale;
    if f > 0.0 {
        (f as usize).min(bins - 1)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Topology;

    fn unit_triangle() -> Primitive {
        let mut p = Primitive::new(Topology::Triangles);
        p.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        p
    }

    fn two_parallel_triangles() -> Primitive {
        // Two triangles in z=1 and z=2 planes; a +Z ray from the
        // origin hits the z=1 triangle first.
        let mut p = Primitive::new(Topology::Triangles);
        p.positions = vec![
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ];
        p
    }

    fn unit_cube() -> Primitive {
        // 12-triangle unit cube spanning [0,1]^3. CCW from outside.
        let mut p = Primitive::new(Topology::Triangles);
        p.positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        p.indices = Some(crate::mesh::Indices::U32(vec![
            // -Z (CCW from -Z side)
            0, 2, 1, 0, 3, 2, // +Z (CCW from +Z side)
            4, 5, 6, 4, 6, 7, // -X
            0, 4, 7, 0, 7, 3, // +X
            1, 2, 6, 1, 6, 5, // -Y
            0, 1, 5, 0, 5, 4, // +Y
            3, 7, 6, 3, 6, 2,
        ]));
        p
    }

    #[test]
    fn build_single_triangle_one_leaf() {
        let p = unit_triangle();
        let bvh = Bvh::build(&p).expect("triangle builds");
        assert_eq!(bvh.triangle_count(), 1);
        assert_eq!(bvh.leaf_count(), 1);
        assert_eq!(bvh.node_count(), 1);
        assert!(bvh.nodes[0].is_leaf());
    }

    #[test]
    fn build_empty_primitive_returns_none() {
        let p = Primitive::new(Topology::Triangles);
        assert!(Bvh::build(&p).is_none());
    }

    #[test]
    fn build_non_triangle_topology_returns_none() {
        let mut p = Primitive::new(Topology::Lines);
        p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(Bvh::build(&p).is_none());
    }

    #[test]
    fn build_all_nan_returns_none() {
        let mut p = Primitive::new(Topology::Triangles);
        p.positions = vec![[f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(Bvh::build(&p).is_none());
    }

    #[test]
    fn build_skips_out_of_range_index() {
        // Two triangles, one with a bogus index — bvh holds exactly
        // one usable triangle.
        let mut p = Primitive::new(Topology::Triangles);
        p.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        p.indices = Some(crate::mesh::Indices::U32(vec![0, 1, 2, 0, 1, 99]));
        let bvh = Bvh::build(&p).expect("one good triangle remains");
        assert_eq!(bvh.triangle_count(), 1);
    }

    #[test]
    fn intersect_matches_brute_force_two_triangles() {
        let p = two_parallel_triangles();
        let bvh = Bvh::build(&p).unwrap();
        let r = Ray::new([0.3333, 0.3333, 0.0], [0.0, 0.0, 1.0]);
        let bvh_hit = bvh.intersect_ray(&p, r, f32::INFINITY).unwrap();
        let bf_hit = p.intersect_ray(r, f32::INFINITY).unwrap();
        assert_eq!(bvh_hit, bf_hit);
        // The closer triangle is the first (z=1), at t≈1.0.
        assert!((bvh_hit.t - 1.0).abs() < 1e-5);
        assert_eq!(bvh_hit.triangle_index, 0);
    }

    #[test]
    fn intersect_matches_brute_force_cube_through_minus_x_face() {
        let p = unit_cube();
        let bvh = Bvh::build(&p).unwrap();
        let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let bvh_hit = bvh.intersect_ray(&p, r, f32::INFINITY).unwrap();
        let bf_hit = p.intersect_ray(r, f32::INFINITY).unwrap();
        // The ray crosses the face diagonal: either adjacent triangle
        // may win the exact tie, but t / facing must agree.
        assert_eq!(bvh_hit.t, bf_hit.t);
        assert_eq!(bvh_hit.front_face, bf_hit.front_face);
        assert!((bvh_hit.t - 1.0).abs() < 1e-5);
        // The -X face is the front face when entering from -X
        // direction +X side.
        assert!(bvh_hit.front_face);
    }

    #[test]
    fn intersect_miss_returns_none() {
        let p = unit_cube();
        let bvh = Bvh::build(&p).unwrap();
        // Ray passes outside the cube entirely.
        let r = Ray::new([-1.0, 5.0, 0.5], [1.0, 0.0, 0.0]);
        assert!(bvh.intersect_ray(&p, r, f32::INFINITY).is_none());
    }

    #[test]
    fn intersect_t_max_culls_hits() {
        let p = unit_cube();
        let bvh = Bvh::build(&p).unwrap();
        let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        // The -X face is at t=1.0; t_max=0.5 culls it.
        assert!(bvh.intersect_ray(&p, r, 0.5).is_none());
    }

    #[test]
    fn intersect_matches_brute_force_across_many_rays() {
        // Fuzz-style cross-check: a grid of rays from the +Z side of
        // the cube shooting -Z. The BVH hit must agree with the
        // brute-force hit on `t` (and therefore on the hit point);
        // a `triangle_index` tie at a shared edge / corner can fall
        // either side depending on visit order, but the geometry
        // must coincide.
        let p = unit_cube();
        let bvh = Bvh::build(&p).unwrap();
        for ix in 0..7 {
            for iy in 0..7 {
                let x = -0.5 + 0.25 * ix as f32;
                let y = -0.5 + 0.25 * iy as f32;
                let r = Ray::new([x, y, 2.0], [0.0, 0.0, -1.0]);
                let bf = p.intersect_ray(r, f32::INFINITY);
                let bv = bvh.intersect_ray(&p, r, f32::INFINITY);
                match (bf, bv) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert!(
                            (a.t - b.t).abs() < 1e-5,
                            "t mismatch at ({}, {}): bf={} bv={}",
                            x,
                            y,
                            a.t,
                            b.t
                        );
                        assert_eq!(a.front_face, b.front_face);
                    }
                    other => panic!("hit/miss disagreement at ({}, {}): {:?}", x, y, other),
                }
            }
        }
    }

    #[test]
    fn any_ray_intersection_true_through_cube() {
        let p = unit_cube();
        let bvh = Bvh::build(&p).unwrap();
        let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        assert!(bvh.any_ray_intersection(&p, r, f32::INFINITY));
    }

    #[test]
    fn any_ray_intersection_false_when_outside() {
        let p = unit_cube();
        let bvh = Bvh::build(&p).unwrap();
        let r = Ray::new([-1.0, 5.0, 0.5], [1.0, 0.0, 0.0]);
        assert!(!bvh.any_ray_intersection(&p, r, f32::INFINITY));
    }

    #[test]
    fn any_ray_intersection_respects_t_max() {
        let p = unit_cube();
        let bvh = Bvh::build(&p).unwrap();
        let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        assert!(!bvh.any_ray_intersection(&p, r, 0.5));
    }

    #[test]
    fn leaf_threshold_is_respected() {
        // 50-triangle fan in z=1 plane around (0.5, 0.5). Tree
        // should have at least 2 leaves and every leaf must hold at
        // most LEAF_THRESHOLD triangles.
        let mut p = Primitive::new(Topology::Triangles);
        // Central vertex first, then 50 perimeter vertices.
        p.positions.push([0.5, 0.5, 1.0]);
        let mut indices = Vec::new();
        for i in 0..50u32 {
            let theta = (i as f32) * std::f32::consts::TAU / 50.0;
            p.positions
                .push([0.5 + 0.4 * theta.cos(), 0.5 + 0.4 * theta.sin(), 1.0]);
            let next = if i == 49 { 1 } else { i + 2 };
            indices.extend_from_slice(&[0, i + 1, next]);
        }
        p.indices = Some(crate::mesh::Indices::U32(indices));
        let bvh = Bvh::build(&p).unwrap();
        assert_eq!(bvh.triangle_count(), 50);
        for node in &bvh.nodes {
            if node.is_leaf() {
                assert!(
                    node.tri_count as usize <= Bvh::LEAF_THRESHOLD,
                    "leaf has {} > {}",
                    node.tri_count,
                    Bvh::LEAF_THRESHOLD
                );
            }
        }
        assert!(bvh.leaf_count() >= 2);
    }

    #[test]
    fn coincident_centroids_still_build() {
        // 16 identical triangles overlaid — every centroid coincides
        // so the centroid extent is zero on every axis. The
        // degenerate-split path must still terminate at the leaf
        // threshold, not infinite-recurse.
        let mut p = Primitive::new(Topology::Triangles);
        p.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let mut indices = Vec::new();
        for _ in 0..16 {
            indices.extend_from_slice(&[0, 1, 2]);
        }
        p.indices = Some(crate::mesh::Indices::U32(indices));
        let bvh = Bvh::build(&p).expect("degenerate centroids still build");
        assert_eq!(bvh.triangle_count(), 16);
        // Every leaf still respects the threshold.
        for node in &bvh.nodes {
            if node.is_leaf() {
                assert!(node.tri_count as usize <= Bvh::LEAF_THRESHOLD);
            }
        }
    }

    #[test]
    fn root_bounds_are_tight() {
        let p = unit_cube();
        let bvh = Bvh::build(&p).unwrap();
        let bounds = bvh.bounds().unwrap();
        assert_eq!(bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.max, [1.0, 1.0, 1.0]);
    }
}
