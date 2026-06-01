//! Bounding-volume hierarchy over a [`crate::Primitive`]'s triangle
//! tessellation — accelerates many-ray workloads from
//! `O(triangle_count)` per ray to roughly `O(log triangle_count)` once
//! the tree is built.
//!
//! # Algorithm
//!
//! The tree is a binary AABB hierarchy built top-down by
//! **median-split on the largest-extent axis** of the parent box.
//! Each interior node stores a tight AABB over its subtree's
//! triangles plus indices into a flat [`Bvh::nodes`] arena pointing
//! at its two children. Each leaf node stores a tight AABB plus a
//! contiguous range `[first_tri, first_tri + tri_count)` into a
//! permuted index array [`Bvh::triangles`]. The permutation is the
//! single mutable side-effect of construction: leaf triangle lists
//! are stored contiguously by re-ordering the index array, so a leaf
//! visit walks a slice rather than a scattered set of indices. This
//! is the classical object-median AABB-tree construction reviewed by
//! Goldsmith & Salmon ("Automatic Creation of Object Hierarchies for
//! Ray Tracing", IEEE CG&A 7(5), 1987) — pre-dating the
//! surface-area-heuristic family that displaced it for very large
//! scenes. We pick object-median because it has no spec gaps, is
//! fully deterministic from the input triangle order, and is the
//! cheapest construction whose query asymptotics are still
//! logarithmic for well-behaved inputs.
//!
//! Splitting recurses while the candidate node holds more than
//! [`Bvh::LEAF_THRESHOLD`] triangles (currently `4`). When every
//! centroid lies on a single point along the chosen axis (zero
//! extent), we fall back to a degenerate "all-left, empty-right"
//! split so the recursion still terminates at the leaf threshold
//! without entering an infinite loop.
//!
//! # Traversal
//!
//! Both queries traverse with an **explicit LIFO stack** rather than
//! recursion — Rust's release-mode tail-call optimisation is not
//! guaranteed and a deeply-built tree could otherwise blow the host
//! stack on pathological input. The stack is sized to the tree's
//! depth and pushed in **near-child-first order**: the slab test
//! (Kay & Kajiya, "Ray Tracing Complex Scenes", SIGGRAPH 1986) at
//! each interior node returns the entry parameter for both children,
//! and the child with the smaller entry parameter is visited first.
//! That ordering lets the closest-hit query shrink its `t_max` bound
//! before descending into the farther subtree — when the near
//! subtree's leaf hits something at `t = t0`, every node in the
//! far subtree whose AABB enters at `t > t0` is pruned without
//! visiting any triangles.
//!
//! The any-hit (shadow-ray) traversal does not bother with near/far
//! ordering — the first leaf hit short-circuits the walk, so the
//! order in which children land on the stack only changes which leaf
//! that is, never whether the answer is `true` vs `false`.
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
//! and triangles whose vertices contain a NaN coordinate are silently
//! skipped during build — same contract as
//! [`crate::Primitive::compute_normals`] /
//! [`crate::Primitive::surface_area`] /
//! [`crate::Primitive::signed_volume`]. A primitive whose every
//! triangle is in one of those classes builds to `None`.

use crate::ray::{intersect_aabb, intersect_triangle, Ray, RayHit};
use crate::scene::BoundingBox;
use crate::Primitive;

/// One node in a [`Bvh`].
///
/// The same struct represents both interior and leaf nodes. A leaf is
/// signalled by `tri_count > 0`; an interior node is signalled by
/// `tri_count == 0` and `left_child < right_child < nodes.len()`. The
/// tight AABB over the node's subtree is in `bounds` either way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNode {
    /// Tight axis-aligned bounding box over every triangle reachable
    /// from this node — the union of all leaf-triangle AABBs in the
    /// subtree.
    pub bounds: BoundingBox,
    /// For an interior node, the index of the left child in
    /// [`Bvh::nodes`]. For a leaf, the index `first_tri` into
    /// [`Bvh::triangles`] where the leaf's triangle slice starts.
    pub left_or_first: u32,
    /// For an interior node, the index of the right child in
    /// [`Bvh::nodes`] (always `left_or_first + 1` in the current
    /// construction, but treated as an explicit field so the layout
    /// can evolve without breaking serialisers). For a leaf, unused
    /// (`0`).
    pub right_child: u32,
    /// `0` for an interior node; the **number of triangles in this
    /// leaf** for a leaf. Use [`BvhNode::is_leaf`] for clarity.
    pub tri_count: u32,
}

impl BvhNode {
    /// `true` if this node is a leaf (holds a contiguous triangle
    /// range in [`Bvh::triangles`] rather than two child indices).
    pub fn is_leaf(&self) -> bool {
        self.tri_count > 0
    }
}

/// Bounding-volume hierarchy built from one [`crate::Primitive`].
///
/// See the module-level docs for the construction and traversal
/// algorithms. The struct is a value type — clone-by-Vec-copy, no
/// shared state with the source primitive, no interior mutability —
/// so a build can be cached alongside its primitive without lifetime
/// gymnastics.
#[derive(Clone, Debug)]
pub struct Bvh {
    /// Flat array of nodes; node `0` is the root.
    pub nodes: Vec<BvhNode>,
    /// Permuted triangle-index array. A leaf node's triangle range
    /// `[first_tri, first_tri + tri_count)` is a slice into this
    /// array. Each entry is the index that should be passed to
    /// [`crate::Primitive::triangle_indices`]`[i]` to recover the
    /// triangle's vertex indices — i.e. the position the triangle
    /// occupied in the source enumeration, not a vertex index.
    pub triangles: Vec<u32>,
}

impl Bvh {
    /// Maximum number of triangles a leaf is allowed to hold.
    ///
    /// Smaller values trade a deeper tree (more nodes, more traversal
    /// work) for fewer triangle tests at each leaf. The chosen value
    /// (`4`) is a long-standing balance for SSE-era CPUs from the
    /// rendering literature (e.g. Wald, Boulos & Shirley, "Ray
    /// Tracing Deformable Scenes Using Dynamic Bounding Volume
    /// Hierarchies", ACM TOG 26(1), 2007). It is exposed as a
    /// constant so callers can compare against it in tests without
    /// hard-coding the literal.
    pub const LEAF_THRESHOLD: usize = 4;

    /// Build a BVH over every triangle of `primitive`.
    ///
    /// Returns `None` when the primitive enumerates zero usable
    /// triangles — either because the topology is non-triangle
    /// (`Lines`/`Points`/`LineStrip`/`LineLoop`), or because every
    /// triangle is silently skipped by the robustness contract
    /// (out-of-range vertex index, or a NaN coordinate on at least
    /// one vertex of the triangle).
    ///
    /// Pure: the source primitive is not mutated.
    pub fn build(primitive: &Primitive) -> Option<Self> {
        let n_pos = primitive.positions.len();
        let all_tris = primitive.triangle_indices();
        if all_tris.is_empty() {
            return None;
        }

        // Per-triangle precomputed AABB + centroid, filtered through
        // the same robustness contract as the rest of the crate.
        let mut tri_bounds: Vec<BoundingBox> = Vec::with_capacity(all_tris.len());
        let mut tri_centroids: Vec<[f32; 3]> = Vec::with_capacity(all_tris.len());
        let mut tri_indices: Vec<u32> = Vec::with_capacity(all_tris.len());

        for (idx, [ia, ib, ic]) in all_tris.iter().enumerate() {
            let (ia, ib, ic) = (*ia as usize, *ib as usize, *ic as usize);
            if ia >= n_pos || ib >= n_pos || ic >= n_pos {
                continue;
            }
            let p0 = primitive.positions[ia];
            let p1 = primitive.positions[ib];
            let p2 = primitive.positions[ic];
            // Any NaN coord on any vertex disqualifies the triangle.
            if !finite_point(p0) || !finite_point(p1) || !finite_point(p2) {
                continue;
            }
            let bbox = BoundingBox::from_point(p0).expand(p1).expand(p2);
            let centroid = [
                (p0[0] + p1[0] + p2[0]) / 3.0,
                (p0[1] + p1[1] + p2[1]) / 3.0,
                (p0[2] + p1[2] + p2[2]) / 3.0,
            ];
            tri_bounds.push(bbox);
            tri_centroids.push(centroid);
            tri_indices.push(idx as u32);
        }

        if tri_indices.is_empty() {
            return None;
        }

        // `tri_indices` is the soon-to-be-permuted leaf-order index
        // array; `tri_bounds` and `tri_centroids` are parallel scratch
        // arrays that follow the same permutation.
        let mut nodes: Vec<BvhNode> = Vec::new();
        let total = tri_indices.len();
        build_recursive(
            &mut nodes,
            &mut tri_indices,
            &mut tri_bounds,
            &mut tri_centroids,
            0,
            total,
        );

        Some(Bvh {
            nodes,
            triangles: tri_indices,
        })
    }

    /// Closest-hit ray query against the BVH.
    ///
    /// Returns `Some(RayHit)` of the smallest-`t` triangle intersection
    /// in `[0, t_max]`, or `None` if nothing in range is struck. The
    /// returned [`crate::RayHit::triangle_index`] is the index in
    /// [`crate::Primitive::triangle_indices`] — identical to the value
    /// [`crate::Primitive::intersect_ray`] would have returned for the
    /// same ray on the same primitive.
    ///
    /// Closest-hit `t` and barycentric coordinates match
    /// [`crate::Primitive::intersect_ray`] on every input ray;
    /// tests cross-validate the two paths. A `triangle_index` tie at
    /// a shared edge or corner (two adjacent triangles strictly
    /// share the same `t`) can land on either contributing triangle
    /// depending on visit order, but the hit point is identical.
    pub fn intersect_ray(&self, primitive: &Primitive, ray: Ray, t_max: f32) -> Option<RayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let all_tris = primitive.triangle_indices();
        let n_pos = primitive.positions.len();

        // The slab test on the root short-circuits the whole walk
        // when the ray misses the scene-wide AABB. The `?` borrows
        // the slab-test option for its presence only — the tuple is
        // discarded; an inner `intersect_aabb` call per child node
        // does the per-step early-out.
        intersect_aabb(
            ray,
            self.nodes[0].bounds.min,
            self.nodes[0].bounds.max,
            t_max,
        )?;

        let mut best: Option<RayHit> = None;
        let mut best_t = t_max;

        // Explicit stack of (node_index, t_enter). `t_enter` is the
        // slab-method entry parameter for the node — if it exceeds
        // the current `best_t`, every triangle in the subtree is
        // already strictly beyond the best closest hit and the node
        // can be dropped without inspecting it. The stack capacity is
        // a heuristic high-water mark for a roughly balanced tree.
        let mut stack: Vec<(u32, f32)> = Vec::with_capacity(64);
        stack.push((0, 0.0));

        while let Some((node_idx, t_enter)) = stack.pop() {
            if t_enter > best_t {
                continue;
            }
            let node = &self.nodes[node_idx as usize];

            if node.is_leaf() {
                let start = node.left_or_first as usize;
                let end = start + node.tri_count as usize;
                for &tri_idx in &self.triangles[start..end] {
                    let tri = all_tris[tri_idx as usize];
                    let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                    // Defensive — the build step already filtered,
                    // but `primitive` is a `&` to a caller-supplied
                    // value: if the same Bvh is reused against a
                    // mutated primitive, we still refuse to panic.
                    if ia >= n_pos || ib >= n_pos || ic >= n_pos {
                        continue;
                    }
                    let p0 = primitive.positions[ia];
                    let p1 = primitive.positions[ib];
                    let p2 = primitive.positions[ic];
                    if let Some((t, u, v, front)) = intersect_triangle(ray, p0, p1, p2, best_t) {
                        let w = 1.0 - u - v;
                        best = Some(RayHit {
                            t,
                            triangle_index: tri_idx as usize,
                            barycentric: [w, u, v],
                            front_face: front,
                        });
                        best_t = t;
                    }
                }
                continue;
            }

            // Interior node — slab-test both children and push the
            // farther one first so the nearer is popped (and walked)
            // first.
            let left = node.left_or_first;
            let right = node.right_child;
            let left_bbox = self.nodes[left as usize].bounds;
            let right_bbox = self.nodes[right as usize].bounds;
            let left_hit = intersect_aabb(ray, left_bbox.min, left_bbox.max, best_t);
            let right_hit = intersect_aabb(ray, right_bbox.min, right_bbox.max, best_t);

            match (left_hit, right_hit) {
                (Some((tl, _)), Some((tr, _))) => {
                    if tl <= tr {
                        stack.push((right, tr));
                        stack.push((left, tl));
                    } else {
                        stack.push((left, tl));
                        stack.push((right, tr));
                    }
                }
                (Some((tl, _)), None) => stack.push((left, tl)),
                (None, Some((tr, _))) => stack.push((right, tr)),
                (None, None) => {}
            }
        }

        best
    }

    /// Shadow-ray early-exit query.
    ///
    /// Returns `true` as soon as any triangle in the BVH is struck at
    /// a parameter `t ∈ [0, t_max]`. Does not search for the closest
    /// hit; once a single triangle's intersection lands, the walk
    /// short-circuits.
    ///
    /// Bit-exact match with [`crate::Primitive::any_ray_intersection`]
    /// in terms of the boolean answer (the same set of rays return
    /// true).
    pub fn any_ray_intersection(&self, primitive: &Primitive, ray: Ray, t_max: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let all_tris = primitive.triangle_indices();
        let n_pos = primitive.positions.len();

        if intersect_aabb(
            ray,
            self.nodes[0].bounds.min,
            self.nodes[0].bounds.max,
            t_max,
        )
        .is_none()
        {
            return false;
        }

        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx as usize];

            if node.is_leaf() {
                let start = node.left_or_first as usize;
                let end = start + node.tri_count as usize;
                for &tri_idx in &self.triangles[start..end] {
                    let tri = all_tris[tri_idx as usize];
                    let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                    if ia >= n_pos || ib >= n_pos || ic >= n_pos {
                        continue;
                    }
                    let p0 = primitive.positions[ia];
                    let p1 = primitive.positions[ib];
                    let p2 = primitive.positions[ic];
                    if intersect_triangle(ray, p0, p1, p2, t_max).is_some() {
                        return true;
                    }
                }
                continue;
            }

            let left = node.left_or_first;
            let right = node.right_child;
            let left_bbox = self.nodes[left as usize].bounds;
            let right_bbox = self.nodes[right as usize].bounds;
            if intersect_aabb(ray, left_bbox.min, left_bbox.max, t_max).is_some() {
                stack.push(left);
            }
            if intersect_aabb(ray, right_bbox.min, right_bbox.max, t_max).is_some() {
                stack.push(right);
            }
        }

        false
    }

    /// Total number of nodes — interior plus leaf.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaf nodes — equivalently, the number of contiguous
    /// triangle ranges in [`Bvh::triangles`].
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Number of triangles indexed by the tree.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Tight AABB of the whole tree — the root's bounds, or `None` if
    /// the tree is empty (which the public constructor never returns,
    /// but the field-driven construction path can in principle).
    pub fn bounds(&self) -> Option<BoundingBox> {
        self.nodes.first().map(|n| n.bounds)
    }
}

/// Recursive median-split builder. Operates on the inclusive-exclusive
/// range `[start, end)` of the (in-place permuted) `tri_indices` /
/// `tri_bounds` / `tri_centroids` parallel arrays. Returns the index
/// of the freshly-pushed node in `nodes`.
fn build_recursive(
    nodes: &mut Vec<BvhNode>,
    tri_indices: &mut [u32],
    tri_bounds: &mut [BoundingBox],
    tri_centroids: &mut [[f32; 3]],
    start: usize,
    end: usize,
) -> u32 {
    debug_assert!(end > start, "build_recursive called on an empty range");
    let count = end - start;

    // Tight bound over this node's triangle range.
    let mut node_bounds = tri_bounds[start];
    for b in &tri_bounds[start + 1..end] {
        node_bounds = node_bounds.union(*b);
    }

    // Leaf cut-off — recursion stops here.
    if count <= Bvh::LEAF_THRESHOLD {
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            bounds: node_bounds,
            left_or_first: start as u32,
            right_child: 0,
            tri_count: count as u32,
        });
        return node_idx;
    }

    // Centroid bound — drives the axis + split choice. Distinct from
    // `node_bounds` because the longest axis of the centroid spread
    // is what matters for splitting, not the longest axis of the
    // triangle bounds (which can be dominated by one very large
    // triangle).
    let mut cmin = tri_centroids[start];
    let mut cmax = cmin;
    for c in &tri_centroids[start + 1..end] {
        for axis in 0..3 {
            if c[axis] < cmin[axis] {
                cmin[axis] = c[axis];
            }
            if c[axis] > cmax[axis] {
                cmax[axis] = c[axis];
            }
        }
    }
    let extent = [cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]];
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };

    // Degenerate split — every centroid coincides on the chosen axis
    // (and therefore on every axis with not-greater extent). Fall
    // back to "all triangles in the left child, an empty right
    // child" by recursing with mid = start + count/2. This is
    // suboptimal (the tree is unbalanced) but it is finite — each
    // recursion strictly shrinks the range, so the leaf threshold is
    // eventually hit and the recursion bottoms out.
    let mid = if extent[axis] <= 0.0 {
        start + count / 2
    } else {
        // Object-median split: partition by centroid coordinate
        // against the **midpoint of the centroid bound** on the
        // chosen axis. Triangles whose centroid coordinate is below
        // the midpoint go left, the rest go right. The partition is
        // an in-place three-way swap loop that keeps the parallel
        // arrays in lockstep.
        let mid_coord = 0.5 * (cmin[axis] + cmax[axis]);
        let mut left = start;
        let mut right = end;
        while left < right {
            if tri_centroids[left][axis] < mid_coord {
                left += 1;
            } else {
                right -= 1;
                tri_indices.swap(left, right);
                tri_bounds.swap(left, right);
                tri_centroids.swap(left, right);
            }
        }
        // Pathological centroid distributions can leave one side
        // empty even with a non-zero extent (e.g. every centroid sits
        // exactly on `mid_coord`). Equal split keeps the recursion
        // making progress in that case too.
        if left == start || left == end {
            start + count / 2
        } else {
            left
        }
    };

    // Reserve the parent slot first so children know its address;
    // children get pushed in pre-order (left-then-right) so a
    // depth-first iterator over `nodes` walks the tree top-down,
    // left-to-right. The parent's left/right child indices are
    // back-patched once both subtrees finish building.
    let parent_idx = nodes.len() as u32;
    nodes.push(BvhNode {
        bounds: node_bounds,
        left_or_first: 0,
        right_child: 0,
        tri_count: 0,
    });

    let left_child = build_recursive(nodes, tri_indices, tri_bounds, tri_centroids, start, mid);
    let right_child = build_recursive(nodes, tri_indices, tri_bounds, tri_centroids, mid, end);

    nodes[parent_idx as usize].left_or_first = left_child;
    nodes[parent_idx as usize].right_child = right_child;

    parent_idx
}

#[inline]
fn finite_point(p: [f32; 3]) -> bool {
    p[0].is_finite() && p[1].is_finite() && p[2].is_finite()
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
        assert_eq!(bvh_hit, bf_hit);
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
