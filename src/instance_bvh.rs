//! Scene-level bounding-volume hierarchy over the per-instance
//! world-AABB snapshot — accelerates many-ray scene-graph queries
//! from `O(reachable_instance_count)` per ray to roughly
//! `O(log reachable_instance_count)` once the tree is built.
//!
//! The seed for this acceleration layer is the
//! [`crate::Scene3D::world_node_bounds`] snapshot from round 216:
//! every reachable node carrying a mesh contributes one entry to a
//! flat list of [`Instance`] descriptors, each pairing the node's
//! world-space mesh AABB with its world matrix, the matrix inverse,
//! and the `(NodeId, MeshId)` pair the per-instance ray query
//! ultimately re-dispatches to.
//!
//! # Algorithm
//!
//! The tree is the same flat-array binary AABB hierarchy as
//! [`crate::Bvh`]: median-split on the largest-extent axis of the
//! centroid bound, leaves at [`InstanceBvh::LEAF_THRESHOLD`] = 4
//! instances. The construction reviewed by Goldsmith & Salmon
//! ("Automatic Creation of Object Hierarchies for Ray Tracing", IEEE
//! CG&A 7(5), 1987) is deterministic from the input instance order,
//! has no spec gaps, and is the cheapest construction whose query
//! asymptotics are still logarithmic for well-behaved scenes. It
//! displaces the per-primitive choice once for the per-instance
//! layer — the two BVHs share the construction so the only
//! difference is the leaf payload (a slice of triangle indices on
//! [`crate::Bvh`], one [`Instance`] per leaf entry here).
//!
//! # Traversal
//!
//! Both queries traverse with an **explicit LIFO stack**: release-mode
//! tail-call optimisation is not guaranteed in Rust and a deeply-built
//! tree on a pathological scene could otherwise blow the host stack.
//! The slab method (Kay & Kajiya, "Ray Tracing Complex Scenes",
//! SIGGRAPH 1986) at each interior node returns the entry parameter
//! for both children; the child with the smaller entry parameter is
//! visited first so the closest-hit query can shrink its `t_max`
//! bound before descending into the farther subtree. The any-hit
//! traversal pushes children in fixed (left-then-right) order — the
//! first leaf hit short-circuits, so the answer (`true` / `false`) is
//! order-invariant.
//!
//! # Per-instance dispatch
//!
//! A leaf visit calls [`crate::Mesh::intersect_ray`] for the closest-hit
//! variant on the mesh-local ray — the world ray transformed through
//! the cached per-instance inverse. Affine change-of-frame leaves the
//! scalar ray parameter `t` invariant
//! (`P_world = M · P_local` ⇒ `O_world + t · D_world = M · (O_local + t · D_local)`),
//! so the world hit point is recoverable via `ray.point_at(t)` without
//! re-transforming through the instance matrix. The any-hit walk
//! delegates the same way; either query returns the union of the
//! per-instance answers exactly as
//! [`crate::Scene3D::intersect_ray`] / [`crate::Scene3D::any_ray_intersection`]
//! would.
//!
//! # Determinism
//!
//! The instance list is gathered by the same depth-first walk as
//! [`crate::Scene3D::world_node_bounds`] — leftmost root first,
//! leftmost child first, cycle nodes visited once at first arrival,
//! shared children resolved through the first parent's chain.
//! Within the BVH walk a tie on `t` between two instances visited at
//! the same effective distance is broken by the earlier-visited
//! instance (matches [`crate::Scene3D::intersect_ray`]'s
//! deterministic-winner convention). The build order matches the
//! gather order; the median-split partition is in-place and
//! deterministic from the centroid bound.
//!
//! # Robustness contract
//!
//! Nodes producing an instance with a non-affine / singular / non-finite
//! world matrix are skipped at gather time (same condition as
//! [`crate::Scene3D::intersect_ray`]'s per-instance affine-inverse
//! guard). Nodes whose mesh has no [`crate::Mesh::bounding_box`]
//! (empty mesh, or every primitive has non-finite bounds) are also
//! skipped — no instance reaches the BVH leaves without a finite AABB.
//! A scene whose every reachable node-mesh instance falls through one
//! of these guards builds to `None`.

use crate::ray::{intersect_aabb, Ray};
use crate::scene::{mat4_affine_inverse, mat4_mul, ray_into_local, BoundingBox, MeshId, NodeId};
use crate::{Scene3D, SceneRayHit};

/// One reachable node-mesh instance in a [`Scene3D`] — the world
/// matrix, its inverse, the mesh-local AABB transformed to world
/// space, and the `(NodeId, MeshId)` re-dispatch keys.
///
/// Built by [`InstanceBvh::build`]; the descriptor is stored in the
/// BVH's permuted leaf order rather than `NodeId.0` order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// The scene-graph node that produced this instance — the same
    /// `NodeId` [`Scene3D::intersect_ray`] would have reported on a
    /// hit at this instance.
    pub node: NodeId,
    /// The mesh that the node carries — equal to
    /// `scene.nodes[node.0].mesh.unwrap()` at build time. Stored so a
    /// caller can look the mesh up without re-walking the node arena.
    pub mesh: MeshId,
    /// World-space AABB of the mesh's vertices, computed at gather
    /// time as the eight-corner refit of the mesh's local AABB
    /// through the node's full ancestor-chain world matrix. The
    /// same value [`Scene3D::world_node_bounds`] reports for this
    /// node's slot.
    pub bounds: BoundingBox,
    /// World matrix for the node — `parent_chain * node.transform.to_matrix()`,
    /// row-major column-vector.
    pub world: [[f32; 4]; 4],
    /// Affine inverse of `world`, computed at gather time. Cached so
    /// the per-ray query path doesn't repeatedly invert the same
    /// matrix once per ray per instance. The bottom row is
    /// `[0, 0, 0, 1]` by construction (the inverse function refuses
    /// non-affine inputs).
    pub world_inv: [[f32; 4]; 4],
}

/// One node in an [`InstanceBvh`].
///
/// Same flat-array shape as [`crate::BvhNode`]: leaf signalled by
/// `instance_count > 0` (range `[first, first + instance_count)` into
/// [`InstanceBvh::instances`]), interior signalled by
/// `instance_count == 0` (`left_or_first` / `right_child` index into
/// [`InstanceBvh::nodes`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceBvhNode {
    /// Tight axis-aligned bounding box over every instance reachable
    /// from this node — the union of the instance world AABBs in the
    /// subtree.
    pub bounds: BoundingBox,
    /// For an interior node, the index of the left child in
    /// [`InstanceBvh::nodes`]. For a leaf, the index `first` into
    /// [`InstanceBvh::instances`] where the leaf's instance slice
    /// starts.
    pub left_or_first: u32,
    /// For an interior node, the index of the right child in
    /// [`InstanceBvh::nodes`]. For a leaf, unused (`0`).
    pub right_child: u32,
    /// `0` for an interior node; the **number of instances in this
    /// leaf** for a leaf. Use [`InstanceBvhNode::is_leaf`] for clarity.
    pub instance_count: u32,
}

impl InstanceBvhNode {
    /// `true` if this node is a leaf (holds a contiguous instance
    /// range in [`InstanceBvh::instances`] rather than two child
    /// indices).
    pub fn is_leaf(&self) -> bool {
        self.instance_count > 0
    }
}

/// Bounding-volume hierarchy over the per-instance world AABBs of a
/// [`Scene3D`] — the next acceleration layer above
/// [`crate::Bvh::intersect_ray`] that round 210 / 216 gestured at.
///
/// See the module-level docs for the construction and traversal
/// algorithms. The struct is a value type — clone-by-Vec-copy, no
/// shared state with the source scene, no interior mutability — so a
/// build can be cached alongside its scene without lifetime
/// gymnastics, but it must be rebuilt when the scene's node
/// transforms or mesh AABBs change.
#[derive(Clone, Debug)]
pub struct InstanceBvh {
    /// Flat array of nodes; node `0` is the root when non-empty.
    pub nodes: Vec<InstanceBvhNode>,
    /// Permuted instance array. A leaf's instance range is a slice
    /// into this array. The order is the build-time leaf-grouping
    /// permutation of the source gather order, not the original
    /// DFS order.
    pub instances: Vec<Instance>,
}

impl InstanceBvh {
    /// Maximum number of instances a leaf is allowed to hold.
    ///
    /// Smaller values trade a deeper tree (more nodes, more traversal
    /// work) for fewer per-instance mesh re-dispatches at each leaf.
    /// The chosen value (`4`) matches [`crate::Bvh::LEAF_THRESHOLD`]
    /// so the two acceleration layers stay symmetrical.
    pub const LEAF_THRESHOLD: usize = 4;

    /// Build an [`InstanceBvh`] over every reachable node-mesh
    /// instance of `scene`.
    ///
    /// Returns `None` when the scene contributes zero usable
    /// instances — empty arenas, no reachable nodes carrying a mesh,
    /// every reachable instance has a singular or non-finite world
    /// matrix, or every reachable mesh has no local AABB.
    ///
    /// Pure: the source scene is not mutated.
    pub fn build(scene: &Scene3D) -> Option<Self> {
        let instances = gather_instances(scene);
        if instances.is_empty() {
            return None;
        }
        let mut indices: Vec<u32> = (0..instances.len() as u32).collect();
        let mut bounds: Vec<BoundingBox> = instances.iter().map(|i| i.bounds).collect();
        let mut centroids: Vec<[f32; 3]> = instances.iter().map(|i| i.bounds.center()).collect();
        let total = indices.len();
        let mut nodes: Vec<InstanceBvhNode> = Vec::new();
        build_recursive(
            &mut nodes,
            &mut indices,
            &mut bounds,
            &mut centroids,
            0,
            total,
        );
        // Materialise the permuted instance list — the leaf slices
        // in `nodes` index directly into `permuted`.
        let permuted: Vec<Instance> = indices.iter().map(|&i| instances[i as usize]).collect();
        Some(InstanceBvh {
            nodes,
            instances: permuted,
        })
    }

    /// Closest-hit world-space ray query across every reachable
    /// node-mesh instance in the source scene.
    ///
    /// Returns `Some(SceneRayHit)` of the smallest-`t` per-instance
    /// hit in `[0, t_max]`, or `None` if nothing in range is struck.
    /// The answer matches [`Scene3D::intersect_ray`]'s
    /// closest-hit walk on the same scene + ray: same `t`, same
    /// world hit point. The `NodeId` / `primitive_index` can
    /// tie-break differently on a strict tie (two coincident
    /// instances reporting the exact same `t`) because the BVH walk
    /// orders by AABB entry parameter rather than DFS order, but the
    /// hit geometry is identical.
    ///
    /// `scene` must be the same scene the BVH was built from — the
    /// per-instance world matrices are cached in the leaves, so the
    /// only consumer of `scene` at query time is the mesh lookup
    /// (`scene.mesh(instance.mesh)`). A different scene with the same
    /// mesh arena layout would still produce a valid result.
    pub fn intersect_ray(&self, scene: &Scene3D, ray: Ray, t_max: f32) -> Option<SceneRayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        // Root slab miss short-circuits the whole walk.
        intersect_aabb(
            ray,
            self.nodes[0].bounds.min,
            self.nodes[0].bounds.max,
            t_max,
        )?;

        let mut best: Option<SceneRayHit> = None;
        let mut best_t = t_max;
        // Explicit stack of (node_index, t_enter). Same shape as
        // `crate::Bvh::intersect_ray`'s traversal: a far subtree
        // whose entry parameter has slipped past the current best
        // hit is dropped without inspecting it.
        let mut stack: Vec<(u32, f32)> = Vec::with_capacity(64);
        stack.push((0, 0.0));

        while let Some((node_idx, t_enter)) = stack.pop() {
            if t_enter > best_t {
                continue;
            }
            let node = &self.nodes[node_idx as usize];

            if node.is_leaf() {
                let start = node.left_or_first as usize;
                let end = start + node.instance_count as usize;
                for inst in &self.instances[start..end] {
                    // Per-instance AABB cull before the (expensive)
                    // local-ray transform + mesh walk.
                    let aabb_hit = intersect_aabb(ray, inst.bounds.min, inst.bounds.max, best_t);
                    if aabb_hit.is_none() {
                        continue;
                    }
                    let Some(mesh) = scene.mesh(inst.mesh) else {
                        continue;
                    };
                    let local_ray = ray_into_local(inst.world_inv, ray);
                    if let Some((prim_idx, hit)) = mesh.intersect_ray(local_ray, best_t) {
                        // hit.t equals the world-frame ray parameter
                        // (affine change of frame is
                        // parameter-preserving). Strict `<` so an
                        // earlier-visited instance wins exact ties —
                        // matches `Scene3D::intersect_ray`.
                        if best.is_none() || hit.t < best_t {
                            best_t = hit.t;
                            best = Some(SceneRayHit {
                                node: inst.node,
                                primitive_index: prim_idx,
                                hit,
                            });
                        }
                    }
                }
                continue;
            }

            // Interior node — slab-test both children, push the
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

    /// Shadow-ray early-exit query over the same reachable
    /// node-mesh instances as [`InstanceBvh::intersect_ray`].
    ///
    /// Returns `true` as soon as any reachable instance reports a hit
    /// within `t_max` for `ray`. The boolean answer matches
    /// [`Scene3D::any_ray_intersection`] on the same scene + ray;
    /// only the order in which instances are visited differs.
    pub fn any_ray_intersection(&self, scene: &Scene3D, ray: Ray, t_max: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
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
                let end = start + node.instance_count as usize;
                for inst in &self.instances[start..end] {
                    if intersect_aabb(ray, inst.bounds.min, inst.bounds.max, t_max).is_none() {
                        continue;
                    }
                    let Some(mesh) = scene.mesh(inst.mesh) else {
                        continue;
                    };
                    let local_ray = ray_into_local(inst.world_inv, ray);
                    if mesh.intersect_ray(local_ray, t_max).is_some() {
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

    /// Number of leaf nodes in the tree.
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Number of instances stored in the tree's leaves (equal to
    /// `instances.len()`).
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Root AABB over every instance in the tree, or `None` if the
    /// tree is empty.
    pub fn bounds(&self) -> Option<BoundingBox> {
        self.nodes.first().map(|n| n.bounds)
    }
}

/// Gather every reachable node-mesh instance from `scene` with a
/// finite local AABB and a non-singular world matrix.
///
/// Walks the [`Scene3D::roots`] forest depth-first in the same
/// leftmost-first LIFO order as
/// [`Scene3D::world_node_bounds`] / [`Scene3D::intersect_ray`]: roots
/// in `roots`-order, children in source order, cycles guarded once
/// at first arrival, shared children resolved via the first parent's
/// chain. The output order is the deterministic build-time gather
/// order; the BVH's permuted leaf layout is a separate permutation
/// applied during `build`.
fn gather_instances(scene: &Scene3D) -> Vec<Instance> {
    let n_nodes = scene.nodes.len();
    if n_nodes == 0 || scene.meshes.is_empty() {
        return Vec::new();
    }
    let identity: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut visited = vec![false; n_nodes];
    let mut out: Vec<Instance> = Vec::new();
    // Roots pushed in reverse so the LIFO pop visits the leftmost
    // root first — matching the rest of the crate's DFS ordering.
    let mut stack: Vec<(NodeId, [[f32; 4]; 4])> =
        scene.roots.iter().rev().map(|r| (*r, identity)).collect();
    while let Some((nid, parent)) = stack.pop() {
        let idx = nid.0 as usize;
        if idx >= n_nodes || visited[idx] {
            continue;
        }
        visited[idx] = true;
        let Some(node) = scene.node(nid) else {
            continue;
        };
        let world = mat4_mul(parent, node.transform.to_matrix());
        if let Some(m_id) = node.mesh {
            if let Some(mesh) = scene.mesh(m_id) {
                if let Some(local) = mesh.bounding_box() {
                    if let Some(world_inv) = mat4_affine_inverse(world) {
                        out.push(Instance {
                            node: nid,
                            mesh: m_id,
                            bounds: local.transform(world),
                            world,
                            world_inv,
                        });
                    }
                }
            }
        }
        // Reverse so leftmost child pops first.
        for child in node.children.iter().rev() {
            stack.push((*child, world));
        }
    }
    out
}

/// Median-split recursive builder mirroring
/// [`crate::bvh`]'s per-primitive construction — the only differences
/// are the leaf payload (one [`Instance`] per slot vs one triangle
/// index) and the node arena type.
fn build_recursive(
    nodes: &mut Vec<InstanceBvhNode>,
    indices: &mut [u32],
    bounds: &mut [BoundingBox],
    centroids: &mut [[f32; 3]],
    start: usize,
    end: usize,
) -> u32 {
    let count = end - start;
    debug_assert!(count > 0);

    // Tight AABB of the candidate range — union of every instance's
    // world AABB.
    let mut node_bounds = bounds[start];
    for b in &bounds[start + 1..end] {
        node_bounds = node_bounds.union(*b);
    }

    // Leaf cut-off.
    if count <= InstanceBvh::LEAF_THRESHOLD {
        let node_idx = nodes.len() as u32;
        nodes.push(InstanceBvhNode {
            bounds: node_bounds,
            left_or_first: start as u32,
            right_child: 0,
            instance_count: count as u32,
        });
        return node_idx;
    }

    // Centroid bound on the candidate range — drives axis + split
    // coordinate (the longest axis of the centroid spread, not the
    // longest axis of the box union, because the union can be
    // dominated by a single large instance with a centroid in the
    // middle of every other instance's centroid).
    let mut cmin = centroids[start];
    let mut cmax = cmin;
    for c in &centroids[start + 1..end] {
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

    // Degenerate split — every centroid coincides on the chosen
    // axis. Equal-count fallback keeps the recursion terminating at
    // the leaf threshold.
    let mid = if extent[axis] <= 0.0 {
        start + count / 2
    } else {
        let mid_coord = 0.5 * (cmin[axis] + cmax[axis]);
        let mut left = start;
        let mut right = end;
        while left < right {
            if centroids[left][axis] < mid_coord {
                left += 1;
            } else {
                right -= 1;
                indices.swap(left, right);
                bounds.swap(left, right);
                centroids.swap(left, right);
            }
        }
        // Empty-side fallback — pathological centroid distributions
        // can leave one side empty even with non-zero extent (e.g.
        // every centroid sits exactly on `mid_coord`).
        if left == start || left == end {
            start + count / 2
        } else {
            left
        }
    };

    // Reserve the parent slot before recursion so its children know
    // its address; back-patched once both subtrees are built.
    let parent_idx = nodes.len() as u32;
    nodes.push(InstanceBvhNode {
        bounds: node_bounds,
        left_or_first: 0,
        right_child: 0,
        instance_count: 0,
    });
    let left_child = build_recursive(nodes, indices, bounds, centroids, start, mid);
    let right_child = build_recursive(nodes, indices, bounds, centroids, mid, end);
    nodes[parent_idx as usize].left_or_first = left_child;
    nodes[parent_idx as usize].right_child = right_child;
    parent_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Topology;
    use crate::scene::Transform;
    use crate::{Mesh, Node, Primitive};

    fn unit_cube_mesh() -> Mesh {
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
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
        ]));
        Mesh::new(Some("cube".to_owned())).with_primitive(p)
    }

    fn one_cube_scene() -> Scene3D {
        let mut s = Scene3D::new();
        let mid = s.add_mesh(unit_cube_mesh());
        let nid = s.add_node(Node::new().with_mesh(mid));
        s.add_root(nid);
        s
    }

    fn grid_scene(n: usize, spacing: f32) -> Scene3D {
        let mut s = Scene3D::new();
        let mid = s.add_mesh(unit_cube_mesh());
        for ix in 0..n {
            let t = Transform::Trs {
                translation: [ix as f32 * spacing, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            };
            let nid = s.add_node(Node::new().with_transform(t).with_mesh(mid));
            s.add_root(nid);
        }
        s
    }

    #[test]
    fn build_empty_scene_returns_none() {
        let s = Scene3D::new();
        assert!(InstanceBvh::build(&s).is_none());
    }

    #[test]
    fn build_scene_with_node_but_no_mesh_returns_none() {
        let mut s = Scene3D::new();
        let nid = s.add_node(Node::new());
        s.add_root(nid);
        assert!(InstanceBvh::build(&s).is_none());
    }

    #[test]
    fn build_one_cube_yields_one_leaf() {
        let s = one_cube_scene();
        let b = InstanceBvh::build(&s).expect("single instance builds");
        assert_eq!(b.instance_count(), 1);
        assert_eq!(b.leaf_count(), 1);
        assert_eq!(b.node_count(), 1);
        assert!(b.nodes[0].is_leaf());
        // Single instance is the only entry.
        assert_eq!(b.instances[0].bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(b.instances[0].bounds.max, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn build_grid_yields_interior_nodes() {
        // Grid of 16 cubes well past the leaf threshold; tree must
        // have interior nodes.
        let s = grid_scene(16, 3.0);
        let b = InstanceBvh::build(&s).expect("16 instances builds");
        assert_eq!(b.instance_count(), 16);
        assert!(b.leaf_count() > 1);
        assert!(b.node_count() > b.leaf_count(), "interior nodes present");
        // Root bounds span the full grid extent (cubes at x=0..1,
        // 3..4, 6..7, ..., 45..46).
        let root = b.bounds().unwrap();
        assert!((root.min[0] - 0.0).abs() < 1e-5);
        assert!((root.max[0] - 46.0).abs() < 1e-5);
    }

    #[test]
    fn detached_node_does_not_appear() {
        let mut s = Scene3D::new();
        let mid = s.add_mesh(unit_cube_mesh());
        let _detached = s.add_node(Node::new().with_mesh(mid));
        // Note: not added as a root, so it's unreachable.
        let attached = s.add_node(Node::new().with_mesh(mid));
        s.add_root(attached);
        let b = InstanceBvh::build(&s).expect("one reachable instance");
        assert_eq!(b.instance_count(), 1);
        assert_eq!(b.instances[0].node, attached);
    }

    #[test]
    fn singular_transform_is_skipped() {
        // Zero-scale collapses the upper-left 3x3 to rank < 3; the
        // affine-inverse guard rejects it, so the instance is
        // skipped at gather time.
        let mut s = Scene3D::new();
        let mid = s.add_mesh(unit_cube_mesh());
        let bad = s.add_node(
            Node::new()
                .with_transform(Transform::Trs {
                    translation: [0.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.0, 1.0, 1.0],
                })
                .with_mesh(mid),
        );
        s.add_root(bad);
        // Adding a good instance so the scene isn't entirely empty.
        let good = s.add_node(
            Node::new()
                .with_transform(Transform::Trs {
                    translation: [5.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                })
                .with_mesh(mid),
        );
        s.add_root(good);
        let b = InstanceBvh::build(&s).expect("one good instance");
        assert_eq!(b.instance_count(), 1);
        assert_eq!(b.instances[0].node, good);
    }

    #[test]
    fn intersect_ray_matches_scene_walk_on_one_cube() {
        let s = one_cube_scene();
        let b = InstanceBvh::build(&s).unwrap();
        let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let bvh_hit = b.intersect_ray(&s, r, f32::INFINITY).unwrap();
        let scene_hit = s.intersect_ray(r, f32::INFINITY).unwrap();
        assert!((bvh_hit.hit.t - scene_hit.hit.t).abs() < 1e-5);
        assert_eq!(bvh_hit.node, scene_hit.node);
    }

    #[test]
    fn intersect_ray_miss_returns_none() {
        let s = one_cube_scene();
        let b = InstanceBvh::build(&s).unwrap();
        let r = Ray::new([-1.0, 5.0, 5.0], [1.0, 0.0, 0.0]);
        assert!(b.intersect_ray(&s, r, f32::INFINITY).is_none());
    }

    #[test]
    fn intersect_ray_t_max_culls_hits() {
        let s = one_cube_scene();
        let b = InstanceBvh::build(&s).unwrap();
        let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        // -X face is at t=1.0; t_max=0.5 culls it.
        assert!(b.intersect_ray(&s, r, 0.5).is_none());
    }

    #[test]
    fn intersect_ray_picks_nearest_instance_in_grid() {
        // Grid along +X; shoot a +X ray from origin. The nearest
        // instance (ix=0) must win regardless of build-time
        // permutation order.
        let s = grid_scene(8, 3.0);
        let b = InstanceBvh::build(&s).unwrap();
        let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let hit = b.intersect_ray(&s, r, f32::INFINITY).unwrap();
        // Cube 0 is at x ∈ [0,1]; -X face is at t=1.0 from origin -1.
        assert!((hit.hit.t - 1.0).abs() < 1e-4);
    }

    #[test]
    fn intersect_ray_matches_scene_walk_on_grid() {
        // Cross-validate every cube hit against the brute-force
        // scene walk. Tie-breaking can land on different instance
        // ids when two cubes tie exactly on `t` (the BVH visits by
        // AABB distance, the scene walk visits by DFS order), but
        // the hit `t` and front-face must agree exactly.
        let s = grid_scene(8, 3.0);
        let b = InstanceBvh::build(&s).unwrap();
        for iy in 0..5 {
            let y = -1.0 + 0.5 * iy as f32;
            let r = Ray::new([-1.0, y, 0.5], [1.0, 0.0, 0.0]);
            let bf = s.intersect_ray(r, f32::INFINITY);
            let bv = b.intersect_ray(&s, r, f32::INFINITY);
            match (bf, bv) {
                (None, None) => {}
                (Some(a), Some(c)) => {
                    assert!((a.hit.t - c.hit.t).abs() < 1e-4);
                    assert_eq!(a.hit.front_face, c.hit.front_face);
                }
                (a, b) => panic!("mismatch: scene={:?} bvh={:?}", a, b),
            }
        }
    }

    #[test]
    fn any_ray_intersection_agrees_with_scene_walk() {
        let s = grid_scene(8, 3.0);
        let b = InstanceBvh::build(&s).unwrap();
        for iy in 0..5 {
            let y = -1.0 + 0.5 * iy as f32;
            let r = Ray::new([-1.0, y, 0.5], [1.0, 0.0, 0.0]);
            assert_eq!(
                b.any_ray_intersection(&s, r, f32::INFINITY),
                s.any_ray_intersection(r, f32::INFINITY)
            );
        }
    }

    #[test]
    fn any_ray_intersection_short_circuits_on_hit() {
        let s = grid_scene(4, 3.0);
        let b = InstanceBvh::build(&s).unwrap();
        let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        assert!(b.any_ray_intersection(&s, r, f32::INFINITY));
    }

    #[test]
    fn any_ray_intersection_miss_returns_false() {
        let s = grid_scene(4, 3.0);
        let b = InstanceBvh::build(&s).unwrap();
        let r = Ray::new([-1.0, 5.0, 5.0], [1.0, 0.0, 0.0]);
        assert!(!b.any_ray_intersection(&s, r, f32::INFINITY));
    }

    #[test]
    fn instance_count_equals_reachable_meshed_nodes() {
        let s = grid_scene(7, 3.0);
        let b = InstanceBvh::build(&s).unwrap();
        assert_eq!(b.instance_count(), 7);
        // Root bounds span the cubes [0,1] .. [18,19].
        let root = b.bounds().unwrap();
        assert!((root.min[0] - 0.0).abs() < 1e-5);
        assert!((root.max[0] - 19.0).abs() < 1e-5);
    }

    #[test]
    fn leaf_threshold_constant_is_four() {
        assert_eq!(InstanceBvh::LEAF_THRESHOLD, 4);
    }

    #[test]
    fn small_scene_at_or_below_threshold_is_a_single_leaf() {
        // 4 instances == LEAF_THRESHOLD, so the build keeps them in
        // one leaf.
        let s = grid_scene(4, 3.0);
        let b = InstanceBvh::build(&s).unwrap();
        assert_eq!(b.node_count(), 1);
        assert_eq!(b.leaf_count(), 1);
        assert!(b.nodes[0].is_leaf());
    }

    #[test]
    fn shared_child_resolved_via_first_parent() {
        // A node listed under two parents resolves via the first
        // parent's chain — matches `world_node_transforms`'s
        // first-parent rule. We just need to confirm the build
        // doesn't double-count.
        let mut s = Scene3D::new();
        let mid = s.add_mesh(unit_cube_mesh());
        let shared = s.add_node(Node::new().with_mesh(mid));
        let p1 = s.add_node(Node::new().with_transform(Transform::Trs {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }));
        let p2 = s.add_node(Node::new().with_transform(Transform::Trs {
            translation: [10.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }));
        // shared is a child of both p1 and p2
        if let Some(p1n) = s.node_mut(p1) {
            p1n.children.push(shared);
        }
        if let Some(p2n) = s.node_mut(p2) {
            p2n.children.push(shared);
        }
        s.add_root(p1);
        s.add_root(p2);
        let b = InstanceBvh::build(&s).unwrap();
        // shared appears once (first-parent rule).
        assert_eq!(b.instance_count(), 1);
    }
}
