//! Two-level scene acceleration structure: a top-level BVH over the
//! scene's reachable node-mesh **instances**, each pointing at shared
//! bottom-level [`crate::Bvh`]es built once per mesh primitive.
//!
//! # Structure
//!
//! * **Instances.** Every reachable node carrying a mesh contributes
//!   one [`Instance`]: the node's world matrix, its affine inverse,
//!   the world-space AABB of the mesh, a mirroring flag and the
//!   `(NodeId, MeshId)` keys. The gather walk is the same
//!   deterministic depth-first walk as
//!   [`crate::Scene3D::world_node_bounds`].
//! * **Bottom level (BLAS).** One [`crate::Bvh`] per primitive of
//!   every instanced mesh, in [`InstanceBvh::mesh_bvhs`]
//!   (`mesh_bvhs[mesh][primitive]`). A mesh instanced by many nodes is
//!   built once — the classic instancing split of a two-level
//!   hierarchy (I. Wald, C. Benthin & P. Slusallek, "A Simple and
//!   Practical Method for Interactive Ray Tracing of Dynamic Scenes",
//!   Saarland University tech report, 2002).
//! * **Top level (TLAS).** A BVH over the instance world AABBs, built
//!   by the same binned-SAH / object-median driver as the per-primitive
//!   tree and stored in the same 32-byte pair layout (see the
//!   [`crate::bvh`] module docs; [`InstanceBvhNode`] is
//!   layout-identical to [`crate::BvhNode`]).
//!
//! # Traversal
//!
//! The world ray is prepared once ([`PreparedRay`]); the TLAS is
//! walked front-to-back with the robust slab test and a fixed stack.
//! At a leaf, each instance whose world AABB the ray enters gets the
//! ray transformed into mesh-local space through the cached inverse
//! and re-prepared, and every primitive BLAS of the mesh is walked
//! with the shared running `t_max`. An affine change of frame leaves
//! the ray parameter invariant
//! (`O_w + t·D_w = M·(O_l + t·D_l)` with an unnormalised local
//! direction), so hits from different instances compare directly.
//!
//! # Hit records
//!
//! [`InstanceBvh::closest_hit`] returns a [`SceneHit`] with
//! everything a shader needs to fetch material + attributes itself:
//! node / mesh / instance / primitive ids, the triangle index in
//! [`crate::Primitive::triangle_indices`] space and its three vertex
//! indices, barycentrics, `t`, the world-space hit point (barycentric
//! reconstruction transformed by the world matrix — more precise than
//! `origin + t·dir` far from the origin), the unit world-space
//! **geometric normal** and the front/back-face flag.
//!
//! The geometric normal follows glTF 2.0 §3.7.4 ("when the determinant
//! of a node's global transform is negative, the winding order is
//! clockwise"): it is `normalize(M⁻ᵀ · (e1 × e2))`, negated for a
//! mirroring (negative-determinant) instance, so it always points to
//! the side the asset author considers *front* after mirroring.
//! `front_face` is `true` when the ray direction opposes it.
//!
//! [`InstanceBvh::shading_normal`], [`InstanceBvh::uv`],
//! [`InstanceBvh::tangent`] and [`InstanceBvh::color`] interpolate
//! vertex attributes at a hit and bring them to world space; the
//! local-space building blocks live on [`crate::Primitive`]
//! (`interpolate_*`).
//!
//! # Robustness contract
//!
//! Nodes producing a non-affine / singular / non-finite world matrix
//! are skipped at gather time, as are meshes with no finite AABB. A
//! scene whose every reachable node-mesh instance falls through one of
//! these guards builds to `None`. Primitives whose triangles are all
//! unusable get no BLAS (`None`) and are never hit.

use crate::bvh::{build_tree, sah_cost, tree_depth, BvhBuildOptions, SlotHit};
use crate::ray::{PreparedRay, Ray, RayHit, RayQuery};
use crate::scene::{mat4_affine_inverse, mat4_mul, ray_into_local, BoundingBox, MeshId, NodeId};
use crate::{Bvh, Primitive, Scene3D, SceneRayHit};

/// One reachable node-mesh instance in a [`Scene3D`].
///
/// Built by [`InstanceBvh::build`]; stored in the TLAS's leaf order
/// (index = [`SceneHit::instance`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// The scene-graph node that produced this instance.
    pub node: NodeId,
    /// The mesh the node carries.
    pub mesh: MeshId,
    /// World-space AABB of the mesh (eight-corner refit of the mesh's
    /// local AABB through `world`) — the same value
    /// [`Scene3D::world_node_bounds`] reports for the node.
    pub bounds: BoundingBox,
    /// World matrix (`parent_chain * node.transform`), row-major,
    /// column-vector convention.
    pub world: [[f32; 4]; 4],
    /// Affine inverse of `world` (bottom row `[0, 0, 0, 1]`).
    pub world_inv: [[f32; 4]; 4],
    /// `true` when `det(world) < 0` (a mirroring transform): the
    /// primitive's winding is flipped in world space (glTF 2.0
    /// §3.7.4), which [`SceneHit::geometric_normal`] /
    /// [`SceneHit::front_face`] and [`InstanceBvh::tangent`] account
    /// for.
    pub mirrored: bool,
}

/// One 32-byte node of an [`InstanceBvh`]'s top level. Layout-identical
/// to [`crate::BvhNode`] (`min`, `left_or_first`, `max`, count); the
/// right child of an interior node is `left_or_first + 1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceBvhNode {
    /// AABB lower corner.
    pub min: [f32; 3],
    /// Interior: left-child index. Leaf: first slot in
    /// [`InstanceBvh::instances`].
    pub left_or_first: u32,
    /// AABB upper corner.
    pub max: [f32; 3],
    /// `0` for an interior node; number of instances in the leaf.
    pub instance_count: u32,
}

impl InstanceBvhNode {
    /// `true` if this node is a leaf.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.instance_count > 0
    }

    /// The node's AABB.
    #[inline]
    pub fn bounds(&self) -> BoundingBox {
        BoundingBox {
            min: self.min,
            max: self.max,
        }
    }

    /// Left child index (interior nodes).
    #[inline]
    pub fn left_child(&self) -> u32 {
        self.left_or_first
    }

    /// Right child index (interior nodes) — `left_or_first + 1`.
    #[inline]
    pub fn right_child(&self) -> u32 {
        self.left_or_first + 1
    }

    /// The node as eight `u32` words in GPU layout order.
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
            self.instance_count,
        ]
    }
}

/// A candidate intersection offered to a filter closure
/// ([`InstanceBvh::closest_hit_filtered`] /
/// [`InstanceBvh::occluded_filtered`]) before it is accepted — enough
/// to look up the material and evaluate an alpha mask at the
/// candidate's texture coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitCandidate {
    /// Index into [`InstanceBvh::instances`].
    pub instance: u32,
    pub node: NodeId,
    pub mesh: MeshId,
    /// Index into the mesh's `primitives`.
    pub primitive_index: usize,
    /// Index into [`crate::Primitive::triangle_indices`].
    pub triangle_index: usize,
    /// The triangle's three vertex indices.
    pub vertex_indices: [u32; 3],
    /// Ray parameter of the candidate.
    pub t: f32,
    /// `[w, u, v]` weights of the three vertices.
    pub barycentric: [f32; 3],
}

/// Closest-hit record of an [`InstanceBvh`] query. See the module docs
/// for the conventions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneHit {
    /// Index into [`InstanceBvh::instances`] (world matrix lookup).
    pub instance: u32,
    /// Scene-graph node of the struck instance.
    pub node: NodeId,
    /// Mesh of the struck instance.
    pub mesh: MeshId,
    /// Index into the mesh's `primitives` (fetch the material there).
    pub primitive_index: usize,
    /// Index into [`crate::Primitive::triangle_indices`] — the
    /// original triangle enumeration order.
    pub triangle_index: usize,
    /// The triangle's three vertex indices into the primitive's
    /// attribute arrays (`positions`, `normals`, `uvs[n]`, …).
    pub vertex_indices: [u32; 3],
    /// Ray parameter; `ray.point_at(t)` ≈ `position`.
    pub t: f32,
    /// `[w, u, v]`: attribute `A` interpolates as
    /// `w·A[v0] + u·A[v1] + v·A[v2]`.
    pub barycentric: [f32; 3],
    /// World-space hit point.
    pub position: [f32; 3],
    /// Unit world-space geometric normal, oriented by the triangle's
    /// winding with glTF mirroring applied (never flipped towards the
    /// ray).
    pub geometric_normal: [f32; 3],
    /// `true` when the ray hits the side `geometric_normal` points to
    /// (`dot(ray.direction, geometric_normal) < 0`).
    pub front_face: bool,
}

/// Two-level (TLAS over instances + per-primitive BLAS) scene BVH.
/// See the module docs.
///
/// A value type with no shared state with the source scene. It must
/// be rebuilt when the set of reachable instances or any mesh's
/// connectivity changes; node transforms and vertex positions can be
/// updated in place with [`InstanceBvh::refit`].
#[derive(Clone, Debug)]
pub struct InstanceBvh {
    /// TLAS nodes; node `0` is the root.
    pub nodes: Vec<InstanceBvhNode>,
    /// Instances in TLAS leaf order.
    pub instances: Vec<Instance>,
    /// Bottom-level BVHs: `mesh_bvhs[mesh][primitive]`. Meshes not
    /// referenced by any instance have an empty list; primitives with
    /// no usable triangles have `None`.
    pub mesh_bvhs: Vec<Vec<Option<Bvh>>>,
}

/// Best-so-far record during a scene traversal.
#[derive(Clone, Copy)]
struct Best {
    instance: u32,
    prim: u32,
    hit: SlotHit,
}

impl InstanceBvh {
    /// Default maximum number of instances per TLAS leaf (same as
    /// [`crate::Bvh::LEAF_THRESHOLD`]).
    pub const LEAF_THRESHOLD: usize = 4;

    /// Build the two-level BVH (binned SAH at both levels) over every
    /// reachable node-mesh instance of `scene`.
    ///
    /// Returns `None` when the scene contributes zero usable instances.
    /// Pure: the source scene is not mutated.
    pub fn build(scene: &Scene3D) -> Option<Self> {
        Self::build_with(scene, &BvhBuildOptions::default())
    }

    /// [`InstanceBvh::build`] with explicit options (applied to both
    /// the TLAS and every BLAS).
    pub fn build_with(scene: &Scene3D, options: &BvhBuildOptions) -> Option<Self> {
        let instances = gather_instances(scene);
        if instances.is_empty() {
            return None;
        }
        let bounds: Vec<BoundingBox> = instances.iter().map(|i| i.bounds).collect();
        let centroids: Vec<[f32; 3]> = bounds.iter().map(|b| b.center()).collect();
        let (nodes, order) = build_tree(&bounds, &centroids, options);
        let nodes = nodes
            .into_iter()
            .map(|n| InstanceBvhNode {
                min: n.min,
                left_or_first: n.left_or_first,
                max: n.max,
                instance_count: n.tri_count,
            })
            .collect();
        let instances: Vec<Instance> = order.iter().map(|&i| instances[i as usize]).collect();

        let mut mesh_bvhs: Vec<Vec<Option<Bvh>>> = vec![Vec::new(); scene.meshes.len()];
        for inst in &instances {
            let m = inst.mesh.0 as usize;
            if mesh_bvhs[m].is_empty() {
                if let Some(mesh) = scene.meshes.get(m) {
                    mesh_bvhs[m] = mesh
                        .primitives
                        .iter()
                        .map(|p| Bvh::build_with(p, options))
                        .collect();
                }
            }
        }
        Some(InstanceBvh {
            nodes,
            instances,
            mesh_bvhs,
        })
    }

    /// Closest-hit query in `[0, t_max]` (Möller-Trumbore), reported in
    /// the legacy [`SceneRayHit`] shape (mesh-local `RayHit` whose
    /// `front_face` is relative to the untransformed local winding).
    /// Same `t` / hit point as [`Scene3D::intersect_ray`]; on an exact
    /// tie between instances the winner may differ. Prefer
    /// [`InstanceBvh::closest_hit`] for rendering.
    pub fn intersect_ray(&self, scene: &Scene3D, ray: Ray, t_max: f32) -> Option<SceneRayHit> {
        let h = self.closest_hit(scene, ray, &RayQuery::new(t_max))?;
        let mirrored = self.instances[h.instance as usize].mirrored;
        Some(SceneRayHit {
            node: h.node,
            primitive_index: h.primitive_index,
            hit: RayHit {
                t: h.t,
                triangle_index: h.triangle_index,
                barycentric: h.barycentric,
                front_face: h.front_face != mirrored,
            },
        })
    }

    /// Shadow-ray query in `[0, t_max]`; same boolean answer as
    /// [`Scene3D::any_ray_intersection`].
    pub fn any_ray_intersection(&self, scene: &Scene3D, ray: Ray, t_max: f32) -> bool {
        self.occluded(scene, ray, &RayQuery::new(t_max))
    }

    /// Closest hit within `query`'s interval, with a full
    /// [`SceneHit`] record.
    pub fn closest_hit(&self, scene: &Scene3D, ray: Ray, query: &RayQuery) -> Option<SceneHit> {
        self.closest_hit_filtered(scene, ray, query, |_| true)
    }

    /// Closest hit where each candidate must also pass `filter`
    /// (return `false` to ignore it, e.g. an alpha-masked texel). The
    /// filter may see candidates that are later superseded, in no
    /// particular order.
    pub fn closest_hit_filtered<F: FnMut(&HitCandidate) -> bool>(
        &self,
        scene: &Scene3D,
        ray: Ray,
        query: &RayQuery,
        mut filter: F,
    ) -> Option<SceneHit> {
        let pr = PreparedRay::new(ray);
        let mut t_max = query.t_max;
        let mut best: Option<Best> = None;
        self.walk(&pr, query.t_min, &mut t_max, |inst_idx, inst, t_max| {
            let Some(mesh) = scene.meshes.get(inst.mesh.0 as usize) else {
                return false;
            };
            let Some(blas) = self.mesh_bvhs.get(inst.mesh.0 as usize) else {
                return false;
            };
            let local = PreparedRay::new(ray_into_local(inst.world_inv, ray));
            for (p, (prim, bvh)) in mesh.primitives.iter().zip(blas).enumerate() {
                let Some(bvh) = bvh else { continue };
                let mut f = |h: &SlotHit| filter(&candidate(inst_idx, inst, p, bvh, h));
                if let Some(h) = bvh.closest_slot(
                    &prim.positions,
                    &local,
                    query.t_min,
                    t_max,
                    query.triangle_test,
                    &mut f,
                ) {
                    best = Some(Best {
                        instance: inst_idx,
                        prim: p as u32,
                        hit: h,
                    });
                }
            }
            false
        });
        let b = best?;
        self.finish_hit(scene, &pr, b)
    }

    /// Any-hit (occlusion / shadow-ray) query inside `query`'s
    /// interval.
    pub fn occluded(&self, scene: &Scene3D, ray: Ray, query: &RayQuery) -> bool {
        self.occluded_filtered(scene, ray, query, |_| true)
    }

    /// Any-hit query where a candidate occludes only if `filter`
    /// accepts it — the hook for alpha-`MASK` materials in shadow
    /// rays.
    pub fn occluded_filtered<F: FnMut(&HitCandidate) -> bool>(
        &self,
        scene: &Scene3D,
        ray: Ray,
        query: &RayQuery,
        mut filter: F,
    ) -> bool {
        let pr = PreparedRay::new(ray);
        let mut t_max = query.t_max;
        self.walk(&pr, query.t_min, &mut t_max, |inst_idx, inst, t_max| {
            let Some(mesh) = scene.meshes.get(inst.mesh.0 as usize) else {
                return false;
            };
            let Some(blas) = self.mesh_bvhs.get(inst.mesh.0 as usize) else {
                return false;
            };
            let local = PreparedRay::new(ray_into_local(inst.world_inv, ray));
            for (p, (prim, bvh)) in mesh.primitives.iter().zip(blas).enumerate() {
                let Some(bvh) = bvh else { continue };
                let mut f = |h: &SlotHit| filter(&candidate(inst_idx, inst, p, bvh, h));
                if bvh.any_slot(
                    &prim.positions,
                    &local,
                    query.t_min,
                    *t_max,
                    query.triangle_test,
                    &mut f,
                ) {
                    return true;
                }
            }
            false
        })
    }

    /// Ordered TLAS walk. `visit(instance_index, instance, t_max)` is
    /// called for every instance whose world AABB the ray enters
    /// within the current `[t_min, t_max]`; it may shrink `t_max`, and
    /// returning `true` stops the walk (any-hit). Returns whether the
    /// walk was stopped.
    #[inline]
    fn walk<V: FnMut(u32, &Instance, &mut f32) -> bool>(
        &self,
        ray: &PreparedRay,
        t_min: f32,
        t_max: &mut f32,
        mut visit: V,
    ) -> bool {
        let nodes = &self.nodes[..];
        let Some(root) = nodes.first() else {
            return false;
        };
        if !ray.is_valid() || ray.slab(&root.min, &root.max, t_min, *t_max).is_none() {
            return false;
        }
        let mut stack: [(u32, f32); 64] = [(0, 0.0); 64];
        let mut spill: Vec<(u32, f32)> = Vec::new();
        let mut sp = 0usize;
        let mut idx = 0usize;
        loop {
            let node = &nodes[idx];
            let mut next: Option<usize> = None;
            if node.instance_count > 0 {
                let first = node.left_or_first as usize;
                let end = (first + node.instance_count as usize).min(self.instances.len());
                for i in first..end {
                    let inst = &self.instances[i];
                    if ray
                        .slab(&inst.bounds.min, &inst.bounds.max, t_min, *t_max)
                        .is_some()
                        && visit(i as u32, inst, t_max)
                    {
                        return true;
                    }
                }
            } else {
                let l = node.left_or_first as usize;
                if let (Some(ln), Some(rn)) = (nodes.get(l), nodes.get(l + 1)) {
                    let tl = ray.slab(&ln.min, &ln.max, t_min, *t_max);
                    let tr = ray.slab(&rn.min, &rn.max, t_min, *t_max);
                    let far = match (tl, tr) {
                        (Some(a), Some(b)) => {
                            if a <= b {
                                next = Some(l);
                                Some(((l + 1) as u32, b))
                            } else {
                                next = Some(l + 1);
                                Some((l as u32, a))
                            }
                        }
                        (Some(_), None) => {
                            next = Some(l);
                            None
                        }
                        (None, Some(_)) => {
                            next = Some(l + 1);
                            None
                        }
                        (None, None) => None,
                    };
                    if let Some(f) = far {
                        if sp < stack.len() {
                            stack[sp] = f;
                            sp += 1;
                        } else {
                            spill.push(f);
                        }
                    }
                }
            }
            if let Some(n) = next {
                idx = n;
                continue;
            }
            loop {
                let popped = if let Some(x) = spill.pop() {
                    Some(x)
                } else if sp > 0 {
                    sp -= 1;
                    Some(stack[sp])
                } else {
                    None
                };
                match popped {
                    None => return false,
                    Some((n, t)) if t <= *t_max => {
                        idx = n as usize;
                        break;
                    }
                    Some(_) => {}
                }
            }
        }
    }

    /// Expand the winning slot hit into a full [`SceneHit`].
    fn finish_hit(&self, scene: &Scene3D, ray: &PreparedRay, b: Best) -> Option<SceneHit> {
        let inst = &self.instances[b.instance as usize];
        let prim = scene
            .meshes
            .get(inst.mesh.0 as usize)?
            .primitives
            .get(b.prim as usize)?;
        let bvh = self.mesh_bvhs[inst.mesh.0 as usize][b.prim as usize].as_ref()?;
        let slot = b.hit.slot as usize;
        let tri = bvh.triangle_vertices[slot];
        let [p0, p1, p2] = crate::bvh::fetch(&prim.positions, tri, prim.positions.len())?;
        let (u, v) = (b.hit.u, b.hit.v);
        let w = 1.0 - u - v;
        let local = [
            w * p0[0] + u * p1[0] + v * p2[0],
            w * p0[1] + u * p1[1] + v * p2[1],
            w * p0[2] + u * p1[2] + v * p2[2],
        ];
        let position = xform_point(&inst.world, local);
        let n_local = cross(sub(p1, p0), sub(p2, p0));
        let mut n = normal_to_world(&inst.world_inv, n_local);
        if inst.mirrored {
            n = [-n[0], -n[1], -n[2]];
        }
        let geometric_normal = normalize_or_zero(n);
        let front_face = dot(ray.ray.direction, geometric_normal) < 0.0;
        Some(SceneHit {
            instance: b.instance,
            node: inst.node,
            mesh: inst.mesh,
            primitive_index: b.prim as usize,
            triangle_index: bvh.triangles[slot] as usize,
            vertex_indices: tri,
            t: b.hit.t,
            barycentric: [w, u, v],
            position,
            geometric_normal,
            front_face,
        })
    }

    /// The primitive a hit landed on.
    pub fn hit_primitive<'s>(&self, scene: &'s Scene3D, hit: &SceneHit) -> Option<&'s Primitive> {
        scene
            .meshes
            .get(hit.mesh.0 as usize)?
            .primitives
            .get(hit.primitive_index)
    }

    /// Unit world-space **shading** normal at `hit`: the barycentric
    /// blend of the primitive's vertex normals, transformed by the
    /// instance's normal matrix (inverse transpose — correct under
    /// non-uniform scale; vertex normals are true normals, so no
    /// mirroring flip is applied). Falls back to
    /// [`SceneHit::geometric_normal`] when the primitive has no
    /// normals or the blend degenerates.
    pub fn shading_normal(&self, scene: &Scene3D, hit: &SceneHit) -> [f32; 3] {
        let Some(inst) = self.instances.get(hit.instance as usize) else {
            return hit.geometric_normal;
        };
        self.hit_primitive(scene, hit)
            .and_then(|p| p.interpolate_normal(hit.vertex_indices, hit.barycentric))
            .map(|n| normalize_or_zero(normal_to_world(&inst.world_inv, n)))
            .filter(|n| n.iter().any(|c| *c != 0.0))
            .unwrap_or(hit.geometric_normal)
    }

    /// Interpolated texture coordinate of UV set `set` at `hit`
    /// (`TEXCOORD_<set>`); `None` when the primitive has no such set.
    pub fn uv(&self, scene: &Scene3D, hit: &SceneHit, set: usize) -> Option<[f32; 2]> {
        self.hit_primitive(scene, hit)?
            .interpolate_uv(hit.vertex_indices, hit.barycentric, set)
    }

    /// Interpolated world-space tangent at `hit`: `xyz` is the blended
    /// vertex tangent transformed by the instance's linear part and
    /// normalised; `w` is the glTF bitangent sign, negated for a
    /// mirroring instance so `bitangent = cross(N, T.xyz) · w` stays
    /// correct in world space. `None` without tangents.
    pub fn tangent(&self, scene: &Scene3D, hit: &SceneHit) -> Option<[f32; 4]> {
        let inst = self.instances.get(hit.instance as usize)?;
        let t = self
            .hit_primitive(scene, hit)?
            .interpolate_tangent(hit.vertex_indices, hit.barycentric)?;
        let m = &inst.world;
        let x = normalize_or_zero([
            m[0][0] * t[0] + m[0][1] * t[1] + m[0][2] * t[2],
            m[1][0] * t[0] + m[1][1] * t[1] + m[1][2] * t[2],
            m[2][0] * t[0] + m[2][1] * t[1] + m[2][2] * t[2],
        ]);
        let w = if inst.mirrored { -t[3] } else { t[3] };
        Some([x[0], x[1], x[2], w])
    }

    /// Interpolated vertex colour of set `set` (`COLOR_<set>`) at
    /// `hit`; `None` when absent.
    pub fn color(&self, scene: &Scene3D, hit: &SceneHit, set: usize) -> Option<[f32; 4]> {
        self.hit_primitive(scene, hit)?
            .interpolate_color(hit.vertex_indices, hit.barycentric, set)
    }

    /// Update the structure in place after node transforms and/or
    /// vertex positions changed (same reachable instances, same mesh
    /// connectivity): recomputes every instance's world matrix,
    /// inverse, mirroring flag and world AABB, refits every BLAS from
    /// the current primitive positions, then refits the TLAS bottom-up.
    ///
    /// Returns `false` when the scene no longer matches the build
    /// (a node lost its mesh / became unreachable / singular, a mesh's
    /// primitive count changed, or a triangle became invalid); the
    /// structure is then still safe to query but may be stale — rebuild
    /// it.
    pub fn refit(&mut self, scene: &Scene3D) -> bool {
        let mut ok = true;
        // Bottom level.
        for (m, blas) in self.mesh_bvhs.iter_mut().enumerate() {
            if blas.is_empty() {
                continue;
            }
            let Some(mesh) = scene.meshes.get(m) else {
                ok = false;
                continue;
            };
            if mesh.primitives.len() != blas.len() {
                ok = false;
                continue;
            }
            for (prim, bvh) in mesh.primitives.iter().zip(blas.iter_mut()) {
                if let Some(bvh) = bvh {
                    ok &= bvh.refit(prim);
                }
            }
        }
        // Instances.
        let worlds = scene.world_node_transforms();
        for inst in &mut self.instances {
            let node = scene.nodes.get(inst.node.0 as usize);
            let world = worlds.get(inst.node.0 as usize).copied().flatten();
            let mesh = scene.meshes.get(inst.mesh.0 as usize);
            match (node, world, mesh) {
                (Some(n), Some(world), Some(mesh)) if n.mesh == Some(inst.mesh) => {
                    match (mat4_affine_inverse(world), mesh.bounding_box()) {
                        (Some(world_inv), Some(local)) => {
                            inst.world = world;
                            inst.world_inv = world_inv;
                            inst.mirrored = det3(&world) < 0.0;
                            inst.bounds = local.transform(world);
                        }
                        _ => ok = false,
                    }
                }
                _ => ok = false,
            }
        }
        // Top level, children-after-parent order ⇒ reverse sweep.
        for i in (0..self.nodes.len()).rev() {
            let node = self.nodes[i];
            let mut acc: Option<BoundingBox> = None;
            if node.is_leaf() {
                let first = node.left_or_first as usize;
                let end = (first + node.instance_count as usize).min(self.instances.len());
                for inst in &self.instances[first..end] {
                    acc = Some(acc.map_or(inst.bounds, |b| b.union(inst.bounds)));
                }
            } else {
                let l = node.left_or_first as usize;
                for c in [l, l + 1] {
                    if let Some(ch) = self.nodes.get(c) {
                        let cb = ch.bounds();
                        acc = Some(acc.map_or(cb, |b| b.union(cb)));
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

    /// Total number of TLAS nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of TLAS leaves.
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Number of instances (equal to `instances.len()`).
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Root AABB over every instance, `None` if empty.
    pub fn bounds(&self) -> Option<BoundingBox> {
        self.nodes.first().map(|n| n.bounds())
    }

    /// TLAS depth (a single leaf has depth 1).
    pub fn depth(&self) -> usize {
        tree_depth(&self.nodes, |n| n.is_leaf(), |n| n.left_or_first)
    }

    /// Root-normalised SAH cost of the TLAS (unit costs).
    pub fn sah_cost(&self) -> f32 {
        sah_cost(&self.nodes, |n| (n.bounds(), n.instance_count))
    }

    /// TLAS nodes as a `u32` word stream in the GPU layout (8 words
    /// per node).
    pub fn node_words(&self) -> Vec<u32> {
        self.nodes.iter().flat_map(|n| n.to_words()).collect()
    }
}

#[inline]
fn candidate(inst_idx: u32, inst: &Instance, prim: usize, bvh: &Bvh, h: &SlotHit) -> HitCandidate {
    let slot = h.slot as usize;
    HitCandidate {
        instance: inst_idx,
        node: inst.node,
        mesh: inst.mesh,
        primitive_index: prim,
        triangle_index: bvh.triangles[slot] as usize,
        vertex_indices: bvh.triangle_vertices[slot],
        t: h.t,
        barycentric: [1.0 - h.u - h.v, h.u, h.v],
    }
}

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
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let l2 = dot(v, v);
    if l2 > 0.0 && l2.is_finite() {
        let inv = 1.0 / l2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0; 3]
    }
}

#[inline]
fn xform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
    ]
}

/// `M⁻ᵀ · n` given `M⁻¹` (multiply by the transpose of the inverse's
/// linear part).
#[inline]
fn normal_to_world(inv: &[[f32; 4]; 4], n: [f32; 3]) -> [f32; 3] {
    [
        inv[0][0] * n[0] + inv[1][0] * n[1] + inv[2][0] * n[2],
        inv[0][1] * n[0] + inv[1][1] * n[1] + inv[2][1] * n[2],
        inv[0][2] * n[0] + inv[1][2] * n[1] + inv[2][2] * n[2],
    ]
}

/// Determinant of the upper-left 3x3 (linear part).
#[inline]
fn det3(m: &[[f32; 4]; 4]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
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
                            mirrored: det3(&world) < 0.0,
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
        // 4 instances == LEAF_THRESHOLD, so the object-median build
        // keeps them in one leaf (the SAH builder may still split
        // well-separated instances when that is cheaper).
        let s = grid_scene(4, 3.0);
        let b = InstanceBvh::build_with(&s, &BvhBuildOptions::object_median()).unwrap();
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
