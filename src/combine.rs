//! Primitive concatenation and material-grouped mesh consolidation.
//!
//! Decoders and authoring tools routinely produce a mesh split into
//! many small [`Primitive`]s — one per material, one per OBJ `g`
//! group, one per glTF accessor batch. For draw-call batching, stream
//! compression, or whole-mesh geometry queries it is often useful to
//! **fuse** primitives that share a draw state back into a single
//! vertex/index pair. This module provides that fusion:
//!
//! - [`Primitive::merge`] concatenates two primitives into one indexed
//!   `Triangles` primitive, re-basing the second's indices onto the end
//!   of the first's vertex pool.
//! - [`Mesh::merge_primitives_by_material`] groups a mesh's primitives
//!   by their [`material`](Primitive::material) reference (plus their
//!   [`variant_mappings`](Primitive::variant_mappings), which are part
//!   of the draw state) and fuses each group, collapsing an
//!   N-primitive mesh into at most one primitive per distinct draw
//!   state.
//!
//! # Attribute reconciliation
//!
//! Two primitives need not carry the same optional attributes: one may
//! have `NORMAL`s and the other not, or they may differ in UV-set
//! count. The merge rule is **union with spec-neutral fill**: an
//! optional attribute appears in the output iff it was present on
//! *either* input, and the side that lacked it contributes a default
//! row per vertex that a renderer treats as "attribute effectively
//! absent":
//!
//! | attribute  | fill row            | rationale                       |
//! |------------|---------------------|---------------------------------|
//! | normal     | `[0, 0, 1]`         | +Z, the canonical up normal     |
//! | tangent    | `[1, 0, 0, 1]`     | +X, right-handed (`w = +1`)     |
//! | uv set     | `[0, 0]`            | origin of texture space         |
//! | colour set | `[1, 1, 1, 1]`     | opaque white (multiplicative id)|
//! | joints     | `[0, 0, 0, 0]`     | bind to joint 0                 |
//! | weights    | `[0, 0, 0, 0]`     | no influence                    |
//!
//! UV and colour *set counts* take the maximum across the two inputs;
//! missing sets on either side are filled the same way. Morph-target
//! deltas are dropped on merge — a fused primitive has no coherent
//! shared target roster — and the result carries no `targets`.
//!
//! The merged `material` is the first input's when both agree (or one
//! is `None`); a genuine material conflict keeps the *first* input's
//! reference (the grouping helpers never create such a conflict because
//! they pre-partition by material). `extras` and
//! `variant_mappings` are taken from the first input — the grouping
//! helpers also partition on the variant mappings, so a fused group is
//! always mapping-homogeneous. Neither entry point mutates its
//! receiver.

use crate::mesh::{Indices, Mesh, Primitive, Topology};

/// Default fill rows for absent optional attributes (see the module
/// table).
const FILL_NORMAL: [f32; 3] = [0.0, 0.0, 1.0];
const FILL_TANGENT: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const FILL_UV: [f32; 2] = [0.0, 0.0];
const FILL_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const FILL_JOINTS: [u16; 4] = [0, 0, 0, 0];
const FILL_WEIGHTS: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

impl Primitive {
    /// Concatenate `self` and `other` into one indexed `Triangles`
    /// primitive.
    ///
    /// Both inputs are de-stripped to triangle lists, their vertex
    /// pools are concatenated (so `other`'s vertices follow `self`'s),
    /// and `other`'s triangle indices are shifted up by
    /// `self.positions.len()` so they reference the relocated pool.
    /// Optional attributes are reconciled by union-with-fill (see the
    /// [module docs](crate::combine)); morph targets are dropped.
    ///
    /// `material` and `extras` come from `self`. The output is always
    /// `Topology::Triangles` with a `U32` index buffer. Non-triangle
    /// topologies contribute no triangles (their `triangle_indices`
    /// is empty) but their vertices still join the pool — exactly the
    /// behaviour of [`to_triangle_list`](Primitive::to_triangle_list)
    /// on each side. Does not mutate either input.
    pub fn merge(&self, other: &Primitive) -> Primitive {
        merge_all(&[other], self)
    }
}

/// Merge `base` with every primitive in `rest`, in order. Factored out
/// so the per-material grouping can fuse a whole group in one pass
/// without quadratic re-walking.
fn merge_all(rest: &[&Primitive], base: &Primitive) -> Primitive {
    // Determine the union attribute shape across base + rest.
    let all: Vec<&Primitive> = std::iter::once(base).chain(rest.iter().copied()).collect();

    let want_normals = all.iter().any(|p| p.normals.is_some());
    let want_tangents = all.iter().any(|p| p.tangents.is_some());
    let want_joints = all.iter().any(|p| p.joints.is_some());
    let want_weights = all.iter().any(|p| p.weights.is_some());
    let uv_sets = all.iter().map(|p| p.uvs.len()).max().unwrap_or(0);
    let color_sets = all.iter().map(|p| p.colors.len()).max().unwrap_or(0);

    let mut out = Primitive::new(Topology::Triangles);
    out.material = base.material;
    out.variant_mappings = base.variant_mappings.clone();
    out.extras = base.extras.clone();
    if want_normals {
        out.normals = Some(Vec::new());
    }
    if want_tangents {
        out.tangents = Some(Vec::new());
    }
    if want_joints {
        out.joints = Some(Vec::new());
    }
    if want_weights {
        out.weights = Some(Vec::new());
    }
    out.uvs = vec![Vec::new(); uv_sets];
    out.colors = vec![Vec::new(); color_sets];

    let mut flat: Vec<u32> = Vec::new();

    for p in &all {
        let base_index = out.positions.len() as u32;
        let vcount = p.positions.len();

        // Append this primitive's vertex rows, filling absent slots.
        out.positions.extend_from_slice(&p.positions);

        if want_normals {
            let dst = out.normals.as_mut().unwrap();
            match &p.normals {
                Some(src) => append_or_fill(dst, src, vcount, FILL_NORMAL),
                None => dst.extend(std::iter::repeat(FILL_NORMAL).take(vcount)),
            }
        }
        if want_tangents {
            let dst = out.tangents.as_mut().unwrap();
            match &p.tangents {
                Some(src) => append_or_fill(dst, src, vcount, FILL_TANGENT),
                None => dst.extend(std::iter::repeat(FILL_TANGENT).take(vcount)),
            }
        }
        if want_joints {
            let dst = out.joints.as_mut().unwrap();
            match &p.joints {
                Some(src) => append_or_fill(dst, src, vcount, FILL_JOINTS),
                None => dst.extend(std::iter::repeat(FILL_JOINTS).take(vcount)),
            }
        }
        if want_weights {
            let dst = out.weights.as_mut().unwrap();
            match &p.weights {
                Some(src) => append_or_fill(dst, src, vcount, FILL_WEIGHTS),
                None => dst.extend(std::iter::repeat(FILL_WEIGHTS).take(vcount)),
            }
        }
        for s in 0..uv_sets {
            match p.uvs.get(s) {
                Some(src) => append_or_fill(&mut out.uvs[s], src, vcount, FILL_UV),
                None => out.uvs[s].extend(std::iter::repeat(FILL_UV).take(vcount)),
            }
        }
        for s in 0..color_sets {
            match p.colors.get(s) {
                Some(src) => append_or_fill(&mut out.colors[s], src, vcount, FILL_COLOR),
                None => out.colors[s].extend(std::iter::repeat(FILL_COLOR).take(vcount)),
            }
        }

        // Shift this primitive's triangles onto the relocated pool.
        for tri in p.triangle_indices() {
            // Drop any out-of-range corner (malformed input) rather than
            // emitting a dangling index.
            if tri.iter().all(|&c| (c as usize) < vcount) {
                flat.push(base_index + tri[0]);
                flat.push(base_index + tri[1]);
                flat.push(base_index + tri[2]);
            }
        }
    }

    out.indices = Some(Indices::U32(flat));
    out
}

/// Append `src` to `dst`, padding with `fill` if `src` is shorter than
/// `vcount` and truncating if longer — keeping the per-vertex buffers
/// exactly `vcount` rows so every attribute stream stays aligned even
/// when a malformed input had a mismatched attribute length.
fn append_or_fill<T: Copy>(dst: &mut Vec<T>, src: &[T], vcount: usize, fill: T) {
    let take = src.len().min(vcount);
    dst.extend_from_slice(&src[..take]);
    if take < vcount {
        dst.extend(std::iter::repeat(fill).take(vcount - take));
    }
}

impl Mesh {
    /// Fuse this mesh's primitives so each distinct draw state —
    /// material reference plus `KHR_materials_variants` mappings — is
    /// drawn by at most one primitive.
    ///
    /// Primitives are partitioned by their
    /// [`material`](Primitive::material) reference (with `None` — the
    /// unmaterialled group — kept as its own bucket) **and** their
    /// [`variant_mappings`](Primitive::variant_mappings) (primitives
    /// whose materials diverge under an active variant must keep
    /// separate draw calls), and each bucket is concatenated via the
    /// same union-with-fill rule as [`Primitive::merge`]. A material
    /// referenced by a single primitive is still rewritten into an
    /// indexed `Triangles` primitive (its `to_triangle_list` form) so
    /// the output is uniform.
    ///
    /// Group order follows first appearance of each draw state in
    /// [`Mesh::primitives`], so the result is deterministic. The mesh's
    /// `name`, `weights`, and `target_names` are preserved; morph
    /// targets are dropped from the fused primitives (consistent with
    /// `merge` — run [`Mesh::morphed`] first, or clear the mesh-level
    /// morph fields, if the inputs carried targets).
    /// Does not mutate `self`. A mesh with no primitives returns a clone
    /// with an empty primitive list.
    pub fn merge_primitives_by_material(&self) -> Mesh {
        let mut groups: Vec<Vec<&Primitive>> = Vec::new();

        for p in &self.primitives {
            // Draw-state key: the base material AND the variant
            // mappings. Two primitives with the same base material
            // but different `KHR_materials_variants` mappings render
            // differently once a variant is active, so they must not
            // fuse.
            match groups.iter().position(|g| {
                g[0].material == p.material && g[0].variant_mappings == p.variant_mappings
            }) {
                Some(idx) => groups[idx].push(p),
                None => groups.push(vec![p]),
            }
        }

        let mut out = self.clone();
        out.primitives = groups
            .into_iter()
            .map(|g| {
                let (base, rest) = g.split_first().expect("group is non-empty");
                merge_all(rest, base)
            })
            .collect();
        out
    }
}
