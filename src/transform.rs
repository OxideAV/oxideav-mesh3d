//! Baking an affine transform into a primitive's geometry, plus
//! orientation flips.
//!
//! [`Scene3D::bake_transforms`](crate::Scene3D::bake_transforms) folds
//! the node hierarchy into per-node world *matrices*; the complementary
//! operation is to fold a matrix into the *vertex data* itself, so the
//! geometry lands in a target space with no transform attached. An
//! exporter writing a hierarchy-free, transform-free format (binary STL
//! stores raw world-space triangles) needs exactly this: take each
//! instance's world matrix and bake it into a copy of the mesh.
//!
//! # Transforming each attribute correctly
//!
//! Positions are **points**: transform with the full affine,
//! `p' = M · (p, 1)`. Normals and tangent directions are **covectors /
//! directions** and do *not* transform with `M` — under a non-uniform
//! scale that would shear them off the surface. The correct transform
//! for a normal is the **inverse-transpose** of the linear part
//! `L = M₃ₓ₃`:
//!
//! ```text
//! n' = normalize( (L⁻¹)ᵀ · n )
//! ```
//!
//! which keeps `n'` perpendicular to every transformed surface tangent.
//! Tangent *directions* are true surface vectors, so they transform
//! with `L` directly (then renormalise); the handedness `w` is a sign
//! and is preserved — except that a transform with **negative
//! determinant** (a mirror / odd number of axis flips) reverses
//! handedness, so `w` is negated in that case to keep the bitangent
//! `cross(n, t.xyz) * t.w` pointing the right way.
//!
//! When `L` is singular (zero or non-finite determinant) the
//! inverse-transpose is undefined; normals are then left unchanged
//! (the safe, finite fallback) while positions still transform.
//!
//! All entry points are pure (no mutation of `self`) and preserve
//! topology, indices, and every non-geometric attribute.

use crate::mesh::{Mesh, Primitive};
use crate::scene::mat4_affine_inverse;

/// Apply the affine `m` to a point `(p, 1)`, returning the transformed
/// `xyz` (perspective divide by the resulting `w` for safety, though an
/// affine row keeps `w == 1`).
fn transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let x = m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3];
    let y = m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3];
    let z = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3];
    let w = m[3][0] * p[0] + m[3][1] * p[1] + m[3][2] * p[2] + m[3][3];
    if w != 0.0 && w != 1.0 && w.is_finite() {
        [x / w, y / w, z / w]
    } else {
        [x, y, z]
    }
}

/// Apply a 3x3 linear part `l` (extracted from the upper-left of a 4x4)
/// to a direction vector.
fn transform_dir(l: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        l[0][0] * v[0] + l[0][1] * v[1] + l[0][2] * v[2],
        l[1][0] * v[0] + l[1][1] * v[1] + l[1][2] * v[2],
        l[2][0] * v[0] + l[2][1] * v[1] + l[2][2] * v[2],
    ]
}

fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len.is_finite() && len > 0.0 {
        Some([v[0] / len, v[1] / len, v[2] / len])
    } else {
        None
    }
}

/// Signed determinant of the upper-left 3x3 of a 4x4.
fn linear_det(m: [[f32; 4]; 4]) -> f32 {
    let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, f) = (m[1][0], m[1][1], m[1][2]);
    let (g, h, i) = (m[2][0], m[2][1], m[2][2]);
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

impl Primitive {
    /// Bake the affine transform `m` into this primitive's geometry,
    /// returning a transformed copy.
    ///
    /// Positions move by the full affine; normals by the
    /// inverse-transpose of the linear part (renormalised); tangent
    /// directions by the linear part (renormalised) with handedness
    /// flipped when `m` mirrors (negative determinant). Topology,
    /// indices, UVs, colours, joints, weights, material, and extras
    /// are unchanged. See the [module docs](crate::transform) for the
    /// derivation.
    ///
    /// Morph-target **deltas** are displacement vectors, not points,
    /// so they move by the direction rules (round 401 — previously
    /// they were passed through untouched, which left a re-based
    /// morphable mesh deforming along the *old* axes): position and
    /// tangent deltas by the linear part, normal deltas by the
    /// inverse-transpose — with **no renormalisation** on any delta
    /// (deltas are not unit-length; scaling them would change the
    /// morph amplitude the author stored). This makes morphing and
    /// transforming commute exactly on positions:
    /// `transformed(m).apply_morph_weights(w)` equals
    /// `apply_morph_weights(w)` then `transformed(m)` — because
    /// `M·(p + Σ wᵢdᵢ) = M·p + Σ wᵢ·(L·dᵢ)`. (For normals the
    /// equivalence is approximate: the base normal is renormalised at
    /// transform time, the runtime renormalises again after adding
    /// deltas.)
    ///
    /// When the linear part is singular / non-finite the normals and
    /// normal deltas are left as-is (finite fallback) while positions,
    /// position deltas, and tangent data still transform.
    /// Does not mutate `self`.
    pub fn transformed(&self, m: [[f32; 4]; 4]) -> Primitive {
        let mut out = self.clone();
        for p in &mut out.positions {
            *p = transform_point(m, *p);
        }

        // Linear part of m.
        let l = [
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]],
        ];

        // Inverse-transpose for normals; None ⇒ leave normals untouched.
        let normal_mat = mat4_affine_inverse(m).map(|inv| {
            // (L⁻¹)ᵀ : transpose the upper-left 3x3 of the inverse.
            [
                [inv[0][0], inv[1][0], inv[2][0]],
                [inv[0][1], inv[1][1], inv[2][1]],
                [inv[0][2], inv[1][2], inv[2][2]],
            ]
        });

        if let (Some(nm), Some(normals)) = (normal_mat, out.normals.as_mut()) {
            for n in normals.iter_mut() {
                let t = transform_dir(nm, *n);
                if let Some(u) = normalize3(t) {
                    *n = u;
                }
            }
        }

        // Tangents: transform xyz by L; preserve / flip handedness w.
        let mirror = linear_det(m) < 0.0;
        if let Some(tangents) = out.tangents.as_mut() {
            for tg in tangents.iter_mut() {
                let dir = transform_dir(l, [tg[0], tg[1], tg[2]]);
                if let Some(u) = normalize3(dir) {
                    tg[0] = u[0];
                    tg[1] = u[1];
                    tg[2] = u[2];
                }
                if mirror {
                    tg[3] = -tg[3];
                }
            }
        }

        // Morph deltas are displacements: position/tangent deltas by
        // L, normal deltas by the inverse-transpose. NO renormalising
        // — a delta's length is the stored morph amplitude.
        for target in &mut out.targets {
            if let Some(dp) = target.position.as_mut() {
                for d in dp.iter_mut() {
                    *d = transform_dir(l, *d);
                }
            }
            if let (Some(nm), Some(dn)) = (normal_mat, target.normal.as_mut()) {
                for d in dn.iter_mut() {
                    *d = transform_dir(nm, *d);
                }
            }
            if let Some(dt) = target.tangent.as_mut() {
                for d in dt.iter_mut() {
                    *d = transform_dir(l, *d);
                }
            }
        }

        out
    }

    /// Reverse this primitive's triangle winding, flipping which side
    /// faces outward.
    ///
    /// For each triangle the last two corners are swapped (the standard
    /// orientation flip), present `normals` are negated, and the
    /// tangent handedness `w` is inverted so the derived bitangent still
    /// matches the flipped surface. Non-triangle topologies have their
    /// winding-agnostic data left as-is (normals / tangents still flip).
    /// Returns an indexed `Triangles` copy for triangle input. Does not
    /// mutate `self`.
    ///
    /// Useful when importing a left-handed or clockwise-wound source
    /// into the crate's right-handed CCW convention, or to turn a
    /// surface inside-out.
    pub fn reverse_winding(&self) -> Primitive {
        let mut out = self.to_triangle_list();
        if let Some(crate::mesh::Indices::U32(idx)) = &mut out.indices {
            for tri in idx.chunks_exact_mut(3) {
                tri.swap(1, 2);
            }
        }
        if let Some(normals) = out.normals.as_mut() {
            for n in normals.iter_mut() {
                n[0] = -n[0];
                n[1] = -n[1];
                n[2] = -n[2];
            }
        }
        if let Some(tangents) = out.tangents.as_mut() {
            for t in tangents.iter_mut() {
                t[3] = -t[3];
            }
        }
        out
    }
}

impl Mesh {
    /// Bake the affine `m` into every primitive of this mesh, returning
    /// a transformed copy (see [`Primitive::transformed`]). `name` and
    /// `weights` are preserved; `self` is not mutated.
    pub fn transformed(&self, m: [[f32; 4]; 4]) -> Mesh {
        let mut out = self.clone();
        for prim in &mut out.primitives {
            *prim = prim.transformed(m);
        }
        out
    }
}
