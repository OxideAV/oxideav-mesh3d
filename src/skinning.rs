//! Linear-blend skinning — joint matrices + vertex deformation.
//!
//! glTF 2.0 §3.7.3 defines skinning in two halves, and this module
//! implements both:
//!
//! 1. **Joint matrices** ([`Scene3D::joint_matrices`]). For every
//!    joint `j` of the skin bound to a node, a single matrix is
//!    computed:
//!
//!    ```text
//!    jointMatrix(j) = globalTransform(joint_j) · inverseBindMatrix(j)
//!    ```
//!
//!    The inverse bind matrix takes a rest-pose vertex from
//!    mesh-local space into joint-local space; the joint's global
//!    (world) transform then carries it to wherever the joint
//!    currently is. Per §3.7.3.2, **the transform of the node the
//!    skinned mesh is attached to is ignored** — only the joint
//!    transforms move a skinned vertex, so the deformed vertices land
//!    directly in scene world space.
//!
//! 2. **Vertex blend** ([`Primitive::skinned`]). Each vertex carries
//!    up to four `(joint, weight)` influences
//!    ([`Primitive::joints`](crate::Primitive::joints) /
//!    [`Primitive::weights`](crate::Primitive::weights)); the
//!    per-vertex transform is the weighted linear sum of the joint
//!    matrices (§3.7.3.3):
//!
//!    ```text
//!    M(v) = Σᵢ weight[v][i] · jointMatrix(joints[v][i])
//!    p'   = M(v) · (p, 1)
//!    ```
//!
//!    Positions move by the full blended affine; normals by the
//!    **inverse-transpose** of its linear part (the same covector rule
//!    [`Primitive::transformed`](crate::Primitive::transformed)
//!    documents, evaluated per vertex because every vertex blends a
//!    different matrix); tangent directions by the linear part with
//!    handedness `w` flipped when the blend mirrors (negative
//!    determinant).
//!
//! # Weight semantics
//!
//! Weights are used **as stored** — the spec requires each vertex's
//! weights to be non-negative and sum to 1, and this module does not
//! silently repair data that violates that (a vertex whose weights sum
//! to `s ≠ 1` deforms toward `s·M̄` exactly as the numbers dictate).
//! [`Primitive::normalize_weights`](crate::Primitive::normalize_weights)
//! is the explicit fixer, and [`Scene3D::validate`](crate::Scene3D::validate)
//! reports out-of-range joints and negative weights. Two safety rails
//! keep malformed rows from poisoning the output:
//!
//! * an influence whose weight is non-finite or `<= 0`, or whose joint
//!   index falls outside the palette, contributes nothing;
//! * a vertex with no surviving influence (all-zero weights — the spec
//!   shape for "not skinned") is left at its rest pose.

use crate::mesh::{Mesh, Primitive};
use crate::scene::{mat4_mul, NodeId, Scene3D};

/// Blend `out += w * m` over the 4x4 (row-major, column-vector).
fn mat4_madd(out: &mut [[f32; 4]; 4], w: f32, m: &[[f32; 4]; 4]) {
    for (or, mr) in out.iter_mut().zip(m.iter()) {
        for (oc, mc) in or.iter_mut().zip(mr.iter()) {
            *oc += w * mc;
        }
    }
}

/// Signed determinant of the upper-left 3x3.
fn linear_det(m: &[[f32; 4]; 4]) -> f32 {
    let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, f) = (m[1][0], m[1][1], m[1][2]);
    let (g, h, i) = (m[2][0], m[2][1], m[2][2]);
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Inverse-transpose of the upper-left 3x3, or `None` when singular /
/// non-finite. Derived from the adjugate: `L⁻¹ = adj(L) / det(L)`, and
/// `(L⁻¹)ᵀ = adj(L)ᵀ / det(L)` where `adj(L)ᵀ` is the cofactor matrix.
fn linear_inverse_transpose(m: &[[f32; 4]; 4]) -> Option<[[f32; 3]; 3]> {
    let det = linear_det(m);
    if !det.is_finite() || det == 0.0 {
        return None;
    }
    let inv_det = 1.0 / det;
    let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, f) = (m[1][0], m[1][1], m[1][2]);
    let (g, h, i) = (m[2][0], m[2][1], m[2][2]);
    // Cofactor matrix (adjugate transpose), scaled by 1/det.
    let out = [
        [
            (e * i - f * h) * inv_det,
            (f * g - d * i) * inv_det,
            (d * h - e * g) * inv_det,
        ],
        [
            (c * h - b * i) * inv_det,
            (a * i - c * g) * inv_det,
            (b * g - a * h) * inv_det,
        ],
        [
            (b * f - c * e) * inv_det,
            (c * d - a * f) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ];
    if out.iter().flatten().all(|v| v.is_finite()) {
        Some(out)
    } else {
        None
    }
}

fn mul3(l: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
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

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

impl Scene3D {
    /// Skinning matrix palette for the skin bound to `node`, in
    /// [`Skeleton::joints`](crate::Skeleton::joints) order — entry `j`
    /// is `globalTransform(joint_j) · inverseBindMatrix(j)` per glTF
    /// 2.0 §3.7.3. Feed the result straight into
    /// [`Primitive::skinned`] / [`Mesh::skinned`]; the deformed
    /// vertices land in **scene world space** (the skinned-mesh node's
    /// own transform is ignored per §3.7.3.2).
    ///
    /// Joint world transforms are taken from the rest scene graph
    /// ([`Scene3D::world_node_transforms`]). To evaluate a skin under
    /// an animated pose, compute the posed world matrices first and
    /// call [`Scene3D::joint_matrices_with`].
    ///
    /// Returns `None` when the request cannot produce a full palette:
    ///
    /// * `node` is out of range, or carries no
    ///   [`Node::skin`](crate::Node::skin);
    /// * the skin's skeleton id dangles, or a joint `NodeId` is out of
    ///   range;
    /// * a joint node is not reachable from [`Scene3D::roots`] (no
    ///   world transform exists — §3.7.3.2 requires the joints of an
    ///   instantiated skin to live in the same scene);
    /// * fewer inverse-bind matrices than joints are stored
    ///   (`0 < ibm.len() < joints.len()`). An **empty**
    ///   `inverse_bind_matrices` means "identity for every joint"
    ///   (the accessor is optional in glTF; identity is its documented
    ///   default), and *extra* trailing matrices beyond the joint
    ///   count are allowed and ignored (§3.7.3.1 requires the count to
    ///   be greater than **or equal to** the joint count).
    pub fn joint_matrices(&self, node: NodeId) -> Option<Vec<[[f32; 4]; 4]>> {
        self.joint_matrices_with(node, &self.world_node_transforms())
    }

    /// [`Scene3D::joint_matrices`] against caller-supplied world
    /// matrices (indexed by `NodeId.0`, `None` = unreachable), instead
    /// of recomputing [`Scene3D::world_node_transforms`] internally.
    ///
    /// Two callers need this split:
    ///
    /// * batch skinning — one `world_node_transforms()` walk shared
    ///   across every skinned node in the scene;
    /// * animation — worlds computed under a sampled pose, so the
    ///   palette tracks the animated joints rather than the rest pose.
    ///
    /// The `None` conditions of [`Scene3D::joint_matrices`] apply,
    /// with "joint not reachable" generalised to "`worlds` holds no
    /// matrix for the joint's index".
    pub fn joint_matrices_with(
        &self,
        node: NodeId,
        worlds: &[Option<[[f32; 4]; 4]>],
    ) -> Option<Vec<[[f32; 4]; 4]>> {
        let node = self.node(node)?;
        let skin = self.skins.get(node.skin?.0 as usize)?;
        let skeleton = self.skeletons.get(skin.skeleton.0 as usize)?;
        let n_joints = skeleton.joints.len();
        let ibms = &skeleton.inverse_bind_matrices;
        if !ibms.is_empty() && ibms.len() < n_joints {
            return None;
        }
        let mut out = Vec::with_capacity(n_joints);
        for (j, joint) in skeleton.joints.iter().enumerate() {
            let world = *worlds.get(joint.0 as usize)?;
            let world = world?;
            let ibm = if ibms.is_empty() { IDENTITY } else { ibms[j] };
            out.push(mat4_mul(world, ibm));
        }
        Some(out)
    }
}

impl Scene3D {
    /// Render-ready world-space geometry for one node — the full glTF
    /// 2.0 §3.7.4 instantiation pipeline in one call:
    ///
    /// 1. **Morph.** When the mesh has morph targets and default
    ///    weights ([`Mesh::weights`](crate::Mesh::weights)), each
    ///    primitive's base attributes are blended via
    ///    [`Primitive::apply_morph_weights`]. An empty weight list
    ///    instantiates the non-morphed state (§3.7.4: all weights
    ///    zero).
    /// 2. **Skin.** When the node carries a
    ///    [`Node::skin`](crate::Node::skin), the palette from
    ///    [`Scene3D::joint_matrices`] deforms every influenced
    ///    primitive via [`Primitive::skinned`] — the result is already
    ///    world space, the node's own transform being ignored per
    ///    §3.7.3.2.
    /// 3. **Transform.** Otherwise the node's world matrix is baked
    ///    into the vertex data via [`Mesh::transformed`]
    ///    ([`Primitive::transformed`] rules for positions / normals /
    ///    tangents).
    ///
    /// The output is a static bake: morph targets and default morph
    /// weights are cleared (their contribution is folded in), and the
    /// skin path consumes the `joints`/`weights` buffers. A primitive
    /// *without* influence data inside a skinned mesh (invalid per
    /// §3.7.3.3 but seen in the wild) falls back to the node's world
    /// transform, so every output vertex lives in one space.
    ///
    /// Returns `None` when `node` is out of range, carries no mesh,
    /// the mesh id dangles, the node is unreachable from
    /// [`Scene3D::roots`] (no world transform — the node is not
    /// instantiated in this scene), or — on the skin path — the joint
    /// palette cannot be built (see [`Scene3D::joint_matrices`]).
    ///
    /// The rest pose is used; to instantiate under an animated pose,
    /// sample the animation into posed world matrices and call
    /// [`Scene3D::world_mesh_with`]. This models the *mesh*'s default
    /// morph weights only — a node-level morph-weight override (glTF
    /// `node.weights`) is not currently representable on
    /// [`Node`](crate::Node).
    pub fn world_mesh(&self, node: NodeId) -> Option<Mesh> {
        self.world_mesh_with(node, &self.world_node_transforms())
    }

    /// [`Scene3D::world_mesh`] against caller-supplied world matrices
    /// (indexed by `NodeId.0`, `None` = unreachable) — one
    /// [`Scene3D::world_node_transforms`] walk can serve every node,
    /// and animated poses supply their own matrices.
    pub fn world_mesh_with(&self, node: NodeId, worlds: &[Option<[[f32; 4]; 4]>]) -> Option<Mesh> {
        let n = self.node(node)?;
        let mesh = self.meshes.get(n.mesh?.0 as usize)?;
        let world = (*worlds.get(node.0 as usize)?)?;
        let palette = if n.skin.is_some() {
            // A skinned node whose palette can't be built is an error,
            // not a fall-through to rigid instancing.
            Some(self.joint_matrices_with(node, worlds)?)
        } else {
            None
        };

        let mut out = mesh.clone();
        for prim in &mut out.primitives {
            // 1. Fold the default morph weights into the base
            //    attributes (no-op when either side is empty).
            if !prim.targets.is_empty() && !mesh.weights.is_empty() {
                let morphed = prim.apply_morph_weights(&mesh.weights);
                prim.positions = morphed.positions;
                prim.normals = morphed.normals;
                prim.tangents = morphed.tangents;
            }
            prim.targets = Vec::new();

            // 2./3. Skin (world space) or bake the world matrix.
            let has_influences = prim.joints.is_some() && prim.weights.is_some();
            *prim = match &palette {
                Some(p) if has_influences => prim.skinned(p),
                _ => prim.transformed(world),
            };
        }
        out.weights = Vec::new();
        Some(out)
    }
}

impl Primitive {
    /// Deform this primitive by linear-blend skinning against a joint
    /// matrix palette (usually [`Scene3D::joint_matrices`]), returning
    /// the deformed copy. Pure — `self` is untouched.
    ///
    /// Per vertex, the blended matrix `M = Σ wᵢ · palette[jᵢ]` is
    /// accumulated over the four influences, skipping any influence
    /// whose weight is non-finite or `<= 0` or whose joint index is
    /// `>= palette.len()`; a vertex with no surviving influence keeps
    /// its rest-pose data. Then:
    ///
    /// * **positions** move by the blended affine `M`;
    /// * **normals** move by the inverse-transpose of `M`'s linear
    ///   part (renormalised) — per-vertex, because every vertex blends
    ///   a different matrix; a singular blend leaves that vertex's
    ///   normal untouched (same fallback as
    ///   [`Primitive::transformed`]);
    /// * **tangent** directions move by `M`'s linear part
    ///   (renormalised), with handedness `w` negated when the blend
    ///   mirrors (`det < 0`);
    /// * **`joints` / `weights` are dropped** (`None`) — the influence
    ///   data has been consumed; the output is a static mesh in the
    ///   palette's target space and re-skinning it would double-apply;
    /// * **morph targets are dropped.** Skinning applies *after*
    ///   morphing (glTF orders morph → skin), so the base attributes a
    ///   delta was authored against no longer exist in the output.
    ///   Apply [`Primitive::apply_morph_weights`] first when morphs
    ///   matter — [`Scene3D::world_mesh`] runs that exact pipeline.
    ///
    /// Topology, indices, UVs, colours, material, and extras are
    /// preserved. A primitive with no `joints` **or** no `weights`
    /// buffer is returned unchanged (minus nothing — it has no
    /// influence data to consume).
    ///
    /// Weights are used as stored (no renormalisation) — see the
    /// [module docs](crate::skinning) and
    /// [`Primitive::normalize_weights`].
    pub fn skinned(&self, palette: &[[[f32; 4]; 4]]) -> Primitive {
        let (Some(joints), Some(weights)) = (self.joints.as_ref(), self.weights.as_ref()) else {
            return self.clone();
        };
        let mut out = self.clone();
        let n = out.positions.len();
        for v in 0..n {
            let (Some(jrow), Some(wrow)) = (joints.get(v), weights.get(v)) else {
                // Malformed row-count mismatch (validate() reports it);
                // vertices past the shorter buffer keep rest pose.
                continue;
            };
            let mut m = [[0.0f32; 4]; 4];
            let mut any = false;
            for i in 0..4 {
                let w = wrow[i];
                let j = jrow[i] as usize;
                if !w.is_finite() || w <= 0.0 || j >= palette.len() {
                    continue;
                }
                mat4_madd(&mut m, w, &palette[j]);
                any = true;
            }
            if !any {
                continue;
            }

            let p = out.positions[v];
            out.positions[v] = [
                m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
                m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
                m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
            ];

            if let Some(normals) = out.normals.as_mut() {
                if let Some(nv) = normals.get_mut(v) {
                    if let Some(it) = linear_inverse_transpose(&m) {
                        if let Some(u) = normalize3(mul3(it, *nv)) {
                            *nv = u;
                        }
                    }
                }
            }

            if let Some(tangents) = out.tangents.as_mut() {
                if let Some(tv) = tangents.get_mut(v) {
                    let l = [
                        [m[0][0], m[0][1], m[0][2]],
                        [m[1][0], m[1][1], m[1][2]],
                        [m[2][0], m[2][1], m[2][2]],
                    ];
                    if let Some(u) = normalize3(mul3(l, [tv[0], tv[1], tv[2]])) {
                        tv[0] = u[0];
                        tv[1] = u[1];
                        tv[2] = u[2];
                    }
                    if linear_det(&m) < 0.0 {
                        tv[3] = -tv[3];
                    }
                }
            }
        }
        out.joints = None;
        out.weights = None;
        out.targets = Vec::new();
        out
    }
}

impl Mesh {
    /// [`Primitive::skinned`] applied to every primitive, returning
    /// the deformed copy. `name` is preserved; the default morph
    /// `weights` are cleared alongside the per-primitive morph targets
    /// (a skinned bake is a static mesh — see [`Primitive::skinned`]
    /// for why targets don't survive). `self` is not mutated.
    pub fn skinned(&self, palette: &[[[f32; 4]; 4]]) -> Mesh {
        let mut out = self.clone();
        for prim in &mut out.primitives {
            *prim = prim.skinned(palette);
        }
        out.weights = Vec::new();
        out
    }
}
