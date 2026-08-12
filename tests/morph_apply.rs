//! Tests for [`Primitive::apply_morph_weights`].
//!
//! Truth ladder: glTF 2.0 §3.7.2.2 (Morph Targets) — formula at lines
//! 3577-3580 of `docs/3d/gltf/gltf-2.0-spec.html`:
//!
//! ```text
//! morphed.POSITION = base.POSITION
//!                  + weights[0] * targets[0].POSITION
//!                  + weights[1] * targets[1].POSITION
//!                  + weights[2] * targets[2].POSITION + ...
//! ```
//!
//! Additional spec constraints exercised:
//!
//! * Line 3589 — "Attributes present in the base mesh primitive but
//!   not included in a given morph target MUST retain their original
//!   values for the morph target." → a `None` slot contributes zero.
//! * Line 3616 — "Note that the W component for handedness is omitted
//!   when targeting `TANGENT` data since handedness cannot be displaced."
//!   → morph TANGENT is `[f32; 3]`, base TANGENT `w` survives unmodified.
//! * Line 3697 — "When `mesh.weights` is undefined, the default
//!   targets' weights are zeros." → empty / short weights vector
//!   leaves the base unchanged for any unmentioned target index.

use oxideav_mesh3d::{MorphTarget, MorphedAttributes, Primitive, Topology};

/// Approximate-equality on a 3-vector, suitable for f32 morph arithmetic.
fn vec3_close(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
    (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
}

fn vec4_close(a: [f32; 4], b: [f32; 4], eps: f32) -> bool {
    (a[0] - b[0]).abs() < eps
        && (a[1] - b[1]).abs() < eps
        && (a[2] - b[2]).abs() < eps
        && (a[3] - b[3]).abs() < eps
}

/// Construct a 3-vertex primitive with optional normals/tangents.
fn base_triangle() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    // Use distinctive handedness w values so we can prove they survive.
    p.tangents = Some(vec![
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, 1.0],
    ]);
    p
}

// ---------------------------------------------------------------------------
// No-op cases — empty weights / no targets / all-zero weights
// ---------------------------------------------------------------------------

#[test]
fn empty_weights_returns_base_verbatim() {
    let p = base_triangle();
    let m = p.apply_morph_weights(&[]);
    assert_eq!(m.positions, p.positions);
    assert_eq!(m.normals.as_deref(), p.normals.as_deref());
    assert_eq!(m.tangents.as_deref(), p.tangents.as_deref());
}

#[test]
fn no_targets_returns_base_even_with_weights() {
    // No morph targets on the primitive; supplying weights must not
    // panic or invent contributions (the loop bound clamps to
    // `targets.len()` which is zero).
    let p = base_triangle();
    let m = p.apply_morph_weights(&[0.5, 0.25, 0.125]);
    assert_eq!(m.positions, p.positions);
}

#[test]
fn all_zero_weights_is_no_op() {
    // Spec line 3697: missing weights default to zero. Explicit zero
    // weight is identical to the missing case.
    let mut p = base_triangle();
    p.targets.push(MorphTarget {
        position: Some(vec![[10.0, 20.0, 30.0]; 3]),
        normal: None,
        tangent: None,
    });
    let m = p.apply_morph_weights(&[0.0]);
    assert_eq!(m.positions, p.positions);
    assert_eq!(m.normals.as_deref(), p.normals.as_deref());
}

// ---------------------------------------------------------------------------
// Single-target POSITION
// ---------------------------------------------------------------------------

#[test]
fn single_target_full_weight_matches_base_plus_delta() {
    let mut p = base_triangle();
    p.targets.push(MorphTarget {
        position: Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        normal: None,
        tangent: None,
    });
    let m = p.apply_morph_weights(&[1.0]);
    assert!(vec3_close(m.positions[0], [0.1, 0.2, 0.3], 1e-6));
    assert!(vec3_close(m.positions[1], [1.4, 0.5, 0.6], 1e-6));
    assert!(vec3_close(m.positions[2], [0.7, 1.8, 0.9], 1e-6));
}

#[test]
fn single_target_half_weight_scales_delta() {
    let mut p = base_triangle();
    p.targets.push(MorphTarget {
        position: Some(vec![[2.0, 0.0, 0.0]; 3]),
        normal: None,
        tangent: None,
    });
    let m = p.apply_morph_weights(&[0.5]);
    // base + 0.5 * 2.0 = base + 1.0
    assert!(vec3_close(m.positions[0], [1.0, 0.0, 0.0], 1e-6));
    assert!(vec3_close(m.positions[1], [2.0, 0.0, 0.0], 1e-6));
    assert!(vec3_close(m.positions[2], [1.0, 1.0, 0.0], 1e-6));
}

#[test]
fn negative_weight_subtracts_delta() {
    // Spec doesn't forbid negative weights — they're legal and produce
    // an "anti-pose" interpolant. This is the natural extrapolation
    // edge case (`weight < 0`).
    let mut p = base_triangle();
    p.targets.push(MorphTarget {
        position: Some(vec![[1.0, 1.0, 1.0]; 3]),
        normal: None,
        tangent: None,
    });
    let m = p.apply_morph_weights(&[-1.0]);
    assert!(vec3_close(m.positions[0], [-1.0, -1.0, -1.0], 1e-6));
    assert!(vec3_close(m.positions[1], [0.0, -1.0, -1.0], 1e-6));
}

// ---------------------------------------------------------------------------
// Multiple targets — weighted sum
// ---------------------------------------------------------------------------

#[test]
fn two_targets_blend_linearly_per_vertex() {
    // Two targets, weights 0.25 and 0.75 → output[k] = base[k]
    // + 0.25 * target_a[k] + 0.75 * target_b[k]
    let mut p = base_triangle();
    p.targets.push(MorphTarget {
        position: Some(vec![[4.0, 0.0, 0.0]; 3]),
        normal: None,
        tangent: None,
    });
    p.targets.push(MorphTarget {
        position: Some(vec![[0.0, 4.0, 0.0]; 3]),
        normal: None,
        tangent: None,
    });
    let m = p.apply_morph_weights(&[0.25, 0.75]);
    // base[0] = (0,0,0); +0.25*4=1 on x; +0.75*4=3 on y
    assert!(vec3_close(m.positions[0], [1.0, 3.0, 0.0], 1e-6));
    // base[1] = (1,0,0); +1+3 = (2, 3, 0)
    assert!(vec3_close(m.positions[1], [2.0, 3.0, 0.0], 1e-6));
    // base[2] = (0,1,0); +(1,3) = (1, 4, 0)
    assert!(vec3_close(m.positions[2], [1.0, 4.0, 0.0], 1e-6));
}

#[test]
fn three_targets_only_first_two_have_weights() {
    // Spec line 3697: missing weights default to zero. Supply 2
    // weights for 3 targets — the third is treated as if its weight
    // were zero.
    let mut p = base_triangle();
    for _ in 0..3 {
        p.targets.push(MorphTarget {
            position: Some(vec![[1.0, 0.0, 0.0]; 3]),
            normal: None,
            tangent: None,
        });
    }
    let m = p.apply_morph_weights(&[0.5, 0.5]);
    // Only the first two targets contribute: base + 0.5 + 0.5 = base + 1
    assert!(vec3_close(m.positions[0], [1.0, 0.0, 0.0], 1e-6));
    assert!(vec3_close(m.positions[1], [2.0, 0.0, 0.0], 1e-6));
    assert!(vec3_close(m.positions[2], [1.0, 1.0, 0.0], 1e-6));
}

#[test]
fn extra_weights_past_target_count_are_ignored() {
    // Inverse of the above: more weights than targets. The trailing
    // weights have no target to apply against and must be silently
    // dropped (don't index out of bounds, don't panic).
    let mut p = base_triangle();
    p.targets.push(MorphTarget {
        position: Some(vec![[1.0, 0.0, 0.0]; 3]),
        normal: None,
        tangent: None,
    });
    let m = p.apply_morph_weights(&[1.0, 99.0, -99.0, 0.5]);
    assert!(vec3_close(m.positions[0], [1.0, 0.0, 0.0], 1e-6));
    assert!(vec3_close(m.positions[1], [2.0, 0.0, 0.0], 1e-6));
}

// ---------------------------------------------------------------------------
// NORMAL slot
// ---------------------------------------------------------------------------

#[test]
fn normal_slot_blends_when_base_and_target_present() {
    let mut p = base_triangle();
    p.targets.push(MorphTarget {
        position: None,
        normal: Some(vec![[1.0, 0.0, 0.0]; 3]),
        tangent: None,
    });
    let m = p.apply_morph_weights(&[0.5]);
    // Position should be unchanged (no position delta on this target).
    assert_eq!(m.positions, p.positions);
    // Normals blended: (0,0,1) + 0.5 * (1,0,0) = (0.5, 0, 1)
    let n = m.normals.unwrap();
    for nk in n.iter().take(3) {
        assert!(vec3_close(*nk, [0.5, 0.0, 1.0], 1e-6));
    }
    // (Note: spec doesn't mandate re-normalisation by the apply step.
    // Renderers re-normalise themselves after blending.)
}

#[test]
fn normal_target_without_base_normals_is_dropped() {
    // Spec line 3586: "For each morph target attribute, an original
    // attribute MUST be present in the mesh primitive." A target that
    // names NORMAL when the base has no normals is malformed input —
    // we leave the output `normals = None` rather than fabricate a
    // synthetic base.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0; 3]; 3];
    p.normals = None;
    p.targets.push(MorphTarget {
        position: None,
        normal: Some(vec![[1.0, 0.0, 0.0]; 3]),
        tangent: None,
    });
    let m = p.apply_morph_weights(&[1.0]);
    assert!(m.normals.is_none());
}

// ---------------------------------------------------------------------------
// TANGENT slot — handedness preservation
// ---------------------------------------------------------------------------

#[test]
fn tangent_handedness_w_survives_unmodified() {
    // Spec §3.7.2.2 line 3616: morph TANGENT is VEC3 (xyz). The base
    // TANGENT's `w` handedness MUST NOT be touched.
    let mut p = base_triangle();
    // Base tangents have alternating w = +1 / -1 / +1.
    let base_w = [
        p.tangents.as_ref().unwrap()[0][3],
        p.tangents.as_ref().unwrap()[1][3],
        p.tangents.as_ref().unwrap()[2][3],
    ];
    p.targets.push(MorphTarget {
        position: None,
        normal: None,
        tangent: Some(vec![[0.1, 0.2, 0.3]; 3]),
    });
    let m = p.apply_morph_weights(&[1.0]);
    let t = m.tangents.unwrap();
    // xyz blended additively:
    assert!(vec4_close(t[0], [1.1, 0.2, 0.3, base_w[0]], 1e-6));
    assert!(vec4_close(t[1], [1.1, 0.2, 0.3, base_w[1]], 1e-6));
    assert!(vec4_close(t[2], [1.1, 0.2, 0.3, base_w[2]], 1e-6));
    // Handedness signs preserved (the load-bearing assertion):
    assert_eq!(t[0][3], 1.0);
    assert_eq!(t[1][3], -1.0);
    assert_eq!(t[2][3], 1.0);
}

// ---------------------------------------------------------------------------
// Combined POSITION + NORMAL + TANGENT in one pass
// ---------------------------------------------------------------------------

#[test]
fn three_attribute_target_blends_all_three() {
    let mut p = base_triangle();
    p.targets.push(MorphTarget {
        position: Some(vec![[1.0, 0.0, 0.0]; 3]),
        normal: Some(vec![[0.0, 1.0, 0.0]; 3]),
        tangent: Some(vec![[0.0, 0.0, 1.0]; 3]),
    });
    let m = p.apply_morph_weights(&[1.0]);
    // Position delta on x.
    assert!(vec3_close(m.positions[0], [1.0, 0.0, 0.0], 1e-6));
    assert!(vec3_close(m.positions[1], [2.0, 0.0, 0.0], 1e-6));
    // Normal delta on y.
    let n = m.normals.unwrap();
    assert!(vec3_close(n[0], [0.0, 1.0, 1.0], 1e-6));
    // Tangent xyz delta on z; w preserved.
    let t = m.tangents.unwrap();
    assert!(vec4_close(t[0], [1.0, 0.0, 1.0, 1.0], 1e-6));
    assert_eq!(t[0][3], 1.0);
    assert_eq!(t[1][3], -1.0);
}

// ---------------------------------------------------------------------------
// Missing-slot semantics — output presence mirrors input presence
// ---------------------------------------------------------------------------

#[test]
fn output_normals_none_when_base_normals_none_even_if_target_has_normal() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0; 3]];
    p.normals = None; // base has no normals
    p.targets.push(MorphTarget {
        position: None,
        normal: Some(vec![[1.0, 1.0, 1.0]]),
        tangent: None,
    });
    let m = p.apply_morph_weights(&[1.0]);
    assert!(m.normals.is_none());
}

#[test]
fn output_tangents_none_when_base_tangents_none() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0; 3]];
    p.normals = Some(vec![[0.0, 0.0, 1.0]]);
    p.tangents = None;
    p.targets.push(MorphTarget {
        position: None,
        normal: None,
        tangent: Some(vec![[1.0, 1.0, 1.0]]),
    });
    let m = p.apply_morph_weights(&[1.0]);
    assert!(m.tangents.is_none());
}

#[test]
fn target_with_only_position_leaves_normals_and_tangents_untouched() {
    let mut p = base_triangle();
    let base_normals = p.normals.clone();
    let base_tangents = p.tangents.clone();
    p.targets.push(MorphTarget {
        position: Some(vec![[1.0, 0.0, 0.0]; 3]),
        normal: None,
        tangent: None,
    });
    let m = p.apply_morph_weights(&[1.0]);
    assert_eq!(m.normals, base_normals);
    assert_eq!(m.tangents, base_tangents);
}

// ---------------------------------------------------------------------------
// Soft-error handling — length mismatch doesn't panic
// ---------------------------------------------------------------------------

#[test]
fn shorter_target_delta_applies_prefix_then_leaves_remainder_alone() {
    let mut p = base_triangle(); // 3 vertices
    p.targets.push(MorphTarget {
        position: Some(vec![[1.0, 0.0, 0.0]]), // only 1 delta!
        normal: None,
        tangent: None,
    });
    let m = p.apply_morph_weights(&[1.0]);
    // Vertex 0 receives the delta; vertices 1, 2 unchanged from base.
    assert!(vec3_close(m.positions[0], [1.0, 0.0, 0.0], 1e-6));
    assert_eq!(m.positions[1], [1.0, 0.0, 0.0]); // base, unchanged
    assert_eq!(m.positions[2], [0.0, 1.0, 0.0]); // base, unchanged
}

#[test]
fn longer_target_delta_ignores_excess_entries() {
    let mut p = base_triangle(); // 3 vertices
    p.targets.push(MorphTarget {
        position: Some(vec![[1.0, 0.0, 0.0]; 10]), // 10 deltas; only 3 used
        normal: None,
        tangent: None,
    });
    let m = p.apply_morph_weights(&[1.0]);
    assert_eq!(m.positions.len(), 3);
    assert!(vec3_close(m.positions[0], [1.0, 0.0, 0.0], 1e-6));
    assert!(vec3_close(m.positions[1], [2.0, 0.0, 0.0], 1e-6));
    assert!(vec3_close(m.positions[2], [1.0, 1.0, 0.0], 1e-6));
}

// ---------------------------------------------------------------------------
// Type ergonomics
// ---------------------------------------------------------------------------

#[test]
fn morphed_attributes_round_trips_through_clone_and_eq() {
    let mut p = base_triangle();
    p.targets.push(MorphTarget {
        position: Some(vec![[1.0, 0.0, 0.0]; 3]),
        normal: None,
        tangent: None,
    });
    let m: MorphedAttributes = p.apply_morph_weights(&[1.0]);
    let cloned = m.clone();
    assert_eq!(cloned, m);
    assert_eq!(cloned.positions, m.positions);
}

#[test]
fn morphed_attributes_reexported_at_crate_root() {
    // Both type paths must resolve to the same type — the re-export
    // must be a `pub use`, not a duplicate definition.
    let _via_root: oxideav_mesh3d::MorphedAttributes = MorphedAttributes {
        positions: vec![],
        normals: None,
        tangents: None,
    };
    let _via_module: oxideav_mesh3d::mesh::MorphedAttributes = MorphedAttributes {
        positions: vec![],
        normals: None,
        tangents: None,
    };
}

// ---------------------------------------------------------------------------
// Primitive::morphed / Mesh::morphed — the static-fold lifts
// ---------------------------------------------------------------------------

#[test]
fn primitive_morphed_folds_the_blend_and_consumes_targets() {
    let mut p = base_triangle();
    p.targets = vec![MorphTarget {
        position: Some(vec![[0.0, 0.0, 4.0]; 3]),
        ..Default::default()
    }];
    let out = p.morphed(&[0.5]);
    assert!(vec3_close(out.positions[0], [0.0, 0.0, 2.0], 1e-6));
    assert!(out.targets.is_empty(), "morph state consumed");
    assert_eq!(p.targets.len(), 1, "self untouched");
}

#[test]
fn primitive_morphed_with_empty_weights_bakes_the_base_state() {
    let mut p = base_triangle();
    p.targets = vec![MorphTarget {
        position: Some(vec![[0.0, 0.0, 4.0]; 3]),
        ..Default::default()
    }];
    let out = p.morphed(&[]);
    assert_eq!(out.positions, p.positions, "all weights zero = base");
    assert!(out.targets.is_empty(), "roster still cleared");
}

#[test]
fn primitive_morphed_carries_every_other_field() {
    let mut p = base_triangle();
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]];
    p.joints = Some(vec![[0, 0, 0, 0]; 3]);
    p.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]);
    p.targets = vec![MorphTarget {
        position: Some(vec![[1.0, 0.0, 0.0]; 3]),
        ..Default::default()
    }];
    let out = p.morphed(&[1.0]);
    assert_eq!(out.topology, p.topology);
    assert_eq!(out.uvs, p.uvs);
    assert_eq!(out.joints, p.joints, "skinning influences pass through");
    assert_eq!(out.weights, p.weights);
    assert!(vec4_close(
        out.tangents.as_ref().unwrap()[1],
        [1.0, 0.0, 0.0, -1.0],
        1e-6
    ));
}

#[test]
fn mesh_morphed_lifts_across_primitives_and_clears_defaults() {
    let mut a = base_triangle();
    a.targets = vec![MorphTarget {
        position: Some(vec![[0.0, 0.0, 4.0]; 3]),
        ..Default::default()
    }];
    let mut b = base_triangle();
    b.targets = vec![MorphTarget {
        position: Some(vec![[2.0, 0.0, 0.0]; 3]),
        ..Default::default()
    }];
    let mesh = oxideav_mesh3d::Mesh::new(Some("blend".to_owned()))
        .with_primitive(a)
        .with_primitive(b)
        .with_weights(vec![0.25]);
    let out = mesh.morphed(&[0.5]);
    assert_eq!(out.name.as_deref(), Some("blend"), "name preserved");
    assert!(vec3_close(
        out.primitives[0].positions[0],
        [0.0, 0.0, 2.0],
        1e-6
    ));
    assert!(vec3_close(
        out.primitives[1].positions[0],
        [1.0, 0.0, 0.0],
        1e-6
    ));
    assert!(out.weights.is_empty(), "consumed defaults cleared");
    assert!(out.primitives.iter().all(|p| p.targets.is_empty()));
    assert_eq!(mesh.weights, vec![0.25], "self untouched");
}
