//! Round 6: typed morph-target field coverage.
//!
//! Pins the new typed surfaces added by this round so consumers can
//! migrate off the `__morph_targets` / `__mesh_weights` extras
//! sentinels with confidence:
//!
//! * `Primitive::targets: Vec<MorphTarget>` — the per-pose delta
//!   buffers (`POSITION`, `NORMAL`, `TANGENT`) per glTF 2.0 §3.7.2.2.
//!   Default is the empty vec (no morph targets).
//! * `MorphTarget { position, normal, tangent }` — typed Option slots
//!   (Vec<[f32; 3]> each); deltas at vertex `i` are added to the base
//!   attribute at vertex `i` scaled by the target's blend weight.
//! * `Mesh::weights: Vec<f32>` — static default morph-blend weights
//!   per spec §3.7.2.2 `mesh.weights`. Empty default; runtime
//!   `AnimationProperty::MorphWeights` channel overrides.
//!
//! Coverage:
//!
//! 1. **Defaults are empty** — `Primitive::new()` and `Mesh::new()` /
//!    `Mesh::default()` start with `targets = []` / `weights = []`
//!    and `MorphTarget::new()` has every slot `None`.
//! 2. **Primitive with two morph targets (POSITION + NORMAL each)
//!    survives clone + PartialEq round-trip.**
//! 3. **Mesh with four blend weights survives clone + PartialEq
//!    round-trip via field comparison.**
//! 4. **Re-export from crate root** — `oxideav_mesh3d::MorphTarget` is
//!    the same type as `oxideav_mesh3d::mesh::MorphTarget`.
//! 5. **Builder helper `Mesh::with_weights`** — chains and overwrites
//!    cleanly; accepts both `Vec<f32>` and `[f32; N]`.
//! 6. **Tangent slot uses [f32; 3] per spec** — handedness `w` is not
//!    morphed, only xyz.

use oxideav_mesh3d::{Mesh, MorphTarget, Primitive, Topology};

// ---------- defaults ----------

#[test]
fn primitive_new_has_empty_targets() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.targets.is_empty(), "expected no morph targets by default");
}

#[test]
fn mesh_new_has_empty_weights() {
    let m = Mesh::new(None);
    assert!(m.weights.is_empty(), "expected no morph weights by default");
}

#[test]
fn mesh_default_has_empty_weights_and_no_primitives() {
    let m = Mesh::default();
    assert!(m.weights.is_empty());
    assert!(m.primitives.is_empty());
    assert!(m.name.is_none());
}

#[test]
fn morph_target_new_is_all_none() {
    let t = MorphTarget::new();
    assert!(t.position.is_none());
    assert!(t.normal.is_none());
    assert!(t.tangent.is_none());
}

#[test]
fn morph_target_default_matches_new() {
    assert_eq!(MorphTarget::default(), MorphTarget::new());
}

// ---------- crate root re-export parity ----------

#[test]
fn morph_target_reexport_from_crate_root() {
    // Construct via the crate root and via the module path; assert
    // they are PartialEq-equivalent (same type underneath).
    let via_root: oxideav_mesh3d::MorphTarget = MorphTarget::new();
    let via_module: oxideav_mesh3d::mesh::MorphTarget =
        oxideav_mesh3d::mesh::MorphTarget::default();
    assert_eq!(via_root, via_module);
}

// ---------- Primitive.targets round-trip ----------

#[test]
fn primitive_with_two_morph_targets_clone_round_trip() {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);

    // Target 0: "smile" — POSITION + NORMAL deltas on three vertices.
    let smile = MorphTarget {
        position: Some(vec![[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]),
        normal: Some(vec![[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]),
        tangent: None,
    };
    // Target 1: "frown" — POSITION + NORMAL with opposite deltas.
    let frown = MorphTarget {
        position: Some(vec![[-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]),
        normal: Some(vec![
            [-0.01, 0.0, 0.0],
            [0.0, -0.01, 0.0],
            [0.0, 0.0, -0.01],
        ]),
        tangent: None,
    };

    prim.targets = vec![smile.clone(), frown.clone()];
    assert_eq!(prim.targets.len(), 2);

    // Clone round-trip via PartialEq on the typed MorphTarget vector.
    let clone = prim.clone();
    assert_eq!(clone.targets, prim.targets);
    assert_eq!(clone.targets[0], smile);
    assert_eq!(clone.targets[1], frown);

    // Per-slot field round-trip — the typed surface must preserve
    // every Option + every f32 component bit-exact.
    assert_eq!(
        clone.targets[0].position.as_deref(),
        Some([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]].as_slice())
    );
    assert_eq!(
        clone.targets[0].normal.as_deref(),
        Some([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]].as_slice())
    );
    assert!(clone.targets[0].tangent.is_none());
}

#[test]
fn morph_target_with_tangent_slot_round_trips() {
    // §3.7.2.2: morph TANGENT delta is VEC3 (xyz only) — handedness
    // `w` on the base TANGENT is not morphed.
    let target = MorphTarget {
        position: None,
        normal: None,
        tangent: Some(vec![[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
    };

    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0]];
    prim.targets.push(target.clone());

    let cloned = prim.clone();
    assert_eq!(cloned.targets.len(), 1);
    assert_eq!(cloned.targets[0], target);
    assert_eq!(
        cloned.targets[0].tangent.as_deref(),
        Some([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]].as_slice())
    );
}

// ---------- Mesh.weights round-trip ----------

#[test]
fn mesh_with_four_weights_round_trips() {
    let m = Mesh::new(Some("face".to_string()))
        .with_primitive(Primitive::new(Topology::Triangles))
        .with_weights(vec![0.0_f32, 0.25, 0.5, 1.0]);
    assert_eq!(m.weights.len(), 4);
    assert_eq!(m.weights, vec![0.0, 0.25, 0.5, 1.0]);

    // Clone equality on the Vec<f32> field captures the round-trip
    // (Mesh itself doesn't derive PartialEq because Primitive's
    // serde_json::Value extras don't implement Eq cheaply).
    let cloned = m.clone();
    assert_eq!(cloned.weights, m.weights);
    assert_eq!(cloned.name.as_deref(), Some("face"));
    assert_eq!(cloned.primitives.len(), 1);
}

#[test]
fn mesh_with_weights_overwrites_previous_values() {
    let m = Mesh::new(None)
        .with_weights(vec![0.1, 0.2])
        .with_weights(vec![0.7, 0.8, 0.9]);
    assert_eq!(m.weights, vec![0.7, 0.8, 0.9]);
}

#[test]
fn mesh_with_weights_accepts_array_literal() {
    // `impl Into<Vec<f32>>` accepts both Vec<f32> and arrays.
    let m = Mesh::new(None).with_weights([0.0_f32, 1.0]);
    assert_eq!(m.weights, vec![0.0, 1.0]);
}

// ---------- recommended migration paths ----------
//
// `Primitive` and `Mesh` are NOT yet `#[non_exhaustive]` (round-7
// candidate, deferred until every downstream caller migrates onto
// builders — same posture as `oxideav_core::Group` per MEMORY). The
// next two tests pin the recommended construction style for new
// callers so they stay forward-compatible when `#[non_exhaustive]`
// lands.

#[test]
fn mesh_constructed_via_builders_has_expected_field_layout() {
    let mut m = Mesh::new(Some("explicit".to_owned())).with_weights(vec![1.0_f32, 2.0]);
    m.primitives.push(Primitive::new(Topology::Triangles));
    assert_eq!(m.weights, vec![1.0, 2.0]);
    assert_eq!(m.primitives.len(), 1);
    assert_eq!(m.name.as_deref(), Some("explicit"));
}

#[test]
fn primitive_constructed_via_new_then_field_assignment() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0]];
    p.targets.push(MorphTarget {
        position: Some(vec![[0.5, 0.0, 0.0]]),
        normal: None,
        tangent: None,
    });
    assert_eq!(p.positions.len(), 1);
    assert_eq!(p.targets.len(), 1);
}
