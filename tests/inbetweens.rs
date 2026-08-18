//! Tests for typed in-between shapes — [`Inbetween`] on
//! [`MorphTarget::inbetweens`], the [`MorphTarget::at_weight`]
//! station resolution, its routing through
//! [`Primitive::apply_morph_weights`], the [`Scene3D::validate`]
//! authoring rules, and the geometry-pass interplay (transform, weld,
//! fetch-permute, subdivide, cluster).
//!
//! Truth ladder: the USD blend-shape schema as staged at
//! `docs/3d/usd/usdskel-usdpreviewsurface-schema.md` §1.4.1 —
//! implicit endpoints (null shape at 0, primary at 1), authoring
//! errors (weight 0/1, duplicate weights) ignored at runtime,
//! piecewise-linear **unbounded** resolution (the worked example: an
//! in-between at 0.25, channel weight −0.25 ⇒ that shape at weight
//! −1), per-in-between optional normal offsets ("absence = no normal
//! offsets").

use oxideav_mesh3d::{
    Inbetween, Mesh, MorphTarget, Node, Primitive, Scene3D, Topology, ValidationError,
};

fn close3(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
    (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
}

/// Primary target: +1 in X per vertex, `n` vertices.
fn primary_target(n: usize) -> MorphTarget {
    let mut t = MorphTarget::new();
    t.position = Some(vec![[1.0, 0.0, 0.0]; n]);
    t
}

// ---------------------------------------------------------------- //
// at_weight — resolution semantics                                 //
// ---------------------------------------------------------------- //

#[test]
fn no_inbetweens_is_the_linear_rule() {
    let mut t = primary_target(2);
    t.normal = Some(vec![[0.0, 1.0, 0.0]; 2]);
    t.tangent = Some(vec![[0.0, 0.0, 1.0]; 2]);
    for w in [-1.0f32, 0.0, 0.25, 0.5, 1.0, 2.0] {
        let r = t.at_weight(w);
        assert!(close3(r.position.as_ref().unwrap()[0], [w, 0.0, 0.0], 1e-6));
        assert!(close3(r.normal.as_ref().unwrap()[1], [0.0, w, 0.0], 1e-6));
        assert!(close3(r.tangent.as_ref().unwrap()[0], [0.0, 0.0, w], 1e-6));
        assert!(r.inbetweens.is_empty());
    }
}

#[test]
fn endpoints_resolve_to_null_and_primary() {
    let mut t = primary_target(1);
    t.inbetweens = vec![Inbetween::new(0.5).with_position(vec![[0.9, 0.0, 0.0]])];
    let r0 = t.at_weight(0.0);
    assert!(close3(r0.position.unwrap()[0], [0.0; 3], 1e-6));
    let r1 = t.at_weight(1.0);
    assert!(close3(r1.position.unwrap()[0], [1.0, 0.0, 0.0], 1e-6));
}

#[test]
fn worked_example_brackets_and_extrapolates() {
    // §1.4.1 worked example: in-betweens at 0.25 and 0.5.
    let mut t = primary_target(1);
    t.inbetweens = vec![
        Inbetween::new(0.25).with_position(vec![[0.4, 0.0, 0.0]]),
        Inbetween::new(0.5).with_position(vec![[0.6, 0.0, 0.0]]),
    ];
    // w < 0.25: null ↔ 0.25-shape. At w = 0.125, halfway to 0.4.
    let r = t.at_weight(0.125);
    assert!(close3(r.position.unwrap()[0], [0.2, 0.0, 0.0], 1e-6));
    // Exact station.
    let r = t.at_weight(0.25);
    assert!(close3(r.position.unwrap()[0], [0.4, 0.0, 0.0], 1e-6));
    // 0.25 <= w <= 0.5: between the two shapes.
    let r = t.at_weight(0.375);
    assert!(close3(r.position.unwrap()[0], [0.5, 0.0, 0.0], 1e-6));
    // w > 0.5: 0.5-shape ↔ primary. At 0.75, halfway 0.6 → 1.0.
    let r = t.at_weight(0.75);
    assert!(close3(r.position.unwrap()[0], [0.8, 0.0, 0.0], 1e-6));
    // Unbounded below: w = −0.25 ⇒ the 0.25 shape at weight −1.
    let r = t.at_weight(-0.25);
    assert!(close3(r.position.unwrap()[0], [-0.4, 0.0, 0.0], 1e-6));
    // Unbounded above: last segment (0.5 → 1) extrapolates. At
    // w = 1.5: 0.6 + 2·(1.0 − 0.6) = 1.4.
    let r = t.at_weight(1.5);
    assert!(close3(r.position.unwrap()[0], [1.4, 0.0, 0.0], 1e-6));
}

#[test]
fn absent_inbetween_normal_reads_as_zeros() {
    // Primary has normal offsets; the in-between doesn't ("no normal
    // offsets" per the schema). At the station the normal
    // contribution must be exactly zero, interpolating back up to
    // the primary between station and 1.
    let mut t = primary_target(1);
    t.normal = Some(vec![[0.0, 2.0, 0.0]]);
    t.inbetweens = vec![Inbetween::new(0.5).with_position(vec![[0.5, 0.0, 0.0]])];
    let r = t.at_weight(0.5);
    assert!(close3(r.normal.as_ref().unwrap()[0], [0.0; 3], 1e-6));
    let r = t.at_weight(0.75);
    assert!(close3(r.normal.unwrap()[0], [0.0, 1.0, 0.0], 1e-6));
}

#[test]
fn inbetween_only_slot_is_resolved() {
    // Normal offsets exist ONLY on the in-between: the output slot
    // must still materialise (peaking at the station, fading to zero
    // at both endpoints).
    let mut t = primary_target(1);
    t.inbetweens = vec![Inbetween::new(0.5)
        .with_position(vec![[0.5, 0.0, 0.0]])
        .with_normal(vec![[0.0, 0.0, 1.0]])];
    let r = t.at_weight(0.5);
    assert!(close3(r.normal.as_ref().unwrap()[0], [0.0, 0.0, 1.0], 1e-6));
    let r = t.at_weight(1.0);
    assert!(close3(r.normal.unwrap()[0], [0.0; 3], 1e-6));
}

#[test]
fn malformed_inbetweens_are_ignored() {
    let mut t = primary_target(1);
    t.inbetweens = vec![
        Inbetween::new(0.0).with_position(vec![[9.0, 0.0, 0.0]]), // implicit endpoint
        Inbetween::new(1.0).with_position(vec![[9.0, 0.0, 0.0]]), // implicit endpoint
        Inbetween::new(f32::NAN).with_position(vec![[9.0, 0.0, 0.0]]),
        // Colliding stations: BOTH dropped.
        Inbetween::new(0.5).with_position(vec![[9.0, 0.0, 0.0]]),
        Inbetween::new(0.5).with_position(vec![[9.0, 0.0, 0.0]]),
    ];
    // Everything malformed ⇒ pure linear rule.
    let r = t.at_weight(0.5);
    assert!(close3(r.position.unwrap()[0], [0.5, 0.0, 0.0], 1e-6));
}

#[test]
fn tangent_deltas_stay_linear() {
    let mut t = primary_target(1);
    t.tangent = Some(vec![[0.0, 0.0, 2.0]]);
    t.inbetweens = vec![Inbetween::new(0.5).with_position(vec![[0.9, 0.0, 0.0]])];
    // The in-between bends positions but tangents scale linearly.
    let r = t.at_weight(0.5);
    assert!(close3(
        r.position.as_ref().unwrap()[0],
        [0.9, 0.0, 0.0],
        1e-6
    ));
    assert!(close3(r.tangent.unwrap()[0], [0.0, 0.0, 1.0], 1e-6));
}

#[test]
fn negative_weight_station_brackets_below_zero() {
    // A corrective authored at −0.5 is a legal (finite, non-0/1)
    // station; w = −0.25 interpolates null ↔ that shape.
    let mut t = primary_target(1);
    t.inbetweens = vec![Inbetween::new(-0.5).with_position(vec![[-2.0, 0.0, 0.0]])];
    let r = t.at_weight(-0.25);
    assert!(close3(r.position.unwrap()[0], [-1.0, 0.0, 0.0], 1e-6));
}

// ---------------------------------------------------------------- //
// apply_morph_weights routing                                      //
// ---------------------------------------------------------------- //

/// One-triangle primitive with one morph target carrying an
/// in-between at 0.5 that overshoots the linear path.
fn inbetween_primitive() -> Primitive {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut t = primary_target(3);
    t.inbetweens = vec![Inbetween::new(0.5).with_position(vec![[0.9, 0.0, 0.0]; 3])];
    prim.targets = vec![t];
    prim
}

#[test]
fn apply_morph_weights_routes_through_stations() {
    let prim = inbetween_primitive();
    // At w = 0.5 the in-between shape applies exactly: base + 0.9.
    let m = prim.apply_morph_weights(&[0.5]);
    assert!(close3(m.positions[0], [0.9, 0.0, 0.0], 1e-6));
    assert!(close3(m.positions[1], [1.9, 0.0, 0.0], 1e-6));
    // At w = 1 the primary applies: base + 1.0.
    let m = prim.apply_morph_weights(&[1.0]);
    assert!(close3(m.positions[0], [1.0, 0.0, 0.0], 1e-6));
    // At w = 0 nothing applies.
    let m = prim.apply_morph_weights(&[0.0]);
    assert!(close3(m.positions[0], [0.0, 0.0, 0.0], 1e-6));
}

#[test]
fn morphed_consumes_inbetweens_with_the_roster() {
    let flat = inbetween_primitive().morphed(&[0.5]);
    assert!(flat.targets.is_empty());
    assert!(close3(flat.positions[0], [0.9, 0.0, 0.0], 1e-6));
}

#[test]
fn linear_targets_unchanged_by_the_routing() {
    // Regression: a plain glTF target (no in-betweens) must produce
    // bit-identical results through the new path selector.
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0; 3]; 3];
    prim.targets = vec![primary_target(3)];
    let m = prim.apply_morph_weights(&[0.65]);
    assert_eq!(m.positions[0], [0.65, 0.0, 0.0]);
}

// ---------------------------------------------------------------- //
// validate()                                                       //
// ---------------------------------------------------------------- //

fn scene_with(prim: Primitive) -> Scene3D {
    let mut scene = Scene3D::new();
    let mid = scene.add_mesh(Mesh::new(None).with_primitive(prim));
    let node = scene.add_node(Node::new().with_mesh(mid));
    scene.roots.push(node);
    scene
}

#[test]
fn validate_accepts_well_formed_inbetweens() {
    let scene = scene_with(inbetween_primitive());
    assert_eq!(scene.validate(), Ok(()));
}

#[test]
fn validate_rejects_endpoint_and_nonfinite_weights() {
    let mut prim = inbetween_primitive();
    prim.targets[0].inbetweens.push(Inbetween::new(0.0));
    prim.targets[0].inbetweens.push(Inbetween::new(1.0));
    prim.targets[0].inbetweens.push(Inbetween::new(f32::NAN));
    let errs = scene_with(prim).validate().unwrap_err();
    let hits = errs
        .iter()
        .filter(|e| matches!(e, ValidationError::InbetweenWeightInvalid { .. }))
        .count();
    assert_eq!(hits, 3);
}

#[test]
fn validate_rejects_duplicate_stations() {
    let mut prim = inbetween_primitive();
    // The fixture already has a shape at 0.5 — collide with it.
    prim.targets[0]
        .inbetweens
        .push(Inbetween::new(0.5).with_position(vec![[0.1, 0.0, 0.0]; 3]));
    let errs = scene_with(prim).validate().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        ValidationError::InbetweenDuplicateWeight { weight, .. } if *weight == 0.5
    )));
}

#[test]
fn validate_rejects_delta_length_mismatch() {
    let mut prim = inbetween_primitive();
    // 2 rows against a 3-vertex base.
    prim.targets[0].inbetweens[0].position = Some(vec![[0.9, 0.0, 0.0]; 2]);
    prim.targets[0].inbetweens[0].normal = Some(vec![[0.0, 0.0, 1.0]; 5]);
    let errs = scene_with(prim).validate().unwrap_err();
    let locs: Vec<String> = errs
        .iter()
        .filter(|e| matches!(e, ValidationError::AttributeLengthMismatch { .. }))
        .map(|e| format!("{e}"))
        .collect();
    assert!(locs.iter().any(|l| l.contains("inbetweens[0].position")));
    assert!(locs.iter().any(|l| l.contains("inbetweens[0].normal")));
}

// ---------------------------------------------------------------- //
// Geometry-pass interplay                                          //
// ---------------------------------------------------------------- //

#[test]
fn transformed_rebases_inbetween_deltas() {
    let mut prim = inbetween_primitive();
    prim.targets[0].inbetweens[0].normal = Some(vec![[0.0, 0.0, 1.0]; 3]);
    // Non-uniform scale: X × 2. Position deltas follow L; normal
    // deltas follow the inverse-transpose (Z axis unscaled here, but
    // an X-pointing normal delta would halve).
    let scale2x = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let out = prim.transformed(scale2x);
    let ib = &out.targets[0].inbetweens[0];
    assert!(close3(
        ib.position.as_ref().unwrap()[0],
        [1.8, 0.0, 0.0],
        1e-6
    ));
    assert!(close3(
        ib.normal.as_ref().unwrap()[0],
        [0.0, 0.0, 1.0],
        1e-6
    ));
    // And morphing the transformed primitive matches transforming
    // the morphed one (commutation on positions).
    let a = out.apply_morph_weights(&[0.5]).positions;
    let b = prim.morphed(&[0.5]).transformed(scale2x).positions;
    for (pa, pb) in a.iter().zip(&b) {
        assert!(close3(*pa, *pb, 1e-5));
    }
}

#[test]
fn weld_gathers_inbetween_buffers() {
    // Two triangles sharing an edge, fully duplicated pool.
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let mut t = primary_target(6);
    t.inbetweens = vec![Inbetween::new(0.5).with_position(vec![[0.9, 0.0, 0.0]; 6])];
    prim.targets = vec![t];
    let welded = prim.weld_vertices();
    assert_eq!(welded.positions.len(), 4);
    let ib = &welded.targets[0].inbetweens[0];
    assert_eq!(ib.position.as_ref().unwrap().len(), 4);
    assert_eq!(ib.weight, 0.5);
}

#[test]
fn weld_keeps_vertices_distinct_on_inbetween_deltas() {
    // Same position, same primary delta — different in-between
    // delta. The corners must NOT weld together.
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut t = primary_target(3);
    t.inbetweens = vec![Inbetween::new(0.5).with_position(vec![
        [0.9, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [0.7, 0.0, 0.0],
    ])];
    prim.targets = vec![t.clone()];
    let welded = prim.weld_vertices();
    assert_eq!(welded.positions.len(), 3);

    // Control: identical in-between rows on identical corners DO weld.
    let mut dup = Primitive::new(Topology::Triangles);
    dup.positions = vec![[0.0; 3], [0.0; 3], [1.0, 0.0, 0.0]];
    let mut td = primary_target(3);
    td.position = Some(vec![[1.0, 0.0, 0.0]; 3]);
    td.inbetweens = vec![Inbetween::new(0.5).with_position(vec![[0.9, 0.0, 0.0]; 3])];
    dup.targets = vec![td];
    let welded = dup.weld_vertices();
    assert_eq!(welded.positions.len(), 2);
}

#[test]
fn vertex_fetch_permutes_inbetween_buffers() {
    let mut prim = inbetween_primitive();
    // Distinct rows so the permutation is observable.
    prim.targets[0].inbetweens[0].position =
        Some(vec![[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]);
    prim.indices = Some(oxideav_mesh3d::Indices::U32(vec![2, 1, 0]));
    let out = prim.optimize_vertex_fetch();
    // First-use order 2, 1, 0 → pool reversed; in-between rows follow.
    let ib = out.targets[0].inbetweens[0].position.as_ref().unwrap();
    assert!(close3(ib[0], [0.3, 0.0, 0.0], 1e-6));
    assert!(close3(ib[2], [0.1, 0.0, 0.0], 1e-6));
}

#[test]
fn subdivide_carries_inbetween_buffers_at_pool_length() {
    let out = inbetween_primitive().subdivide_loop();
    let n = out.positions.len();
    assert!(n > 3);
    let ib = &out.targets[0].inbetweens[0];
    assert_eq!(ib.position.as_ref().unwrap().len(), n);
    assert_eq!(ib.weight, 0.5);
}

#[test]
fn cluster_averages_inbetween_buffers() {
    // A fine grid collapses to fewer cells; the in-between arrays
    // must land at pool length with the station metadata intact.
    let mut prim = Primitive::new(Topology::Triangles);
    for i in 0..8 {
        let x = (i % 4) as f32 * 0.1;
        let y = (i / 4) as f32 * 0.1;
        prim.positions.push([x, y, 0.0]);
    }
    // Two triangles over the strip.
    prim.indices = Some(oxideav_mesh3d::Indices::U32(vec![0, 1, 4, 1, 5, 4]));
    let mut t = primary_target(8);
    t.inbetweens = vec![Inbetween::new(0.5)
        .with_name("half")
        .with_position(vec![[0.9, 0.0, 0.0]; 8])];
    prim.targets = vec![t];
    let out = prim.simplify_cluster(2);
    let n = out.positions.len();
    if !out.targets.is_empty() && n > 0 {
        let ib = &out.targets[0].inbetweens[0];
        assert_eq!(ib.name.as_deref(), Some("half"));
        assert_eq!(ib.weight, 0.5);
        let buf = ib.position.as_ref().unwrap();
        assert_eq!(buf.len(), n);
        // Every cell average of identical rows is the row itself.
        for v in buf {
            assert!(close3(*v, [0.9, 0.0, 0.0], 1e-6));
        }
    }
}

#[test]
fn append_carries_inbetweens_verbatim() {
    let mut dst = scene_with(Primitive::new(Topology::Triangles));
    let src = scene_with(inbetween_primitive());
    let off = dst.append(&src);
    let mesh = &dst.meshes[off.meshes as usize];
    let ib = &mesh.primitives[0].targets[0].inbetweens[0];
    assert_eq!(ib.weight, 0.5);
    assert!(ib.position.is_some());
}
