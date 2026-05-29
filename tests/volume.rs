//! Tests for `Primitive::signed_volume` / `Primitive::volume`,
//! `Mesh::signed_volume` / `Mesh::volume`, and
//! `Scene3D::signed_volume` / `Scene3D::volume`.
//!
//! Every closed-form expected value below is derived from elementary
//! geometry — the divergence-theorem reduction
//! `V = (1/6) Σ (P_a · (P_b × P_c))` evaluated on a closed
//! triangle tessellation matches the textbook volume of the solid the
//! tessellation encloses.

use oxideav_mesh3d::{Indices, Mesh, MorphTarget, Primitive, Scene3D, Topology};

/// Build the canonical axis-aligned unit cube (1×1×1) on the +X+Y+Z
/// octant, every facet wound CCW seen from outside. Surface area is
/// 6.0 (already tested) and signed volume is 1.0.
fn unit_cube_ccw() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // +Z face (top, normal +Z) — CCW seen from +Z.
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        // -Z face (bottom, normal -Z) — CCW seen from -Z.
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        // +X face (right, normal +X) — CCW seen from +X.
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        // -X face (left, normal -X) — CCW seen from -X.
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        // +Y face (back, normal +Y) — CCW seen from +Y.
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // -Y face (front, normal -Y) — CCW seen from -Y.
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    p
}

/// Same cube with every facet's winding reversed (CW seen from
/// outside) — its signed volume should be -1.0 and unsigned 1.0.
fn unit_cube_cw() -> Primitive {
    let mut p = unit_cube_ccw();
    // Swap last two corners of every triangle to reverse winding.
    for tri in p.positions.chunks_exact_mut(3) {
        tri.swap(1, 2);
    }
    p
}

// ── Primitive::signed_volume — closed-form expected values ─────────

/// The canonical axis-aligned unit cube has signed volume 1.0.
#[test]
fn unit_cube_volume_one() {
    let p = unit_cube_ccw();
    assert!(
        (p.signed_volume() - 1.0).abs() < 1e-10,
        "got {}",
        p.signed_volume()
    );
    assert!((p.volume() - 1.0).abs() < 1e-10);
}

/// An axis-aligned 2×3×4 box has signed volume 24.0.
#[test]
fn axis_aligned_box_volume() {
    let mut p = unit_cube_ccw();
    // Stretch x by 2, y by 3, z by 4.
    for v in p.positions.iter_mut() {
        v[0] *= 2.0;
        v[1] *= 3.0;
        v[2] *= 4.0;
    }
    assert!(
        (p.signed_volume() - 24.0).abs() < 1e-9,
        "got {}",
        p.signed_volume()
    );
}

/// A regular tetrahedron with corner at the origin and three unit
/// edges along x, y, z has volume = 1/6 (the canonical tetrahedron
/// constant — base × height / 3 for an axis-aligned tetra is
/// (1/2)·1·1·1 / 3 = 1/6).
#[test]
fn unit_tetrahedron_volume() {
    let mut p = Primitive::new(Topology::Triangles);
    // Corners: O=(0,0,0), A=(1,0,0), B=(0,1,0), C=(0,0,1).
    // Four faces, each wound CCW seen from outside:
    p.positions = vec![
        // Bottom: O, B, A (normal -Z).
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        // Front: O, A, C (normal -Y).
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        // Left: O, C, B (normal -X).
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        // Slanted face: A, B, C (normal in (+,+,+)).
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let v = p.signed_volume();
    assert!((v - 1.0 / 6.0).abs() < 1e-10, "got {}", v);
}

/// Inside-out winding (CW seen from outside) flips the sign but not the
/// magnitude — the unsigned `volume()` is winding-invariant.
#[test]
fn reversed_winding_flips_sign() {
    let ccw = unit_cube_ccw();
    let cw = unit_cube_cw();
    assert!((ccw.signed_volume() - 1.0).abs() < 1e-10);
    assert!((cw.signed_volume() + 1.0).abs() < 1e-10);
    // Magnitudes identical.
    assert!((ccw.volume() - cw.volume()).abs() < 1e-10);
    assert!((ccw.volume() - 1.0).abs() < 1e-10);
}

/// Translation by a constant vector should not change the signed
/// volume of a closed surface (the divergence-theorem cancellation
/// keeps origin out of the answer).
#[test]
fn translation_invariant_for_closed_surface() {
    let base = unit_cube_ccw();
    let mut shifted = unit_cube_ccw();
    for v in shifted.positions.iter_mut() {
        v[0] += 100.0;
        v[1] -= 50.0;
        v[2] += 7.5;
    }
    let b = base.signed_volume();
    let s = shifted.signed_volume();
    // The `O(n)` summation loses some bits of precision once the
    // operands are 100s of times larger than the result, so this
    // tolerance is wider than the 1e-10 used elsewhere.
    assert!((b - s).abs() < 1e-4, "base {} shifted {}", b, s);
}

/// Scaling every position by `k` scales the volume by `k³` — this is
/// the dimensional check (volume is cubic units, not linear).
#[test]
fn scaling_cubes_the_volume() {
    let base = unit_cube_ccw();
    let base_v = base.signed_volume();

    let mut scaled = unit_cube_ccw();
    let k = 3.0_f32;
    for v in scaled.positions.iter_mut() {
        v[0] *= k;
        v[1] *= k;
        v[2] *= k;
    }
    let scaled_v = scaled.signed_volume();

    let k = k as f64;
    assert!(
        ((scaled_v / base_v) - (k * k * k)).abs() < 1e-9,
        "ratio {}",
        scaled_v / base_v
    );
}

/// A single isolated triangle (open surface) reports a non-zero
/// "signed volume" arithmetically, but that's not a physical volume —
/// the doc says it's only meaningful for closed manifolds. Pin the
/// arithmetic anyway so a regression flips it from -1/6 (the tet to
/// the origin) to something silently wrong.
#[test]
fn single_triangle_arithmetic_pin() {
    let mut p = Primitive::new(Topology::Triangles);
    // (1,0,0) (0,1,0) (0,0,1) — `P_a · (P_b × P_c)` = 1.
    p.positions = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // V_tri = 1/6.
    assert!((p.signed_volume() - 1.0 / 6.0).abs() < 1e-12);
}

// ── Edge cases — empties, degenerates, NaN, Inf, out-of-range ──────

/// Empty primitive — no positions, no triangles — returns 0.0.
#[test]
fn empty_primitive_zero() {
    let p = Primitive::new(Topology::Triangles);
    assert_eq!(p.signed_volume(), 0.0);
    assert_eq!(p.volume(), 0.0);
}

/// Incomplete trailing 1–2 vertices that don't complete a triangle
/// are dropped (`triangle_indices` contract).
#[test]
fn incomplete_trailing_vertices_dropped() {
    let mut p = Primitive::new(Topology::Triangles);
    // The (1,0,0)(0,1,0)(0,0,1) triangle alone, plus two leftover
    // vertices.
    p.positions = vec![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
    ];
    assert!((p.signed_volume() - 1.0 / 6.0).abs() < 1e-12);
}

/// Degenerate triangle (collinear corners) contributes zero, not NaN.
#[test]
fn degenerate_triangle_contributes_zero() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    assert_eq!(p.signed_volume(), 0.0);
    assert_eq!(p.volume(), 0.0);
}

/// Coincident corners (all three at the same point) → zero.
#[test]
fn coincident_corners_zero() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[1.0, 1.0, 1.0]; 3];
    assert_eq!(p.signed_volume(), 0.0);
}

/// A degenerate triangle mixed with a closed cube still gives the
/// cube's correct volume — the bad triangle drops out silently.
#[test]
fn one_degenerate_among_valid() {
    let mut p = unit_cube_ccw();
    // Append a degenerate triangle far from the cube.
    p.positions
        .extend_from_slice(&[[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [12.0, 10.0, 10.0]]);
    assert!((p.signed_volume() - 1.0).abs() < 1e-9);
}

/// NaN-bearing positions don't poison the sum — the affected
/// triangle is skipped.
#[test]
fn nan_face_skipped() {
    let mut p = unit_cube_ccw();
    p.positions
        .extend_from_slice(&[[f32::NAN, 0.0, 0.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]);
    let v = p.signed_volume();
    assert!(v.is_finite());
    assert!((v - 1.0).abs() < 1e-9);
}

/// Infinity-bearing positions don't poison the sum either.
#[test]
fn inf_face_skipped() {
    let mut p = unit_cube_ccw();
    p.positions.extend_from_slice(&[
        [f32::INFINITY, 0.0, 0.0],
        [f32::INFINITY, 1.0, 0.0],
        [f32::INFINITY, 0.0, 1.0],
    ]);
    let v = p.signed_volume();
    assert!(v.is_finite());
    assert!((v - 1.0).abs() < 1e-9);
}

/// Out-of-range index dereference (malformed primitive) — the bad
/// face is skipped, valid faces still contribute.
#[test]
fn out_of_range_index_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 99]));
    let v = p.signed_volume();
    assert!(v.is_finite());
    assert!((v - 1.0 / 6.0).abs() < 1e-12);
}

// ── Topology integration ──────────────────────────────────────────

/// Non-triangle topologies (lines/points) report zero volume — they
/// have no surface, so no enclosed volume.
#[test]
fn non_triangle_topologies_zero() {
    for top in [
        Topology::Lines,
        Topology::LineStrip,
        Topology::LineLoop,
        Topology::Points,
    ] {
        let mut p = Primitive::new(top);
        p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(p.signed_volume(), 0.0, "topology {:?} should be 0.0", top);
    }
}

/// TriangleStrip alternating-winding rule is honoured — volume of a
/// strip-encoded shape matches the flat-list version.
#[test]
fn triangle_strip_matches_list() {
    // Make a 2-triangle "strip" with the same first 4 vertices used
    // for triangle_strip alternation. The geometric content matches
    // its de-stripped (`to_triangle_list`) form, so signed_volume
    // must agree.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = positions.clone();
    let list = strip.to_triangle_list();
    assert!(
        (strip.signed_volume() - list.signed_volume()).abs() < 1e-12,
        "strip {} list {}",
        strip.signed_volume(),
        list.signed_volume()
    );
}

/// TriangleFan-encoded shape matches its `to_triangle_list` version.
#[test]
fn triangle_fan_matches_list() {
    let mut fan = Primitive::new(Topology::TriangleFan);
    fan.positions = vec![
        // anchor
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let list = fan.to_triangle_list();
    assert!((fan.signed_volume() - list.signed_volume()).abs() < 1e-12);
}

/// `signed_volume` on a `to_triangle_list`-normalised closed cube
/// matches the original — connectivity normalisation is
/// volume-preserving.
#[test]
fn destrip_preserves_volume() {
    let cube = unit_cube_ccw();
    let list = cube.to_triangle_list();
    assert!((cube.signed_volume() - list.signed_volume()).abs() < 1e-10);
}

/// `signed_volume` after `weld_vertices` matches the soup — welding
/// preserves geometry, only collapses duplicates.
#[test]
fn weld_preserves_volume() {
    let cube = unit_cube_ccw();
    let welded = cube.weld_vertices();
    assert!((cube.signed_volume() - welded.signed_volume()).abs() < 1e-10);
    assert!((cube.signed_volume() - 1.0).abs() < 1e-10);
}

// ── Indexed-buffer parity ──────────────────────────────────────────

/// Indexed cube (8 verts + 36 indices) produces the same volume as
/// the non-indexed soup version.
#[test]
fn indexed_cube_matches_soup() {
    let soup = unit_cube_ccw();
    // 8 canonical corners, indexed.
    let mut indexed = Primitive::new(Topology::Triangles);
    indexed.positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [1.0, 1.0, 0.0], // 2
        [0.0, 1.0, 0.0], // 3
        [0.0, 0.0, 1.0], // 4
        [1.0, 0.0, 1.0], // 5
        [1.0, 1.0, 1.0], // 6
        [0.0, 1.0, 1.0], // 7
    ];
    indexed.indices = Some(Indices::U16(vec![
        // +Z (top)
        4, 5, 6, 4, 6, 7, // -Z (bottom)
        3, 2, 1, 3, 1, 0, // +X (right)
        1, 2, 6, 1, 6, 5, // -X (left)
        4, 7, 3, 4, 3, 0, // +Y (back)
        2, 3, 7, 2, 7, 6, // -Y (front)
        0, 1, 5, 0, 5, 4,
    ]));
    assert!((soup.signed_volume() - indexed.signed_volume()).abs() < 1e-10);
    assert!((indexed.signed_volume() - 1.0).abs() < 1e-10);
}

// ── Stress / precision ──────────────────────────────────────────

/// Million-tetrahedron arithmetic stress: each tet contributes
/// `V_tri = 1/6` of signed volume, so the total should sum to a clean
/// `500_000.0 / 3.0` (no f32 drift — accumulator is `f64`).
#[test]
fn large_count_no_drift() {
    let n_tris = 1_000_000;
    let mut positions = Vec::with_capacity(n_tris * 3);
    for _ in 0..n_tris {
        // Tetra-pointing triangle with unit basis vectors.
        positions.push([1.0, 0.0, 0.0]);
        positions.push([0.0, 1.0, 0.0]);
        positions.push([0.0, 0.0, 1.0]);
    }
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = positions;
    let v = p.signed_volume();
    let expected = (n_tris as f64) / 6.0; // 166_666.666…
    assert!(
        (v - expected).abs() < 1e-6,
        "got {} expected {}",
        v,
        expected
    );
}

/// Morph deltas are not folded in (per spec — `signed_volume` is the
/// *base* attribute reduction).
#[test]
fn morph_targets_ignored() {
    let mut p = unit_cube_ccw();
    // Add a morph target that, if applied, would inflate every corner
    // outward — the base volume should still be 1.0.
    let n = p.positions.len();
    p.targets.push(MorphTarget {
        position: Some(vec![[1.0, 1.0, 1.0]; n]),
        normal: None,
        tangent: None,
    });
    assert!((p.signed_volume() - 1.0).abs() < 1e-10);
}

// ── Mesh aggregation ──────────────────────────────────────────────

/// `Mesh::signed_volume` sums every primitive's contribution.
#[test]
fn mesh_volume_sum() {
    // Two cubes side by side — total volume 2.0.
    let p1 = unit_cube_ccw();
    let mut p2 = unit_cube_ccw();
    for v in p2.positions.iter_mut() {
        v[0] += 5.0;
    }
    let mesh = Mesh::new(Some("two-cubes".to_owned()))
        .with_primitive(p1)
        .with_primitive(p2);
    assert!((mesh.signed_volume() - 2.0).abs() < 1e-9);
    assert!((mesh.volume() - 2.0).abs() < 1e-9);
}

/// A mesh whose only primitive is a line strip has zero volume.
#[test]
fn mesh_lines_only_zero() {
    let mut p = Primitive::new(Topology::LineStrip);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let mesh = Mesh::new(None).with_primitive(p);
    assert_eq!(mesh.signed_volume(), 0.0);
    assert_eq!(mesh.volume(), 0.0);
}

/// Empty mesh (no primitives) → 0.0.
#[test]
fn empty_mesh_zero() {
    let mesh = Mesh::new(None);
    assert_eq!(mesh.signed_volume(), 0.0);
    assert_eq!(mesh.volume(), 0.0);
}

/// Two primitives with opposite-sign signed volumes: `Mesh::volume`
/// reports `|Σ signed|`, so they cancel — this is the documented
/// "single-shell" assumption. Pin it so a future change to
/// "sum-of-abs" reduction flags here.
#[test]
fn mesh_volume_cancels_opposite_winding() {
    let ccw = unit_cube_ccw(); // +1
    let cw = unit_cube_cw(); // -1
    let mesh = Mesh::new(None).with_primitive(ccw).with_primitive(cw);
    assert!(mesh.signed_volume().abs() < 1e-9);
    assert!(mesh.volume().abs() < 1e-9);
}

// ── Scene aggregation ──────────────────────────────────────────────

/// `Scene3D::signed_volume` sums every mesh's contribution.
#[test]
fn scene_volume_sum() {
    let mut scene = Scene3D::new();

    // Cube of volume 1.0.
    scene
        .meshes
        .push(Mesh::new(None).with_primitive(unit_cube_ccw()));

    // 2×3×4 box (volume 24.0).
    let mut box234 = unit_cube_ccw();
    for v in box234.positions.iter_mut() {
        v[0] *= 2.0;
        v[1] *= 3.0;
        v[2] *= 4.0;
    }
    scene.meshes.push(Mesh::new(None).with_primitive(box234));

    // Total 1.0 + 24.0 = 25.0.
    assert!((scene.signed_volume() - 25.0).abs() < 1e-9);
    assert!((scene.volume() - 25.0).abs() < 1e-9);
}

/// Empty scene → 0.0.
#[test]
fn empty_scene_zero() {
    let scene = Scene3D::new();
    assert_eq!(scene.signed_volume(), 0.0);
    assert_eq!(scene.volume(), 0.0);
}

/// Same mesh instanced by two nodes is **not** double-counted —
/// `Scene3D::signed_volume` walks meshes once, not node instances.
#[test]
fn instanced_mesh_counted_once() {
    let mut scene = Scene3D::new();
    scene
        .meshes
        .push(Mesh::new(None).with_primitive(unit_cube_ccw()));

    let mesh_id = oxideav_mesh3d::MeshId(0);
    let node1 = oxideav_mesh3d::Node {
        mesh: Some(mesh_id),
        ..oxideav_mesh3d::Node::default()
    };
    let node2 = oxideav_mesh3d::Node {
        mesh: Some(mesh_id),
        ..oxideav_mesh3d::Node::default()
    };
    scene.nodes.push(node1);
    scene.nodes.push(node2);

    // One cube, not two.
    assert!((scene.signed_volume() - 1.0).abs() < 1e-9);
}
