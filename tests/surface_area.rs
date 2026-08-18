//! Tests for `Primitive::surface_area`, `Mesh::surface_area`, and
//! `Scene3D::surface_area`.
//!
//! Every closed-form expected value below is derived from elementary
//! geometry — half the cross-product magnitude is one triangle's area,
//! summed over the de-stripped triangle list (matching
//! [`Primitive::triangle_indices`]).

use oxideav_mesh3d::{Indices, Mesh, MorphTarget, Primitive, Scene3D, Topology};

/// One unit right-triangle (legs of length 1 on the x and y axes)
/// has area exactly 0.5.
#[test]
fn unit_right_triangle() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert!((p.surface_area() - 0.5).abs() < 1e-12);
}

/// Unit square (2 right-triangles) has area exactly 1.0.
#[test]
fn unit_square_two_triangles() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    assert!((p.surface_area() - 1.0).abs() < 1e-12);
}

/// A unit square stitched as an indexed quad (4 verts, 6 indices) has
/// the same area as the non-indexed two-triangle version.
#[test]
fn unit_square_indexed() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
    assert!((p.surface_area() - 1.0).abs() < 1e-12);
}

/// A triangle tilted into 3-space (not axis-aligned) still gets its
/// true (slant) area — `√3 / 4` for an equilateral triangle of side 1.
#[test]
fn equilateral_triangle_unit_side() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, (3.0_f32).sqrt() / 2.0, 0.0],
    ];
    let expected = (3.0_f64).sqrt() / 4.0;
    assert!((p.surface_area() - expected).abs() < 1e-7);
}

/// Reversed winding (clockwise instead of CCW) returns the same
/// **unsigned** area — half the cross-product *magnitude*, not the
/// signed area.
#[test]
fn area_is_unsigned_winding_invariant() {
    let mut ccw = Primitive::new(Topology::Triangles);
    ccw.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut cw = Primitive::new(Topology::Triangles);
    cw.positions = vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    assert!((ccw.surface_area() - cw.surface_area()).abs() < 1e-12);
    assert!((ccw.surface_area() - 0.5).abs() < 1e-12);
}

/// An axis-aligned unit cube has surface area exactly 6.0 (six
/// 1×1 faces, each two triangles).
#[test]
fn unit_cube_surface_area() {
    // 12 triangles wound CCW seen from outside.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // +Z face (top)
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        // -Z face (bottom)
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        // +X face (right)
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        // -X face (left)
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        // +Y face (back)
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // -Y face (front)
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    assert!((p.surface_area() - 6.0).abs() < 1e-10);
}

/// Empty primitive — no positions, no triangles — returns 0.0.
#[test]
fn empty_primitive_zero() {
    let p = Primitive::new(Topology::Triangles);
    assert_eq!(p.surface_area(), 0.0);
}

/// A trailing 1–2 vertices that don't complete a triangle are dropped
/// (the triangle_indices contract), so the area covers only complete
/// triples.
#[test]
fn incomplete_trailing_vertices_dropped() {
    let mut p = Primitive::new(Topology::Triangles);
    // One full triangle (area 0.5) + two leftover vertices.
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
    ];
    assert!((p.surface_area() - 0.5).abs() < 1e-12);
}

/// Degenerate triangle (three collinear corners) contributes zero,
/// not a NaN — matches the `degenerate_triangles` contract.
#[test]
fn degenerate_triangle_contributes_zero() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // Collinear along x.
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ];
    assert_eq!(p.surface_area(), 0.0);
}

/// Coincident corners (all three at the same point) → zero.
#[test]
fn coincident_corners_zero() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[1.0, 1.0, 1.0]; 3];
    assert_eq!(p.surface_area(), 0.0);
}

/// A degenerate triangle mixed with valid triangles still gives the
/// correct total — the bad one drops out silently.
#[test]
fn one_degenerate_among_valid() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // Valid triangle (area 0.5).
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        // Degenerate (collinear).
        [0.0, 0.0, 5.0],
        [1.0, 0.0, 5.0],
        [2.0, 0.0, 5.0],
        // Valid triangle (area 0.5).
        [0.0, 0.0, 9.0],
        [1.0, 0.0, 9.0],
        [0.0, 1.0, 9.0],
    ];
    assert!((p.surface_area() - 1.0).abs() < 1e-12);
}

/// NaN-bearing positions don't poison the sum — the affected triangle
/// is skipped (its cross product is non-finite).
#[test]
fn nan_face_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        // NaN corner — this triangle is unevaluable.
        [f32::NAN, 0.0, 0.0],
        [1.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
    ];
    let a = p.surface_area();
    assert!(a.is_finite());
    assert!((a - 0.5).abs() < 1e-12);
}

/// Infinity-bearing positions don't poison the sum either — Inf − Inf
/// = NaN in the edge difference, and the NaN guard drops the face.
#[test]
fn inf_face_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
        [f32::INFINITY, 1.0, 0.0],
        [f32::INFINITY, 0.0, 1.0],
    ];
    let a = p.surface_area();
    assert!(a.is_finite());
    assert!((a - 0.5).abs() < 1e-12);
}

/// Out-of-range index dereference (malformed primitive) — the bad
/// face is skipped, the remaining face contributes its real area.
#[test]
fn out_of_range_index_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 99]));
    let a = p.surface_area();
    assert!(a.is_finite());
    assert!((a - 0.5).abs() < 1e-12);
}

/// TriangleStrip alternating-winding rule is honoured — area is the
/// same as the equivalent flat-list version.
#[test]
fn triangle_strip_matches_list() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
    ];
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = positions.clone();

    let list = strip.to_triangle_list();
    let mut list_p = Primitive::new(Topology::Triangles);
    list_p.positions = positions;
    list_p.indices = list.indices;
    assert!((strip.surface_area() - list_p.surface_area()).abs() < 1e-10);
}

/// TriangleFan with a square (4 vertices) on the +XY plane has the
/// same area (1.0) as the two-triangle list.
#[test]
fn triangle_fan_unit_square() {
    let mut fan = Primitive::new(Topology::TriangleFan);
    fan.positions = vec![
        // anchor
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    // Two triangles: (0,1,2) + (0,2,3) — area 0.5 + 0.5 = 1.0.
    assert!((fan.surface_area() - 1.0).abs() < 1e-12);
}

/// Lines / LineStrip / LineLoop / Points all return 0.0 — they have
/// no surface.
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
        assert_eq!(p.surface_area(), 0.0, "topology {:?} should be 0.0", top);
    }
}

/// A million-triangle stress test: each triangle has area 0.5, so the
/// total should be exactly 500_000.0 (no f32 drift — accumulator is
/// `f64`).
#[test]
fn large_count_no_drift() {
    let n_tris = 1_000_000;
    let mut positions = Vec::with_capacity(n_tris * 3);
    for _ in 0..n_tris {
        positions.push([0.0, 0.0, 0.0]);
        positions.push([1.0, 0.0, 0.0]);
        positions.push([0.0, 1.0, 0.0]);
    }
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = positions;
    let a = p.surface_area();
    assert!((a - 500_000.0).abs() < 1e-6, "got {}", a);
}

/// Morph deltas are not folded in (per spec — `bounding_box` already
/// excludes them; `surface_area` is the *base* attribute reduction).
/// A morph target that would, if applied, double the area, leaves
/// `surface_area` unchanged.
#[test]
fn morph_targets_ignored() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.targets.push(MorphTarget::with_deltas(
        Some(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        None,
        None,
    ));
    assert!((p.surface_area() - 0.5).abs() < 1e-12);
}

/// Mesh::surface_area sums every primitive.
#[test]
fn mesh_surface_area_sum() {
    let mut p1 = Primitive::new(Topology::Triangles);
    p1.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [0.0, 2.0, 5.0]];
    let mesh = Mesh::new(Some("two-prims".to_owned()))
        .with_primitive(p1)
        .with_primitive(p2);
    assert!((mesh.surface_area() - (0.5 + 2.0)).abs() < 1e-12);
}

/// A mesh whose only primitive is a line strip has zero surface area.
#[test]
fn mesh_lines_only_zero() {
    let mut p = Primitive::new(Topology::LineStrip);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let mesh = Mesh::new(None).with_primitive(p);
    assert_eq!(mesh.surface_area(), 0.0);
}

/// Empty mesh (no primitives) returns 0.0.
#[test]
fn empty_mesh_zero() {
    let mesh = Mesh::new(None);
    assert_eq!(mesh.surface_area(), 0.0);
}

/// Scene3D::surface_area sums every mesh.
#[test]
fn scene_surface_area_sum() {
    let mut scene = Scene3D::new();

    let mut p1 = Primitive::new(Topology::Triangles);
    p1.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    scene.meshes.push(Mesh::new(None).with_primitive(p1));

    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
    scene.meshes.push(Mesh::new(None).with_primitive(p2));

    // 0.5 + 6.0 = 6.5
    assert!((scene.surface_area() - 6.5).abs() < 1e-12);
}

/// Empty scene → 0.0.
#[test]
fn empty_scene_zero() {
    let scene = Scene3D::new();
    assert_eq!(scene.surface_area(), 0.0);
}

/// Same mesh instanced by two nodes is **not** double-counted —
/// `Scene3D::surface_area` walks meshes once, not node instances.
/// (Transform-aware totals require walking nodes; see the doc note.)
#[test]
fn instanced_mesh_counted_once() {
    let mut scene = Scene3D::new();

    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    scene.meshes.push(Mesh::new(None).with_primitive(p));

    // Two node instances of the same mesh — but Scene3D::surface_area
    // sums meshes, not node instances, so we get 0.5 not 1.0.
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

    assert!((scene.surface_area() - 0.5).abs() < 1e-12);
}

/// Scaling all positions by k scales the area by k² — this is the
/// dimensional check (area is square units, not linear).
#[test]
fn scaling_squares_the_area() {
    let mut base = Primitive::new(Topology::Triangles);
    base.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let base_a = base.surface_area();

    let mut scaled = Primitive::new(Topology::Triangles);
    let k = 3.0_f32;
    scaled.positions = vec![[0.0, 0.0, 0.0], [k, 0.0, 0.0], [0.0, k, 0.0]];
    let scaled_a = scaled.surface_area();

    let k = k as f64;
    assert!(((scaled_a / base_a) - (k * k)).abs() < 1e-12);
}

/// Rotating a triangle into a non-axis-aligned orientation preserves
/// its area (rotation is an isometry). A 30° rotation around z of the
/// unit right-triangle leaves area = 0.5.
#[test]
fn rotation_preserves_area() {
    let theta: f32 = std::f32::consts::FRAC_PI_6;
    let (c, s) = (theta.cos(), theta.sin());
    let rot = |p: [f32; 3]| [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]];

    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        rot([0.0, 0.0, 0.0]),
        rot([1.0, 0.0, 0.0]),
        rot([0.0, 1.0, 0.0]),
    ];
    assert!((p.surface_area() - 0.5).abs() < 1e-6);
}

/// Translation preserves area (edges are the same).
#[test]
fn translation_preserves_area() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [100.0, 200.0, 300.0],
        [101.0, 200.0, 300.0],
        [100.0, 201.0, 300.0],
    ];
    assert!((p.surface_area() - 0.5).abs() < 1e-7);
}

/// `surface_area` on a `to_triangle_list`-normalised primitive matches
/// the original strip — connectivity normalisation is area-preserving.
#[test]
fn destrip_preserves_area() {
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let list = strip.to_triangle_list();
    assert!((strip.surface_area() - list.surface_area()).abs() < 1e-10);
}

/// `surface_area` after `weld_vertices` matches the soup — welding
/// preserves geometry, only collapses duplicates.
#[test]
fn weld_preserves_area() {
    let mut soup = Primitive::new(Topology::Triangles);
    soup.positions = vec![
        // First triangle (area 0.5).
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        // Adjacent triangle sharing the (1,0,0)–(0,1,0) edge.
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let welded = soup.weld_vertices();
    assert!((soup.surface_area() - welded.surface_area()).abs() < 1e-12);
    // Sanity: two unit right-triangles → 1.0.
    assert!((soup.surface_area() - 1.0).abs() < 1e-12);
}
