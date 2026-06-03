//! Tests for `Primitive::surface_centroid`, `Mesh::surface_centroid`,
//! and `Scene3D::surface_centroid`.
//!
//! Every closed-form expected value below is derived from elementary
//! geometry — the area-weighted average of the per-triangle centroids
//! `(P_a + P_b + P_c) / 3` (Marsden & Tromba, *Vector Calculus*,
//! chapter on triangle integrals — a linear function integrated over
//! a triangle equals `area * value_at_centroid`).

use oxideav_mesh3d::{Indices, Mesh, MorphTarget, Primitive, Scene3D, Topology};

fn approx_eq3(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
}

/// A single triangle's surface centroid is the average of its three
/// corners.
#[test]
fn single_triangle_centroid_is_corner_mean() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 6.0, 0.0]];
    let c = p.surface_centroid().expect("triangle has area");
    assert!(approx_eq3(c, [1.0, 2.0, 0.0], 1e-12), "got {:?}", c);
}

/// Translating every vertex by a constant moves the centroid by the
/// same vector.
#[test]
fn centroid_translation_equivariant() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let c0 = p.surface_centroid().unwrap();
    let mut q = Primitive::new(Topology::Triangles);
    q.positions = vec![[10.0, 20.0, 30.0], [11.0, 20.0, 30.0], [10.0, 21.0, 30.0]];
    let c1 = q.surface_centroid().unwrap();
    assert!(
        approx_eq3(c1, [c0[0] + 10.0, c0[1] + 20.0, c0[2] + 30.0], 1e-12),
        "translated centroid mismatch: c0={:?} c1={:?}",
        c0,
        c1
    );
}

/// Unit square (two right-triangles) centroid sits at the square's
/// geometric centre.
#[test]
fn unit_square_centroid_is_centre() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let c = p.surface_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.0], 1e-12), "got {:?}", c);
}

/// Same centred unit square via an index buffer should produce the
/// same centroid as the explicit version above.
#[test]
fn indexed_unit_square_matches_unindexed() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 2, 3]));
    let c = p.surface_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.0], 1e-12));
}

/// A non-symmetric triangulation should still produce the same
/// continuous centroid because the area-weighted recombination is
/// invariant under retriangulation of the same patch. Two triangles
/// covering a 2x1 rectangle along the long axis have centroid (1, 0.5).
#[test]
fn rectangular_two_triangles_centroid() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let c = p.surface_centroid().unwrap();
    assert!(approx_eq3(c, [1.0, 0.5, 0.0], 1e-12), "got {:?}", c);
}

/// Subdividing a triangle barycentrically into three sub-triangles
/// must produce the same centroid as the original — proves the
/// area-weighted combination is invariant under retriangulation.
#[test]
fn centroid_invariant_under_barycentric_subdivision() {
    let mut original = Primitive::new(Topology::Triangles);
    original.positions = vec![[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 9.0, 0.0]];
    let c_full = original.surface_centroid().unwrap();

    // Subdivide by the centroid into three sub-triangles.
    let mid = [2.0, 3.0, 0.0];
    let mut sub = Primitive::new(Topology::Triangles);
    sub.positions = vec![
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        mid,
        [6.0, 0.0, 0.0],
        [0.0, 9.0, 0.0],
        mid,
        [0.0, 9.0, 0.0],
        [0.0, 0.0, 0.0],
        mid,
    ];
    let c_sub = sub.surface_centroid().unwrap();
    assert!(
        approx_eq3(c_full, c_sub, 1e-12),
        "{:?} vs {:?}",
        c_full,
        c_sub
    );
}

/// Empty positions buffer → `None`.
#[test]
fn empty_primitive_returns_none() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.surface_centroid().is_none());
}

/// A primitive with three collinear vertices contributes zero area and
/// returns `None`.
#[test]
fn degenerate_triangle_returns_none() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    assert!(p.surface_centroid().is_none());
}

/// Non-triangle topologies (lines/points) return `None`.
#[test]
fn lines_topology_returns_none() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(p.surface_centroid().is_none());
}

#[test]
fn points_topology_returns_none() {
    let mut p = Primitive::new(Topology::Points);
    p.positions = vec![[1.0, 2.0, 3.0]];
    assert!(p.surface_centroid().is_none());
}

#[test]
fn line_strip_topology_returns_none() {
    let mut p = Primitive::new(Topology::LineStrip);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
    assert!(p.surface_centroid().is_none());
}

/// Out-of-range index entries are skipped — the surviving triangle
/// dictates the centroid.
#[test]
fn out_of_range_indices_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    // Two triangles in the index list — one references vertex 99,
    // which doesn't exist; the other is the valid unit right-triangle.
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 99]));
    let c = p.surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// A NaN coordinate on one triangle disqualifies that triangle; the
/// rest still contribute.
#[test]
fn nan_coord_triangle_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [f32::NAN, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    let c = p.surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// Triangle strip topology (alternating winding) should produce a
/// centroid identical to the equivalent triangle-list expansion.
#[test]
fn triangle_strip_matches_list_expansion() {
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let c_strip = strip.surface_centroid().unwrap();

    let mut list = Primitive::new(Topology::Triangles);
    list.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        // Second triangle from the strip: the strip's odd-index
        // alternates winding, so triangle (1, 2, 3) in strip order is
        // emitted as (2, 1, 3) for CCW.
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let c_list = list.surface_centroid().unwrap();
    assert!(
        approx_eq3(c_strip, c_list, 1e-12),
        "{:?} vs {:?}",
        c_strip,
        c_list
    );
}

/// Triangle fan: matches the corresponding triangle-list.
#[test]
fn triangle_fan_matches_list_expansion() {
    let mut fan = Primitive::new(Topology::TriangleFan);
    fan.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let c_fan = fan.surface_centroid().unwrap();

    let mut list = Primitive::new(Topology::Triangles);
    list.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let c_list = list.surface_centroid().unwrap();
    assert!(
        approx_eq3(c_fan, c_list, 1e-12),
        "{:?} vs {:?}",
        c_fan,
        c_list
    );
}

/// Axis-aligned unit cube — surface centroid at the cube's centre.
fn unit_cube_ccw() -> Primitive {
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
        // +X face
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        // -X face
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        // +Y face
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        // -Y face
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    p
}

#[test]
fn unit_cube_centroid_is_centre() {
    let p = unit_cube_ccw();
    let c = p.surface_centroid().unwrap();
    assert!(approx_eq3(c, [0.5, 0.5, 0.5], 1e-12), "got {:?}", c);
}

/// Skew a cube into a rectangular box; the centroid follows the
/// translation/scale linearly.
#[test]
fn scaled_cube_centroid() {
    let mut p = unit_cube_ccw();
    for v in &mut p.positions {
        v[0] *= 4.0;
        v[1] *= 2.0;
        v[2] *= 8.0;
    }
    let c = p.surface_centroid().unwrap();
    assert!(approx_eq3(c, [2.0, 1.0, 4.0], 1e-9), "got {:?}", c);
}

/// A primitive whose every triangle is degenerate returns `None`.
#[test]
fn all_degenerate_returns_none() {
    let mut p = Primitive::new(Topology::Triangles);
    // Three coincident triangles, every one a single point.
    p.positions = vec![
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
    ];
    assert!(p.surface_centroid().is_none());
}

/// Empty index buffer (no triangles) returns `None`.
#[test]
fn empty_index_buffer_returns_none() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.indices = Some(Indices::U32(vec![]));
    assert!(p.surface_centroid().is_none());
}

/// Each component of the centroid is finite even when only some
/// triangles are well-formed.
#[test]
fn finite_result_with_partly_corrupt_buffer() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    let c = p.surface_centroid().unwrap();
    assert!(c[0].is_finite() && c[1].is_finite() && c[2].is_finite());
}

/// Tangents, morph targets, and other side-channel fields don't
/// affect the centroid — only `positions` + `triangle_indices()`.
#[test]
fn morph_targets_do_not_affect_centroid() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let c_bare = p.surface_centroid().unwrap();

    p.targets.push(MorphTarget {
        position: Some(vec![[10.0, 10.0, 10.0]; 3]),
        normal: None,
        tangent: None,
    });
    let c_with_targets = p.surface_centroid().unwrap();
    assert!(approx_eq3(c_bare, c_with_targets, 1e-12));
}

// ---- Mesh-level tests ----

/// A mesh holding one primitive returns that primitive's centroid.
#[test]
fn mesh_single_primitive_passthrough() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
    let mesh = Mesh::new(Some("tri".to_owned())).with_primitive(p);
    let c = mesh.surface_centroid().unwrap();
    assert!(approx_eq3(c, [1.0, 1.0, 0.0], 1e-12), "got {:?}", c);
}

/// A mesh with two equal-area primitives — centroid is the midpoint
/// of the two per-primitive centroids.
#[test]
fn mesh_two_equal_area_primitives_midpoint() {
    let mut p1 = Primitive::new(Topology::Triangles);
    p1.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[10.0, 10.0, 0.0], [11.0, 10.0, 0.0], [10.0, 11.0, 0.0]];
    let mesh = Mesh::new(Some("two".to_owned()))
        .with_primitive(p1)
        .with_primitive(p2);
    let c = mesh.surface_centroid().unwrap();
    // Per-primitive centroids: (1/3, 1/3, 0) and (10+1/3, 10+1/3, 0);
    // equal areas → midpoint is (5+1/3, 5+1/3, 0).
    let exp = [5.0 + 1.0 / 3.0, 5.0 + 1.0 / 3.0, 0.0];
    assert!(approx_eq3(c, exp, 1e-12), "got {:?}", c);
}

/// Unequal areas — the larger primitive pulls the centroid toward
/// itself proportionally to its area.
#[test]
fn mesh_unequal_areas_weights_proportionally() {
    // p1: small right-triangle at origin, area 0.5
    let mut p1 = Primitive::new(Topology::Triangles);
    p1.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    // p2: large right-triangle at (10, 10), legs of length 3 → area 4.5
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[10.0, 10.0, 0.0], [13.0, 10.0, 0.0], [10.0, 13.0, 0.0]];
    let mesh = Mesh::new(Some("uneq".to_owned()))
        .with_primitive(p1)
        .with_primitive(p2);
    let c = mesh.surface_centroid().unwrap();

    // p1 centroid (1/3, 1/3, 0) with weight 0.5
    // p2 centroid (11, 11, 0) with weight 4.5
    // sum_area = 5
    let exp_x = (0.5 * (1.0 / 3.0) + 4.5 * 11.0) / 5.0;
    let exp_y = (0.5 * (1.0 / 3.0) + 4.5 * 11.0) / 5.0;
    assert!(approx_eq3(c, [exp_x, exp_y, 0.0], 1e-12), "got {:?}", c);
}

/// Mesh with no primitives returns `None`.
#[test]
fn mesh_no_primitives_returns_none() {
    let mesh = Mesh::new(Some("empty".to_owned()));
    assert!(mesh.surface_centroid().is_none());
}

/// Mesh with one degenerate primitive and one good one — the good one
/// dictates the centroid.
#[test]
fn mesh_degenerate_primitive_skipped() {
    let mut bad = Primitive::new(Topology::Triangles);
    bad.positions = vec![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];
    let mut good = Primitive::new(Topology::Triangles);
    good.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mesh = Mesh::new(Some("mix".to_owned()))
        .with_primitive(bad)
        .with_primitive(good);
    let c = mesh.surface_centroid().unwrap();
    assert!(approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12));
}

/// A mesh-level non-triangle primitive doesn't pull the centroid.
#[test]
fn mesh_lines_primitive_does_not_contribute() {
    let mut lines = Primitive::new(Topology::Lines);
    lines.positions = vec![[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]];
    let mut tri = Primitive::new(Topology::Triangles);
    tri.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mesh = Mesh::new(Some("mix".to_owned()))
        .with_primitive(lines)
        .with_primitive(tri);
    let c = mesh.surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

// ---- Scene3D-level tests ----

#[test]
fn scene_empty_returns_none() {
    let s = Scene3D::new();
    assert!(s.surface_centroid().is_none());
}

#[test]
fn scene_single_mesh_passthrough() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
    let mesh = Mesh::new(Some("tri".to_owned())).with_primitive(p);
    let mut s = Scene3D::new();
    s.add_mesh(mesh);
    let c = s.surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [2.0 / 3.0, 4.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// Two equal-area meshes — centroid is the midpoint.
#[test]
fn scene_two_equal_area_meshes_midpoint() {
    let mut p1 = Primitive::new(Topology::Triangles);
    p1.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mesh1 = Mesh::new(Some("a".to_owned())).with_primitive(p1);

    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]];
    let mesh2 = Mesh::new(Some("b".to_owned())).with_primitive(p2);

    let mut s = Scene3D::new();
    s.add_mesh(mesh1);
    s.add_mesh(mesh2);
    let c = s.surface_centroid().unwrap();
    // p1 centroid (1/3, 1/3, 0), p2 centroid (10+1/3, 1/3, 0), equal
    // areas → midpoint (5+1/3, 1/3, 0).
    assert!(
        approx_eq3(c, [5.0 + 1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// Scene-level walk is mesh-once, not instance-per-node: adding the
/// same mesh under two nodes does NOT change the centroid (because
/// scene_centroid walks the meshes arena, not the node graph).
#[test]
fn scene_centroid_walks_meshes_not_nodes() {
    use oxideav_mesh3d::{Node, Transform};
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mesh = Mesh::new(Some("tri".to_owned())).with_primitive(p);

    let mut s = Scene3D::new();
    let mid = s.add_mesh(mesh);

    // Instance the same mesh once at the origin, once at (100, 100, 0).
    let n0 = s.add_node(Node::new().with_mesh(mid));
    let mut translated = Node::new().with_mesh(mid);
    translated.transform = Transform::Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [100.0, 100.0, 0.0, 1.0],
    ]);
    let n1 = s.add_node(translated);
    s.add_root(n0);
    s.add_root(n1);

    // Scene centroid still equals the mesh's centroid — the second
    // node's transform is invisible to `Scene3D::surface_centroid`.
    let c = s.surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// Scene-level finiteness — every component finite for finite input.
#[test]
fn scene_centroid_components_finite() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    // Note: these three points are colinear in 3D! Make non-colinear:
    p.positions[2] = [7.0, 8.0, 100.0];
    let mesh = Mesh::new(Some("m".to_owned())).with_primitive(p);
    let mut s = Scene3D::new();
    s.add_mesh(mesh);
    let c = s.surface_centroid().unwrap();
    assert!(c[0].is_finite() && c[1].is_finite() && c[2].is_finite());
}

/// Scene with one degenerate mesh and one good one — the good one
/// dictates.
#[test]
fn scene_degenerate_mesh_skipped() {
    let mut bad_p = Primitive::new(Topology::Triangles);
    bad_p.positions = vec![[5.0, 5.0, 5.0]; 3];
    let bad_mesh = Mesh::new(Some("bad".to_owned())).with_primitive(bad_p);

    let mut good_p = Primitive::new(Topology::Triangles);
    good_p.positions = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
    let good_mesh = Mesh::new(Some("good".to_owned())).with_primitive(good_p);

    let mut s = Scene3D::new();
    s.add_mesh(bad_mesh);
    s.add_mesh(good_mesh);
    let c = s.surface_centroid().unwrap();
    assert!(approx_eq3(c, [1.0, 1.0, 0.0], 1e-12), "got {:?}", c);
}

/// Three equal-area meshes forming an equilateral triangle: their
/// area-weighted centroid is the geometric centre of the triangle.
#[test]
fn scene_three_equal_meshes_at_corners() {
    fn unit_tri_at(origin: [f32; 3]) -> Mesh {
        let mut p = Primitive::new(Topology::Triangles);
        p.positions = vec![
            origin,
            [origin[0] + 1.0, origin[1], origin[2]],
            [origin[0], origin[1] + 1.0, origin[2]],
        ];
        Mesh::new(None::<String>).with_primitive(p)
    }
    let mut s = Scene3D::new();
    s.add_mesh(unit_tri_at([0.0, 0.0, 0.0]));
    s.add_mesh(unit_tri_at([10.0, 0.0, 0.0]));
    s.add_mesh(unit_tri_at([5.0, 10.0, 0.0]));
    let c = s.surface_centroid().unwrap();
    // Per-mesh centroids: (1/3, 1/3, 0), (10+1/3, 1/3, 0),
    // (5+1/3, 10+1/3, 0). Equal weights → arithmetic mean.
    let exp_x = (1.0 / 3.0 + 10.0 + 1.0 / 3.0 + 5.0 + 1.0 / 3.0) / 3.0;
    let exp_y = (1.0 / 3.0 + 1.0 / 3.0 + 10.0 + 1.0 / 3.0) / 3.0;
    assert!(approx_eq3(c, [exp_x, exp_y, 0.0], 1e-12), "got {:?}", c);
}
