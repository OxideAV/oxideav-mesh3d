//! Tests for the transform-aware area-weighted surface centroid:
//! [`Primitive::world_surface_centroid`], [`Mesh::world_surface_centroid`],
//! and [`Scene3D::world_surface_centroid`].
//!
//! Every closed-form expected value below is derived from elementary
//! affine geometry: under a linear transform `M`, a triangle's
//! post-transform centroid is `M * local_centroid` and its post-
//! transform area is `|cross(M_3·E1, M_3·E2)| / 2`. The area-weighted
//! continuous identity `C = (Σ area_i · centroid_i) / Σ area_i`
//! (Marsden & Tromba, *Vector Calculus*, chapter on triangle integrals)
//! then fixes the recombination.

use oxideav_mesh3d::{Indices, Mesh, Node, NodeId, Primitive, Scene3D, Topology, Transform};

fn approx_eq3(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn translation_matrix(t: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn scale_matrix(s: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [s[0], 0.0, 0.0, 0.0],
        [0.0, s[1], 0.0, 0.0],
        [0.0, 0.0, s[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn translation_node(t: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: t,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn scale_node(s: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: s,
    }
}

// ──────────────────────────────────────────────────────────────────
// Primitive::world_surface_centroid
// ──────────────────────────────────────────────────────────────────

/// Identity world matrix must reproduce the local centroid bit-for-bit.
#[test]
fn primitive_world_centroid_identity_matches_local() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 6.0, 0.0]];
    let c_local = p.surface_centroid().unwrap();
    let c_world = p.world_surface_centroid(IDENTITY).unwrap();
    assert!(
        approx_eq3(c_local, c_world, 1e-12),
        "{:?} vs {:?}",
        c_local,
        c_world
    );
    assert!(approx_eq3(c_world, [1.0, 2.0, 0.0], 1e-12));
}

/// Pure-translation world matrix: post-transform centroid equals
/// local centroid plus the translation column.
#[test]
fn primitive_world_centroid_pure_translation_equivariant() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let c_local = p.surface_centroid().unwrap();
    let m = translation_matrix([10.0, 20.0, 30.0]);
    let c_world = p.world_surface_centroid(m).unwrap();
    let expected = [c_local[0] + 10.0, c_local[1] + 20.0, c_local[2] + 30.0];
    assert!(approx_eq3(c_world, expected, 1e-12), "got {:?}", c_world);
}

/// Uniform scale around the origin: the centroid scales from the
/// origin by the same factor.
#[test]
fn primitive_world_centroid_uniform_scale_scales_centroid() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let c_local = p.surface_centroid().unwrap();
    let m = scale_matrix([3.0, 3.0, 3.0]);
    let c_world = p.world_surface_centroid(m).unwrap();
    let expected = [c_local[0] * 3.0, c_local[1] * 3.0, c_local[2] * 3.0];
    assert!(approx_eq3(c_world, expected, 1e-12), "got {:?}", c_world);
}

/// Non-uniform diagonal scale: each axis scales independently. The
/// per-triangle area weighting can shift, but for a single triangle
/// the post-transform centroid is still the per-axis scale of the
/// local centroid.
#[test]
fn primitive_world_centroid_nonuniform_scale_single_triangle() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let c_local = p.surface_centroid().unwrap();
    let m = scale_matrix([2.0, 3.0, 5.0]);
    let c_world = p.world_surface_centroid(m).unwrap();
    let expected = [c_local[0] * 2.0, c_local[1] * 3.0, c_local[2] * 5.0];
    assert!(approx_eq3(c_world, expected, 1e-12), "got {:?}", c_world);
}

/// Non-triangle topology → `None`, regardless of transform.
#[test]
fn primitive_world_centroid_non_triangle_topology_none() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(p.world_surface_centroid(IDENTITY).is_none());
    assert!(p
        .world_surface_centroid(translation_matrix([5.0, 0.0, 0.0]))
        .is_none());
}

/// Empty positions → `None`.
#[test]
fn primitive_world_centroid_empty_none() {
    let p = Primitive::new(Topology::Triangles);
    assert!(p.world_surface_centroid(IDENTITY).is_none());
}

/// A scale matrix that flattens every axis to 0 collapses every
/// triangle to zero area → `None`.
#[test]
fn primitive_world_centroid_degenerate_transform_none() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let m = scale_matrix([0.0, 0.0, 0.0]);
    assert!(p.world_surface_centroid(m).is_none());
}

/// Out-of-range indices are skipped, not panicked.
#[test]
fn primitive_world_centroid_out_of_range_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    // First triangle indexes positions 0/1/2 (good), second indexes 5/6/7
    // (out of range, must be skipped).
    p.indices = Some(Indices::U32(vec![0, 1, 2, 5, 6, 7]));
    let c = p.world_surface_centroid(IDENTITY).unwrap();
    assert!(approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12));
}

/// NaN entries in `world` make every transformed corner non-finite;
/// the result must be `None`, not a panic.
#[test]
fn primitive_world_centroid_nan_matrix_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut bad = IDENTITY;
    bad[0][0] = f32::NAN;
    assert!(p.world_surface_centroid(bad).is_none());
}

/// Mirror scale `[-1, 1, 1]` (single-axis mirror): area magnitude
/// unchanged, centroid x-component flips sign relative to the origin.
#[test]
fn primitive_world_centroid_mirror_scale_flips_axis() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
    let c_local = p.surface_centroid().unwrap();
    let c_world = p
        .world_surface_centroid(scale_matrix([-1.0, 1.0, 1.0]))
        .unwrap();
    assert!(
        approx_eq3(c_world, [-c_local[0], c_local[1], c_local[2]], 1e-12),
        "got {:?}",
        c_world
    );
}

/// Indexed and unindexed builds of the same triangle list produce
/// equal world centroids.
#[test]
fn primitive_world_centroid_indexed_matches_unindexed() {
    let mut a = Primitive::new(Topology::Triangles);
    a.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let mut b = Primitive::new(Topology::Triangles);
    b.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    b.indices = Some(Indices::U32(vec![0, 1, 2, 0, 2, 3]));
    let m = translation_matrix([7.0, -3.0, 2.0]);
    let ca = a.world_surface_centroid(m).unwrap();
    let cb = b.world_surface_centroid(m).unwrap();
    assert!(approx_eq3(ca, cb, 1e-12), "{:?} vs {:?}", ca, cb);
}

/// `TriangleStrip` and the equivalent triangle list produce identical
/// world centroids.
#[test]
fn primitive_world_centroid_strip_matches_list() {
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let mut list = Primitive::new(Topology::Triangles);
    list.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let m = scale_matrix([2.0, 3.0, 1.0]);
    let cs = strip.world_surface_centroid(m).unwrap();
    let cl = list.world_surface_centroid(m).unwrap();
    assert!(approx_eq3(cs, cl, 1e-12), "{:?} vs {:?}", cs, cl);
}

/// A pure rotation around the X axis by 90° moves the local centroid
/// to a rotated one. Build the matrix from raw rotation entries to
/// avoid leaning on `Transform::Trs`.
#[test]
fn primitive_world_centroid_rotation_preserves_magnitudes() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    // 90° rotation around +X — (y, z) → (-z, y).
    let r: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let c_local = p.surface_centroid().unwrap();
    let c_world = p.world_surface_centroid(r).unwrap();
    let expected = [c_local[0], -c_local[2], c_local[1]];
    assert!(approx_eq3(c_world, expected, 1e-6), "got {:?}", c_world);
}

// ──────────────────────────────────────────────────────────────────
// Mesh::world_surface_centroid
// ──────────────────────────────────────────────────────────────────

/// A single-primitive mesh passes the centroid through.
#[test]
fn mesh_world_centroid_single_primitive_passthrough() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
    let mesh = Mesh::new(Some("tri".to_owned())).with_primitive(p.clone());
    let local_c = p.world_surface_centroid(IDENTITY).unwrap();
    let mesh_c = mesh.world_surface_centroid(IDENTITY).unwrap();
    assert!(approx_eq3(local_c, mesh_c, 1e-12));
    assert!(approx_eq3(mesh_c, [1.0, 1.0, 0.0], 1e-12));
}

/// Mesh with two equal-area primitives under a uniform scale: the
/// post-transform centroid is the midpoint of each primitive's
/// post-transform centroid.
#[test]
fn mesh_world_centroid_two_equal_area_under_scale() {
    let mut p1 = Primitive::new(Topology::Triangles);
    p1.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[10.0, 10.0, 0.0], [11.0, 10.0, 0.0], [10.0, 11.0, 0.0]];
    let mesh = Mesh::new(Some("two".to_owned()))
        .with_primitive(p1)
        .with_primitive(p2);
    let m = scale_matrix([2.0, 2.0, 2.0]);
    let c = mesh.world_surface_centroid(m).unwrap();
    // Local midpoint of centroids (1/3, 1/3, 0) + (10+1/3, 10+1/3, 0)
    // → (5+1/3, 5+1/3, 0); under uniform scale 2 the centroid sits at
    // 2× that midpoint.
    let exp = [2.0 * (5.0 + 1.0 / 3.0), 2.0 * (5.0 + 1.0 / 3.0), 0.0];
    assert!(approx_eq3(c, exp, 1e-12), "got {:?}", c);
}

/// Mesh with no primitives returns `None`.
#[test]
fn mesh_world_centroid_no_primitives_none() {
    let mesh = Mesh::new(Some("empty".to_owned()));
    assert!(mesh.world_surface_centroid(IDENTITY).is_none());
}

/// A mesh whose primitives are all degenerate returns `None`.
#[test]
fn mesh_world_centroid_all_degenerate_none() {
    let mut p1 = Primitive::new(Topology::Triangles);
    p1.positions = vec![[1.0, 1.0, 1.0]; 3];
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[2.0, 2.0, 2.0]; 3];
    let mesh = Mesh::new(Some("bad".to_owned()))
        .with_primitive(p1)
        .with_primitive(p2);
    assert!(mesh.world_surface_centroid(IDENTITY).is_none());
}

/// One degenerate primitive + one good one: the good one dictates the
/// post-transform centroid.
#[test]
fn mesh_world_centroid_degenerate_primitive_skipped() {
    let mut bad = Primitive::new(Topology::Triangles);
    bad.positions = vec![[5.0, 5.0, 5.0]; 3];
    let mut good = Primitive::new(Topology::Triangles);
    good.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mesh = Mesh::new(Some("mix".to_owned()))
        .with_primitive(bad)
        .with_primitive(good);
    let m = translation_matrix([1.0, 2.0, 3.0]);
    let c = mesh.world_surface_centroid(m).unwrap();
    assert!(
        approx_eq3(c, [1.0 + 1.0 / 3.0, 2.0 + 1.0 / 3.0, 3.0], 1e-12),
        "got {:?}",
        c
    );
}

/// A mesh-level non-triangle primitive contributes nothing.
#[test]
fn mesh_world_centroid_lines_primitive_skipped() {
    let mut lines = Primitive::new(Topology::Lines);
    lines.positions = vec![[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]];
    let mut tri = Primitive::new(Topology::Triangles);
    tri.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mesh = Mesh::new(Some("mix".to_owned()))
        .with_primitive(lines)
        .with_primitive(tri);
    let m = translation_matrix([10.0, 0.0, 0.0]);
    let c = mesh.world_surface_centroid(m).unwrap();
    assert!(
        approx_eq3(c, [10.0 + 1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

// ──────────────────────────────────────────────────────────────────
// Scene3D::world_surface_centroid
// ──────────────────────────────────────────────────────────────────

#[test]
fn scene_world_centroid_empty_none() {
    let s = Scene3D::new();
    assert!(s.world_surface_centroid().is_none());
}

/// A scene with meshes but no nodes returns `None` — the walk is
/// per-instance over the scene-graph, not per-resource.
#[test]
fn scene_world_centroid_no_nodes_none() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut s = Scene3D::new();
    s.add_mesh(Mesh::new(None::<String>).with_primitive(p));
    assert!(s.world_surface_centroid().is_none());
}

/// A mesh reached via a single identity-transformed node returns the
/// mesh's local centroid.
#[test]
fn scene_world_centroid_identity_root_matches_local() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None::<String>).with_primitive(p));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);
    let c = s.world_surface_centroid().unwrap();
    assert!(approx_eq3(c, [1.0, 1.0, 0.0], 1e-12), "got {:?}", c);
}

/// Detached meshes (no node references them) contribute nothing.
#[test]
fn scene_world_centroid_detached_mesh_skipped() {
    let mut p1 = Primitive::new(Topology::Triangles);
    p1.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![
        [100.0, 100.0, 100.0],
        [101.0, 100.0, 100.0],
        [100.0, 101.0, 100.0],
    ];

    let mut s = Scene3D::new();
    let attached = s.add_mesh(Mesh::new(None::<String>).with_primitive(p1));
    let _detached = s.add_mesh(Mesh::new(None::<String>).with_primitive(p2));
    let nid = s.add_node(Node::new().with_mesh(attached));
    s.add_root(nid);
    let c = s.world_surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// A second instance with a non-trivial translation moves the
/// world-centroid (unlike `Scene3D::surface_centroid` which walks
/// meshes once and ignores per-node transforms).
#[test]
fn scene_world_centroid_two_instances_translate_apart() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None::<String>).with_primitive(p));
    let n0 = s.add_node(Node::new().with_mesh(mid));
    let n1 = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_node([10.0, 0.0, 0.0])),
    );
    s.add_root(n0);
    s.add_root(n1);
    let c = s.world_surface_centroid().unwrap();
    // Equal-area instances at centroid (1/3, 1/3, 0) and
    // (10+1/3, 1/3, 0) → midpoint (5+1/3, 1/3, 0).
    assert!(
        approx_eq3(c, [5.0 + 1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// Per-node scale alters both the per-instance centroid (scaled from
/// the origin) and the per-instance area weighting.
#[test]
fn scene_world_centroid_node_scale_weights_centroid() {
    let mut small = Primitive::new(Topology::Triangles);
    small.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut big = Primitive::new(Topology::Triangles);
    big.positions = vec![[10.0, 10.0, 0.0], [11.0, 10.0, 0.0], [10.0, 11.0, 0.0]];

    let mut s = Scene3D::new();
    let m_small = s.add_mesh(Mesh::new(None::<String>).with_primitive(small));
    let m_big = s.add_mesh(Mesh::new(None::<String>).with_primitive(big));

    // big mesh under uniform scale 2 → its post-transform area becomes
    // 4× (area scales s²), and its centroid shifts from (10+1/3, 10+1/3, 0)
    // to (2·(10+1/3), 2·(10+1/3), 0).
    let n_small = s.add_node(Node::new().with_mesh(m_small));
    let n_big = s.add_node(
        Node::new()
            .with_mesh(m_big)
            .with_transform(scale_node([2.0, 2.0, 1.0])),
    );
    s.add_root(n_small);
    s.add_root(n_big);

    let c = s.world_surface_centroid().unwrap();
    // Manual closed form:
    // - small centroid (1/3, 1/3, 0), world area 0.5.
    // - big local area = 0.5; scale [2,2,1] → post-transform area
    //   factor for a +Z-facing triangle is |sx·sy| = 4 → 2.0.
    //   Centroid: (2·(10+1/3), 2·(10+1/3), 0).
    let s_area = 0.5;
    let b_area = 2.0;
    let s_c = [1.0 / 3.0, 1.0 / 3.0, 0.0];
    let b_c = [2.0 * (10.0 + 1.0 / 3.0), 2.0 * (10.0 + 1.0 / 3.0), 0.0];
    let total = s_area + b_area;
    let exp = [
        (s_area * s_c[0] + b_area * b_c[0]) / total,
        (s_area * s_c[1] + b_area * b_c[1]) / total,
        (s_area * s_c[2] + b_area * b_c[2]) / total,
    ];
    assert!(approx_eq3(c, exp, 1e-10), "got {:?} exp {:?}", c, exp);
}

/// An ancestor-chain transform composes: child's effective transform is
/// parent × child.
#[test]
fn scene_world_centroid_ancestor_chain_translates() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None::<String>).with_primitive(p));

    let child = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_node([1.0, 0.0, 0.0])),
    );
    let mut parent = Node::new().with_transform(translation_node([10.0, 20.0, 30.0]));
    parent.children.push(child);
    let parent_id = s.add_node(parent);
    s.add_root(parent_id);

    let c = s.world_surface_centroid().unwrap();
    // Local centroid (1/3, 1/3, 0); add the [1+10, 0+20, 0+30]
    // composed translation.
    let expected = [11.0 + 1.0 / 3.0, 20.0 + 1.0 / 3.0, 30.0];
    assert!(approx_eq3(c, expected, 1e-12), "got {:?}", c);
}

/// A cycle in the scene graph still resolves; each node is visited at
/// most once.
#[test]
fn scene_world_centroid_cycle_visits_once() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None::<String>).with_primitive(p));

    let a = NodeId(0);
    let b = NodeId(1);
    let mut node_a = Node::new().with_mesh(mid);
    node_a.children.push(b);
    let mut node_b = Node::new();
    node_b.children.push(a);
    let _ = s.add_node(node_a);
    let _ = s.add_node(node_b);
    s.add_root(a);

    let c = s.world_surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// A node with a degenerate transform (zero scale on every axis)
/// collapses to zero area and contributes nothing; the remaining
/// reachable instance dictates the centroid.
#[test]
fn scene_world_centroid_degenerate_transform_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None::<String>).with_primitive(p));

    let live = s.add_node(Node::new().with_mesh(mid));
    let dead = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(scale_node([0.0, 0.0, 0.0])),
    );
    s.add_root(live);
    s.add_root(dead);

    let c = s.world_surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// Out-of-range mesh id on a node is silently skipped.
#[test]
fn scene_world_centroid_out_of_range_mesh_id_skipped() {
    use oxideav_mesh3d::MeshId;
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None::<String>).with_primitive(p));

    let good = s.add_node(Node::new().with_mesh(mid));
    let bad = s.add_node(Node::new().with_mesh(MeshId(99)));
    s.add_root(good);
    s.add_root(bad);

    let c = s.world_surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [1.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}

/// Three identity-instanced equal-area meshes at corners of an
/// equilateral arrangement: the world-centroid is the arithmetic mean
/// of the per-mesh centroids.
#[test]
fn scene_world_centroid_three_instances_average() {
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
    let m0 = s.add_mesh(unit_tri_at([0.0, 0.0, 0.0]));
    let m1 = s.add_mesh(unit_tri_at([10.0, 0.0, 0.0]));
    let m2 = s.add_mesh(unit_tri_at([5.0, 10.0, 0.0]));
    let n0 = s.add_node(Node::new().with_mesh(m0));
    let n1 = s.add_node(Node::new().with_mesh(m1));
    let n2 = s.add_node(Node::new().with_mesh(m2));
    s.add_root(n0);
    s.add_root(n1);
    s.add_root(n2);

    let c = s.world_surface_centroid().unwrap();
    let exp_x = (1.0 / 3.0 + 10.0 + 1.0 / 3.0 + 5.0 + 1.0 / 3.0) / 3.0;
    let exp_y = (1.0 / 3.0 + 1.0 / 3.0 + 10.0 + 1.0 / 3.0) / 3.0;
    assert!(approx_eq3(c, [exp_x, exp_y, 0.0], 1e-12), "got {:?}", c);
}

/// The world centroid is finite for any finite input.
#[test]
fn scene_world_centroid_components_finite() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 100.0]];
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None::<String>).with_primitive(p));
    let n = s.add_node(
        Node::new()
            .with_mesh(mid)
            .with_transform(translation_node([1.5, -2.5, 0.25])),
    );
    s.add_root(n);
    let c = s.world_surface_centroid().unwrap();
    assert!(c[0].is_finite() && c[1].is_finite() && c[2].is_finite());
}

/// Cross-check against `Scene3D::surface_centroid` under an
/// identity-rooted scene with a single instance: the two answers must
/// agree.
#[test]
fn scene_world_centroid_identity_matches_local_centroid() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None::<String>).with_primitive(p));
    let nid = s.add_node(Node::new().with_mesh(mid));
    s.add_root(nid);

    let local = s.surface_centroid().unwrap();
    let world = s.world_surface_centroid().unwrap();
    assert!(
        approx_eq3(local, world, 1e-12),
        "{:?} vs {:?}",
        local,
        world
    );
}

/// Same mesh under two identity-transformed nodes: the world centroid
/// equals the mesh's local centroid (both instances coincide).
#[test]
fn scene_world_centroid_two_identity_instances_match_single() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]];

    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None::<String>).with_primitive(p));
    let a = s.add_node(Node::new().with_mesh(mid));
    let b = s.add_node(Node::new().with_mesh(mid));
    s.add_root(a);
    s.add_root(b);

    let c = s.world_surface_centroid().unwrap();
    assert!(
        approx_eq3(c, [2.0 / 3.0, 4.0 / 3.0, 0.0], 1e-12),
        "got {:?}",
        c
    );
}
