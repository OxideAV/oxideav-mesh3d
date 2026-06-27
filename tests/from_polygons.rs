//! Primitive::from_polygons coverage: n-gon triangulation, concave
//! faces, winding inheritance, and malformed-face handling.

use oxideav_mesh3d::{Primitive, Topology};

#[test]
fn quad_becomes_two_triangles() {
    // Unit square in the z=0 plane, CCW.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let faces = vec![vec![0, 1, 2, 3]];
    let prim = Primitive::from_polygons(positions, &faces);
    assert_eq!(prim.topology, Topology::Triangles);
    assert_eq!(prim.triangle_indices().len(), 2);
    // Area is preserved: the unit square has area 1.
    assert!((prim.surface_area() - 1.0).abs() < 1e-9);
}

#[test]
fn pentagon_becomes_three_triangles() {
    // Regular-ish convex pentagon in z=0.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.5, 1.5, 0.0],
        [1.0, 2.5, 0.0],
        [-0.5, 1.5, 0.0],
    ];
    let faces = vec![vec![0, 1, 2, 3, 4]];
    let prim = Primitive::from_polygons(positions, &faces);
    // An n-gon triangulates into n-2 triangles.
    assert_eq!(prim.triangle_indices().len(), 3);
}

#[test]
fn concave_polygon_triangulates_to_its_own_area() {
    // An L-shaped (concave) hexagon in z=0. A naive fan would overlap;
    // the ear clip must produce non-overlapping triangles whose total
    // area equals the L's area.
    //  (0,0)-(2,0)-(2,1)-(1,1)-(1,2)-(0,2)
    let positions = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    let faces = vec![vec![0, 1, 2, 3, 4, 5]];
    let prim = Primitive::from_polygons(positions, &faces);
    assert_eq!(prim.triangle_indices().len(), 4); // 6-gon -> 4 tris
                                                  // The L area: a 2x2 square minus the 1x1 top-right notch = 3.
    assert!(
        (prim.surface_area() - 3.0).abs() < 1e-9,
        "L area = {}",
        prim.surface_area()
    );
}

#[test]
fn triangle_face_keeps_its_winding() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let faces = vec![vec![0, 1, 2]];
    let prim = Primitive::from_polygons(positions, &faces);
    let tris = prim.triangle_indices();
    assert_eq!(tris.len(), 1);
    // The single triangle uses the three distinct corners in the same
    // cyclic (winding-preserving) order — possibly rotated by the ear
    // clip's start cursor, never reversed.
    let t = tris[0];
    let rotations = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];
    assert!(
        rotations.contains(&t),
        "triangle {t:?} is not a winding-preserving rotation of [0,1,2]"
    );
    // +z winding survives.
    assert!(prim.compute_normals().iter().all(|n| n[2] > 0.5));
}

#[test]
fn multiple_faces_share_one_pool() {
    // Two quads sharing the vertex pool.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ];
    let faces = vec![vec![0, 1, 2, 3], vec![1, 4, 5, 2]];
    let prim = Primitive::from_polygons(positions, &faces);
    assert_eq!(prim.triangle_indices().len(), 4); // 2 quads -> 4 tris
    assert_eq!(prim.positions.len(), 6); // pool unchanged
    assert!((prim.surface_area() - 2.0).abs() < 1e-9); // two unit quads
}

#[test]
fn winding_is_inherited_from_the_face() {
    // A CCW quad about +z should produce +z-facing normals.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let ccw = Primitive::from_polygons(positions.clone(), &[vec![0, 1, 2, 3]]);
    let ccw_n = ccw.compute_normals();
    assert!(ccw_n.iter().all(|n| n[2] > 0.5), "CCW should face +z");

    // The reverse winding faces -z.
    let cw = Primitive::from_polygons(positions, &[vec![3, 2, 1, 0]]);
    let cw_n = cw.compute_normals();
    assert!(cw_n.iter().all(|n| n[2] < -0.5), "CW should face -z");
}

#[test]
fn out_of_range_face_is_skipped() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    // First face valid; second references vertex 99.
    let faces = vec![vec![0, 1, 2, 3], vec![0, 1, 99]];
    let prim = Primitive::from_polygons(positions, &faces);
    // Only the valid quad survives → 2 triangles.
    assert_eq!(prim.triangle_indices().len(), 2);
}

#[test]
fn degenerate_and_tiny_faces_are_dropped() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], // collinear with the first two
    ];
    // A 2-corner "face" and an all-collinear triangle both yield nothing.
    let faces = vec![vec![0, 1], vec![0, 1, 2]];
    let prim = Primitive::from_polygons(positions, &faces);
    assert_eq!(prim.triangle_indices().len(), 0);
}

#[test]
fn empty_faces_yield_empty_primitive() {
    let prim = Primitive::from_polygons(vec![[0.0, 0.0, 0.0]], &[]);
    assert_eq!(prim.triangle_indices().len(), 0);
    assert_eq!(prim.positions.len(), 1);
}

#[test]
fn non_planar_quad_triangulates() {
    // A saddle-ish non-planar quad: Newell projection should still
    // triangulate it into 2 triangles with positive total area.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.5],
    ];
    let prim = Primitive::from_polygons(positions, &[vec![0, 1, 2, 3]]);
    assert_eq!(prim.triangle_indices().len(), 2);
    assert!(prim.surface_area() > 0.0);
}
