//! Discrete curvature coverage: flat-surface zero curvature, the
//! cube-corner angle defect, the Gauss-Bonnet total-defect identity for
//! closed meshes, and degenerate-input handling.

use std::f64::consts::PI;

use oxideav_mesh3d::{Indices, Primitive, Topology};

fn unit_cube() -> Primitive {
    // 8 corners of the unit cube.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    // 12 triangles, outward-facing winding.
    let idx: Vec<u32> = vec![
        // -z bottom
        0, 2, 1, 0, 3, 2, // +z top
        4, 5, 6, 4, 6, 7, // -y front
        0, 1, 5, 0, 5, 4, // +y back
        3, 7, 6, 3, 6, 2, // -x left
        0, 4, 7, 0, 7, 3, // +x right
        1, 2, 6, 1, 6, 5,
    ];
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = positions;
    p.indices = Some(Indices::U32(idx));
    p
}

fn octahedron() -> Primitive {
    let positions = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let idx: Vec<u32> = vec![
        0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
    ];
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = positions;
    p.indices = Some(Indices::U32(idx));
    p
}

#[test]
fn flat_grid_has_near_zero_curvature() {
    // A planar 3x3 grid: the interior vertex sits in a flat
    // neighbourhood, so both curvatures should be ~0 there.
    let mut positions = Vec::new();
    for gy in 0..3 {
        for gx in 0..3 {
            positions.push([gx as f32, gy as f32, 0.0]);
        }
    }
    let mut idx: Vec<u32> = Vec::new();
    for cy in 0..2u32 {
        for cx in 0..2u32 {
            let v00 = cy * 3 + cx;
            let v10 = v00 + 1;
            let v01 = v00 + 3;
            let v11 = v01 + 1;
            idx.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
        }
    }
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = positions;
    p.indices = Some(Indices::U32(idx));

    let c = p.curvature();
    // Locate the interior vertex (the one at (1,1)).
    let interior = c
        .welded
        .positions
        .iter()
        .position(|q| (q[0] - 1.0).abs() < 1e-4 && (q[1] - 1.0).abs() < 1e-4)
        .expect("interior vertex");
    assert!(
        c.gaussian[interior].abs() < 1e-9,
        "flat gaussian = {}",
        c.gaussian[interior]
    );
    assert!(
        c.mean[interior].abs() < 1e-6,
        "flat mean = {}",
        c.mean[interior]
    );
}

#[test]
fn cube_corner_defect_is_half_pi() {
    // Every corner of a cube meets three faces, each contributing a
    // π/2 corner angle → angle sum 3π/2 → defect 2π − 3π/2 = π/2.
    let cube = unit_cube();
    let c = cube.curvature();
    assert_eq!(c.len(), 8);
    for i in 0..c.len() {
        // integrated defect at vertex i = K(i) * A(i)
        let defect = c.gaussian[i] * c.area[i];
        assert!(
            (defect - PI / 2.0).abs() < 1e-9,
            "corner {i} defect {defect} != π/2"
        );
        // Positive Gaussian curvature at a convex corner.
        assert!(c.gaussian[i] > 0.0);
    }
}

#[test]
fn closed_cube_satisfies_gauss_bonnet() {
    // Total angle defect of a closed genus-0 surface = 2π·χ = 2π·2 = 4π
    // (the cube has Euler characteristic 2).
    let cube = unit_cube();
    let c = cube.curvature();
    let total = c.total_angle_defect();
    assert!(
        (total - 4.0 * PI).abs() < 1e-7,
        "cube total defect {total} != 4π"
    );
}

#[test]
fn closed_octahedron_satisfies_gauss_bonnet() {
    let oct = octahedron();
    let c = oct.curvature();
    let total = c.total_angle_defect();
    assert!(
        (total - 4.0 * PI).abs() < 1e-7,
        "octahedron total defect {total} != 4π"
    );
    // Every vertex of the octahedron is convex → positive Gaussian.
    for &k in &c.gaussian {
        assert!(k > 0.0);
    }
}

#[test]
fn mixed_area_partitions_total_surface_area() {
    // The mixed Voronoi areas must sum to the mesh's total surface area.
    let cube = unit_cube();
    let c = cube.curvature();
    let area_sum: f64 = c.area.iter().sum();
    let total = cube.surface_area();
    assert!(
        (area_sum - total).abs() < 1e-9,
        "voronoi sum {area_sum} != surface area {total}"
    );
}

#[test]
fn octahedron_has_nonzero_mean_curvature() {
    let oct = octahedron();
    let c = oct.curvature();
    // A non-planar closed surface has curved vertices → mean > 0.
    assert!(c.mean.iter().all(|&h| h.is_finite()));
    assert!(c.mean.iter().any(|&h| h > 0.0));
}

#[test]
fn non_triangle_input_is_empty() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let c = p.curvature();
    assert!(c.is_empty());
    assert_eq!(c.len(), 0);
}

#[test]
fn empty_input_is_empty() {
    let p = Primitive::new(Topology::Triangles);
    let c = p.curvature();
    assert!(c.is_empty());
}

#[test]
fn degenerate_triangle_does_not_poison_output() {
    // A collinear triangle has no area; it contributes nothing and the
    // result stays finite (zero curvature everywhere).
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2]));
    let c = p.curvature();
    assert!(c.gaussian.iter().all(|k| k.is_finite()));
    assert!(c.mean.iter().all(|h| h.is_finite()));
    assert!(c.gaussian.iter().all(|&k| k == 0.0));
}
