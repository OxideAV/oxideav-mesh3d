//! Laplacian + Taubin smoothing coverage: convergence toward the
//! neighbour centroid, boundary preservation, the Taubin anti-shrinkage
//! property, connectivity invariance, and degenerate-input handling.

use oxideav_mesh3d::{Indices, Primitive, SmoothOptions, Topology};

/// A flat 3×3 grid of vertices in the z=0 plane, triangulated into a
/// regular 2×2 quad mesh (8 triangles). The single interior vertex
/// (index 4, the centre) is lifted off the plane by `bump` so a
/// smoothing pass has something to flatten. Returns the indexed
/// `Triangles` primitive.
fn bumped_grid(bump: f32) -> Primitive {
    // Layout (x, y), z=0 except centre:
    //   6 7 8
    //   3 4 5
    //   0 1 2
    let mut positions = Vec::new();
    for gy in 0..3 {
        for gx in 0..3 {
            positions.push([gx as f32, gy as f32, 0.0]);
        }
    }
    positions[4][2] = bump; // lift the centre vertex

    // Two triangles per cell, consistent CCW winding.
    let mut idx: Vec<u32> = Vec::new();
    for cy in 0..2u32 {
        for cx in 0..2u32 {
            let v00 = cy * 3 + cx;
            let v10 = v00 + 1;
            let v01 = v00 + 3;
            let v11 = v01 + 1;
            idx.extend_from_slice(&[v00, v10, v11]);
            idx.extend_from_slice(&[v00, v11, v01]);
        }
    }

    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = positions;
    prim.indices = Some(Indices::U32(idx));
    prim
}

#[test]
fn zero_iterations_is_identity() {
    let prim = bumped_grid(0.5);
    let opts = SmoothOptions {
        iterations: 0,
        preserve_boundary: true,
    };
    let out = prim.smooth_laplacian(1.0, opts);
    // Same vertex count, centre still bumped.
    assert_eq!(out.positions.len(), prim.positions.len());
    assert!((out.positions[4][2] - 0.5).abs() < 1e-6);
}

#[test]
fn lambda_zero_is_identity() {
    let prim = bumped_grid(0.5);
    let out = prim.smooth_laplacian(0.0, SmoothOptions::default());
    assert!((out.positions[4][2] - 0.5).abs() < 1e-6);
}

#[test]
fn laplacian_flattens_a_lone_bump() {
    // The centre vertex's one-ring is its 4 edge-neighbours plus the 2
    // diagonal corners reachable by triangle edges, all at z=0. With
    // lambda = 1 the centre snaps exactly onto their centroid (z=0).
    let prim = bumped_grid(1.0);
    let out = prim.smooth_laplacian(1.0, SmoothOptions::default());
    // Interior vertex 4 should be flattened essentially to the plane.
    assert!(
        out.positions[4][2].abs() < 1e-5,
        "centre z = {} not flattened",
        out.positions[4][2]
    );
}

#[test]
fn smaller_lambda_smooths_less_per_step() {
    let prim = bumped_grid(1.0);
    let gentle = prim.smooth_laplacian(0.5, SmoothOptions::default());
    let full = prim.smooth_laplacian(1.0, SmoothOptions::default());
    // The gentle pass leaves more of the bump than the aggressive one.
    assert!(gentle.positions[4][2] > full.positions[4][2]);
    assert!(gentle.positions[4][2] > 0.0);
}

#[test]
fn iterating_reduces_the_bump_monotonically() {
    let prim = bumped_grid(1.0);
    let one = prim.smooth_laplacian(
        0.5,
        SmoothOptions {
            iterations: 1,
            preserve_boundary: true,
        },
    );
    let three = prim.smooth_laplacian(
        0.5,
        SmoothOptions {
            iterations: 3,
            preserve_boundary: true,
        },
    );
    assert!(three.positions[4][2] < one.positions[4][2]);
}

#[test]
fn boundary_vertices_are_pinned_by_default() {
    // Move a corner off-plane; with preserve_boundary it must not move.
    let mut prim = bumped_grid(0.0);
    prim.positions[0] = [0.0, 0.0, 0.7]; // corner (boundary)
    let out = prim.smooth_laplacian(1.0, SmoothOptions::default());
    // Find the welded vertex nearest the original corner location's xy
    // and confirm its z is preserved. The corner is the only vertex at
    // (0,0,*); locate it.
    let corner = out
        .positions
        .iter()
        .find(|p| p[0].abs() < 1e-4 && p[1].abs() < 1e-4)
        .expect("corner present");
    assert!(
        (corner[2] - 0.7).abs() < 1e-5,
        "boundary corner moved: z = {}",
        corner[2]
    );
}

#[test]
fn boundary_can_be_smoothed_when_not_preserved() {
    let mut prim = bumped_grid(0.0);
    prim.positions[0] = [0.0, 0.0, 0.7];
    let pinned = prim.smooth_laplacian(
        1.0,
        SmoothOptions {
            iterations: 1,
            preserve_boundary: true,
        },
    );
    let out = prim.smooth_laplacian(
        1.0,
        SmoothOptions {
            iterations: 1,
            preserve_boundary: false,
        },
    );
    // All grid positions are distinct, so weld preserves vertex 0 (the
    // lifted corner) as welded index 0. Pinned: untouched. Unpinned: the
    // corner is pulled toward its one-ring, dropping below 0.7.
    assert!((pinned.positions[0][2] - 0.7).abs() < 1e-5);
    assert!(out.positions[0][2] < 0.7);
}

#[test]
fn connectivity_is_preserved() {
    let prim = bumped_grid(1.0);
    let before = prim.weld_vertices().triangle_indices();
    let out = prim.smooth_laplacian(0.5, SmoothOptions::default());
    let after = out.triangle_indices();
    // Smoothing welds then only moves points: triangle topology matches
    // the welded source exactly.
    assert_eq!(before, after);
}

#[test]
fn taubin_preserves_volume_better_than_laplacian() {
    // A unit-ish closed octahedron. Repeated Laplacian smoothing shrinks
    // it toward the centroid; Taubin's negative pass counteracts that.
    let positions = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    // 8 faces of the octahedron, outward CCW.
    let idx: Vec<u32> = vec![
        0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
    ];
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = positions;
    prim.indices = Some(Indices::U32(idx));

    let v0 = prim.volume();
    assert!(v0 > 0.0);

    let opts = SmoothOptions {
        iterations: 10,
        preserve_boundary: true,
    };
    let lap = prim.smooth_laplacian(0.5, opts).volume();
    let taub = prim.smooth_taubin(0.33, -0.34, opts).volume();

    // Laplacian collapses the closed surface toward its centroid; the
    // Taubin inflate pass counteracts that, so Taubin retains
    // substantially more volume at the same iteration count.
    assert!(lap > 0.0 && taub > 0.0);
    assert!(lap < taub, "lap {lap} should be < taubin {taub}");
    assert!(
        taub > 2.0 * lap,
        "taubin {taub} should keep markedly more than laplacian {lap} (v0={v0})"
    );
}

#[test]
fn non_triangle_input_yields_empty_triangles() {
    let mut prim = Primitive::new(Topology::Lines);
    prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let out = prim.smooth_laplacian(1.0, SmoothOptions::default());
    assert_eq!(out.topology, Topology::Triangles);
    assert_eq!(out.triangle_indices().len(), 0);
}

#[test]
fn empty_input_yields_empty() {
    let prim = Primitive::new(Topology::Triangles);
    let out = prim.smooth_taubin(0.33, -0.34, SmoothOptions::default());
    assert_eq!(out.positions.len(), 0);
    assert_eq!(out.triangle_indices().len(), 0);
}
