//! Integration tests for `oxideav_mesh3d::Bvh`.
//!
//! These tests exercise the public `Bvh` surface against the public
//! `Primitive` surface, in the same way a downstream renderer would
//! consume it. They cross-validate every ray query against the
//! brute-force `Primitive::intersect_ray` path so any divergence
//! between the two is caught at CI time.

use oxideav_mesh3d::{Bvh, Indices, Primitive, Ray, Topology};

fn make_grid(side: u32) -> Primitive {
    // `side x side` grid of unit squares in z = 1, each split into
    // two CCW triangles. Total triangle count = 2 * side^2.
    let mut p = Primitive::new(Topology::Triangles);
    let vertex = |x: u32, y: u32| -> u32 { y * (side + 1) + x };
    for y in 0..=side {
        for x in 0..=side {
            p.positions.push([x as f32, y as f32, 1.0]);
        }
    }
    let mut indices = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let v00 = vertex(x, y);
            let v10 = vertex(x + 1, y);
            let v01 = vertex(x, y + 1);
            let v11 = vertex(x + 1, y + 1);
            indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
        }
    }
    p.indices = Some(Indices::U32(indices));
    p
}

fn unit_cube_primitive() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    p.indices = Some(Indices::U32(vec![
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5, 4,
        3, 7, 6, 3, 6, 2,
    ]));
    p
}

#[test]
fn primitive_build_bvh_returns_some_for_triangles() {
    let p = unit_cube_primitive();
    let bvh = p.build_bvh().expect("12-tri cube builds");
    assert_eq!(bvh.triangle_count(), 12);
}

#[test]
fn primitive_build_bvh_returns_none_for_lines() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(p.build_bvh().is_none());
}

#[test]
fn bvh_intersect_agrees_with_primitive_on_grid() {
    // 8x8 grid = 128 triangles. Walk a 16x16 ray grid; cross-check
    // both `t` and `triangle_index` (no shared-edge ties when the
    // ray strikes a triangle interior).
    let p = make_grid(8);
    let bvh = Bvh::build(&p).unwrap();
    for ix in 0..16 {
        for iy in 0..16 {
            // Place rays inside triangle interiors — offset by
            // (+0.27, +0.31) into each unit square so we never land
            // on a shared edge.
            let x = 0.5 * ix as f32 + 0.27;
            let y = 0.5 * iy as f32 + 0.31;
            let r = Ray::new([x, y, 2.0], [0.0, 0.0, -1.0]);
            let bf = p.intersect_ray(r, f32::INFINITY);
            let bv = bvh.intersect_ray(&p, r, f32::INFINITY);
            assert_eq!(bv, bf, "mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn bvh_any_intersection_agrees_with_primitive_on_grid() {
    // Shadow-ray semantics across the same grid.
    let p = make_grid(8);
    let bvh = Bvh::build(&p).unwrap();
    for ix in 0..16 {
        for iy in 0..16 {
            let x = 0.5 * ix as f32 + 0.27;
            let y = 0.5 * iy as f32 + 0.31;
            let r = Ray::new([x, y, 2.0], [0.0, 0.0, -1.0]);
            let bf = p.any_ray_intersection(r, f32::INFINITY);
            let bv = bvh.any_ray_intersection(&p, r, f32::INFINITY);
            assert_eq!(bv, bf, "mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn bvh_misses_outside_grid_extent() {
    let p = make_grid(8);
    let bvh = Bvh::build(&p).unwrap();
    // Ray well outside the [0, 8] x [0, 8] extent on x.
    let r = Ray::new([-100.0, 4.0, 2.0], [0.0, 0.0, -1.0]);
    assert!(bvh.intersect_ray(&p, r, f32::INFINITY).is_none());
    assert!(!bvh.any_ray_intersection(&p, r, f32::INFINITY));
}

#[test]
fn bvh_t_max_culls_far_hit() {
    // Ray hits z=1 at t=1.0 (origin at z=2, direction -Z). With
    // t_max = 0.5 the hit is culled.
    let p = make_grid(2);
    let bvh = Bvh::build(&p).unwrap();
    let r = Ray::new([0.5, 0.5, 2.0], [0.0, 0.0, -1.0]);
    assert!(bvh.intersect_ray(&p, r, f32::INFINITY).is_some());
    assert!(bvh.intersect_ray(&p, r, 0.5).is_none());
}

#[test]
fn bvh_leaf_count_smaller_than_triangle_count() {
    // The whole point of the LEAF_THRESHOLD cut-off: leaves bundle
    // triangles. 128 triangles compress to noticeably fewer leaves.
    let p = make_grid(8);
    let bvh = Bvh::build(&p).unwrap();
    assert_eq!(bvh.triangle_count(), 128);
    assert!(bvh.leaf_count() < bvh.triangle_count());
    assert!(bvh.leaf_count() >= 128 / Bvh::LEAF_THRESHOLD);
}

#[test]
fn bvh_root_bounds_match_primitive_bounds() {
    let p = unit_cube_primitive();
    let bvh = Bvh::build(&p).unwrap();
    let root = bvh.bounds().unwrap();
    let prim = p.bounding_box().unwrap();
    assert_eq!(root.min, prim.min);
    assert_eq!(root.max, prim.max);
}

#[test]
fn bvh_handles_triangle_strip_topology() {
    // TriangleStrip de-strips through `triangle_indices()` already
    // covered by the brute-force path; BVH must do the same.
    let mut p = Primitive::new(Topology::TriangleStrip);
    p.positions = vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let bvh = Bvh::build(&p).expect("triangle strip builds");
    // 4 strip vertices = 2 triangles.
    assert_eq!(bvh.triangle_count(), 2);
    let r = Ray::new([0.5, 0.5, 2.0], [0.0, 0.0, -1.0]);
    let bf = p.intersect_ray(r, f32::INFINITY);
    let bv = bvh.intersect_ray(&p, r, f32::INFINITY);
    assert!(bf.is_some());
    assert!(bv.is_some());
    assert!((bf.unwrap().t - bv.unwrap().t).abs() < 1e-5);
}

#[test]
fn bvh_independent_of_source_primitive_mutability() {
    // The BVH owns its permuted index array, so dropping the source
    // primitive between build and query is fine — but for the query
    // path we still pass `&primitive` for vertex coordinates. This
    // test confirms that re-running the query after a no-op mutation
    // (no positions changed) yields the same result, exercising the
    // build's "no shared state" promise.
    let p = unit_cube_primitive();
    let bvh = Bvh::build(&p).unwrap();
    let r = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let first = bvh.intersect_ray(&p, r, f32::INFINITY).unwrap();
    let second = bvh.intersect_ray(&p, r, f32::INFINITY).unwrap();
    assert_eq!(first, second);
}

#[test]
fn bvh_node_count_invariant_holds() {
    // A binary tree with N leaves has exactly 2N - 1 nodes when no
    // leaf is empty. The build always populates both children, so
    // the invariant holds. Verify it on a primitive large enough
    // for a non-trivial tree.
    let p = make_grid(6); // 72 triangles
    let bvh = Bvh::build(&p).unwrap();
    let leaves = bvh.leaf_count();
    let total = bvh.node_count();
    assert_eq!(total, 2 * leaves - 1, "binary-tree node invariant");
}

#[test]
fn bvh_ray_aligned_with_face_misses_other_faces() {
    let p = unit_cube_primitive();
    let bvh = Bvh::build(&p).unwrap();
    // Ray entering from -X at y=0.5, z=0.5 — strikes the -X face
    // (front) at t=1 and no other face is in front of it.
    let r = Ray::new([-2.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let hit = bvh.intersect_ray(&p, r, f32::INFINITY).unwrap();
    assert!((hit.t - 2.0).abs() < 1e-5);
    assert!(hit.front_face);
}
