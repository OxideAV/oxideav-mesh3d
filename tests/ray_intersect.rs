//! Tests for the ray-mesh intersection primitive:
//! [`Ray`] + [`crate::ray::intersect_triangle`] +
//! [`crate::ray::intersect_aabb`] + [`Primitive::intersect_ray`] +
//! [`Primitive::any_ray_intersection`] + [`Mesh::intersect_ray`] +
//! [`BoundingBox::intersect_ray`].
//!
//! Spec/derivation is in the `ray` module doc-comment.

use oxideav_mesh3d::{BoundingBox, Indices, Mesh, Primitive, Ray, RayHit, Topology};

// ------------------------------------------------------------------
// Fixtures
// ------------------------------------------------------------------

/// Unit triangle in the z = 1 plane, CCW from +Z.
fn unit_z1_triangle() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    p
}

/// Unit cube as a 12-triangle soup, vertex ordering CCW from outside.
/// Min corner (0,0,0), max corner (1,1,1).
fn unit_cube_soup() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    // 8 corners
    let c = [
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [0.0, 1.0, 0.0], // 2
        [1.0, 1.0, 0.0], // 3
        [0.0, 0.0, 1.0], // 4
        [1.0, 0.0, 1.0], // 5
        [0.0, 1.0, 1.0], // 6
        [1.0, 1.0, 1.0], // 7
    ];
    // 6 faces × 2 triangles, each wound CCW from outside the cube.
    // -Z face (looking down +Z from outside, which is -Z direction):
    //   normal -Z, CCW from outside means CW when viewed from +Z.
    let faces: [[usize; 3]; 12] = [
        // -Z (bottom): outside normal -Z; CCW from outside = (0,2,1), (1,2,3)
        [0, 2, 1],
        [1, 2, 3],
        // +Z (top): outside normal +Z; CCW from outside = (4,5,6),(5,7,6)
        [4, 5, 6],
        [5, 7, 6],
        // -Y front: outside normal -Y; CCW from outside = (0,1,4),(1,5,4)
        [0, 1, 4],
        [1, 5, 4],
        // +Y back: outside normal +Y; CCW from outside = (2,6,3),(3,6,7)
        [2, 6, 3],
        [3, 6, 7],
        // -X left: outside normal -X; CCW from outside = (0,4,2),(2,4,6)
        [0, 4, 2],
        [2, 4, 6],
        // +X right: outside normal +X; CCW from outside = (1,3,5),(3,7,5)
        [1, 3, 5],
        [3, 7, 5],
    ];
    p.positions = c.to_vec();
    let mut idx = Vec::with_capacity(36);
    for f in &faces {
        idx.extend_from_slice(&[f[0] as u32, f[1] as u32, f[2] as u32]);
    }
    p.indices = Some(Indices::U32(idx));
    p
}

// ------------------------------------------------------------------
// Ray basics
// ------------------------------------------------------------------

#[test]
fn ray_point_at_origin() {
    let r = Ray::new([2.0, -3.0, 7.0], [0.5, 0.5, 0.5]);
    assert_eq!(r.point_at(0.0), [2.0, -3.0, 7.0]);
}

#[test]
fn ray_point_at_two() {
    let r = Ray::new([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
    assert_eq!(r.point_at(2.0), [2.0, 4.0, 6.0]);
}

#[test]
fn ray_negative_t_extrapolates_backwards() {
    let r = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    assert_eq!(r.point_at(-1.0), [-1.0, 0.0, 0.0]);
}

// ------------------------------------------------------------------
// Primitive::intersect_ray
// ------------------------------------------------------------------

#[test]
fn primitive_centre_hit_returns_hit() {
    // Triangle in z = 1 with CCW-from-+Z winding ⇒ outward normal +Z;
    // ray shooting +Z from below hits the BACK of the triangle.
    let tri = unit_z1_triangle();
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    let hit = tri.intersect_ray(ray, f32::INFINITY).expect("centre hit");
    assert!((hit.t - 1.0).abs() < 1e-5);
    assert_eq!(hit.triangle_index, 0);
    assert!(
        !hit.front_face,
        "ray shooting along the normal hits the back"
    );
    // Barycentric sums to ~1
    let sum: f32 = hit.barycentric.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn primitive_miss_outside_simplex() {
    let tri = unit_z1_triangle();
    let ray = Ray::new([2.0, 2.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(tri.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn primitive_miss_parallel() {
    let tri = unit_z1_triangle();
    let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    assert!(tri.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn primitive_miss_behind_origin() {
    let tri = unit_z1_triangle();
    // Origin above the triangle, ray pointing further away.
    let ray = Ray::new([0.3, 0.3, 2.0], [0.0, 0.0, 1.0]);
    assert!(tri.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn primitive_front_face_hit_reported() {
    // Ray coming from above (+Z side, the front), shooting -Z, hits
    // the front face.
    let tri = unit_z1_triangle();
    let ray = Ray::new([0.3, 0.3, 2.0], [0.0, 0.0, -1.0]);
    let hit = tri.intersect_ray(ray, f32::INFINITY).expect("front hit");
    assert!(hit.front_face, "ray opposing the normal hits the front");
}

#[test]
fn primitive_t_max_culls() {
    let tri = unit_z1_triangle();
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    // Hit is at t = 1.0; cap at 0.5.
    assert!(tri.intersect_ray(ray, 0.5).is_none());
}

#[test]
fn primitive_closest_hit_among_overlapping() {
    // Two parallel triangles on z = 1 and z = 2.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
    ];
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    let hit = p.intersect_ray(ray, f32::INFINITY).expect("closest hit");
    assert!((hit.t - 1.0).abs() < 1e-5, "closest = z=1, got t={}", hit.t);
    assert_eq!(hit.triangle_index, 0, "first triangle");
}

#[test]
fn primitive_closest_hit_far_triangle_when_near_culled() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
    ];
    // Start at z = 1.5, between the two triangles, going +Z.
    let ray = Ray::new([0.3, 0.3, 1.5], [0.0, 0.0, 1.0]);
    let hit = p.intersect_ray(ray, f32::INFINITY).expect("far hit");
    assert!((hit.t - 0.5).abs() < 1e-5);
    assert_eq!(hit.triangle_index, 1, "second triangle (z=2)");
}

#[test]
fn primitive_cube_x_ray_hits_two_faces_returns_nearest() {
    let cube = unit_cube_soup();
    // Origin at x = -1, shoot +X through centre.
    let ray = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let hit = cube.intersect_ray(ray, f32::INFINITY).expect("cube hit");
    // Near face is at x = 0, so t = 1.0
    assert!((hit.t - 1.0).abs() < 1e-5);
    // Hit point on -X face → triangle index 8 or 9.
    assert!(hit.triangle_index == 8 || hit.triangle_index == 9);
    assert!(hit.front_face, "ray strikes the outward-facing -X side");
}

#[test]
fn primitive_cube_diagonal_hit() {
    let cube = unit_cube_soup();
    let ray = Ray::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let hit = cube
        .intersect_ray(ray, f32::INFINITY)
        .expect("diagonal hit");
    // Should hit the cube near the (0, 0, 0) corner.
    let pt = ray.point_at(hit.t);
    // Hit on one of the three near faces — at least one coord is ~0.
    assert!(pt[0] <= 1e-4 || pt[1] <= 1e-4 || pt[2] <= 1e-4);
    assert!(hit.front_face);
}

#[test]
fn primitive_indexed_u16_resolves_through_indices() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [10.0, 10.0, 10.0], // unused
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    p.indices = Some(Indices::U16(vec![1, 2, 3]));
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    let hit = p.intersect_ray(ray, f32::INFINITY).expect("indexed hit");
    assert!((hit.t - 1.0).abs() < 1e-5);
}

#[test]
fn primitive_indexed_u32_resolves_through_indices() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2]));
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    assert!(p.intersect_ray(ray, f32::INFINITY).is_some());
}

#[test]
fn primitive_out_of_range_index_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 99])); // 99 out of range
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    assert!(p.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn primitive_degenerate_triangle_skipped() {
    // Three collinear corners → degenerate; ray should miss everything.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]];
    let ray = Ray::new([0.5, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(p.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn primitive_nan_position_skipped() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[f32::NAN, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    assert!(p.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn primitive_zero_length_ray_misses() {
    let tri = unit_z1_triangle();
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 0.0]);
    // det = 0 → epsilon check culls.
    assert!(tri.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn primitive_lines_topology_returns_none() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let ray = Ray::new([0.5, -1.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(p.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn primitive_points_topology_returns_none() {
    let mut p = Primitive::new(Topology::Points);
    p.positions = vec![[0.0, 0.0, 0.0]];
    let ray = Ray::new([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(p.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn primitive_strip_topology_matches_list() {
    // 4-vertex strip = 2 triangles in CCW alternating winding.
    let mut strip = Primitive::new(Topology::TriangleStrip);
    strip.positions = vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let ray = Ray::new([0.7, 0.7, 0.0], [0.0, 0.0, 1.0]);
    // Hit should land on the second triangle of the strip.
    let hit = strip.intersect_ray(ray, f32::INFINITY).expect("strip hit");
    assert!((hit.t - 1.0).abs() < 1e-5);
}

#[test]
fn primitive_fan_topology_matches_list() {
    let mut fan = Primitive::new(Topology::TriangleFan);
    fan.positions = vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let ray = Ray::new([0.5, 0.5, 0.0], [0.0, 0.0, 1.0]);
    let hit = fan.intersect_ray(ray, f32::INFINITY).expect("fan hit");
    assert!((hit.t - 1.0).abs() < 1e-5);
}

#[test]
fn primitive_barycentric_sums_to_one_at_centre() {
    let tri = unit_z1_triangle();
    let ray = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = tri.intersect_ray(ray, f32::INFINITY).expect("hit");
    let sum: f32 = hit.barycentric.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // All non-negative inside the simplex.
    for c in hit.barycentric {
        assert!((0.0..=1.0).contains(&c), "bary coord {} out of range", c);
    }
}

#[test]
fn primitive_barycentric_reconstructs_hit_point() {
    let tri = unit_z1_triangle();
    let ray = Ray::new([0.2, 0.5, 0.0], [0.0, 0.0, 1.0]);
    let hit = tri.intersect_ray(ray, f32::INFINITY).expect("hit");
    let p0 = [0.0_f32, 0.0, 1.0];
    let p1 = [1.0_f32, 0.0, 1.0];
    let p2 = [0.0_f32, 1.0, 1.0];
    let [w, u, v] = hit.barycentric;
    let recon = [
        w * p0[0] + u * p1[0] + v * p2[0],
        w * p0[1] + u * p1[1] + v * p2[1],
        w * p0[2] + u * p1[2] + v * p2[2],
    ];
    let actual = ray.point_at(hit.t);
    for axis in 0..3 {
        assert!(
            (recon[axis] - actual[axis]).abs() < 1e-4,
            "axis {} recon {} actual {}",
            axis,
            recon[axis],
            actual[axis]
        );
    }
}

#[test]
fn primitive_empty_returns_none() {
    let p = Primitive::new(Topology::Triangles);
    let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(p.intersect_ray(ray, f32::INFINITY).is_none());
}

// ------------------------------------------------------------------
// Primitive::any_ray_intersection
// ------------------------------------------------------------------

#[test]
fn any_ray_intersection_true_on_hit() {
    let tri = unit_z1_triangle();
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    assert!(tri.any_ray_intersection(ray, f32::INFINITY));
}

#[test]
fn any_ray_intersection_false_on_miss() {
    let tri = unit_z1_triangle();
    let ray = Ray::new([2.0, 2.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(!tri.any_ray_intersection(ray, f32::INFINITY));
}

#[test]
fn any_ray_intersection_respects_t_max() {
    let tri = unit_z1_triangle();
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    assert!(!tri.any_ray_intersection(ray, 0.5));
    assert!(tri.any_ray_intersection(ray, 1.5));
}

#[test]
fn any_ray_intersection_lines_false() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let ray = Ray::new([0.5, -1.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(!p.any_ray_intersection(ray, f32::INFINITY));
}

// ------------------------------------------------------------------
// Mesh::intersect_ray
// ------------------------------------------------------------------

#[test]
fn mesh_intersect_ray_routes_to_primitive() {
    let mesh = Mesh::new(Some("m".into())).with_primitive(unit_z1_triangle());
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    let (prim_idx, hit) = mesh.intersect_ray(ray, f32::INFINITY).expect("hit");
    assert_eq!(prim_idx, 0);
    assert!((hit.t - 1.0).abs() < 1e-5);
}

#[test]
fn mesh_intersect_ray_picks_closest_across_primitives() {
    // Two triangles in two primitives — second is closer.
    let mut far = Primitive::new(Topology::Triangles);
    far.positions = vec![[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]];
    let near = unit_z1_triangle();
    let mesh = Mesh::new(None).with_primitive(far).with_primitive(near);
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    let (prim_idx, hit) = mesh.intersect_ray(ray, f32::INFINITY).expect("hit");
    assert_eq!(prim_idx, 1, "near primitive wins");
    assert!((hit.t - 1.0).abs() < 1e-5);
}

#[test]
fn mesh_intersect_ray_empty_returns_none() {
    let mesh = Mesh::new(None);
    let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(mesh.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn mesh_intersect_ray_all_miss_returns_none() {
    let mesh = Mesh::new(None).with_primitive(unit_z1_triangle());
    let ray = Ray::new([2.0, 2.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(mesh.intersect_ray(ray, f32::INFINITY).is_none());
}

// ------------------------------------------------------------------
// BoundingBox::intersect_ray
// ------------------------------------------------------------------

#[test]
fn aabb_intersect_through_centre_axis_aligned() {
    let bb = BoundingBox {
        min: [0.0, 0.0, 0.0],
        max: [1.0, 1.0, 1.0],
    };
    let ray = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let (te, tx) = bb.intersect_ray(ray, f32::INFINITY).expect("hit");
    assert!((te - 1.0).abs() < 1e-6);
    assert!((tx - 2.0).abs() < 1e-6);
}

#[test]
fn aabb_intersect_diagonal() {
    let bb = BoundingBox {
        min: [0.0, 0.0, 0.0],
        max: [1.0, 1.0, 1.0],
    };
    // Unit direction along the diagonal; entry at the near corner.
    let ray = Ray::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let (te, tx) = bb.intersect_ray(ray, f32::INFINITY).expect("hit");
    assert!((te - 1.0).abs() < 1e-6, "entry t = {}", te);
    assert!((tx - 2.0).abs() < 1e-6, "exit t = {}", tx);
}

#[test]
fn aabb_intersect_origin_inside_box_t_enter_zero() {
    let bb = BoundingBox {
        min: [0.0, 0.0, 0.0],
        max: [1.0, 1.0, 1.0],
    };
    let ray = Ray::new([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let (te, tx) = bb.intersect_ray(ray, f32::INFINITY).expect("hit");
    assert_eq!(te, 0.0);
    assert!((tx - 0.5).abs() < 1e-6);
}

#[test]
fn aabb_intersect_miss_to_the_side() {
    let bb = BoundingBox {
        min: [0.0, 0.0, 0.0],
        max: [1.0, 1.0, 1.0],
    };
    let ray = Ray::new([-1.0, 2.0, 0.5], [1.0, 0.0, 0.0]);
    assert!(bb.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn aabb_intersect_t_max_culls_far_box() {
    let bb = BoundingBox {
        min: [10.0, 0.0, 0.0],
        max: [11.0, 1.0, 1.0],
    };
    let ray = Ray::new([0.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    // box at t ∈ [10, 11]; cap at 5.
    assert!(bb.intersect_ray(ray, 5.0).is_none());
}

#[test]
fn aabb_intersect_axis_parallel_outside_slab_misses() {
    let bb = BoundingBox {
        min: [0.0, 0.0, 0.0],
        max: [1.0, 1.0, 1.0],
    };
    let ray = Ray::new([2.0, -1.0, 0.5], [0.0, 1.0, 0.0]);
    assert!(bb.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn aabb_intersect_axis_parallel_inside_slab_passes() {
    let bb = BoundingBox {
        min: [0.0, 0.0, 0.0],
        max: [1.0, 1.0, 1.0],
    };
    let ray = Ray::new([0.5, -1.0, 0.5], [0.0, 1.0, 0.0]);
    let (te, tx) = bb.intersect_ray(ray, f32::INFINITY).expect("hit");
    assert!((te - 1.0).abs() < 1e-6);
    assert!((tx - 2.0).abs() < 1e-6);
}

// ------------------------------------------------------------------
// Combined: AABB early-out + per-primitive intersect
// ------------------------------------------------------------------

#[test]
fn aabb_culls_before_primitive_walk() {
    // Cheap acceleration pattern: skip the per-triangle walk when the
    // primitive's AABB misses the ray.
    let cube = unit_cube_soup();
    let bb = cube.bounding_box().expect("cube has positions");
    // Ray that doesn't even enter the AABB.
    let ray = Ray::new([-1.0, 5.0, 0.5], [1.0, 0.0, 0.0]);
    assert!(bb.intersect_ray(ray, f32::INFINITY).is_none());
    // Sanity: the primitive-level walk also misses (same result).
    assert!(cube.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn aabb_passes_then_primitive_hits() {
    let cube = unit_cube_soup();
    let bb = cube.bounding_box().unwrap();
    let ray = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let (te, _tx) = bb.intersect_ray(ray, f32::INFINITY).expect("AABB hit");
    let prim_hit = cube.intersect_ray(ray, f32::INFINITY).expect("prim hit");
    // Prim hit's t is on a face inside the AABB, so t_enter ≤ prim_t.
    assert!(te <= prim_hit.t + 1e-5);
}

// ------------------------------------------------------------------
// RayHit determinism
// ------------------------------------------------------------------

#[test]
fn ray_hit_is_clone_copy_partial_eq() {
    let h = RayHit {
        t: 1.0,
        triangle_index: 7,
        barycentric: [0.3, 0.3, 0.4],
        front_face: true,
    };
    let h2 = h;
    assert_eq!(h, h2);
}

#[test]
fn ray_is_clone_copy_partial_eq() {
    let r = Ray::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let r2 = r;
    assert_eq!(r, r2);
}

#[test]
fn primitive_intersect_ray_is_deterministic() {
    let cube = unit_cube_soup();
    let ray = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let a = cube.intersect_ray(ray, f32::INFINITY);
    let b = cube.intersect_ray(ray, f32::INFINITY);
    assert_eq!(a, b);
}
