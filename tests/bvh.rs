//! Tests for the [`Bvh`] ray acceleration structure.
//!
//! The contract is that `Bvh::intersect_ray` agrees bit-for-bit with
//! the brute-force `Primitive::intersect_ray` on the same input —
//! same closest `t`, same `triangle_index`, same `barycentric`, same
//! `front_face`. The BVH only changes how we get there (O(log T)
//! traversal vs O(T) sweep). These tests pin that equivalence over
//! every interesting topology / robustness case the brute-force
//! query handles, plus structural properties of the build itself
//! (node counts, bounds, leaf threshold, deterministic build /
//! query, empty / degenerate inputs).
//!
//! Algorithm / derivation references in the `bvh` module doc-comment.

use oxideav_mesh3d::{
    BoundingBox, Bvh, BvhNode, Indices, Mesh, Primitive, Ray, RayHit, Topology,
    DEFAULT_LEAF_THRESHOLD,
};

// ------------------------------------------------------------------
// Fixtures
// ------------------------------------------------------------------

/// One unit triangle in z = 1.
fn one_triangle() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    p
}

/// 12-triangle cube soup, CCW-from-outside winding.
fn unit_cube_soup() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    let c = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let faces: [[usize; 3]; 12] = [
        [0, 2, 1],
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 6],
        [0, 1, 4],
        [1, 5, 4],
        [2, 6, 3],
        [3, 6, 7],
        [0, 4, 2],
        [2, 4, 6],
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

/// A row of `n` axis-aligned unit triangles laid along +X in the
/// z = 0 plane, each at x ∈ [i, i+1]. Useful for forcing a deep
/// tree: the triangles' centroids increase monotonically, so a
/// median split on X subdivides perfectly.
fn triangle_row(n: usize) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    let mut positions = Vec::with_capacity(3 * n);
    let mut indices = Vec::with_capacity(3 * n);
    for i in 0..n {
        let x = i as f32;
        let base = positions.len() as u32;
        positions.push([x, 0.0, 0.0]);
        positions.push([x + 1.0, 0.0, 0.0]);
        positions.push([x, 1.0, 0.0]);
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);
    }
    p.positions = positions;
    p.indices = Some(Indices::U32(indices));
    p
}

// ------------------------------------------------------------------
// Build basics
// ------------------------------------------------------------------

#[test]
fn build_empty_primitive_is_empty() {
    let p = Primitive::new(Topology::Triangles);
    let bvh = Bvh::build(&p);
    assert!(bvh.is_empty());
    assert_eq!(bvh.node_count(), 0);
    assert_eq!(bvh.triangle_count(), 0);
    assert!(bvh.bounds().is_none());
}

#[test]
fn build_lines_topology_is_empty() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let bvh = Bvh::build(&p);
    assert!(bvh.is_empty(), "non-triangle topology produces empty BVH");
}

#[test]
fn build_points_topology_is_empty() {
    let mut p = Primitive::new(Topology::Points);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let bvh = Bvh::build(&p);
    assert!(bvh.is_empty());
}

#[test]
fn build_single_triangle_is_one_leaf() {
    let p = one_triangle();
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.node_count(), 1);
    assert_eq!(bvh.triangle_count(), 1);
    let root = bvh.nodes[0];
    assert!(root.is_leaf());
    assert_eq!(root.triangle_count, 1);
}

#[test]
fn build_cube_root_bounds_match_unit_cube() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    let bounds = bvh.bounds().expect("cube has bounds");
    assert_eq!(bounds.min, [0.0, 0.0, 0.0]);
    assert_eq!(bounds.max, [1.0, 1.0, 1.0]);
}

#[test]
fn build_cube_triangle_count_is_12() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 12);
}

#[test]
fn build_cube_tree_has_more_than_one_node_at_default_threshold() {
    // Default leaf threshold = 4; 12 triangles forces internal nodes.
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    assert!(
        bvh.node_count() > 1,
        "12 > 4 triangles ⇒ tree, not single leaf"
    );
}

#[test]
fn build_with_large_leaf_threshold_collapses_to_single_leaf() {
    let p = unit_cube_soup();
    let bvh = Bvh::build_with_leaf_threshold(&p, 100);
    assert_eq!(bvh.node_count(), 1);
    let root = bvh.nodes[0];
    assert!(root.is_leaf());
    assert_eq!(root.triangle_count, 12);
}

#[test]
fn build_with_threshold_zero_raised_to_one() {
    // Threshold 0 would never terminate; the build silently raises
    // it to 1. Use a row of triangles with distinct centroids so the
    // recursion bottoms out at one triangle per leaf (a degenerate-
    // centroid cluster, e.g. two cube faces sharing a centroid, can
    // legitimately collapse to a multi-triangle leaf — see
    // `build_with_coincident_centroids_collapses_to_leaf`).
    let p = triangle_row(8);
    let bvh = Bvh::build_with_leaf_threshold(&p, 0);
    assert_eq!(bvh.triangle_count(), 8);
    for node in &bvh.nodes {
        if node.is_leaf() {
            assert_eq!(node.triangle_count, 1);
        }
    }
}

#[test]
fn build_skips_degenerate_collinear_triangles() {
    let mut p = Primitive::new(Topology::Triangles);
    // One valid triangle + one collinear degenerate.
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        // collinear:
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 3, 4, 5]));
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 1, "degenerate dropped from BVH");
}

#[test]
fn build_skips_coincident_corner_triangles() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        // duplicate-corner triangle (zero area):
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 3, 4, 5]));
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 1);
}

#[test]
fn build_skips_nan_position_triangles() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        // NaN triangle:
        [f32::NAN, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 3, 4, 5]));
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 1);
}

#[test]
fn build_skips_out_of_range_index_triangles() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    // Second triangle references index 99 → out of range, dropped.
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 99]));
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 1);
}

#[test]
fn build_is_deterministic() {
    let p = unit_cube_soup();
    let a = Bvh::build(&p);
    let b = Bvh::build(&p);
    assert_eq!(a.nodes.len(), b.nodes.len());
    assert_eq!(a.triangle_indices, b.triangle_indices);
    // Pointwise node equality:
    for (na, nb) in a.nodes.iter().zip(b.nodes.iter()) {
        assert_eq!(na, nb);
    }
}

#[test]
fn build_row_yields_balanced_tree() {
    // 16 triangles laid out along +X; default leaf threshold = 4 ⇒
    // expected depth ≈ log2(16 / 4) = 2 internal levels above
    // leaves, so ≈ 4 leaves of 4 triangles each.
    let p = triangle_row(16);
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 16);
    let leaf_count = bvh.nodes.iter().filter(|n| n.is_leaf()).count();
    assert_eq!(
        leaf_count, 4,
        "expected exactly 4 leaves at threshold 4 over 16 sorted entries"
    );
}

#[test]
fn build_default_leaf_threshold_constant() {
    assert_eq!(DEFAULT_LEAF_THRESHOLD, 4);
}

#[test]
fn node_is_leaf_distinguishes_internal_and_leaf() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    let leaves: Vec<&BvhNode> = bvh.nodes.iter().filter(|n| n.is_leaf()).collect();
    let internals: Vec<&BvhNode> = bvh.nodes.iter().filter(|n| !n.is_leaf()).collect();
    assert!(!leaves.is_empty());
    assert!(!internals.is_empty());
    for leaf in &leaves {
        assert_eq!(leaf.left_child, u32::MAX);
        assert_eq!(leaf.right_child, u32::MAX);
    }
    for internal in &internals {
        assert_eq!(internal.triangle_count, 0);
        assert_ne!(internal.left_child, u32::MAX);
        assert_ne!(internal.right_child, u32::MAX);
    }
}

// ------------------------------------------------------------------
// intersect_ray — equivalence with Primitive::intersect_ray
// ------------------------------------------------------------------

/// Bit-exact RayHit equality.
fn hits_equal(a: Option<RayHit>, b: Option<RayHit>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(ha), Some(hb)) => {
            ha.t == hb.t
                && ha.triangle_index == hb.triangle_index
                && ha.barycentric == hb.barycentric
                && ha.front_face == hb.front_face
        }
        _ => false,
    }
}

#[test]
fn bvh_hits_match_brute_force_on_single_triangle() {
    let p = one_triangle();
    let bvh = Bvh::build(&p);
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    let bvh_hit = bvh.intersect_ray(&p, ray, f32::INFINITY);
    let brute = p.intersect_ray(ray, f32::INFINITY);
    assert!(bvh_hit.is_some());
    assert!(hits_equal(bvh_hit, brute));
}

#[test]
fn bvh_misses_match_brute_force_on_single_triangle() {
    let p = one_triangle();
    let bvh = Bvh::build(&p);
    let ray = Ray::new([2.0, 2.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(bvh.intersect_ray(&p, ray, f32::INFINITY).is_none());
    assert!(p.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn bvh_hits_match_brute_force_on_cube_through_origin() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    // Ray shooting +X through the centre of the -X face.
    let ray = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let bvh_hit = bvh.intersect_ray(&p, ray, f32::INFINITY);
    let brute = p.intersect_ray(ray, f32::INFINITY);
    assert!(bvh_hit.is_some());
    assert!(hits_equal(bvh_hit, brute));
    // Specifically: closest hit is the -X face at t = 1.0.
    let h = bvh_hit.unwrap();
    assert!((h.t - 1.0).abs() < 1e-5);
    assert!(h.front_face);
}

#[test]
fn bvh_hits_match_brute_force_on_cube_diagonal_offset() {
    // Pure body-diagonal ray hits the cube corner — three triangles
    // meet there so a barycentric-zero vertex tie is possible and
    // first-wins picks vary between brute-force and BVH walks. Offset
    // the ray slightly so it lands strictly in the interior of one
    // triangle, where the closest-hit is unique.
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    let ray = Ray::new([-1.0, -0.7, -0.3], [1.0, 1.0, 1.0]);
    let bvh_hit = bvh.intersect_ray(&p, ray, f32::INFINITY);
    let brute = p.intersect_ray(ray, f32::INFINITY);
    assert!(bvh_hit.is_some(), "bvh = {:?}", bvh_hit);
    assert!(
        hits_equal(bvh_hit, brute),
        "bvh = {:?}, brute = {:?}",
        bvh_hit,
        brute
    );
}

#[test]
fn bvh_misses_match_brute_force_on_cube_outside() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    // Ray passing well outside the cube along +X.
    let ray = Ray::new([-1.0, 5.0, 5.0], [1.0, 0.0, 0.0]);
    assert!(bvh.intersect_ray(&p, ray, f32::INFINITY).is_none());
    assert!(p.intersect_ray(ray, f32::INFINITY).is_none());
}

#[test]
fn bvh_equivalence_on_six_faces_of_cube() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    // Six axis-aligned rays, one through each face of the cube.
    let rays = [
        Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]), // -X face
        Ray::new([2.0, 0.5, 0.5], [-1.0, 0.0, 0.0]), // +X face
        Ray::new([0.5, -1.0, 0.5], [0.0, 1.0, 0.0]), // -Y face
        Ray::new([0.5, 2.0, 0.5], [0.0, -1.0, 0.0]), // +Y face
        Ray::new([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]), // -Z face
        Ray::new([0.5, 0.5, 2.0], [0.0, 0.0, -1.0]), // +Z face
    ];
    for r in rays {
        let bvh_hit = bvh.intersect_ray(&p, r, f32::INFINITY);
        let brute = p.intersect_ray(r, f32::INFINITY);
        assert!(bvh_hit.is_some(), "ray {:?} should hit", r);
        assert!(
            hits_equal(bvh_hit, brute),
            "ray {:?}: bvh = {:?}, brute = {:?}",
            r,
            bvh_hit,
            brute
        );
        let h = bvh_hit.unwrap();
        assert!(
            h.front_face,
            "ray {:?} approaches face from outside ⇒ front",
            r
        );
    }
}

#[test]
fn bvh_equivalence_on_triangle_row() {
    let p = triangle_row(32);
    let bvh = Bvh::build(&p);
    // Probe several rays.
    for i in 0..32 {
        let x = i as f32 + 0.5;
        let ray = Ray::new([x, 0.5, 1.0], [0.0, 0.0, -1.0]);
        let bvh_hit = bvh.intersect_ray(&p, ray, f32::INFINITY);
        let brute = p.intersect_ray(ray, f32::INFINITY);
        // Both should hit the i-th triangle at t = 1.
        // Equality is bit-exact because the math path is the same
        // Möller-Trumbore call on the same vertex data.
        assert!(
            hits_equal(bvh_hit, brute),
            "i={}: bvh = {:?}, brute = {:?}",
            i,
            bvh_hit,
            brute
        );
        let h = bvh_hit.unwrap();
        assert_eq!(h.triangle_index, i);
    }
}

#[test]
fn bvh_picks_closest_hit_with_two_planes() {
    // Two parallel triangles at z = 1 and z = 2.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        // z = 2 (farther):
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        // z = 1 (closer):
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 3, 4, 5]));
    let bvh = Bvh::build(&p);
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    let hit = bvh.intersect_ray(&p, ray, f32::INFINITY).expect("hit");
    // Closest t is the z = 1 plane.
    assert!((hit.t - 1.0).abs() < 1e-5);
    let brute = p.intersect_ray(ray, f32::INFINITY).unwrap();
    assert!(hits_equal(Some(hit), Some(brute)));
}

#[test]
fn bvh_picks_closest_hit_with_two_planes_swapped_order() {
    // Same as above but triangles in the other order in the buffer.
    // The BVH should still pick the geometrically-closest hit.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 3, 4, 5]));
    let bvh = Bvh::build(&p);
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    let hit = bvh.intersect_ray(&p, ray, f32::INFINITY).expect("hit");
    assert!((hit.t - 1.0).abs() < 1e-5);
}

#[test]
fn bvh_respects_t_max_cull() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    let ray = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    // Closest hit at t = 1. t_max = 0.5 should cull it.
    assert!(bvh.intersect_ray(&p, ray, 0.5).is_none());
    // t_max = 1.5 should report it.
    assert!(bvh.intersect_ray(&p, ray, 1.5).is_some());
}

#[test]
fn bvh_hit_on_empty_returns_none() {
    let p = Primitive::new(Topology::Triangles);
    let bvh = Bvh::build(&p);
    let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    assert!(bvh.intersect_ray(&p, ray, f32::INFINITY).is_none());
}

#[test]
fn bvh_strip_topology_matches_brute_force() {
    // TriangleStrip with 5 vertices = 3 triangles.
    let mut p = Primitive::new(Topology::TriangleStrip);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 3);
    let ray = Ray::new([0.3, 0.3, 1.0], [0.0, 0.0, -1.0]);
    let bvh_hit = bvh.intersect_ray(&p, ray, f32::INFINITY);
    let brute = p.intersect_ray(ray, f32::INFINITY);
    assert!(hits_equal(bvh_hit, brute));
}

#[test]
fn bvh_fan_topology_matches_brute_force() {
    let mut p = Primitive::new(Topology::TriangleFan);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 2);
    let ray = Ray::new([0.5, 0.5, 1.0], [0.0, 0.0, -1.0]);
    let bvh_hit = bvh.intersect_ray(&p, ray, f32::INFINITY);
    let brute = p.intersect_ray(ray, f32::INFINITY);
    assert!(hits_equal(bvh_hit, brute));
}

#[test]
fn bvh_indexed_u16_matches_brute_force() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 1, 3, 2]));
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 2);
    let ray = Ray::new([0.5, 0.5, 1.0], [0.0, 0.0, -1.0]);
    let bvh_hit = bvh.intersect_ray(&p, ray, f32::INFINITY);
    let brute = p.intersect_ray(ray, f32::INFINITY);
    assert!(hits_equal(bvh_hit, brute));
}

#[test]
fn bvh_query_is_deterministic() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    let ray = Ray::new([-1.0, 0.3, 0.7], [1.0, 0.1, 0.1]);
    let h1 = bvh.intersect_ray(&p, ray, f32::INFINITY);
    let h2 = bvh.intersect_ray(&p, ray, f32::INFINITY);
    let h3 = bvh.intersect_ray(&p, ray, f32::INFINITY);
    assert!(hits_equal(h1, h2));
    assert!(hits_equal(h2, h3));
}

// ------------------------------------------------------------------
// any_intersection
// ------------------------------------------------------------------

#[test]
fn bvh_any_intersection_true_on_hit() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    let ray = Ray::new([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!(bvh.any_intersection(&p, ray, f32::INFINITY));
}

#[test]
fn bvh_any_intersection_false_on_miss() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    let ray = Ray::new([-1.0, 5.0, 5.0], [1.0, 0.0, 0.0]);
    assert!(!bvh.any_intersection(&p, ray, f32::INFINITY));
}

#[test]
fn bvh_any_intersection_respects_t_max() {
    let p = one_triangle();
    let bvh = Bvh::build(&p);
    // Hit at t = 1.
    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);
    assert!(bvh.any_intersection(&p, ray, 2.0));
    assert!(!bvh.any_intersection(&p, ray, 0.5));
}

#[test]
fn bvh_any_intersection_false_on_empty() {
    let p = Primitive::new(Topology::Triangles);
    let bvh = Bvh::build(&p);
    let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    assert!(!bvh.any_intersection(&p, ray, f32::INFINITY));
}

#[test]
fn bvh_any_intersection_false_on_lines() {
    let mut p = Primitive::new(Topology::Lines);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let bvh = Bvh::build(&p);
    let ray = Ray::new([0.5, 0.0, -1.0], [0.0, 0.0, 1.0]);
    assert!(!bvh.any_intersection(&p, ray, f32::INFINITY));
}

// ------------------------------------------------------------------
// Structural / type properties
// ------------------------------------------------------------------

#[test]
fn bvh_is_clone_and_debug() {
    let p = one_triangle();
    let bvh = Bvh::build(&p);
    let cloned = bvh.clone();
    assert_eq!(bvh.node_count(), cloned.node_count());
    let s = format!("{:?}", bvh);
    assert!(s.contains("Bvh"));
}

#[test]
fn bvh_node_is_copy_clone_debug_partial_eq() {
    let n1 = BvhNode {
        bounds: BoundingBox {
            min: [0.0; 3],
            max: [1.0; 3],
        },
        first_triangle: 0,
        triangle_count: 1,
        left_child: u32::MAX,
        right_child: u32::MAX,
    };
    let n2 = n1;
    assert_eq!(n1, n2);
    // Explicit Clone impl exercise — Copy promotes the call, so use
    // the trait path to call it without the clippy `clone_on_copy`
    // lint.
    let n3 = Clone::clone(&n1);
    assert_eq!(n1, n3);
    let s = format!("{:?}", n1);
    assert!(s.contains("BvhNode"));
}

#[test]
fn bvh_node_internal_carries_no_triangles() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    for node in &bvh.nodes {
        if !node.is_leaf() {
            assert_eq!(node.triangle_count, 0);
            assert_eq!(node.first_triangle, u32::MAX);
        }
    }
}

#[test]
fn bvh_leaf_triangle_slice_in_range() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    for node in &bvh.nodes {
        if node.is_leaf() {
            let begin = node.first_triangle as usize;
            let end = begin + node.triangle_count as usize;
            assert!(end <= bvh.triangle_indices.len());
        }
    }
}

#[test]
fn bvh_node_bounds_contain_children_bounds() {
    // Standard BVH invariant: every internal node's AABB must
    // contain both children's AABBs.
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    for node in &bvh.nodes {
        if !node.is_leaf() {
            let left = &bvh.nodes[node.left_child as usize];
            let right = &bvh.nodes[node.right_child as usize];
            for axis in 0..3 {
                assert!(node.bounds.min[axis] <= left.bounds.min[axis] + 1e-6);
                assert!(node.bounds.max[axis] >= left.bounds.max[axis] - 1e-6);
                assert!(node.bounds.min[axis] <= right.bounds.min[axis] + 1e-6);
                assert!(node.bounds.max[axis] >= right.bounds.max[axis] - 1e-6);
            }
        }
    }
}

#[test]
fn bvh_triangle_indices_permutation_is_complete() {
    // Every valid source triangle should appear exactly once in the
    // BVH's permutation (modulo degenerates dropped at build).
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    let mut seen: Vec<u32> = bvh.triangle_indices.clone();
    seen.sort();
    let expected: Vec<u32> = (0..12).collect();
    assert_eq!(seen, expected);
}

#[test]
fn bvh_root_bounds_match_primitive_bounding_box() {
    let p = unit_cube_soup();
    let bvh = Bvh::build(&p);
    let bvh_b = bvh.bounds().unwrap();
    let p_b = p.bounding_box().unwrap();
    assert_eq!(bvh_b.min, p_b.min);
    assert_eq!(bvh_b.max, p_b.max);
}

// ------------------------------------------------------------------
// Edge cases — coincident centroids, all-collinear cluster
// ------------------------------------------------------------------

#[test]
fn build_with_coincident_centroids_collapses_to_leaf() {
    // Three identical (overlapping) triangles share a centroid;
    // there's no meaningful split axis ⇒ a single leaf.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 2, 0, 1, 2]));
    let bvh = Bvh::build(&p);
    assert_eq!(bvh.triangle_count(), 3);
    // Degenerate-centroid case collapses to a single leaf because
    // there's no axis with non-zero centroid extent.
    assert_eq!(bvh.node_count(), 1);
    let root = bvh.nodes[0];
    assert!(root.is_leaf());
    assert_eq!(root.triangle_count, 3);
}

#[test]
fn build_with_one_triangle_above_threshold() {
    // Three triangles + threshold 2 ⇒ root is internal, two leaves.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [20.0, 0.0, 0.0],
        [21.0, 0.0, 0.0],
        [20.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]));
    let bvh = Bvh::build_with_leaf_threshold(&p, 2);
    assert!(bvh.node_count() >= 3);
    assert_eq!(bvh.triangle_count(), 3);
}

// ------------------------------------------------------------------
// Mesh-equivalent path: walk via Bvh on every primitive
// ------------------------------------------------------------------

#[test]
fn bvh_per_primitive_walk_matches_mesh_intersect_ray() {
    // Build a mesh with two primitives and confirm the per-primitive
    // BVH walk picks the same closest hit as Mesh::intersect_ray.
    let mut a = Primitive::new(Topology::Triangles);
    a.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    let mut b = Primitive::new(Topology::Triangles);
    b.positions = vec![[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]];
    let mesh = Mesh::new(None)
        .with_primitive(a.clone())
        .with_primitive(b.clone());

    let bvh_a = Bvh::build(&a);
    let bvh_b = Bvh::build(&b);

    let ray = Ray::new([0.3, 0.3, 0.0], [0.0, 0.0, 1.0]);

    // Brute mesh:
    let (mesh_idx, mesh_hit) = mesh.intersect_ray(ray, f32::INFINITY).expect("hit");
    assert_eq!(mesh_idx, 0);

    // BVH walk:
    let h_a = bvh_a.intersect_ray(&a, ray, f32::INFINITY);
    let h_b = bvh_b.intersect_ray(&b, ray, f32::INFINITY);
    let bvh_closest = match (h_a, h_b) {
        (Some(ha), Some(hb)) => {
            if ha.t <= hb.t {
                (0, ha)
            } else {
                (1, hb)
            }
        }
        (Some(ha), None) => (0, ha),
        (None, Some(hb)) => (1, hb),
        (None, None) => unreachable!("both should hit"),
    };
    assert_eq!(bvh_closest.0, mesh_idx);
    assert!(hits_equal(Some(bvh_closest.1), Some(mesh_hit)));
}
