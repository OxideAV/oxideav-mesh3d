//! Round 285 — parametric extruded solids: `Profile2D` triangulation
//! plus closed-manifold extrusion (the swept-solid tessellation
//! kernel an IFC-style format producer consumes).

use oxideav_mesh3d::{Indices, Primitive, Profile2D};

// ---------------------------------------------------------------
// helpers
// ---------------------------------------------------------------

fn square() -> Vec<[f32; 2]> {
    vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
}

/// L-shape: unit square minus its top-right 0.5 x 0.5 corner.
fn l_shape() -> Vec<[f32; 2]> {
    vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [0.5, 0.5],
        [0.5, 1.0],
        [0.0, 1.0],
    ]
}

fn inner_square() -> Vec<[f32; 2]> {
    vec![[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
}

/// Flattened original vertex list the triangulation indices refer to.
fn flat(profile: &Profile2D) -> Vec<[f32; 2]> {
    let mut v = profile.outer.clone();
    for h in &profile.holes {
        v.extend_from_slice(h);
    }
    v
}

/// Signed area of one triangle over the flattened vertex list.
fn tri_area(verts: &[[f32; 2]], t: [u32; 3]) -> f64 {
    let p = |i: u32| {
        let v = verts[i as usize];
        [f64::from(v[0]), f64::from(v[1])]
    };
    let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
    ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
}

/// Sum of (signed) triangle areas; asserts every triangle is CCW.
fn ccw_area_sum(profile: &Profile2D, tris: &[[u32; 3]]) -> f64 {
    let verts = flat(profile);
    let mut sum = 0.0;
    for &t in tris {
        let a = tri_area(&verts, t);
        assert!(a > 0.0, "triangle {t:?} is not CCW (area {a})");
        sum += a;
    }
    sum
}

fn assert_close(a: f64, b: f64, eps: f64, what: &str) {
    assert!((a - b).abs() <= eps, "{what}: {a} vs {b}");
}

fn closed(prim: &Primitive) -> bool {
    prim.edge_manifold_report().is_closed_manifold()
}

// ---------------------------------------------------------------
// triangulate
// ---------------------------------------------------------------

#[test]
fn square_triangulates_to_two_ccw_triangles() {
    let p = Profile2D::new(square());
    let tris = p.triangulate().expect("square triangulates");
    assert_eq!(tris.len(), 2);
    assert_close(ccw_area_sum(&p, &tris), 1.0, 1e-12, "square area");
}

#[test]
fn cw_outer_input_is_normalised() {
    let mut pts = square();
    pts.reverse(); // clockwise input
    let p = Profile2D::new(pts);
    let tris = p.triangulate().expect("CW square still triangulates");
    assert_eq!(tris.len(), 2);
    // Triangles still come out CCW.
    assert_close(ccw_area_sum(&p, &tris), 1.0, 1e-12, "CW square area");
}

#[test]
fn triangle_profile_is_one_triangle() {
    let p = Profile2D::new(vec![[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]);
    let tris = p.triangulate().expect("triangle triangulates");
    assert_eq!(tris.len(), 1);
    assert_close(ccw_area_sum(&p, &tris), 3.0, 1e-12, "triangle area");
}

#[test]
fn concave_l_shape_triangulates() {
    let p = Profile2D::new(l_shape());
    let tris = p.triangulate().expect("L-shape triangulates");
    assert_eq!(tris.len(), 4); // n - 2
    assert_close(ccw_area_sum(&p, &tris), 0.75, 1e-12, "L-shape area");
}

#[test]
fn too_few_vertices_is_none() {
    assert!(Profile2D::new(vec![]).triangulate().is_none());
    assert!(Profile2D::new(vec![[0.0, 0.0]]).triangulate().is_none());
    assert!(Profile2D::new(vec![[0.0, 0.0], [1.0, 0.0]])
        .triangulate()
        .is_none());
}

#[test]
fn collinear_outer_is_none() {
    let p = Profile2D::new(vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
    assert!(p.triangulate().is_none(), "zero-area outer loop");
}

#[test]
fn non_finite_coordinate_is_none() {
    let p = Profile2D::new(vec![[0.0, 0.0], [1.0, f32::NAN], [0.0, 1.0]]);
    assert!(p.triangulate().is_none());
    let p =
        Profile2D::new(square()).with_hole(vec![[0.25, 0.25], [f32::INFINITY, 0.25], [0.5, 0.5]]);
    assert!(p.triangulate().is_none());
}

#[test]
fn closing_duplicate_point_is_ignored() {
    let mut pts = square();
    pts.push([0.0, 0.0]); // explicit closure, wire-format style
    let p = Profile2D::new(pts);
    let tris = p.triangulate().expect("closed polyline triangulates");
    assert_eq!(tris.len(), 2);
    // The duplicate slot (index 4) is never referenced.
    assert!(tris.iter().flatten().all(|&i| i < 4));
    assert_close(ccw_area_sum(&p, &tris), 1.0, 1e-12, "area");
}

#[test]
fn consecutive_duplicate_point_is_ignored() {
    let p = Profile2D::new(vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0], // exact repeat
        [1.0, 1.0],
        [0.0, 1.0],
    ]);
    let tris = p.triangulate().expect("duplicate point tolerated");
    assert_eq!(tris.len(), 2);
    assert_close(ccw_area_sum(&p, &tris), 1.0, 1e-12, "area");
}

#[test]
fn collinear_midpoint_is_kept_and_covered() {
    // Square with an extra vertex in the middle of the bottom edge.
    let p = Profile2D::new(vec![
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]);
    let tris = p.triangulate().expect("collinear midpoint tolerated");
    assert_eq!(tris.len(), 3); // n - 2 with the midpoint participating
    assert_close(ccw_area_sum(&p, &tris), 1.0, 1e-12, "area");
}

#[test]
fn square_with_hole_triangulates() {
    let p = Profile2D::new(square()).with_hole(inner_square());
    let tris = p.triangulate().expect("holed square triangulates");
    // n + 2h - 2 = 8 + 2 - 2.
    assert_eq!(tris.len(), 8);
    assert_close(ccw_area_sum(&p, &tris), 0.75, 1e-12, "annulus area");
    // Hole vertices (flattened indices 4..8) are referenced.
    assert!(tris.iter().flatten().any(|&i| i >= 4));
    assert!(tris.iter().flatten().all(|&i| i < 8));
}

#[test]
fn hole_winding_is_normalised() {
    let mut hole = inner_square();
    hole.reverse(); // pass the hole CW vs CCW — both must work
    let a = Profile2D::new(square()).with_hole(inner_square());
    let b = Profile2D::new(square()).with_hole(hole);
    let ta = a.triangulate().expect("CCW hole");
    let tb = b.triangulate().expect("CW hole");
    assert_close(
        ccw_area_sum(&a, &ta),
        ccw_area_sum(&b, &tb),
        1e-12,
        "hole winding parity",
    );
}

#[test]
fn two_holes_triangulate() {
    let p = Profile2D::new(vec![[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0]])
        .with_hole(vec![[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
        .with_hole(vec![[2.25, 0.25], [2.75, 0.25], [2.75, 0.75], [2.25, 0.75]]);
    let tris = p.triangulate().expect("two holes triangulate");
    // n + 2h - 2 = 12 + 4 - 2.
    assert_eq!(tris.len(), 14);
    assert_close(ccw_area_sum(&p, &tris), 3.0 - 0.5, 1e-12, "two-hole area");
}

#[test]
fn degenerate_hole_is_none() {
    let p = Profile2D::new(square()).with_hole(vec![[0.25, 0.25], [0.75, 0.75]]);
    assert!(p.triangulate().is_none(), "2-point hole");
    let p = Profile2D::new(square()).with_hole(vec![[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]]);
    assert!(p.triangulate().is_none(), "collinear hole");
}

#[test]
fn hole_right_of_outer_is_none() {
    // The +x bridge ray from the hole never crosses the outer ring.
    let p = Profile2D::new(square()).with_hole(vec![
        [2.0, 0.25],
        [2.5, 0.25],
        [2.5, 0.75],
        [2.0, 0.75],
    ]);
    assert!(p.triangulate().is_none(), "hole outside the boundary");
}

#[test]
fn area_accessor() {
    assert_close(Profile2D::new(square()).area(), 1.0, 1e-12, "square");
    assert_close(
        Profile2D::new(square()).with_hole(inner_square()).area(),
        0.75,
        1e-12,
        "annulus",
    );
    let mut cw = square();
    cw.reverse();
    assert_close(Profile2D::new(cw).area(), 1.0, 1e-12, "winding-agnostic");
    assert_close(Profile2D::new(vec![]).area(), 0.0, 0.0, "empty");
}

#[test]
fn vertex_count_accessor() {
    let p = Profile2D::new(square()).with_hole(inner_square());
    assert_eq!(p.vertex_count(), 8);
    assert_eq!(Profile2D::default().vertex_count(), 0);
}

// ---------------------------------------------------------------
// extrude
// ---------------------------------------------------------------

#[test]
fn unit_cube_extrusion() {
    let prim = Profile2D::new(square())
        .extrude([0.0, 0.0, 1.0], 1.0)
        .expect("cube extrudes");
    assert_eq!(prim.positions.len(), 8);
    // 2 + 2 cap triangles + 4 edges x 2 wall triangles.
    assert_eq!(prim.triangle_count(), 12);
    assert!(matches!(prim.indices, Some(Indices::U16(_))));
    assert!(closed(&prim), "cube is a closed two-manifold");
    assert!(prim.boundary_edges().is_empty());
    assert_close(prim.signed_volume(), 1.0, 1e-6, "cube volume");
    assert_close(prim.surface_area(), 6.0, 1e-6, "cube surface");
}

#[test]
fn position_layout_is_bottom_ring_then_top_ring() {
    let prim = Profile2D::new(square())
        .extrude([0.0, 0.0, 1.0], 2.5)
        .expect("extrudes");
    for (i, p) in square().iter().enumerate() {
        assert_eq!(prim.positions[i], [p[0], p[1], 0.0], "bottom {i}");
        assert_eq!(prim.positions[i + 4], [p[0], p[1], 2.5], "top {i}");
    }
}

#[test]
fn downward_extrusion_stays_outward_facing() {
    let prim = Profile2D::new(square())
        .extrude([0.0, 0.0, -1.0], 1.0)
        .expect("downward extrudes");
    assert!(closed(&prim));
    // Winding is flipped so the enclosed volume is still positive.
    assert_close(prim.signed_volume(), 1.0, 1e-6, "downward volume");
    assert_eq!(prim.positions[4][2], -1.0, "top ring sits below");
}

#[test]
fn oblique_extrusion_shears_the_prism() {
    // direction (1, 0, 1)/sqrt(2), depth sqrt(2) => offset (1, 0, 1):
    // a sheared prism with base area 1 and vertical height 1.
    let prim = Profile2D::new(square())
        .extrude([1.0, 0.0, 1.0], std::f32::consts::SQRT_2)
        .expect("oblique extrudes");
    assert!(closed(&prim));
    assert_close(prim.signed_volume(), 1.0, 1e-5, "sheared volume");
    let bb = prim.bounding_box().expect("bounds");
    assert_close(f64::from(bb.max[0]), 2.0, 1e-6, "sheared +x extent");
    assert_close(f64::from(bb.max[2]), 1.0, 1e-6, "z extent");
}

#[test]
fn direction_is_normalised_before_depth_applies() {
    // |direction| = 5 must not multiply the translation length.
    let prim = Profile2D::new(square())
        .extrude([0.0, 0.0, 5.0], 2.0)
        .expect("extrudes");
    assert_eq!(prim.positions[4][2], 2.0);
    assert_close(prim.signed_volume(), 2.0, 1e-6, "volume");
}

#[test]
fn invalid_depth_is_none() {
    let p = Profile2D::new(square());
    assert!(p.extrude([0.0, 0.0, 1.0], 0.0).is_none(), "zero depth");
    assert!(p.extrude([0.0, 0.0, 1.0], -1.0).is_none(), "negative depth");
    assert!(p.extrude([0.0, 0.0, 1.0], f32::NAN).is_none(), "NaN depth");
    assert!(
        p.extrude([0.0, 0.0, 1.0], f32::INFINITY).is_none(),
        "infinite depth"
    );
}

#[test]
fn invalid_direction_is_none() {
    let p = Profile2D::new(square());
    assert!(p.extrude([0.0, 0.0, 0.0], 1.0).is_none(), "zero direction");
    // Perpendicular to the z axis (dz == 0) sweeps no solid.
    assert!(p.extrude([1.0, 0.0, 0.0], 1.0).is_none(), "in-plane");
    assert!(p.extrude([0.0, 1.0, 0.0], 1.0).is_none(), "in-plane y");
    assert!(p.extrude([f32::NAN, 0.0, 1.0], 1.0).is_none(), "NaN");
}

#[test]
fn degenerate_profile_extrusion_is_none() {
    assert!(Profile2D::new(vec![])
        .extrude([0.0, 0.0, 1.0], 1.0)
        .is_none());
    let collinear = Profile2D::new(vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
    assert!(collinear.extrude([0.0, 0.0, 1.0], 1.0).is_none());
}

#[test]
fn hollow_section_extrudes_watertight() {
    // 1 x 1 outer, centred 0.5 x 0.5 cavity, depth 2 — the hollow
    // box-section column case.
    let prim = Profile2D::new(square())
        .with_hole(inner_square())
        .extrude([0.0, 0.0, 1.0], 2.0)
        .expect("hollow section extrudes");
    assert_eq!(prim.positions.len(), 16);
    // Caps 8 x 2 + outer walls 4 x 2 + inner walls 4 x 2.
    assert_eq!(prim.triangle_count(), 32);
    assert!(closed(&prim), "hollow section is watertight");
    assert_close(prim.signed_volume(), 0.75 * 2.0, 1e-6, "hollow volume");
    // 2 caps of 0.75 + outer skirt 4 * 2 + inner skirt 4 * 0.5 * 2.
    assert_close(prim.surface_area(), 1.5 + 8.0 + 4.0, 1e-5, "hollow surface");
}

#[test]
fn concave_profile_extrudes_watertight() {
    let prim = Profile2D::new(l_shape())
        .extrude([0.0, 0.0, 1.0], 3.0)
        .expect("L-shape extrudes");
    assert!(closed(&prim));
    assert_close(prim.signed_volume(), 0.75 * 3.0, 1e-6, "L volume");
}

#[test]
fn two_hole_profile_extrudes_watertight() {
    let prim = Profile2D::new(vec![[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0]])
        .with_hole(vec![[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
        .with_hole(vec![[2.25, 0.25], [2.75, 0.25], [2.75, 0.75], [2.25, 0.75]])
        .extrude([0.0, 0.0, 1.0], 1.0)
        .expect("two-hole slab extrudes");
    assert!(closed(&prim));
    assert_close(prim.signed_volume(), 2.5, 1e-6, "two-hole volume");
}

#[test]
fn closed_polyline_profile_extrudes_watertight() {
    // Explicitly-closed outer loop (last == first) — the wall pass
    // and the cap pass must agree on the cleaned loop.
    let mut pts = square();
    pts.push([0.0, 0.0]);
    let prim = Profile2D::new(pts)
        .extrude([0.0, 0.0, 1.0], 1.0)
        .expect("closed polyline extrudes");
    assert!(closed(&prim), "duplicate closure stays watertight");
    assert_close(prim.signed_volume(), 1.0, 1e-6, "volume");
    // The duplicate vertex slots exist but are never referenced.
    assert_eq!(prim.positions.len(), 10);
}

#[test]
fn attribute_slots_are_left_default() {
    let prim = Profile2D::new(square())
        .extrude([0.0, 0.0, 1.0], 1.0)
        .expect("extrudes");
    assert!(prim.normals.is_none());
    assert!(prim.tangents.is_none());
    assert!(prim.uvs.is_empty());
    assert!(prim.colors.is_empty());
    assert!(prim.material.is_none());
    assert!(prim.targets.is_empty());
    assert!(prim.extras.is_empty());
    // Recompute post-passes plug straight in.
    assert_eq!(prim.compute_normals().len(), prim.positions.len());
}

#[test]
fn extrusion_introduces_no_duplicate_vertices() {
    let prim = Profile2D::new(square())
        .extrude([0.0, 0.0, 1.0], 1.0)
        .expect("extrudes");
    let welded = prim.weld_vertices();
    assert_eq!(welded.positions.len(), 8, "vertex pool already minimal");
    assert!(closed(&welded));
}

#[test]
fn volume_centroid_of_extruded_cube_is_its_centre() {
    let prim = Profile2D::new(square())
        .extrude([0.0, 0.0, 1.0], 1.0)
        .expect("extrudes");
    let c = prim.volume_centroid().expect("closed solid has a centre");
    assert_close(c[0], 0.5, 1e-6, "cx");
    assert_close(c[1], 0.5, 1e-6, "cy");
    assert_close(c[2], 0.5, 1e-6, "cz");
}

#[test]
fn large_profile_promotes_to_u32_indices() {
    // 33 000-gon: 66 000 pooled vertices > 65 536 forces U32.
    let n = 33_000usize;
    let pts: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            let a = (i as f64) / (n as f64) * std::f64::consts::TAU;
            [a.cos() as f32, a.sin() as f32]
        })
        .collect();
    let prim = Profile2D::new(pts)
        .extrude([0.0, 0.0, 1.0], 1.0)
        .expect("large polygon extrudes");
    assert_eq!(prim.positions.len(), 2 * n);
    assert!(matches!(prim.indices, Some(Indices::U32(_))));
    assert!(closed(&prim), "large prism is watertight");
    // Regular n-gon area: n/2 * sin(tau/n) * r^2.
    let area = (n as f64) / 2.0 * (std::f64::consts::TAU / (n as f64)).sin();
    assert_close(prim.signed_volume(), area, 1e-3, "cylinder volume");
}
