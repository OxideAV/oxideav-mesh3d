//! Integration tests for `Scene3D::intersect_ray` /
//! `Scene3D::any_ray_intersection` — world-space ray queries against
//! reachable node-mesh instances.
//!
//! Validation strategy:
//! * Build a Scene3D with one or more node-mesh instances at known
//!   transforms.
//! * Shoot a hand-constructed ray whose hit point in world space we
//!   know.
//! * Confirm `Scene3D::intersect_ray` returns a `SceneRayHit` whose
//!   `t` reconstructs the expected world hit point via
//!   `ray.point_at(t)`.
//! * Cross-validate against the underlying `Mesh::intersect_ray` /
//!   `Primitive::intersect_ray` brute-force paths after manually
//!   transforming the ray.
//! * Exercise the closest-hit shrinking across multiple instances,
//!   the `t_max` boundary, the any-hit short-circuit, the singular /
//!   non-finite / cycle / detached cases.

use oxideav_mesh3d::{
    Mesh, Node, NodeId, Primitive, Ray, Scene3D, SceneRayHit, Topology, Transform,
};

const EPS_HIT: f32 = 1e-4;

fn approx_point(a: [f32; 3], b: [f32; 3]) -> bool {
    (a[0] - b[0]).abs() <= EPS_HIT
        && (a[1] - b[1]).abs() <= EPS_HIT
        && (a[2] - b[2]).abs() <= EPS_HIT
}

/// Single CCW triangle at z = 1, spanning (0,0)..(1,1) in xy.
/// A ray at origin (x, y, 0) shooting in +Z hits z=1 at parameter t=1
/// when the (x, y) lies inside the triangle.
fn unit_triangle_at_z1() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    p
}

fn quad_at_z1() -> Primitive {
    // Two CCW triangles forming the unit square at z=1.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    p
}

fn mesh_with(prim: Primitive) -> Mesh {
    Mesh::new(None::<String>).with_primitive(prim)
}

fn translation(tx: f32, ty: f32, tz: f32) -> Transform {
    Transform::Trs {
        translation: [tx, ty, tz],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn scale(sx: f32, sy: f32, sz: f32) -> Transform {
    Transform::Trs {
        translation: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [sx, sy, sz],
    }
}

fn identity() -> Transform {
    Transform::identity()
}

fn add_root_mesh_node(scene: &mut Scene3D, mesh: Mesh, transform: Transform) -> NodeId {
    let m = scene.add_mesh(mesh);
    let node = Node::new().with_transform(transform).with_mesh(m);
    let nid = scene.add_node(node);
    scene.add_root(nid);
    nid
}

#[test]
fn empty_scene_returns_none() {
    let scene = Scene3D::new();
    let r = Ray::new([0.5, 0.5, 0.0], [0.0, 0.0, 1.0]);
    assert!(scene.intersect_ray(r, f32::INFINITY).is_none());
    assert!(!scene.any_ray_intersection(r, f32::INFINITY));
}

#[test]
fn scene_without_reachable_mesh_returns_none() {
    let mut scene = Scene3D::new();
    let _m = scene.add_mesh(mesh_with(unit_triangle_at_z1()));
    // Mesh exists but no node references it / is rooted.
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    assert!(scene.intersect_ray(r, f32::INFINITY).is_none());
    assert!(!scene.any_ray_intersection(r, f32::INFINITY));
}

#[test]
fn identity_instance_hits_at_local_t() {
    let mut scene = Scene3D::new();
    let nid = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene
        .intersect_ray(r, f32::INFINITY)
        .expect("hits the triangle");
    assert_eq!(hit.node, nid);
    assert_eq!(hit.primitive_index, 0);
    // Identity world transform: world t == local t == 1.
    assert!((hit.hit.t - 1.0).abs() < EPS_HIT, "t = {}", hit.hit.t);
    assert!(approx_point(r.point_at(hit.hit.t), [0.25, 0.25, 1.0]));
}

#[test]
fn translation_moves_hit_in_world_space() {
    // Mesh lies in local-z=1. Translate the node by +Z=2 so the world
    // surface is at z=3. A ray from (0.25, 0.25, 0) shooting +Z must
    // hit at t=3.
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        translation(0.0, 0.0, 2.0),
    );
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    assert!((hit.hit.t - 3.0).abs() < EPS_HIT, "t = {}", hit.hit.t);
    assert!(approx_point(r.point_at(hit.hit.t), [0.25, 0.25, 3.0]));
}

#[test]
fn uniform_scale_stretches_world_t() {
    // Mesh in local-z=1; scale node by 5. World surface at z=5; ray
    // from (1.25, 1.25, 0) shooting +Z hits at t=5. The (1.25, 1.25)
    // world point sits inside the scaled triangle (which now spans
    // (0,0)..(5,5)) by margin.
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        scale(5.0, 5.0, 5.0),
    );
    let r = Ray::new([1.25, 1.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    assert!((hit.hit.t - 5.0).abs() < EPS_HIT, "t = {}", hit.hit.t);
    assert!(approx_point(r.point_at(hit.hit.t), [1.25, 1.25, 5.0]));
}

#[test]
fn non_uniform_scale_preserves_t_invariance() {
    // Non-uniform scale (1, 1, 2). Mesh-local z=1 → world z=2.
    // Ray from origin straight up hits at t=2.
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        scale(1.0, 1.0, 2.0),
    );
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    assert!((hit.hit.t - 2.0).abs() < EPS_HIT, "t = {}", hit.hit.t);
}

#[test]
fn rotation_90_about_x_swaps_yz() {
    // Rotate 90° about the X axis (rotation quaternion (sin(pi/4), 0, 0, cos(pi/4))).
    // Local mesh at z=1 (triangle in xy plane). After +90° about X,
    // local z maps to world y. So the surface lies at world y=1, in
    // the world xz plane.
    let s = (std::f32::consts::FRAC_PI_4).sin();
    let c = (std::f32::consts::FRAC_PI_4).cos();
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        Transform::Trs {
            translation: [0.0, 0.0, 0.0],
            rotation: [s, 0.0, 0.0, c],
            scale: [1.0, 1.0, 1.0],
        },
    );
    // The rotation `(x, y, z) -> (x, -z, y)` (right-hand +X rotation
    // by 90°) maps the local triangle vertices `(0,0,1), (1,0,1),
    // (0,1,1)` to world `(0,-1,0), (1,-1,0), (0,-1,1)` — the
    // triangle now lies in the world plane `y = -1`. A barycentric
    // interior point `(local 0.25, 0.25, 1)` lands at world
    // `(0.25, -1, 0.25)`. Shoot a +Y ray from `(0.25, -5, 0.25)`,
    // expect a hit at `t = 4`.
    let r = Ray::new([0.25, -5.0, 0.25], [0.0, 1.0, 0.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    assert!((hit.hit.t - 4.0).abs() < EPS_HIT, "t = {}", hit.hit.t);
    assert!(approx_point(r.point_at(hit.hit.t), [0.25, -1.0, 0.25]));
}

#[test]
fn closest_hit_among_two_instances() {
    // Two stacked triangle copies, one at world z=1, the other at z=3.
    // The +Z ray from origin should land on the nearer one (t=1).
    let mut scene = Scene3D::new();
    let nid_near = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    let _nid_far = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        translation(0.0, 0.0, 2.0),
    );
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits near");
    assert_eq!(hit.node, nid_near);
    assert!((hit.hit.t - 1.0).abs() < EPS_HIT);
}

#[test]
fn closest_hit_is_independent_of_insertion_order() {
    // Same as above but insert the far instance first. The ray must
    // still pick the near one.
    let mut scene = Scene3D::new();
    let _far = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        translation(0.0, 0.0, 2.0),
    );
    let nid_near = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    assert_eq!(hit.node, nid_near);
    assert!((hit.hit.t - 1.0).abs() < EPS_HIT);
}

#[test]
fn t_max_excludes_far_hits() {
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        translation(0.0, 0.0, 5.0),
    );
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    // t_max < 6 excludes the hit at t=6.
    assert!(scene.intersect_ray(r, 5.5).is_none());
    assert!(!scene.any_ray_intersection(r, 5.5));
    // t_max > 6 includes it.
    assert!(scene.intersect_ray(r, 7.0).is_some());
    assert!(scene.any_ray_intersection(r, 7.0));
}

#[test]
fn ray_pointing_away_from_triangle_misses() {
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    // Ray from above pointing further up — origin past the triangle.
    let r = Ray::new([0.25, 0.25, 2.0], [0.0, 0.0, 1.0]);
    assert!(scene.intersect_ray(r, f32::INFINITY).is_none());
    assert!(!scene.any_ray_intersection(r, f32::INFINITY));
}

#[test]
fn ray_missing_in_xy_plane_misses() {
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    // Shoot at a (x, y) outside the triangle's barycentric simplex.
    let r = Ray::new([0.9, 0.9, 0.0], [0.0, 0.0, 1.0]);
    assert!(scene.intersect_ray(r, f32::INFINITY).is_none());
    assert!(!scene.any_ray_intersection(r, f32::INFINITY));
}

#[test]
fn any_ray_intersection_short_circuits() {
    // Two instances; either alone hit by the ray. The any-hit should
    // return true. (Determinism of *which* gets visited is not tested
    // here — see the closest-hit tests for ordering.)
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    let _ = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        translation(0.0, 0.0, 2.0),
    );
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    assert!(scene.any_ray_intersection(r, f32::INFINITY));
    // Same scene, ray that misses everything.
    let miss = Ray::new([5.0, 5.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(!scene.any_ray_intersection(miss, f32::INFINITY));
}

#[test]
fn singular_zero_scale_axis_skipped() {
    // Two instances: one with a zero-X scale (singular), one identity.
    // The singular instance must be silently skipped; the identity
    // instance still hits.
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        scale(0.0, 1.0, 1.0),
    );
    let nid_good = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        translation(0.0, 0.0, 1.0),
    );
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene
        .intersect_ray(r, f32::INFINITY)
        .expect("good instance hits");
    assert_eq!(hit.node, nid_good);
}

#[test]
fn detached_node_unreachable_from_roots_skipped() {
    // Build a node attached to a mesh, but never add it to roots.
    // It must not produce hits even though the mesh+node are
    // present in the scene.
    let mut scene = Scene3D::new();
    let m = scene.add_mesh(mesh_with(unit_triangle_at_z1()));
    let detached = Node::new().with_transform(identity()).with_mesh(m);
    let _det_id = scene.add_node(detached);
    // No add_root call — the node is detached.
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    assert!(scene.intersect_ray(r, f32::INFINITY).is_none());
    assert!(!scene.any_ray_intersection(r, f32::INFINITY));
}

#[test]
fn cycle_guard_visits_each_node_once() {
    // Build a cycle: root → A → A. The DFS must terminate; the hit
    // on A is reported once (the second visit short-circuits via the
    // visited bitmap).
    let mut scene = Scene3D::new();
    let m = scene.add_mesh(mesh_with(unit_triangle_at_z1()));
    let a = scene.add_node(Node::new().with_transform(identity()).with_mesh(m));
    // Splice A into its own children — pure cycle.
    if let Some(node) = scene.node_mut(a) {
        node.children.push(a);
    }
    scene.add_root(a);
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits A");
    assert_eq!(hit.node, a);
    assert!((hit.hit.t - 1.0).abs() < EPS_HIT);
}

#[test]
fn child_inherits_parent_translation() {
    // root translates (0,0,5); child also identity-attached carrying
    // the mesh. The world surface for the child sits at z=6.
    let mut scene = Scene3D::new();
    let m = scene.add_mesh(mesh_with(unit_triangle_at_z1()));
    let child = scene.add_node(Node::new().with_transform(identity()).with_mesh(m));
    let root = scene.add_node(Node::new().with_transform(translation(0.0, 0.0, 5.0)));
    if let Some(node) = scene.node_mut(root) {
        node.children.push(child);
    }
    scene.add_root(root);
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits child");
    assert_eq!(hit.node, child);
    assert!((hit.hit.t - 6.0).abs() < EPS_HIT, "t = {}", hit.hit.t);
}

#[test]
fn nested_transforms_compose_translation_and_scale() {
    // root: scale 2 → child: translate (0,0,1) → mesh at local z=1.
    // World z of the mesh surface = 2 * (0 + 1 * 1) = 2 * 1
    // (translation of (0,0,1) at scale-2 parent gives world z=2;
    // mesh at local z=1 of child gives world z=4). Wait: the
    // composition is M_root * M_child. M_root scales by 2 (so any
    // child point gets multiplied by 2). M_child translates +1 in z.
    // mesh-local point (x, y, 1) maps to child-local
    // (x, y, 1+1)=(x, y, 2) then root scales to (2x, 2y, 4).
    // So world surface at z=4, in (x∈[0,2], y∈[0,2]).
    let mut scene = Scene3D::new();
    let m = scene.add_mesh(mesh_with(unit_triangle_at_z1()));
    let child = scene.add_node(
        Node::new()
            .with_transform(translation(0.0, 0.0, 1.0))
            .with_mesh(m),
    );
    let root = scene.add_node(Node::new().with_transform(scale(2.0, 2.0, 2.0)));
    if let Some(node) = scene.node_mut(root) {
        node.children.push(child);
    }
    scene.add_root(root);
    // (0.5, 0.5) world is inside the scaled triangle (which spans
    // (0,0)..(2,2)). Shoot +Z.
    let r = Ray::new([0.5, 0.5, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    assert_eq!(hit.node, child);
    assert!((hit.hit.t - 4.0).abs() < EPS_HIT, "t = {}", hit.hit.t);
}

#[test]
fn multi_primitive_mesh_reports_primitive_index() {
    // Mesh with two primitives, one at local z=1 and one at local z=2.
    // The +Z ray from origin hits the nearer (z=1, primitive 0) first.
    let mut mesh = Mesh::new(None::<String>).with_primitive(unit_triangle_at_z1());
    let mut p2 = Primitive::new(Topology::Triangles);
    p2.positions = vec![[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]];
    mesh.primitives.push(p2);

    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(&mut scene, mesh, identity());
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    assert_eq!(hit.primitive_index, 0);
    assert!((hit.hit.t - 1.0).abs() < EPS_HIT);
}

#[test]
fn closest_hit_t_param_equals_mesh_local_t() {
    // For an identity-transformed instance, world t and mesh-local t
    // must match the brute-force `Mesh::intersect_ray` answer
    // bit-for-bit. This guards the affine-change-of-frame
    // t-invariance claim.
    let mesh = mesh_with(quad_at_z1());
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(&mut scene, mesh.clone(), identity());
    let r = Ray::new([0.7, 0.4, 0.0], [0.0, 0.0, 1.0]);
    let scene_hit: SceneRayHit = scene.intersect_ray(r, f32::INFINITY).expect("scene hits");
    let (prim_idx, mesh_hit) = mesh.intersect_ray(r, f32::INFINITY).expect("mesh hits");
    assert_eq!(scene_hit.primitive_index, prim_idx);
    assert_eq!(scene_hit.hit.t, mesh_hit.t);
    assert_eq!(scene_hit.hit.triangle_index, mesh_hit.triangle_index);
    assert_eq!(scene_hit.hit.front_face, mesh_hit.front_face);
}

#[test]
fn deterministic_winner_on_coincident_instances() {
    // Two identity-transformed copies of the same triangle at z=1 —
    // both produce a hit at exactly t=1. The leftmost-first DFS
    // ordering must pick the first-added node deterministically
    // across repeat calls.
    let mut scene = Scene3D::new();
    let first = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    let _second = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    for _ in 0..5 {
        let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
        assert_eq!(hit.node, first);
        assert!((hit.hit.t - 1.0).abs() < EPS_HIT);
    }
}

#[test]
fn any_hit_returns_false_when_only_singular_instances_cover_ray() {
    // Single instance, scale collapses z to 0 — the matrix is
    // singular. The instance is silently skipped; any_hit must
    // return false even though a non-singular sibling would have hit.
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(
        &mut scene,
        mesh_with(unit_triangle_at_z1()),
        scale(1.0, 1.0, 0.0),
    );
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 1.0]);
    assert!(!scene.any_ray_intersection(r, f32::INFINITY));
    assert!(scene.intersect_ray(r, f32::INFINITY).is_none());
}

#[test]
fn zero_direction_ray_does_not_panic() {
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    let r = Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, 0.0]);
    // The exact answer is "no hit"; the contract is just that we
    // don't crash on the degenerate input.
    let _ = scene.intersect_ray(r, f32::INFINITY);
    let _ = scene.any_ray_intersection(r, f32::INFINITY);
}

#[test]
fn nan_ray_does_not_panic() {
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(&mut scene, mesh_with(unit_triangle_at_z1()), identity());
    let r = Ray::new([f32::NAN, 0.25, 0.0], [0.0, 0.0, 1.0]);
    // No panic, returns None.
    assert!(scene.intersect_ray(r, f32::INFINITY).is_none());
    assert!(!scene.any_ray_intersection(r, f32::INFINITY));
}

#[test]
fn translated_then_scaled_chain_is_t_invariant() {
    // Combine a translation parent with a scale child carrying the
    // mesh. World position = parent.t + scale * mesh_local. Validate
    // the t-invariance of the world-vs-mesh-local change of frame
    // across the composed transform.
    let mut scene = Scene3D::new();
    let m = scene.add_mesh(mesh_with(unit_triangle_at_z1()));
    let child = scene.add_node(
        Node::new()
            .with_transform(scale(3.0, 3.0, 3.0))
            .with_mesh(m),
    );
    let root = scene.add_node(Node::new().with_transform(translation(0.0, 0.0, 1.0)));
    if let Some(node) = scene.node_mut(root) {
        node.children.push(child);
    }
    scene.add_root(root);
    // Mesh-local point (0.25, 0.25, 1) → child-local (0.75, 0.75, 3)
    // → world (0.75, 0.75, 4). Shoot +Z from (0.75, 0.75, 0).
    let r = Ray::new([0.75, 0.75, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    assert_eq!(hit.node, child);
    assert!((hit.hit.t - 4.0).abs() < EPS_HIT, "t = {}", hit.hit.t);
    assert!(approx_point(r.point_at(hit.hit.t), [0.75, 0.75, 4.0]));
}

#[test]
fn many_instances_returns_globally_closest() {
    // 8 instances spaced along +Z (z = 1, 2, 3, …, 8). The closest hit
    // from a -Z ray fired from above (z=10) must be the z=8 instance.
    let mut scene = Scene3D::new();
    let mut last_node = None;
    for k in 1..=8 {
        let nid = add_root_mesh_node(
            &mut scene,
            mesh_with(unit_triangle_at_z1()),
            translation(0.0, 0.0, (k - 1) as f32),
        );
        if k == 8 {
            last_node = Some(nid);
        }
    }
    let r = Ray::new([0.25, 0.25, 10.0], [0.0, 0.0, -1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    // Surface for k-th instance lies at z = 1 + (k-1) = k.
    // From z=10 shooting -Z, the closest is k=8 → z=8 → t=2.
    assert_eq!(hit.node, last_node.unwrap());
    assert!((hit.hit.t - 2.0).abs() < EPS_HIT, "t = {}", hit.hit.t);
}

#[test]
fn world_hit_point_round_trips_through_scene_ray_hit() {
    // Build a non-trivial instance (translation + non-uniform scale)
    // and verify ray.point_at(scene_hit.hit.t) lands at the analytic
    // world position.
    let mut scene = Scene3D::new();
    let _ = add_root_mesh_node(
        &mut scene,
        mesh_with(quad_at_z1()),
        Transform::Trs {
            translation: [10.0, -5.0, 7.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [2.0, 3.0, 0.5],
        },
    );
    // Local mesh at z=1; world surface lands at z = 7 + 0.5 * 1 = 7.5,
    // covering x ∈ [10, 12], y ∈ [-5, -2]. Pick (11, -3.5, 0) and
    // shoot +Z.
    let r = Ray::new([11.0, -3.5, 0.0], [0.0, 0.0, 1.0]);
    let hit = scene.intersect_ray(r, f32::INFINITY).expect("hits");
    assert!((hit.hit.t - 7.5).abs() < EPS_HIT, "t = {}", hit.hit.t);
    assert!(
        approx_point(r.point_at(hit.hit.t), [11.0, -3.5, 7.5]),
        "world hit point should reconstruct"
    );
}
