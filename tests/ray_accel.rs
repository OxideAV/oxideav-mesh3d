//! Cross-validation of the accelerated ray queries (SAH / median
//! `Bvh`, two-level `InstanceBvh`) against brute force on random
//! geometry, plus degenerate inputs, watertightness, instancing with
//! non-uniform scale / mirroring, filters, refit and attribute
//! interpolation. Deterministic xorshift PRNG — no extra deps.

use oxideav_mesh3d::{
    Bvh, BvhBuildOptions, BvhNode, Indices, InstanceBvh, Mesh, Node, PreparedRay, Primitive, Ray,
    RayQuery, Scene3D, Topology, Transform, TriangleTest,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn s(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn v(&mut self) -> [f32; 3] {
        [self.s(), self.s(), self.s()]
    }
}

fn random_soup(rng: &mut Rng, n: usize, size: f32) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    for _ in 0..n {
        let c = rng.v();
        for _ in 0..3 {
            let d = rng.v();
            p.positions
                .push([c[0] + size * d[0], c[1] + size * d[1], c[2] + size * d[2]]);
        }
    }
    p
}

fn random_ray(rng: &mut Rng) -> Ray {
    let o = [rng.s() * 3.0, rng.s() * 3.0, rng.s() * 3.0];
    let target = rng.v();
    Ray::new(o, [target[0] - o[0], target[1] - o[1], target[2] - o[2]])
}

/// Brute-force closest hit over a primitive with an explicit test.
fn brute(p: &Primitive, ray: Ray, q: &RayQuery) -> Option<(usize, f32)> {
    let pr = PreparedRay::new(ray);
    let mut best: Option<(usize, f32)> = None;
    let mut t_max = q.t_max;
    for (i, [a, b, c]) in p.triangle_indices().into_iter().enumerate() {
        let (a, b, c) = (a as usize, b as usize, c as usize);
        if a >= p.positions.len() || b >= p.positions.len() || c >= p.positions.len() {
            continue;
        }
        if let Some((t, ..)) = pr.intersect_triangle(
            q.triangle_test,
            p.positions[a],
            p.positions[b],
            p.positions[c],
            q.t_min,
            t_max,
        ) {
            t_max = t;
            best = Some((i, t));
        }
    }
    best
}

fn options() -> [BvhBuildOptions; 3] {
    let mut wide = BvhBuildOptions::sah();
    wide.sah_bins = 4;
    wide.max_leaf_size = 1;
    [
        BvhBuildOptions::sah(),
        BvhBuildOptions::object_median(),
        wide,
    ]
}

#[test]
fn bvh_matches_brute_force_on_random_soups() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for size in [0.02, 0.3, 1.5] {
        let p = random_soup(&mut rng, 600, size);
        for opts in options() {
            let bvh = Bvh::build_with(&p, &opts).unwrap();
            assert_eq!(bvh.triangle_count(), 600);
            for test in [TriangleTest::MollerTrumbore, TriangleTest::Watertight] {
                for _ in 0..400 {
                    let ray = random_ray(&mut rng);
                    let q = RayQuery::new(if rng.s() > 0.5 { 2.0 } else { f32::INFINITY })
                        .with_t_min(if rng.s() > 0.5 { 0.3 } else { 0.0 })
                        .with_triangle_test(test);
                    let want = brute(&p, ray, &q);
                    let got = bvh.closest_hit(&p, &PreparedRay::new(ray), &q);
                    match (want, got) {
                        (None, None) => {}
                        (Some((_, t)), Some(h)) => {
                            assert_eq!(t, h.t, "{opts:?} {test:?}");
                            assert!(h.t >= q.t_min && h.t <= q.t_max);
                        }
                        other => panic!("{opts:?} {test:?} disagreement {other:?}"),
                    }
                    assert_eq!(
                        want.is_some(),
                        bvh.occluded(&p, &PreparedRay::new(ray), &q),
                        "any-hit disagrees"
                    );
                }
            }
        }
    }
}

#[test]
fn hit_record_reconstructs_point() {
    let mut rng = Rng(77);
    let p = random_soup(&mut rng, 300, 0.4);
    let bvh = Bvh::build(&p).unwrap();
    let tris = p.triangle_indices();
    let mut n = 0;
    for _ in 0..500 {
        let ray = random_ray(&mut rng);
        if let Some(h) = bvh.intersect_ray(&p, ray, f32::INFINITY) {
            n += 1;
            let q = p
                .interpolate_position(tris[h.triangle_index], h.barycentric)
                .unwrap();
            let r = ray.point_at(h.t);
            for k in 0..3 {
                assert!((q[k] - r[k]).abs() < 1e-4);
            }
        }
    }
    assert!(n > 50);
}

#[test]
fn degenerate_inputs() {
    // Coincident triangles, zero-area slivers, and a point-triangle.
    let mut p = Primitive::new(Topology::Triangles);
    for _ in 0..40 {
        p.positions
            .extend_from_slice(&[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]);
    }
    for i in 0..40 {
        let x = i as f32 * 0.01;
        p.positions
            .extend_from_slice(&[[x, 0.0, 0.0], [x, 0.0, 0.0], [x, 0.0, 0.0]]);
        p.positions
            .extend_from_slice(&[[x, 0.0, 2.0], [x + 1.0, 0.0, 2.0], [x + 2.0, 0.0, 2.0]]);
    }
    for opts in options() {
        let bvh = Bvh::build_with(&p, &opts).unwrap();
        assert!(bvh.depth() <= 64);
        for test in [TriangleTest::MollerTrumbore, TriangleTest::Watertight] {
            let q = RayQuery::default().with_triangle_test(test);
            let hit = bvh
                .closest_hit(
                    &p,
                    &PreparedRay::new(Ray::new([0.2, 0.2, -1.0], [0.0, 0.0, 1.0])),
                    &q,
                )
                .unwrap();
            assert!((hit.t - 2.0).abs() < 1e-6);
            // Degenerate rays never hit and never panic.
            for r in [
                Ray::new([0.2, 0.2, -1.0], [0.0, 0.0, 0.0]),
                Ray::new([f32::NAN, 0.2, -1.0], [0.0, 0.0, 1.0]),
                Ray::new([0.2, 0.2, -1.0], [f32::INFINITY, 0.0, 1.0]),
            ] {
                assert!(bvh.closest_hit(&p, &PreparedRay::new(r), &q).is_none());
                assert!(!bvh.occluded(&p, &PreparedRay::new(r), &q));
            }
        }
    }
}

#[test]
fn axis_parallel_rays_on_slab_planes() {
    // Axis-aligned unit quad at z = 0; rays running exactly in the
    // planes x = 0 / y = 0 (the 0·∞ = NaN slab case) must still hit.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let bvh = Bvh::build(&p).unwrap();
    for (o, d) in [
        ([0.5, 0.5, 1.0], [0.0, 0.0, -1.0]),
        ([0.5, 0.5, -1.0], [-0.0, 0.0, 1.0]),
        ([0.0, 0.5, 1.0], [0.0, -0.0, -1.0]),
    ] {
        let q = RayQuery::default().with_triangle_test(TriangleTest::Watertight);
        let h = bvh.closest_hit(&p, &PreparedRay::new(Ray::new(o, d)), &q);
        assert!(h.is_some(), "missed from {o:?} {d:?}");
    }
}

/// Closed, finely triangulated unit cube: rays from the centre aimed
/// exactly at grid vertices / along edges must never escape.
#[test]
fn watertight_closed_mesh_has_no_leaks() {
    let n = 8u32;
    let mut p = Primitive::new(Topology::Triangles);
    let mut idx = Vec::new();
    // Six faces, each an (n+1)^2 vertex grid on [-1, 1]^2.
    for face in 0..6 {
        let base = p.positions.len() as u32;
        for j in 0..=n {
            for i in 0..=n {
                let a = -1.0 + 2.0 * i as f32 / n as f32;
                let b = -1.0 + 2.0 * j as f32 / n as f32;
                let v = match face {
                    0 => [1.0, a, b],
                    1 => [-1.0, b, a],
                    2 => [b, 1.0, a],
                    3 => [a, -1.0, b],
                    4 => [a, b, 1.0],
                    _ => [b, a, -1.0],
                };
                p.positions.push(v);
            }
        }
        for j in 0..n {
            for i in 0..n {
                let a = base + j * (n + 1) + i;
                let b = a + n + 1;
                idx.extend_from_slice(&[a, a + 1, b + 1, a, b + 1, b]);
            }
        }
    }
    p.indices = Some(Indices::U32(idx));
    let bvh = Bvh::build(&p).unwrap();
    let q = RayQuery::default().with_triangle_test(TriangleTest::Watertight);
    let origin = [0.013, -0.007, 0.004];
    let mut tested = 0;
    for v in p.positions.clone() {
        // Toward every vertex, and toward every edge midpoint-ish
        // direction formed with the next vertex.
        let d = [v[0] - origin[0], v[1] - origin[1], v[2] - origin[2]];
        let h = bvh.closest_hit(&p, &PreparedRay::new(Ray::new(origin, d)), &q);
        assert!(h.is_some(), "leak towards vertex {v:?}");
        tested += 1;
    }
    // Rays from the exact centre through edge-shared points.
    for k in 0..=(n * 4) {
        let a = -1.0 + 2.0 * k as f32 / (n * 4) as f32;
        for d in [[1.0, a, 0.0], [a, 1.0, 0.0], [0.0, a, -1.0], [1.0, 1.0, a]] {
            let h = bvh.closest_hit(&p, &PreparedRay::new(Ray::new([0.0, 0.0, 0.0], d)), &q);
            assert!(h.is_some(), "leak along {d:?}");
            tested += 1;
        }
    }
    assert!(tested > 400);
}

#[test]
fn filter_rejects_candidates() {
    let mut rng = Rng(99);
    let p = random_soup(&mut rng, 400, 0.3);
    let bvh = Bvh::build(&p).unwrap();
    // Reference: brute force on the odd triangles only.
    let mut odd = Primitive::new(Topology::Triangles);
    let tris = p.triangle_indices();
    for (i, t) in tris.iter().enumerate() {
        if i % 2 == 1 {
            for &v in t {
                odd.positions.push(p.positions[v as usize]);
            }
        }
    }
    let q = RayQuery::default();
    for _ in 0..400 {
        let ray = random_ray(&mut rng);
        let pr = PreparedRay::new(ray);
        let got = bvh.closest_hit_filtered(&p, &pr, &q, |h| h.triangle_index % 2 == 1);
        let want = brute(&odd, ray, &q);
        assert_eq!(got.map(|h| h.t), want.map(|w| w.1));
        if let Some(h) = got {
            assert_eq!(h.triangle_index % 2, 1);
        }
        assert_eq!(
            bvh.occluded_filtered(&p, &pr, &q, |h| h.triangle_index % 2 == 1),
            want.is_some()
        );
        assert!(!bvh.occluded_filtered(&p, &pr, &q, |_| false));
    }
}

#[test]
fn refit_tracks_moved_vertices() {
    let mut rng = Rng(5);
    let mut p = random_soup(&mut rng, 300, 0.2);
    let mut bvh = Bvh::build(&p).unwrap();
    // Wobble every vertex.
    for v in &mut p.positions {
        v[0] = v[0] * 1.3 + 0.4;
        v[1] += 0.1 * v[2];
    }
    assert!(bvh.refit(&p));
    let root = bvh.bounds().unwrap();
    let pb = p.bounding_box().unwrap();
    assert_eq!(root.min, pb.min);
    assert_eq!(root.max, pb.max);
    let q = RayQuery::default();
    for _ in 0..400 {
        let ray = random_ray(&mut rng);
        let got = bvh.closest_hit(&p, &PreparedRay::new(ray), &q).map(|h| h.t);
        assert_eq!(got, brute(&p, ray, &q).map(|w| w.1));
    }
    // A vertex going NaN is reported, and queries stay safe.
    p.positions[0] = [f32::NAN; 3];
    assert!(!bvh.refit(&p));
    let _ = bvh.closest_hit(&p, &PreparedRay::new(random_ray(&mut rng)), &q);
}

#[test]
fn adversarial_distribution_depth_is_bounded() {
    // Exponentially spaced tiny triangles: SAH would peel one per
    // level without the balancing fallback.
    let mut p = Primitive::new(Topology::Triangles);
    for i in 0..3000 {
        let x = 1.0001f32.powi(i * 8) - 1.0;
        p.positions
            .extend_from_slice(&[[x, 0.0, 0.0], [x, 1e-3, 0.0], [x, 0.0, 1e-3]]);
    }
    for opts in options() {
        let bvh = Bvh::build_with(&p, &opts).unwrap();
        assert!(bvh.depth() <= 64, "{opts:?}: depth {}", bvh.depth());
    }
}

#[test]
fn gpu_layout_is_32_bytes() {
    assert_eq!(std::mem::size_of::<BvhNode>(), 32);
    assert_eq!(std::mem::size_of::<oxideav_mesh3d::InstanceBvhNode>(), 32);
    let mut rng = Rng(3);
    let p = random_soup(&mut rng, 50, 0.2);
    let bvh = Bvh::build(&p).unwrap();
    let words = bvh.node_words();
    assert_eq!(words.len(), 8 * bvh.node_count());
    assert_eq!(f32::from_bits(words[0]), bvh.nodes[0].min[0]);
    assert_eq!(words[7], bvh.nodes[0].tri_count);
    // Pair layout: children follow their parent.
    for (i, n) in bvh.nodes.iter().enumerate() {
        if !n.is_leaf() {
            assert!(n.left_child() as usize > i);
            assert_eq!(n.right_child(), n.left_child() + 1);
        }
    }
    assert_eq!(bvh.node_count(), 2 * bvh.leaf_count() - 1);
}

#[test]
fn sah_beats_median_on_sah_cost() {
    let mut rng = Rng(11);
    // Clustered input: SAH should find a noticeably cheaper tree.
    let mut p = random_soup(&mut rng, 2000, 0.01);
    let big = random_soup(&mut rng, 30, 0.9);
    p.positions.extend(big.positions);
    let sah = Bvh::build_with(&p, &BvhBuildOptions::sah()).unwrap();
    let med = Bvh::build_with(&p, &BvhBuildOptions::object_median()).unwrap();
    assert!(
        sah.sah_cost() < med.sah_cost(),
        "sah {} vs median {}",
        sah.sah_cost(),
        med.sah_cost()
    );
}

// ---------------------------------------------------------------- scene

fn sphere(seg_u: u32, seg_v: u32) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    for j in 0..=seg_v {
        let th = std::f32::consts::PI * j as f32 / seg_v as f32;
        for i in 0..=seg_u {
            let ph = std::f32::consts::TAU * i as f32 / seg_u as f32;
            let n = [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()];
            p.positions.push(n);
            normals.push(n);
            uvs.push([i as f32 / seg_u as f32, j as f32 / seg_v as f32]);
            tangents.push([-ph.sin(), 0.0, ph.cos(), 1.0]);
        }
    }
    let mut idx = Vec::new();
    for j in 0..seg_v {
        for i in 0..seg_u {
            let a = j * (seg_u + 1) + i;
            let b = a + seg_u + 1;
            // CCW seen from outside.
            idx.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    p.indices = Some(Indices::U32(idx));
    p.normals = Some(normals);
    p.uvs = vec![uvs];
    p.tangents = Some(tangents);
    p.colors = vec![vec![[1.0, 0.5, 0.25, 1.0]; p.positions.len()]];
    p
}

fn trs(t: [f32; 3], r: [f32; 4], s: [f32; 3]) -> Transform {
    Transform::Trs {
        translation: t,
        rotation: r,
        scale: s,
    }
}

/// A scene with non-uniform scale, mirroring (negative scale) and
/// nested transforms, plus a second multi-primitive mesh.
fn tricky_scene() -> Scene3D {
    let mut s = Scene3D::new();
    let sph = s.add_mesh(Mesh::new(None).with_primitive(sphere(24, 12)));
    let mut rng = Rng(21);
    let soup_mesh = Mesh::new(None)
        .with_primitive(random_soup(&mut rng, 60, 0.3))
        .with_primitive(random_soup(&mut rng, 60, 0.3));
    let soup = s.add_mesh(soup_mesh);
    let h = std::f32::consts::FRAC_1_SQRT_2;
    let specs: [([f32; 3], [f32; 4], [f32; 3], _); 6] = [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0], sph),
        ([2.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [2.0, 0.5, 1.0], sph),
        (
            [-2.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            sph,
        ),
        ([0.0, 2.5, 0.0], [0.0, h, 0.0, h], [1.0, -0.7, 1.5], sph),
        ([0.0, -2.5, 0.5], [h, 0.0, 0.0, h], [1.0, 1.0, -2.0], soup),
        (
            [1.5, 1.5, -1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.6, 0.6, 0.6],
            soup,
        ),
    ];
    let parent = s.add_node(Node::new().with_transform(trs(
        [0.1, -0.2, 0.3],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    )));
    s.add_root(parent);
    for (t, r, sc, m) in specs {
        let n = s.add_node(Node::new().with_transform(trs(t, r, sc)).with_mesh(m));
        s.nodes[parent.0 as usize].children.push(n);
    }
    s
}

/// World-baked triangle soup + (node, prim, tri) provenance.
fn bake(s: &Scene3D) -> (Primitive, Vec<(u32, usize, usize)>) {
    let worlds = s.world_node_transforms();
    let mut soup = Primitive::new(Topology::Triangles);
    let mut prov = Vec::new();
    for (ni, node) in s.nodes.iter().enumerate() {
        let (Some(m), Some(w)) = (node.mesh, worlds[ni]) else {
            continue;
        };
        for (pi, prim) in s.meshes[m.0 as usize].primitives.iter().enumerate() {
            for (ti, tri) in prim.triangle_indices().into_iter().enumerate() {
                for v in tri {
                    let p = prim.positions[v as usize];
                    soup.positions.push([
                        w[0][0] * p[0] + w[0][1] * p[1] + w[0][2] * p[2] + w[0][3],
                        w[1][0] * p[0] + w[1][1] * p[1] + w[1][2] * p[2] + w[1][3],
                        w[2][0] * p[0] + w[2][1] * p[1] + w[2][2] * p[2] + w[2][3],
                    ]);
                }
                prov.push((ni as u32, pi, ti));
            }
        }
    }
    (soup, prov)
}

fn scene_ray(rng: &mut Rng) -> Ray {
    let o = [rng.s() * 6.0, rng.s() * 6.0, 6.0 + rng.s()];
    let t = [rng.s() * 3.5, rng.s() * 3.5, rng.s()];
    Ray::new(o, [t[0] - o[0], t[1] - o[1], t[2] - o[2]])
}

#[test]
fn instanced_scene_matches_world_baked_brute_force() {
    let s = tricky_scene();
    let (soup, prov) = bake(&s);
    let mut rng = Rng(0xfeed);
    for opts in options() {
        let accel = InstanceBvh::build_with(&s, &opts).unwrap();
        assert_eq!(accel.instance_count(), 6);
        let mut hits = 0;
        for _ in 0..1500 {
            let ray = scene_ray(&mut rng);
            for test in [TriangleTest::MollerTrumbore, TriangleTest::Watertight] {
                let q = RayQuery::default().with_triangle_test(test);
                let got = accel.closest_hit(&s, ray, &q);
                // Brute force always in watertight mode on the baked
                // soup (robust reference); compare with a tolerance
                // since baking rounds differently.
                let want = brute(
                    &soup,
                    ray,
                    &RayQuery::default().with_triangle_test(TriangleTest::Watertight),
                );
                match (got, want) {
                    (None, None) => {}
                    (Some(h), Some((ti, t))) => {
                        hits += 1;
                        assert!((h.t - t).abs() < 1e-3 * t.max(1.0), "{} vs {}", h.t, t);
                        let (n, pi, tri) = prov[ti];
                        if (h.t - t).abs() < 1e-6 {
                            // Not a near-tie: provenance must agree.
                            assert_eq!((h.node.0, h.primitive_index), (n, pi));
                            assert_eq!(h.triangle_index, tri);
                        }
                        // World point.
                        let r = ray.point_at(h.t);
                        for (a, b) in h.position.iter().zip(r) {
                            assert!((a - b).abs() < 1e-3);
                        }
                        // Geometric normal == normalised world-space
                        // winding normal, sign-flipped when mirrored.
                        let a = soup.positions[3 * ti];
                        let b = soup.positions[3 * ti + 1];
                        let c = soup.positions[3 * ti + 2];
                        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                        let cr = [
                            e1[1] * e2[2] - e1[2] * e2[1],
                            e1[2] * e2[0] - e1[0] * e2[2],
                            e1[0] * e2[1] - e1[1] * e2[0],
                        ];
                        let l = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                        if l > 1e-6 && (h.t - t).abs() < 1e-6 {
                            // glTF: a mirroring instance's front side is
                            // its clockwise world winding.
                            let sign = if accel.instances[h.instance as usize].mirrored {
                                -1.0
                            } else {
                                1.0
                            };
                            let cr = [sign * cr[0], sign * cr[1], sign * cr[2]];
                            let g = h.geometric_normal;
                            let d = (g[0] * cr[0] + g[1] * cr[1] + g[2] * cr[2]) / l;
                            assert!(d > 0.999, "normal mismatch {g:?} vs {cr:?}");
                            let facing = ray.direction[0] * cr[0]
                                + ray.direction[1] * cr[1]
                                + ray.direction[2] * cr[2];
                            assert_eq!(h.front_face, facing < 0.0);
                        }
                    }
                    (Some(h), None) => {
                        // Only acceptable as a grazing / edge
                        // disagreement between frames.
                        panic!("accel hit {h:?} but baked brute force missed");
                    }
                    (None, Some((_, t))) => panic!("accel missed, baked hit at {t}"),
                }
                assert_eq!(accel.occluded(&s, ray, &q), want.is_some());
            }
        }
        assert!(hits > 300, "too few hits: {hits}");
    }
}

#[test]
fn legacy_scene_queries_still_agree_with_scene_walk() {
    let s = tricky_scene();
    let accel = InstanceBvh::build(&s).unwrap();
    let mut rng = Rng(0xbeef);
    for _ in 0..500 {
        let ray = scene_ray(&mut rng);
        let a = accel.intersect_ray(&s, ray, f32::INFINITY);
        let b = s.intersect_ray(ray, f32::INFINITY);
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            assert!((a.hit.t - b.hit.t).abs() < 1e-5);
            if a.node == b.node && a.hit.triangle_index == b.hit.triangle_index {
                assert_eq!(a.hit.front_face, b.hit.front_face);
            }
        }
        assert_eq!(
            accel.any_ray_intersection(&s, ray, 7.0),
            s.any_ray_intersection(ray, 7.0)
        );
    }
}

#[test]
fn scene_filter_and_shadow_t_max() {
    let s = tricky_scene();
    let accel = InstanceBvh::build(&s).unwrap();
    // Straight down the axis through the centre sphere.
    let ray = Ray::new([0.1, -0.2, 5.0], [0.0, 0.0, -1.0]);
    let q = RayQuery::default();
    let h = accel.closest_hit(&s, ray, &q).unwrap();
    assert!((h.t - (5.0 - 0.3 - 1.0)).abs() < 2e-2, "t = {}", h.t);
    assert!(h.front_face);
    // Shadow ray stopping short of the sphere is unoccluded.
    assert!(!accel.occluded(&s, ray, &RayQuery::new(h.t * 0.99)));
    assert!(accel.occluded(&s, ray, &RayQuery::new(h.t * 1.01)));
    // Filtering out the first instance's node lets the ray reach the
    // sphere's back face.
    let first = h.node;
    let h2 = accel
        .closest_hit_filtered(&s, ray, &q, |c| !(c.node == first && c.t < h.t + 1e-3))
        .unwrap();
    assert!(h2.t > h.t + 1.0);
    assert!(!h2.front_face);
    assert!(!accel.occluded_filtered(&s, ray, &q, |_| false));
    // t_min skips the near surface too.
    let h3 = accel
        .closest_hit(&s, ray, &RayQuery::default().with_t_min(h.t + 1e-3))
        .unwrap();
    assert!((h3.t - h2.t).abs() < 1e-5);
}

#[test]
fn attribute_interpolation_in_world_space() {
    let s = tricky_scene();
    let accel = InstanceBvh::build(&s).unwrap();
    let mut rng = Rng(0xabc);
    let mut checked = 0;
    for _ in 0..2000 {
        let ray = scene_ray(&mut rng);
        let Some(h) = accel.closest_hit(&s, ray, &RayQuery::default()) else {
            continue;
        };
        let prim = accel.hit_primitive(&s, &h).unwrap();
        if prim.normals.is_none() {
            assert_eq!(accel.shading_normal(&s, &h), h.geometric_normal);
            assert!(accel.uv(&s, &h, 0).is_none());
            continue;
        }
        checked += 1;
        // Smooth normal of a finely tessellated (possibly scaled /
        // mirrored) sphere stays close to the faceted geometric one
        // and points outward (same side) even for mirrored instances.
        let n = accel.shading_normal(&s, &h);
        let g = h.geometric_normal;
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-4);
        assert!(
            n[0] * g[0] + n[1] * g[1] + n[2] * g[2] > 0.8,
            "{n:?} vs {g:?}"
        );
        let uv = accel.uv(&s, &h, 0).unwrap();
        assert!((-1e-4..=1.0001).contains(&uv[0]) && (-1e-4..=1.0001).contains(&uv[1]));
        let c = accel.color(&s, &h, 0).unwrap();
        for (got, want) in c.iter().zip([1.0, 0.5, 0.25, 1.0]) {
            assert!((got - want).abs() < 1e-5);
        }
        let t = accel.tangent(&s, &h).unwrap();
        let mirrored = accel.instances[h.instance as usize].mirrored;
        assert_eq!(t[3], if mirrored { -1.0 } else { 1.0 });
    }
    assert!(checked > 100);
}

#[test]
fn scene_refit_follows_transform_and_vertex_edits() {
    let mut s = tricky_scene();
    let mut accel = InstanceBvh::build(&s).unwrap();
    // Move one node and squash the sphere mesh's vertices.
    let moved = accel.instances[0].node;
    s.nodes[moved.0 as usize].transform = trs([0.0, 0.0, -3.0], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    for v in &mut s.meshes[0].primitives[0].positions {
        v[1] *= 0.5;
    }
    assert!(accel.refit(&s));
    let fresh = InstanceBvh::build(&s).unwrap();
    let mut rng = Rng(0x5eed);
    for _ in 0..800 {
        let ray = scene_ray(&mut rng);
        let a = accel
            .closest_hit(&s, ray, &RayQuery::default())
            .map(|h| h.t);
        let b = fresh
            .closest_hit(&s, ray, &RayQuery::default())
            .map(|h| h.t);
        assert_eq!(a, b);
    }
    // Detaching a mesh makes refit report a stale structure.
    s.nodes[moved.0 as usize].mesh = None;
    assert!(!accel.refit(&s));
}
