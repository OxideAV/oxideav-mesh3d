//! Ray-query throughput benchmark: object-median vs binned-SAH BVH,
//! Möller-Trumbore vs watertight leaves, per-primitive and two-level
//! instanced scenes.
//!
//! ```text
//! cargo run --release --example ray_bench [-- <ray_count>]
//! ```
//!
//! Single-threaded; reports build time, tree statistics, and closest-
//! hit / any-hit throughput in millions of rays per second. Geometry
//! and rays come from a fixed-seed xorshift generator so runs are
//! comparable.

use oxideav_mesh3d::{
    Bvh, BvhBuildOptions, BvhBuildStrategy, Indices, InstanceBvh, Mesh, Node, PreparedRay,
    Primitive, Ray, RayQuery, Scene3D, Topology, Transform, TriangleTest,
};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// UV sphere with a radial ripple — smooth, well-distributed surface.
fn bumpy_sphere(seg_u: u32, seg_v: u32) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    for j in 0..=seg_v {
        let th = std::f32::consts::PI * j as f32 / seg_v as f32;
        for i in 0..=seg_u {
            let ph = std::f32::consts::TAU * i as f32 / seg_u as f32;
            let r = 1.0 + 0.05 * (7.0 * ph).sin() * (5.0 * th).cos();
            p.positions.push([
                r * th.sin() * ph.cos(),
                r * th.cos(),
                r * th.sin() * ph.sin(),
            ]);
        }
    }
    let mut idx = Vec::new();
    for j in 0..seg_v {
        for i in 0..seg_u {
            let a = j * (seg_u + 1) + i;
            let b = a + seg_u + 1;
            idx.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    p.indices = Some(Indices::U32(idx));
    p
}

/// Small random triangles in a cube plus a few huge ones — the case
/// where centroid-median splits do badly.
fn mixed_soup(n: usize, rng: &mut Rng) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    for k in 0..n {
        let size = if k % 500 == 0 { 1.5 } else { 0.03 };
        let c = [
            rng.f() * 2.0 - 1.0,
            rng.f() * 2.0 - 1.0,
            rng.f() * 2.0 - 1.0,
        ];
        for _ in 0..3 {
            p.positions.push([
                c[0] + size * (rng.f() - 0.5),
                c[1] + size * (rng.f() - 0.5),
                c[2] + size * (rng.f() - 0.5),
            ]);
        }
    }
    p
}

fn rays(n: usize, rng: &mut Rng, spread: f32) -> Vec<Ray> {
    (0..n)
        .map(|_| {
            let o = [
                3.0 * spread * (rng.f() * 2.0 - 1.0),
                3.0 * spread * (rng.f() * 2.0 - 1.0),
                3.0 * spread,
            ];
            let t = [
                spread * (rng.f() * 2.0 - 1.0),
                spread * (rng.f() * 2.0 - 1.0),
                rng.f() * 2.0 - 1.0,
            ];
            Ray::new(o, [t[0] - o[0], t[1] - o[1], t[2] - o[2]])
        })
        .collect()
}

/// Coherent pinhole-camera rays: a `side x side` image looking down
/// -Z from z = 3 at the unit cube.
fn camera_rays(side: usize) -> Vec<Ray> {
    let mut out = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let u = (x as f32 + 0.5) / side as f32 * 2.0 - 1.0;
            let v = (y as f32 + 0.5) / side as f32 * 2.0 - 1.0;
            out.push(Ray::new([0.0, 0.0, 3.0], [0.4 * u, 0.4 * v, -1.0]));
        }
    }
    out
}

fn mrays(n: usize, secs: f64) -> f64 {
    n as f64 / secs / 1e6
}

fn label(o: &BvhBuildOptions) -> &'static str {
    match o.strategy {
        BvhBuildStrategy::ObjectMedian => "median",
        BvhBuildStrategy::BinnedSah => "sah",
    }
}

fn bench_prim(name: &str, p: &Primitive, rays: &[Ray]) {
    for opts in [BvhBuildOptions::object_median(), BvhBuildOptions::sah()] {
        let t0 = Instant::now();
        let bvh = Bvh::build_with(p, &opts).unwrap();
        let build = t0.elapsed().as_secs_f64();
        let prepared: Vec<PreparedRay> = rays.iter().map(|r| PreparedRay::new(*r)).collect();
        for test in [TriangleTest::MollerTrumbore, TriangleTest::Watertight] {
            let q = RayQuery::default().with_triangle_test(test);
            let t0 = Instant::now();
            let mut hits = 0usize;
            for r in &prepared {
                hits += bvh.closest_hit(p, r, &q).is_some() as usize;
            }
            let ch = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let mut occ = 0usize;
            for r in &prepared {
                occ += bvh.occluded(p, r, &q) as usize;
            }
            let ah = t0.elapsed().as_secs_f64();
            println!(
                "{name:7} {:6} {:12} tris={:7} nodes={:7} depth={:2} sah={:7.1} build={:7.1}ms  closest={:6.2} Mrays/s  any={:6.2} Mrays/s  hits={hits}/{occ}",
                label(&opts),
                format!("{test:?}"),
                p.triangle_count(),
                bvh.node_count(),
                bvh.depth(),
                bvh.sah_cost(),
                build * 1e3,
                mrays(rays.len(), ch),
                mrays(rays.len(), ah),
            );
        }
    }
}

fn bench_scene(rays: &[Ray]) {
    let mut s = Scene3D::new();
    let mid = s.add_mesh(Mesh::new(None).with_primitive(bumpy_sphere(64, 32)));
    for i in 0..16 {
        for j in 0..16 {
            let t = Transform::Trs {
                translation: [-0.9375 + 0.125 * i as f32, -0.9375 + 0.125 * j as f32, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.05, 0.06, 0.05],
            };
            let n = s.add_node(Node::new().with_transform(t).with_mesh(mid));
            s.add_root(n);
        }
    }
    let tris = s.meshes[0].primitives[0].triangle_count();
    for opts in [BvhBuildOptions::object_median(), BvhBuildOptions::sah()] {
        let t0 = Instant::now();
        let accel = InstanceBvh::build_with(&s, &opts).unwrap();
        let build = t0.elapsed().as_secs_f64();
        let q = RayQuery::default();
        let t0 = Instant::now();
        let mut hits = 0usize;
        for r in rays {
            hits += accel.closest_hit(&s, *r, &q).is_some() as usize;
        }
        let ch = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let mut occ = 0usize;
        for r in rays {
            occ += accel.occluded(&s, *r, &q) as usize;
        }
        let ah = t0.elapsed().as_secs_f64();
        println!(
            "scene   {:6} 256 inst x {tris} tris  tlas_nodes={} build={:6.1}ms  closest={:6.2} Mrays/s  any={:6.2} Mrays/s  hits={hits}/{occ}",
            label(&opts),
            accel.node_count(),
            build * 1e3,
            mrays(rays.len(), ch),
            mrays(rays.len(), ah),
        );
    }
    // Brute-force scene walk for reference (fewer rays — it is slow).
    let few = &rays[..rays.len().min(2000)];
    let t0 = Instant::now();
    let mut hits = 0usize;
    for r in few {
        hits += s.intersect_ray(*r, f32::INFINITY).is_some() as usize;
    }
    println!(
        "scene   brute-force Scene3D::intersect_ray  closest={:6.4} Mrays/s  hits={hits}/{}",
        mrays(few.len(), t0.elapsed().as_secs_f64()),
        few.len()
    );
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200_000);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let rs = rays(n, &mut rng, 1.0);
    let cam = camera_rays((n as f64).sqrt() as usize);
    let sphere = bumpy_sphere(512, 256);
    let soup = mixed_soup(100_000, &mut rng);
    println!("-- incoherent random rays ({n})");
    bench_prim("sphere", &sphere, &rs);
    bench_prim("soup", &soup, &rs);
    bench_scene(&rs);
    println!("-- coherent camera rays ({})", cam.len());
    bench_prim("sphere", &sphere, &cam);
    bench_prim("soup", &soup, &cam);
    bench_scene(&cam);
}
