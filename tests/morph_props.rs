//! Property tests over the round-448 morph surfaces — seeded LCG,
//! fully deterministic, no external dependencies (same harness style
//! as `tests/skinning_props.rs`).
//!
//! Invariants:
//!
//! * `AnimationSampler::morph_weights` round-trips its frames
//!   losslessly and samples exactly at keyframes; Linear samples stay
//!   inside the bracketing frames' componentwise hull;
//! * the cubic constructor round-trips its `(in, value, out)` triples
//!   and samples the centre values at keyframes;
//! * `MorphTarget::at_weight` reproduces every station's shape
//!   exactly at its weight, the endpoints at `0`/`1`, and is affine
//!   between adjacent stations;
//! * a target without in-betweens resolves to the linear rule for
//!   any weight;
//! * `apply_morph_weights` equals the manual `base + Σ at_weight(wᵢ)`
//!   station sum;
//! * morphing commutes with `transformed` on positions when
//!   in-betweens are present;
//! * `optimize_vertex_fetch` leaves the *morphed* drawn triangles
//!   invariant (the in-between buffers permute with the pool).

use oxideav_mesh3d::{
    AnimationSampler, Inbetween, Interpolation, MorphTarget, Primitive, SampledValue, Topology,
    Transform,
};

/// Minimal deterministic LCG (numerical-recipes constants).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [lo, hi).
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next_u32() as f32 / u32::MAX as f32)
    }
    fn usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n.max(1)
    }
    fn vec3(&mut self, lo: f32, hi: f32) -> [f32; 3] {
        [self.f32(lo, hi), self.f32(lo, hi), self.f32(lo, hi)]
    }
    fn quat(&mut self) -> [f32; 4] {
        loop {
            let q = [
                self.f32(-1.0, 1.0),
                self.f32(-1.0, 1.0),
                self.f32(-1.0, 1.0),
                self.f32(-1.0, 1.0),
            ];
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if n > 0.1 {
                return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
            }
        }
    }
    /// Strictly increasing keyframe table.
    fn keyframes(&mut self, n: usize) -> Vec<f32> {
        let mut t = 0.0f32;
        (0..n)
            .map(|_| {
                t += self.f32(0.05, 1.0);
                t
            })
            .collect()
    }
    fn frame(&mut self, stride: usize) -> Vec<f32> {
        (0..stride).map(|_| self.f32(-2.0, 2.0)).collect()
    }
}

fn close(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

fn close3(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
    close(a[0], b[0], eps) && close(a[1], b[1], eps) && close(a[2], b[2], eps)
}

fn unwrap_scalar(s: SampledValue) -> Vec<f32> {
    match s {
        SampledValue::Scalar(v) => v,
        other => panic!("expected Scalar, got {other:?}"),
    }
}

// ---------------------------------------------------------------- //
// Sampler synthesis                                                //
// ---------------------------------------------------------------- //

#[test]
fn prop_morph_weights_round_trip_and_hull() {
    let mut rng = Lcg::new(0x448_0001);
    for _ in 0..300 {
        let n = 1 + rng.usize(6);
        let stride = 1 + rng.usize(4);
        let keyframes = rng.keyframes(n);
        let frames: Vec<Vec<f32>> = (0..n).map(|_| rng.frame(stride)).collect();
        let interp = if rng.usize(2) == 0 {
            Interpolation::Step
        } else {
            Interpolation::Linear
        };
        let s = AnimationSampler::morph_weights(keyframes.clone(), frames.clone(), interp)
            .expect("well-formed by construction");
        // Lossless read-back.
        assert_eq!(s.morph_weight_stride(), Some(stride));
        let back = s.morph_weight_frames().unwrap();
        for (a, b) in back.iter().zip(&frames) {
            assert_eq!(*a, b.as_slice());
        }
        // Exact keyframe samples.
        for (k, t) in keyframes.iter().enumerate() {
            assert_eq!(unwrap_scalar(s.sample(*t).unwrap()), frames[k]);
        }
        // Interior samples stay in the bracketing hull (Step holds
        // the previous frame; Linear blends the two).
        for _ in 0..8 {
            let k = rng.usize(n.max(2) - 1);
            if k + 1 >= n {
                continue;
            }
            let u = rng.f32(0.0, 1.0);
            let t = keyframes[k] + u * (keyframes[k + 1] - keyframes[k]);
            let v = unwrap_scalar(s.sample(t).unwrap());
            for c in 0..stride {
                let lo = frames[k][c].min(frames[k + 1][c]) - 1e-4;
                let hi = frames[k][c].max(frames[k + 1][c]) + 1e-4;
                assert!(
                    v[c] >= lo && v[c] <= hi,
                    "sample escaped the bracketing hull: {} vs [{lo}, {hi}]",
                    v[c]
                );
            }
        }
    }
}

#[test]
fn prop_cubic_round_trip() {
    let mut rng = Lcg::new(0x448_0002);
    for _ in 0..200 {
        let n = 1 + rng.usize(5);
        let stride = 1 + rng.usize(3);
        let keyframes = rng.keyframes(n);
        let ins: Vec<Vec<f32>> = (0..n).map(|_| rng.frame(stride)).collect();
        let vals: Vec<Vec<f32>> = (0..n).map(|_| rng.frame(stride)).collect();
        let outs: Vec<Vec<f32>> = (0..n).map(|_| rng.frame(stride)).collect();
        let s = AnimationSampler::morph_weights_cubic(
            keyframes.clone(),
            ins.clone(),
            vals.clone(),
            outs.clone(),
        )
        .expect("well-formed by construction");
        assert_eq!(s.morph_weight_stride(), Some(stride));
        for k in 0..n {
            let (a, v, b) = s.morph_weight_cubic_frame(k).unwrap();
            assert_eq!(a, ins[k].as_slice());
            assert_eq!(v, vals[k].as_slice());
            assert_eq!(b, outs[k].as_slice());
            // C.1: exact keyframe returns the centre value verbatim.
            assert_eq!(unwrap_scalar(s.sample(keyframes[k]).unwrap()), vals[k]);
        }
    }
}

// ---------------------------------------------------------------- //
// at_weight resolution                                             //
// ---------------------------------------------------------------- //

/// Random target: `n` vertices, optional normal slot, `k` in-betweens
/// at distinct stations drawn from (0, 1).
fn random_target(rng: &mut Lcg, n: usize, k: usize) -> MorphTarget {
    let mut t = MorphTarget::new();
    t.position = Some((0..n).map(|_| rng.vec3(-1.0, 1.0)).collect());
    if rng.usize(2) == 0 {
        t.normal = Some((0..n).map(|_| rng.vec3(-0.5, 0.5)).collect());
    }
    let mut stations: Vec<f32> = Vec::new();
    while stations.len() < k {
        // Quantised draw keeps stations distinct and away from 0/1.
        let w = (1 + rng.usize(98)) as f32 / 100.0;
        if !stations.contains(&w) {
            stations.push(w);
        }
    }
    t.inbetweens = stations
        .into_iter()
        .map(|w| {
            let mut ib =
                Inbetween::new(w).with_position((0..n).map(|_| rng.vec3(-1.0, 1.0)).collect());
            if rng.usize(2) == 0 {
                ib = ib.with_normal((0..n).map(|_| rng.vec3(-0.5, 0.5)).collect());
            }
            ib
        })
        .collect();
    t
}

#[test]
fn prop_at_weight_hits_stations_and_endpoints() {
    let mut rng = Lcg::new(0x448_0003);
    for _ in 0..200 {
        let n = 1 + rng.usize(6);
        let k = rng.usize(4);
        let t = random_target(&mut rng, n, k);
        // Endpoints.
        let r0 = t.at_weight(0.0);
        for v in r0.position.as_ref().unwrap() {
            assert!(close3(*v, [0.0; 3], 1e-6));
        }
        let r1 = t.at_weight(1.0);
        for (a, b) in r1
            .position
            .as_ref()
            .unwrap()
            .iter()
            .zip(t.position.as_ref().unwrap())
        {
            assert!(close3(*a, *b, 1e-5));
        }
        // Every station reproduces its shape exactly.
        for ib in &t.inbetweens {
            let r = t.at_weight(ib.weight);
            let got = r.position.as_ref().unwrap();
            let want = ib.position.as_ref().unwrap();
            for (a, b) in got.iter().zip(want) {
                assert!(close3(*a, *b, 1e-5));
            }
        }
    }
}

#[test]
fn prop_at_weight_affine_between_adjacent_stations() {
    let mut rng = Lcg::new(0x448_0004);
    for _ in 0..200 {
        let n = 1 + rng.usize(5);
        let k = 1 + rng.usize(3);
        let t = random_target(&mut rng, n, k);
        // Full sorted ladder incl. endpoints.
        let mut ws: Vec<f32> = t.inbetweens.iter().map(|ib| ib.weight).collect();
        ws.push(0.0);
        ws.push(1.0);
        ws.sort_by(f32::total_cmp);
        let seg = rng.usize(ws.len() - 1);
        let (a, b) = (ws[seg], ws[seg + 1]);
        let mid = 0.5 * (a + b);
        let ra = t.at_weight(a);
        let rb = t.at_weight(b);
        let rm = t.at_weight(mid);
        for i in 0..n {
            let pa = ra.position.as_ref().unwrap()[i];
            let pb = rb.position.as_ref().unwrap()[i];
            let pm = rm.position.as_ref().unwrap()[i];
            let want = [
                0.5 * (pa[0] + pb[0]),
                0.5 * (pa[1] + pb[1]),
                0.5 * (pa[2] + pb[2]),
            ];
            assert!(close3(pm, want, 1e-4));
        }
    }
}

#[test]
fn prop_no_inbetweens_is_linear_for_any_weight() {
    let mut rng = Lcg::new(0x448_0005);
    for _ in 0..300 {
        let n = 1 + rng.usize(6);
        let t = random_target(&mut rng, n, 0);
        let w = rng.f32(-2.0, 2.0);
        let r = t.at_weight(w);
        for (out, base) in r
            .position
            .as_ref()
            .unwrap()
            .iter()
            .zip(t.position.as_ref().unwrap())
        {
            assert!(close3(*out, [w * base[0], w * base[1], w * base[2]], 1e-4));
        }
    }
}

// ---------------------------------------------------------------- //
// apply_morph_weights routing                                      //
// ---------------------------------------------------------------- //

fn random_morph_primitive(rng: &mut Lcg, n: usize, n_targets: usize) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = (0..n).map(|_| rng.vec3(-3.0, 3.0)).collect();
    p.targets = (0..n_targets)
        .map(|_| {
            let k = rng.usize(3);
            random_target(rng, n, k)
        })
        .collect();
    p
}

#[test]
fn prop_apply_equals_station_sum() {
    let mut rng = Lcg::new(0x448_0006);
    for _ in 0..200 {
        let n = 3 * (1 + rng.usize(3));
        let nt = 1 + rng.usize(3);
        let p = random_morph_primitive(&mut rng, n, nt);
        let weights: Vec<f32> = (0..p.targets.len()).map(|_| rng.f32(-1.5, 1.5)).collect();
        let m = p.apply_morph_weights(&weights);
        for i in 0..n {
            let mut want = p.positions[i];
            for (t, &w) in p.targets.iter().zip(&weights) {
                if w == 0.0 {
                    continue;
                }
                let r = t.at_weight(w);
                let d = r.position.as_ref().unwrap()[i];
                want = [want[0] + d[0], want[1] + d[1], want[2] + d[2]];
            }
            assert!(close3(m.positions[i], want, 1e-3));
        }
    }
}

#[test]
fn prop_morph_transform_commute_with_inbetweens() {
    let mut rng = Lcg::new(0x448_0007);
    for _ in 0..150 {
        let n = 3;
        let nt = 1 + rng.usize(2);
        let p = random_morph_primitive(&mut rng, n, nt);
        let weights: Vec<f32> = (0..p.targets.len()).map(|_| rng.f32(-1.0, 1.5)).collect();
        let m = Transform::Trs {
            translation: rng.vec3(-5.0, 5.0),
            rotation: rng.quat(),
            scale: [rng.f32(0.2, 3.0), rng.f32(0.2, 3.0), rng.f32(0.2, 3.0)],
        }
        .to_matrix();
        let a = p.transformed(m).morphed(&weights).positions;
        let b = p.morphed(&weights).transformed(m).positions;
        for (pa, pb) in a.iter().zip(&b) {
            assert!(close3(*pa, *pb, 2e-3), "{pa:?} vs {pb:?}");
        }
    }
}

#[test]
fn prop_vertex_fetch_preserves_morphed_triangles() {
    let mut rng = Lcg::new(0x448_0008);
    for _ in 0..150 {
        let n = 4 + rng.usize(6);
        let nt = 1 + rng.usize(2);
        let mut p = random_morph_primitive(&mut rng, n, nt);
        // Random triangle list over the pool.
        let tri_count = 1 + rng.usize(5);
        let idx: Vec<u32> = (0..tri_count * 3).map(|_| rng.usize(n) as u32).collect();
        p.indices = Some(oxideav_mesh3d::Indices::U32(idx));
        let weights: Vec<f32> = (0..p.targets.len()).map(|_| rng.f32(-1.0, 1.5)).collect();

        let before = p.morphed(&weights);
        let after = p.optimize_vertex_fetch().morphed(&weights);
        let ta = before.triangle_indices();
        let tb = after.triangle_indices();
        assert_eq!(ta.len(), tb.len());
        for (fa, fb) in ta.iter().zip(&tb) {
            for c in 0..3 {
                let va = before.positions[fa[c] as usize];
                let vb = after.positions[fb[c] as usize];
                assert!(close3(va, vb, 1e-4));
            }
        }
    }
}
