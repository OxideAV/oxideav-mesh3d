//! Keyframed property animation.
//!
//! Modelled on glTF 2.0 §3.6 animation: each [`Animation`] is a bag
//! of [`AnimationChannel`]s. A channel binds an [`AnimationSampler`]
//! (time → value series) to one [`AnimationTarget`] (a node + which
//! property of that node — translation, rotation, scale, or morph
//! weights). All keyframe times are seconds from `t = 0`.

use crate::scene::NodeId;

/// Property of a node that an animation channel drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
    /// Per-morph-target weight vector. Sample length must equal the
    /// number of morph targets on the bound mesh.
    MorphWeights,
}

/// What a channel writes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnimationTarget {
    pub node: NodeId,
    pub property: AnimationProperty,
}

/// How values between keyframes are interpolated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// Hold the previous keyframe's value until the next one.
    Step,
    /// Linear interpolation. For quaternions, use NLERP (or SLERP if
    /// the renderer prefers); the typed model carries the raw
    /// keyframe values either way.
    Linear,
    /// glTF cubic-spline: each keyframe stores `(in_tangent, value, out_tangent)`,
    /// so the values vector is 3x as long as the keyframes vector.
    /// The format crate is responsible for laying that triple out.
    CubicSpline,
}

/// Time-aligned value series for one channel.
#[derive(Clone, Debug)]
pub struct AnimationSampler {
    /// Keyframe times in seconds — strictly increasing.
    pub keyframes: Vec<f32>,
    pub values: AnimationValues,
    pub interpolation: Interpolation,
}

/// One evaluated sampler value at a specific timestamp, returned by
/// [`AnimationSampler::sample`].
///
/// Variants mirror [`AnimationValues`] but each carries a single value
/// — [`Vec3`](Self::Vec3) is one 3-vector, [`Quat`](Self::Quat) one
/// 4-vector (xyzw), [`Scalar`](Self::Scalar) one boxed `Vec<f32>` since
/// `MorphWeights` samplers stride the per-frame value table by the
/// per-mesh morph-target count and there is no fixed-width arity that
/// covers every animation.
#[derive(Clone, Debug, PartialEq)]
pub enum SampledValue {
    Vec3([f32; 3]),
    /// Unit quaternion in xyzw order. The result of
    /// [`Interpolation::CubicSpline`] on a rotation channel is **not**
    /// auto-normalised by [`AnimationSampler::sample`]; callers
    /// targeting a rotation property normalise themselves (the glTF
    /// 2.0 spec Appendix C.5 implementation note requires this).
    Quat([f32; 4]),
    Scalar(Vec<f32>),
}

impl AnimationSampler {
    /// Evaluate the sampler at time `t` (seconds, in the same frame as
    /// [`AnimationSampler::keyframes`]).
    ///
    /// Implements glTF 2.0 Appendix C interpolation modes:
    ///
    /// * **C.1 Clamping.** When `t` falls outside the keyframe range
    ///   `[keyframes[0], keyframes[n-1]]`, the nearest endpoint value
    ///   is returned verbatim (no extrapolation). When `t` matches a
    ///   keyframe timestamp exactly, that keyframe's value is returned
    ///   as-is (no interpolation arithmetic).
    /// * **C.2 Step.** `v_t = v_k` — the previous keyframe's value
    ///   holds until the next keyframe overrides it.
    /// * **C.3 Linear (non-rotation).** `v_t = (1-t)*v_k + t*v_{k+1}`,
    ///   componentwise.
    /// * **C.4 Spherical linear (rotation).** SLERP between two unit
    ///   quaternions, with the shortest-arc sign correction
    ///   `s = sign(v_k . v_{k+1})` and the small-angle fallback to
    ///   componentwise lerp when `|sin(a)| < EPS`.
    /// * **C.5 Cubic spline.** Hermite blend
    ///   `v_t = (2t^3 - 3t^2 + 1)*v_k + t_d*(t^3 - 2t^2 + t)*b_k
    ///        + (-2t^3 + 3t^2)*v_{k+1} + t_d*(t^3 - t^2)*a_{k+1}`
    ///   over the segment of duration `t_d = t_{k+1} - t_k`. The
    ///   sampler's `values` array stores `[a_0, v_0, b_0, a_1, v_1,
    ///   b_1, ...]` per glTF §3.6 (in-tangent, value, out-tangent).
    ///
    /// **MorphWeights handling.** For [`AnimationValues::Scalar`] the
    /// per-frame stride is `values.len() / (keyframes.len() * factor)`
    /// where `factor = 3` for [`Interpolation::CubicSpline`] and `1`
    /// otherwise. The returned [`SampledValue::Scalar`] carries one
    /// `Vec<f32>` of length `stride` (one weight per morph target).
    ///
    /// Returns `None` when the sampler is internally malformed —
    /// empty `keyframes`, mismatched `values.len()` against the
    /// keyframe table per [`Interpolation`], or a non-positive
    /// per-frame stride for `Scalar`. A well-formed sampler always
    /// returns `Some` for any finite `t`.
    pub fn sample(&self, t: f32) -> Option<SampledValue> {
        let n = self.keyframes.len();
        if n == 0 {
            return None;
        }
        let factor: usize = match self.interpolation {
            Interpolation::CubicSpline => 3,
            _ => 1,
        };
        // Per-frame stride: how many `values` entries one keyframe
        // consumes. For Vec3 / Quat / non-MorphWeights Scalar this is
        // always `factor`; for MorphWeights it's `factor * morph_count`.
        let total = self.values.len();
        if total == 0 || total % (n * factor) != 0 {
            return None;
        }
        let per_frame_scalars = total / (n * factor);
        if per_frame_scalars == 0 {
            return None;
        }

        // Pre-/post-clamp + exact-keyframe fast paths (C.1).
        if t <= self.keyframes[0] {
            return Some(self.value_at(0, factor, per_frame_scalars));
        }
        if t >= self.keyframes[n - 1] {
            return Some(self.value_at(n - 1, factor, per_frame_scalars));
        }
        // Find segment k such that keyframes[k] <= t < keyframes[k+1].
        // n >= 2 here because the pre/post-clamp branches above
        // covered n == 1 (every t maps to keyframe 0).
        let k = match self
            .keyframes
            .binary_search_by(|kt| kt.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(exact) => return Some(self.value_at(exact, factor, per_frame_scalars)),
            Err(insert_at) => insert_at - 1,
        };
        let t_k = self.keyframes[k];
        let t_k1 = self.keyframes[k + 1];
        let t_d = t_k1 - t_k;
        // Guard against duplicate keyframes (validate() rejects these
        // but a fuzz harness might hand us one).
        if t_d <= 0.0 {
            return Some(self.value_at(k, factor, per_frame_scalars));
        }
        let u = (t - t_k) / t_d;

        match self.interpolation {
            Interpolation::Step => Some(self.value_at(k, factor, per_frame_scalars)),
            Interpolation::Linear => self.lerp(k, u, per_frame_scalars),
            Interpolation::CubicSpline => self.cubic(k, u, t_d, per_frame_scalars),
        }
    }

    fn value_at(&self, k: usize, factor: usize, stride: usize) -> SampledValue {
        // CubicSpline storage is [in, value, out, in, value, out, ...]
        // so we pick the centre triple of frame k. For Step/Linear
        // factor == 1 and the centre offset is zero.
        let centre_off = if factor == 3 { 1 } else { 0 };
        let base = (k * factor + centre_off) * stride;
        match &self.values {
            AnimationValues::Vec3(v) => SampledValue::Vec3(v[k * factor + centre_off]),
            AnimationValues::Quat(v) => SampledValue::Quat(v[k * factor + centre_off]),
            AnimationValues::Scalar(v) => SampledValue::Scalar(v[base..base + stride].to_vec()),
        }
    }

    fn lerp(&self, k: usize, u: f32, stride: usize) -> Option<SampledValue> {
        match &self.values {
            AnimationValues::Vec3(v) => {
                let a = v[k];
                let b = v[k + 1];
                Some(SampledValue::Vec3([
                    (1.0 - u) * a[0] + u * b[0],
                    (1.0 - u) * a[1] + u * b[1],
                    (1.0 - u) * a[2] + u * b[2],
                ]))
            }
            AnimationValues::Quat(v) => {
                let q0 = v[k];
                let q1 = v[k + 1];
                Some(SampledValue::Quat(slerp(q0, q1, u)))
            }
            AnimationValues::Scalar(v) => {
                let a_off = k * stride;
                let b_off = (k + 1) * stride;
                let mut out = Vec::with_capacity(stride);
                for i in 0..stride {
                    out.push((1.0 - u) * v[a_off + i] + u * v[b_off + i]);
                }
                Some(SampledValue::Scalar(out))
            }
        }
    }

    fn cubic(&self, k: usize, u: f32, t_d: f32, stride: usize) -> Option<SampledValue> {
        // glTF Appendix C.5:
        //   v_t = (2u^3 - 3u^2 + 1) v_k
        //       + t_d (u^3 - 2u^2 + u) b_k
        //       + (-2u^3 + 3u^2) v_{k+1}
        //       + t_d (u^3 - u^2) a_{k+1}
        let uu = u * u;
        let uuu = uu * u;
        let c_v_k = 2.0 * uuu - 3.0 * uu + 1.0;
        let c_b_k = t_d * (uuu - 2.0 * uu + u);
        let c_v_k1 = -2.0 * uuu + 3.0 * uu;
        let c_a_k1 = t_d * (uuu - uu);
        // Indices into the `[in, value, out, in, value, out, ...]`
        // stream. Each keyframe consumes 3 stored entries.
        let off_b_k = k * 3 + 2; // out-tangent of keyframe k
        let off_a_k1 = (k + 1) * 3; // in-tangent of keyframe k+1
        let off_v_k = k * 3 + 1; // value of keyframe k
        let off_v_k1 = (k + 1) * 3 + 1; // value of keyframe k+1
        match &self.values {
            AnimationValues::Vec3(v) => {
                let v_k = v[off_v_k];
                let v_k1 = v[off_v_k1];
                let b_k = v[off_b_k];
                let a_k1 = v[off_a_k1];
                Some(SampledValue::Vec3([
                    c_v_k * v_k[0] + c_b_k * b_k[0] + c_v_k1 * v_k1[0] + c_a_k1 * a_k1[0],
                    c_v_k * v_k[1] + c_b_k * b_k[1] + c_v_k1 * v_k1[1] + c_a_k1 * a_k1[1],
                    c_v_k * v_k[2] + c_b_k * b_k[2] + c_v_k1 * v_k1[2] + c_a_k1 * a_k1[2],
                ]))
            }
            AnimationValues::Quat(v) => {
                let v_k = v[off_v_k];
                let v_k1 = v[off_v_k1];
                let b_k = v[off_b_k];
                let a_k1 = v[off_a_k1];
                // Spec C.5: the cubic blend is componentwise; rotation
                // re-normalisation is the caller's responsibility.
                Some(SampledValue::Quat([
                    c_v_k * v_k[0] + c_b_k * b_k[0] + c_v_k1 * v_k1[0] + c_a_k1 * a_k1[0],
                    c_v_k * v_k[1] + c_b_k * b_k[1] + c_v_k1 * v_k1[1] + c_a_k1 * a_k1[1],
                    c_v_k * v_k[2] + c_b_k * b_k[2] + c_v_k1 * v_k1[2] + c_a_k1 * a_k1[2],
                    c_v_k * v_k[3] + c_b_k * b_k[3] + c_v_k1 * v_k1[3] + c_a_k1 * a_k1[3],
                ]))
            }
            AnimationValues::Scalar(v) => {
                let base_b_k = off_b_k * stride;
                let base_a_k1 = off_a_k1 * stride;
                let base_v_k = off_v_k * stride;
                let base_v_k1 = off_v_k1 * stride;
                let mut out = Vec::with_capacity(stride);
                for i in 0..stride {
                    out.push(
                        c_v_k * v[base_v_k + i]
                            + c_b_k * v[base_b_k + i]
                            + c_v_k1 * v[base_v_k1 + i]
                            + c_a_k1 * v[base_a_k1 + i],
                    );
                }
                Some(SampledValue::Scalar(out))
            }
        }
    }
}

impl AnimationSampler {
    /// Build a well-formed [`AnimationProperty::MorphWeights`] sampler
    /// from per-keyframe weight vectors — the synthesis path for
    /// authoring a sampled morph-weight animation purely through the
    /// typed model (no manual stride flattening).
    ///
    /// `frames[k]` is the full weight vector at `keyframes[k]` — one
    /// `f32` per morph target of the mesh the channel will drive
    /// (glTF 2.0 §3.6: a weights sampler carries `count(targets)`
    /// floats per keyframe). The vectors are flattened row-major into
    /// [`AnimationValues::Scalar`] exactly as
    /// [`AnimationSampler::sample`] expects.
    ///
    /// Returns `None` — instead of a sampler [`Scene3D::validate`]
    /// would reject — when:
    ///
    /// * `keyframes` is empty, non-finite, or not strictly increasing
    ///   (§3.6 requires strictly-increasing timestamps);
    /// * `frames.len() != keyframes.len()`;
    /// * the frames don't share one non-zero length (the per-frame
    ///   stride must be constant and at least 1);
    /// * `interpolation` is [`Interpolation::CubicSpline`] — cubic
    ///   keyframes carry tangent triples, use
    ///   [`AnimationSampler::morph_weights_cubic`].
    ///
    /// Weight *values* are stored verbatim (negative and >1 weights
    /// are meaningful morph inputs and pass through).
    ///
    /// [`Scene3D::validate`]: crate::Scene3D::validate
    pub fn morph_weights(
        keyframes: Vec<f32>,
        frames: Vec<Vec<f32>>,
        interpolation: Interpolation,
    ) -> Option<Self> {
        if matches!(interpolation, Interpolation::CubicSpline) {
            return None;
        }
        let stride = uniform_stride(&keyframes, &[&frames])?;
        let mut values = Vec::with_capacity(frames.len() * stride);
        for f in &frames {
            values.extend_from_slice(f);
        }
        Some(Self {
            keyframes,
            values: AnimationValues::Scalar(values),
            interpolation,
        })
    }

    /// Cubic-spline companion of [`AnimationSampler::morph_weights`]:
    /// build a [`Interpolation::CubicSpline`] `MorphWeights` sampler
    /// from parallel per-keyframe in-tangent / value / out-tangent
    /// weight vectors.
    ///
    /// Per glTF 2.0 §3.6 each cubic keyframe stores the triple
    /// `(in_tangent, value, out_tangent)`; this constructor lays the
    /// three parallel tables out in the interleaved
    /// `[a_0, v_0, b_0, a_1, v_1, b_1, …]` order the sampler reads
    /// (each element `stride` weights wide).
    ///
    /// Returns `None` under the same structural rules as
    /// [`AnimationSampler::morph_weights`], with all **three** tables
    /// required to match the keyframe count and share one non-zero
    /// stride.
    pub fn morph_weights_cubic(
        keyframes: Vec<f32>,
        in_tangents: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
        out_tangents: Vec<Vec<f32>>,
    ) -> Option<Self> {
        let stride = uniform_stride(&keyframes, &[&in_tangents, &values, &out_tangents])?;
        let mut flat = Vec::with_capacity(keyframes.len() * 3 * stride);
        for k in 0..keyframes.len() {
            flat.extend_from_slice(&in_tangents[k]);
            flat.extend_from_slice(&values[k]);
            flat.extend_from_slice(&out_tangents[k]);
        }
        Some(Self {
            keyframes,
            values: AnimationValues::Scalar(flat),
            interpolation: Interpolation::CubicSpline,
        })
    }

    /// Per-keyframe weight-vector width of a `MorphWeights`-shaped
    /// sampler — how many `f32`s one keyframe carries (equals the
    /// morph-target count of the driven mesh on a conforming channel).
    ///
    /// `None` when the values are not [`AnimationValues::Scalar`] or
    /// the sampler is structurally malformed (empty keyframes, value
    /// table not an exact multiple of the keyframe table under the
    /// interpolation's storage factor) — the same well-formedness
    /// test [`AnimationSampler::sample`] applies.
    pub fn morph_weight_stride(&self) -> Option<usize> {
        let AnimationValues::Scalar(v) = &self.values else {
            return None;
        };
        let n = self.keyframes.len();
        if n == 0 {
            return None;
        }
        let factor: usize = match self.interpolation {
            Interpolation::CubicSpline => 3,
            _ => 1,
        };
        let total = v.len();
        if total == 0 || total % (n * factor) != 0 {
            return None;
        }
        Some(total / (n * factor))
    }

    /// The **stored** weight vector at keyframe `k` — no
    /// interpolation. For [`Interpolation::CubicSpline`] this is the
    /// centre *value* element of the keyframe's
    /// `(in_tangent, value, out_tangent)` triple.
    ///
    /// The read-back side of [`AnimationSampler::morph_weights`]: an
    /// exporter that needs the authored key values (rather than a
    /// resampling) walks `(0..keyframes.len())` with this. `None`
    /// under the [`morph_weight_stride`] conditions or when `k` is
    /// out of range.
    ///
    /// [`morph_weight_stride`]: AnimationSampler::morph_weight_stride
    pub fn morph_weight_frame(&self, k: usize) -> Option<&[f32]> {
        let stride = self.morph_weight_stride()?;
        if k >= self.keyframes.len() {
            return None;
        }
        let AnimationValues::Scalar(v) = &self.values else {
            return None;
        };
        let (factor, centre) = match self.interpolation {
            Interpolation::CubicSpline => (3, 1),
            _ => (1, 0),
        };
        let base = (k * factor + centre) * stride;
        v.get(base..base + stride)
    }

    /// Every stored per-keyframe weight vector, in keyframe order —
    /// [`AnimationSampler::morph_weight_frame`] over the whole
    /// sampler. `Some` is always exactly `keyframes.len()` slices
    /// long. Lossless against [`AnimationSampler::morph_weights`]:
    /// `morph_weights(t, f, i)?.morph_weight_frames()` yields `f`.
    pub fn morph_weight_frames(&self) -> Option<Vec<&[f32]>> {
        self.morph_weight_stride()?;
        (0..self.keyframes.len())
            .map(|k| self.morph_weight_frame(k))
            .collect()
    }

    /// The full `(in_tangent, value, out_tangent)` weight-vector
    /// triple stored at keyframe `k` of a
    /// [`Interpolation::CubicSpline`] `MorphWeights` sampler — the
    /// lossless read-back of
    /// [`AnimationSampler::morph_weights_cubic`]. `None` for
    /// non-cubic samplers, non-`Scalar` values, malformed storage, or
    /// `k` out of range.
    pub fn morph_weight_cubic_frame(&self, k: usize) -> Option<(&[f32], &[f32], &[f32])> {
        if !matches!(self.interpolation, Interpolation::CubicSpline) {
            return None;
        }
        let stride = self.morph_weight_stride()?;
        if k >= self.keyframes.len() {
            return None;
        }
        let AnimationValues::Scalar(v) = &self.values else {
            return None;
        };
        let base = k * 3 * stride;
        Some((
            v.get(base..base + stride)?,
            v.get(base + stride..base + 2 * stride)?,
            v.get(base + 2 * stride..base + 3 * stride)?,
        ))
    }
}

/// Shared structural gate of the two `morph_weights*` constructors:
/// keyframes non-empty / finite / strictly increasing, every parallel
/// table exactly `keyframes.len()` rows, and all rows across all
/// tables one common non-zero width. Returns that width.
fn uniform_stride(keyframes: &[f32], tables: &[&Vec<Vec<f32>>]) -> Option<usize> {
    if keyframes.is_empty() {
        return None;
    }
    let mut prev = f32::NEG_INFINITY;
    for &t in keyframes {
        if !t.is_finite() || t <= prev {
            return None;
        }
        prev = t;
    }
    let stride = tables.first()?.first()?.len();
    if stride == 0 {
        return None;
    }
    for table in tables {
        if table.len() != keyframes.len() || table.iter().any(|row| row.len() != stride) {
            return None;
        }
    }
    Some(stride)
}

/// Spherical linear interpolation between two unit quaternions in
/// xyzw order, per glTF 2.0 Appendix C.4.
///
/// Picks the short arc by flipping `q1`'s sign when the dot product
/// is negative, then falls back to componentwise lerp when the angle
/// `a = acos(|q0 . q1|)` is so small that `sin(a)` underflows.
fn slerp(q0: [f32; 4], q1: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    // Short-arc: if the cosine is negative, negate one endpoint so we
    // interpolate the short way round.
    let q1 = if dot < 0.0 {
        dot = -dot;
        [-q1[0], -q1[1], -q1[2], -q1[3]]
    } else {
        q1
    };
    // Clamp to safe acos domain (numerical noise can take dot just
    // outside [-1, 1]).
    let dot = dot.clamp(-1.0, 1.0);
    let a = dot.acos();
    let sin_a = a.sin();
    if sin_a.abs() < 1.0e-6 {
        // Near-identical endpoints: collapse to componentwise lerp +
        // re-normalise so callers always get a unit quaternion. (The
        // spec's note: "when a is close to zero, spherical linear
        // interpolation turns into regular linear interpolation".)
        let r = [
            (1.0 - t) * q0[0] + t * q1[0],
            (1.0 - t) * q0[1] + t * q1[1],
            (1.0 - t) * q0[2] + t * q1[2],
            (1.0 - t) * q0[3] + t * q1[3],
        ];
        let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
        if n > 0.0 {
            [r[0] / n, r[1] / n, r[2] / n, r[3] / n]
        } else {
            r
        }
    } else {
        let w0 = ((1.0 - t) * a).sin() / sin_a;
        let w1 = (t * a).sin() / sin_a;
        [
            w0 * q0[0] + w1 * q1[0],
            w0 * q0[1] + w1 * q1[1],
            w0 * q0[2] + w1 * q1[2],
            w0 * q0[3] + w1 * q1[3],
        ]
    }
}

/// Type of each keyframe value. The variant must match the bound
/// [`AnimationProperty`]: `Translation`/`Scale` → [`Vec3`](AnimationValues::Vec3),
/// `Rotation` → [`Quat`](AnimationValues::Quat), `MorphWeights` →
/// [`Scalar`](AnimationValues::Scalar) (concatenated weight vectors).
#[derive(Clone, Debug, PartialEq)]
pub enum AnimationValues {
    Vec3(Vec<[f32; 3]>),
    Quat(Vec<[f32; 4]>),
    Scalar(Vec<f32>),
}

impl AnimationValues {
    /// Number of stored values, regardless of variant arity.
    pub fn len(&self) -> usize {
        match self {
            Self::Vec3(v) => v.len(),
            Self::Quat(v) => v.len(),
            Self::Scalar(v) => v.len(),
        }
    }

    /// `true` if no values are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One animated property — `(target, sampler)`.
#[derive(Clone, Debug)]
pub struct AnimationChannel {
    pub target: AnimationTarget,
    pub sampler: AnimationSampler,
}

impl AnimationChannel {
    /// Channel binding `sampler` to `property` of `node`.
    pub fn new(node: NodeId, property: AnimationProperty, sampler: AnimationSampler) -> Self {
        Self {
            target: AnimationTarget { node, property },
            sampler,
        }
    }
}

/// Named animation — a bag of channels played back together.
#[derive(Clone, Debug, Default)]
pub struct Animation {
    pub name: Option<String>,
    pub channels: Vec<AnimationChannel>,
}

impl Animation {
    /// Empty animation with the given name.
    pub fn new(name: impl Into<Option<String>>) -> Self {
        Self {
            name: name.into(),
            channels: Vec::new(),
        }
    }

    /// Push a channel binding `sampler` to `property` of `node` and
    /// return `self` for chaining — the builder companion of
    /// [`AnimationChannel::new`].
    ///
    /// ```
    /// use oxideav_mesh3d::{
    ///     Animation, AnimationProperty, AnimationSampler, Interpolation, NodeId,
    /// };
    ///
    /// let sampler = AnimationSampler::morph_weights(
    ///     vec![0.0, 1.0],
    ///     vec![vec![0.0, 0.0], vec![1.0, 0.5]],
    ///     Interpolation::Linear,
    /// )
    /// .unwrap();
    /// let anim = Animation::new("blink".to_owned()).with_channel(
    ///     NodeId(0),
    ///     AnimationProperty::MorphWeights,
    ///     sampler,
    /// );
    /// assert_eq!(anim.channels.len(), 1);
    /// ```
    pub fn with_channel(
        mut self,
        node: NodeId,
        property: AnimationProperty,
        sampler: AnimationSampler,
    ) -> Self {
        self.channels
            .push(AnimationChannel::new(node, property, sampler));
        self
    }

    /// The channel driving `property` of `node`, or `None` when this
    /// animation doesn't animate that slot.
    ///
    /// glTF 2.0 §5.6 forbids two channels of one animation targeting
    /// the same node + property; on out-of-spec duplicates the **last**
    /// matching channel is returned — the same later-channel-wins rule
    /// [`Animation::sample_pose`](crate::Animation) applies when
    /// building a [`Pose`](crate::Pose).
    pub fn channel_for(
        &self,
        node: NodeId,
        property: AnimationProperty,
    ) -> Option<&AnimationChannel> {
        self.channels
            .iter()
            .rev()
            .find(|ch| ch.target.node == node && ch.target.property == property)
    }
}
