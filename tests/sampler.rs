//! Tests for the typed sampler surface: undefined-vs-explicit
//! filters, effective fallbacks, mip-behaviour decomposition, and
//! wrap-mode reference semantics.

use oxideav_mesh3d::{MagFilter, MinFilter, MipmapMode, Sampler, WrapMode};

// ---------------------------------------------------------------
// Defaults / round-trip distinguishability

#[test]
fn default_sampler_leaves_filters_undefined_and_repeats() {
    let s = Sampler::default_sampler();
    assert_eq!(s.mag_filter, None);
    assert_eq!(s.min_filter, None);
    assert_eq!(s.wrap_s, WrapMode::Repeat);
    assert_eq!(s.wrap_t, WrapMode::Repeat);
    assert_eq!(s, Sampler::default());
}

#[test]
fn wrap_mode_default_is_repeat() {
    assert_eq!(WrapMode::default(), WrapMode::Repeat);
}

#[test]
fn undefined_filter_stays_distinguishable_from_explicit_choice() {
    // A file that spelled out trilinear must not compare equal to one
    // that omitted the filter, even though both render the same under
    // the crate's fallback.
    let implicit = Sampler::default_sampler();
    let explicit = Sampler::default_sampler()
        .with_mag_filter(MagFilter::Linear)
        .with_min_filter(MinFilter::LinearMipLinear);
    assert_ne!(implicit, explicit);
    assert_eq!(
        implicit.effective_mag_filter(),
        explicit.effective_mag_filter()
    );
    assert_eq!(
        implicit.effective_min_filter(),
        explicit.effective_min_filter()
    );
}

#[test]
fn effective_filters_fall_back_to_trilinear() {
    let s = Sampler::default_sampler();
    assert_eq!(s.effective_mag_filter(), MagFilter::Linear);
    assert_eq!(s.effective_min_filter(), MinFilter::LinearMipLinear);
    assert!(s.uses_mipmaps());
}

#[test]
fn explicit_filters_pass_through_effective_accessors() {
    let s = Sampler::default_sampler()
        .with_mag_filter(MagFilter::Nearest)
        .with_min_filter(MinFilter::NearestMipNearest);
    assert_eq!(s.effective_mag_filter(), MagFilter::Nearest);
    assert_eq!(s.effective_min_filter(), MinFilter::NearestMipNearest);
}

#[test]
fn builders_set_each_axis() {
    let s = Sampler::default_sampler().with_wrap(WrapMode::ClampToEdge, WrapMode::MirroredRepeat);
    assert_eq!(s.wrap_s, WrapMode::ClampToEdge);
    assert_eq!(s.wrap_t, WrapMode::MirroredRepeat);
    // Filters untouched by the wrap builder.
    assert_eq!(s.mag_filter, None);
}

// ---------------------------------------------------------------
// Mip-behaviour decomposition

#[test]
fn min_filter_mip_axis_decomposition() {
    use MinFilter::*;
    let table: [(MinFilter, MipmapMode, MagFilter); 6] = [
        (Nearest, MipmapMode::Disabled, MagFilter::Nearest),
        (Linear, MipmapMode::Disabled, MagFilter::Linear),
        (NearestMipNearest, MipmapMode::Nearest, MagFilter::Nearest),
        (LinearMipNearest, MipmapMode::Nearest, MagFilter::Linear),
        (NearestMipLinear, MipmapMode::Linear, MagFilter::Nearest),
        (LinearMipLinear, MipmapMode::Linear, MagFilter::Linear),
    ];
    for (filter, mip, base) in table {
        assert_eq!(filter.mipmap_mode(), mip, "{filter:?}");
        assert_eq!(filter.base_filter(), base, "{filter:?}");
        assert_eq!(
            filter.uses_mipmaps(),
            mip != MipmapMode::Disabled,
            "{filter:?}"
        );
    }
}

#[test]
fn sampler_uses_mipmaps_follows_the_effective_filter() {
    let base_only = Sampler::default_sampler().with_min_filter(MinFilter::Linear);
    assert!(!base_only.uses_mipmaps());
    let mipped = Sampler::default_sampler().with_min_filter(MinFilter::NearestMipLinear);
    assert!(mipped.uses_mipmaps());
}

// ---------------------------------------------------------------
// Wrap-mode reference semantics

fn assert_close(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-6, "{a} != {b}");
}

#[test]
fn clamp_to_edge_clamps() {
    let w = WrapMode::ClampToEdge;
    assert_close(w.wrap(-0.5), 0.0);
    assert_close(w.wrap(0.0), 0.0);
    assert_close(w.wrap(0.25), 0.25);
    assert_close(w.wrap(1.0), 1.0);
    assert_close(w.wrap(1.5), 1.0);
}

#[test]
fn repeat_takes_the_fractional_part() {
    let w = WrapMode::Repeat;
    assert_close(w.wrap(0.25), 0.25);
    assert_close(w.wrap(1.25), 0.25);
    assert_close(w.wrap(-0.25), 0.75);
    assert_close(w.wrap(3.0), 0.0);
    assert_close(w.wrap(1.0), 0.0);
}

#[test]
fn mirrored_repeat_is_a_period_two_triangle_wave() {
    let w = WrapMode::MirroredRepeat;
    assert_close(w.wrap(0.25), 0.25);
    // Second tile runs backwards.
    assert_close(w.wrap(1.25), 0.75);
    assert_close(w.wrap(1.75), 0.25);
    // Period 2: 2.25 lands with 0.25 again.
    assert_close(w.wrap(2.25), 0.25);
    // Negative side mirrors too.
    assert_close(w.wrap(-0.25), 0.25);
    assert_close(w.wrap(-1.25), 0.75);
    // Boundary values.
    assert_close(w.wrap(1.0), 1.0);
    assert_close(w.wrap(2.0), 0.0);
}

#[test]
fn wrap_mode_continuity_at_tile_seams() {
    // MirroredRepeat is continuous across every integer seam; Repeat
    // jumps. Spot-check just inside/outside the 1.0 seam.
    let m = WrapMode::MirroredRepeat;
    assert!((m.wrap(1.0 - 1e-4) - m.wrap(1.0 + 1e-4)).abs() < 1e-3);
    let r = WrapMode::Repeat;
    assert!((r.wrap(1.0 - 1e-4) - r.wrap(1.0 + 1e-4)).abs() > 0.9);
}

#[test]
fn non_finite_coordinates_wrap_to_zero() {
    for w in [
        WrapMode::ClampToEdge,
        WrapMode::Repeat,
        WrapMode::MirroredRepeat,
    ] {
        assert_eq!(w.wrap(f32::NAN), 0.0, "{w:?}");
        assert_eq!(w.wrap(f32::INFINITY), 0.0, "{w:?}");
        assert_eq!(w.wrap(f32::NEG_INFINITY), 0.0, "{w:?}");
    }
}

#[test]
fn wrap_uv_applies_each_axis_independently() {
    let s = Sampler::default_sampler().with_wrap(WrapMode::ClampToEdge, WrapMode::Repeat);
    let out = s.wrap_uv([1.5, 1.25]);
    assert_close(out[0], 1.0); // clamped
    assert_close(out[1], 0.25); // repeated
}

// ---------------------------------------------------------------
// Seeded properties (house-style LCG, no dependencies)

fn lcg(state: &mut u64) -> f32 {
    // Numerical Recipes LCG constants; take the high bits.
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*state >> 40) as u32; // 24 bits
    bits as f32 / (1 << 24) as f32
}

#[test]
fn wrap_output_always_lands_in_unit_range_and_is_idempotent() {
    let modes = [
        WrapMode::ClampToEdge,
        WrapMode::Repeat,
        WrapMode::MirroredRepeat,
    ];
    let mut state = 0x5EED_u64;
    for _ in 0..1000 {
        let x = (lcg(&mut state) - 0.5) * 16.0; // [-8, 8)
        for w in modes {
            let y = w.wrap(x);
            assert!((0.0..=1.0).contains(&y), "{w:?}.wrap({x}) = {y}");
            // A wrapped coordinate is already in the sampled range,
            // so wrapping again must be a no-op.
            let z = w.wrap(y);
            assert!(
                (y - z).abs() < 1e-6,
                "{w:?} not idempotent at {x}: {y} -> {z}"
            );
        }
    }
}

#[test]
fn repeat_wrap_is_period_one_and_mirrored_is_period_two() {
    let mut state = 0xACE_u64;
    for _ in 0..500 {
        let x = (lcg(&mut state) - 0.5) * 8.0;
        let r = WrapMode::Repeat;
        assert!((r.wrap(x) - r.wrap(x + 1.0)).abs() < 1e-5, "Repeat at {x}");
        let m = WrapMode::MirroredRepeat;
        assert!(
            (m.wrap(x) - m.wrap(x + 2.0)).abs() < 1e-5,
            "Mirrored at {x}"
        );
        // Mirroring: reflecting about an even integer is invariant.
        assert!((m.wrap(-x) - m.wrap(x)).abs() < 1e-5, "Mirror sym at {x}");
    }
}
