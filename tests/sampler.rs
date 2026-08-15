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
