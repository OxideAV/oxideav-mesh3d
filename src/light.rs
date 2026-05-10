//! Punctual light sources (KHR_lights_punctual analogue).
//!
//! Three light types — directional (sun-like, no position),
//! point (omni-directional from a position), and spot (cone). Each
//! carries an RGB colour in linear space and an intensity in glTF's
//! photometric units (lumens for point/spot, lux for directional).
//! The owning node's transform supplies position + orientation.

/// Light source.
#[derive(Clone, Copy, Debug)]
pub enum Light {
    /// Sun-like directional light. Emits parallel rays along
    /// `-node.local_z`.
    Directional {
        color: [f32; 3],
        /// Illuminance in lux (lm·m^-2).
        intensity: f32,
    },
    /// Omni-directional point light at the node's origin.
    Point {
        color: [f32; 3],
        /// Luminous intensity in candela (lm·sr^-1).
        intensity: f32,
        /// Distance at which the light's contribution falls to zero.
        /// `None` means physical inverse-square falloff with no cutoff.
        range: Option<f32>,
    },
    /// Cone-bounded light along `-node.local_z`.
    Spot {
        color: [f32; 3],
        intensity: f32,
        range: Option<f32>,
        /// Inner cone angle in radians — full intensity inside.
        inner_cone_angle: f32,
        /// Outer cone angle in radians — zero intensity outside.
        /// Must be > `inner_cone_angle`.
        outer_cone_angle: f32,
    },
}
