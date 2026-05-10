//! Camera projection definitions.
//!
//! Two projection shapes per glTF 2.0 §3.10: perspective (the common
//! case) and orthographic (CAD / engineering). The view transform
//! comes from whichever scene-graph node owns the camera reference;
//! these enums carry only the projection-matrix inputs.

/// Camera projection.
#[derive(Clone, Copy, Debug)]
pub enum Camera {
    /// Standard pinhole projection.
    ///
    /// `aspect_ratio = None` means "use the framebuffer's native
    /// aspect" — most renderers do this for windowed playback.
    /// `zfar = None` means "infinite far plane" (reverse-Z friendly,
    /// per glTF spec §3.10.2.1).
    Perspective {
        aspect_ratio: Option<f32>,
        /// Vertical field of view in radians.
        yfov: f32,
        znear: f32,
        zfar: Option<f32>,
    },
    /// Orthographic projection. `xmag` / `ymag` are the half-widths
    /// of the view rectangle in scene units.
    Orthographic {
        xmag: f32,
        ymag: f32,
        znear: f32,
        zfar: f32,
    },
}

impl Camera {
    /// Construct a perspective camera with the glTF default
    /// `aspect_ratio = None`, `zfar = None` (infinite far plane).
    pub fn perspective(yfov: f32, znear: f32) -> Self {
        Self::Perspective {
            aspect_ratio: None,
            yfov,
            znear,
            zfar: None,
        }
    }

    /// Construct an orthographic camera.
    pub fn orthographic(xmag: f32, ymag: f32, znear: f32, zfar: f32) -> Self {
        Self::Orthographic {
            xmag,
            ymag,
            znear,
            zfar,
        }
    }
}
