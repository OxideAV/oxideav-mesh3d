//! Barycentric vertex-attribute interpolation at a ray hit, in the
//! primitive's local space.
//!
//! Given a triangle's three vertex indices `[i0, i1, i2]` (from
//! [`crate::SceneHit::vertex_indices`], a [`crate::Bvh`]'s
//! [`crate::Bvh::triangle_vertices`], or
//! [`crate::Primitive::triangle_indices`]) and barycentric weights
//! `[w, u, v]` (`w = 1 - u - v`), an attribute `A` interpolates
//! linearly as `w·A[i0] + u·A[i1] + v·A[i2]` — the same perspective-
//! correct-in-object-space interpolation rasterisers perform. World-
//! space versions (normal matrix, mirroring) are provided by
//! [`crate::InstanceBvh::shading_normal`] /
//! [`crate::InstanceBvh::tangent`] etc.
//!
//! Every helper returns `None` when the attribute is absent or a
//! vertex index is out of range, never panicking.

use crate::Primitive;

#[inline]
fn lerp3<const N: usize>(data: &[[f32; N]], tri: [u32; 3], bary: [f32; 3]) -> Option<[f32; N]> {
    let a = data.get(tri[0] as usize)?;
    let b = data.get(tri[1] as usize)?;
    let c = data.get(tri[2] as usize)?;
    let mut out = [0.0f32; N];
    for k in 0..N {
        out[k] = bary[0] * a[k] + bary[1] * b[k] + bary[2] * c[k];
    }
    Some(out)
}

impl Primitive {
    /// Interpolated position (local space).
    pub fn interpolate_position(&self, tri: [u32; 3], bary: [f32; 3]) -> Option<[f32; 3]> {
        lerp3(&self.positions, tri, bary)
    }

    /// Unit local-space geometric normal `normalize((p1 - p0) × (p2 - p0))`
    /// of the triangle (CCW-front, glTF winding). `None` for a
    /// degenerate triangle or out-of-range index.
    pub fn triangle_normal(&self, tri: [u32; 3]) -> Option<[f32; 3]> {
        let p0 = *self.positions.get(tri[0] as usize)?;
        let p1 = *self.positions.get(tri[1] as usize)?;
        let p2 = *self.positions.get(tri[2] as usize)?;
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        normalize(n)
    }

    /// Interpolated vertex normal, renormalised (local space). `None`
    /// without `normals` or when the blend is zero-length.
    pub fn interpolate_normal(&self, tri: [u32; 3], bary: [f32; 3]) -> Option<[f32; 3]> {
        normalize(lerp3(self.normals.as_deref()?, tri, bary)?)
    }

    /// Interpolated texture coordinate of UV set `set`.
    pub fn interpolate_uv(&self, tri: [u32; 3], bary: [f32; 3], set: usize) -> Option<[f32; 2]> {
        lerp3(self.uvs.get(set)?, tri, bary)
    }

    /// Interpolated tangent: `xyz` blended and renormalised, `w` (the
    /// glTF bitangent sign) taken from the vertex with the largest
    /// weight (it is not meaningful to blend a sign).
    pub fn interpolate_tangent(&self, tri: [u32; 3], bary: [f32; 3]) -> Option<[f32; 4]> {
        let t = self.tangents.as_deref()?;
        let b = lerp3(t, tri, bary)?;
        let xyz = normalize([b[0], b[1], b[2]])?;
        let dominant = if bary[0] >= bary[1] && bary[0] >= bary[2] {
            tri[0]
        } else if bary[1] >= bary[2] {
            tri[1]
        } else {
            tri[2]
        };
        let w = t.get(dominant as usize)?[3];
        Some([xyz[0], xyz[1], xyz[2], if w < 0.0 { -1.0 } else { 1.0 }])
    }

    /// Interpolated vertex colour of set `set` (RGBA, linear).
    pub fn interpolate_color(&self, tri: [u32; 3], bary: [f32; 3], set: usize) -> Option<[f32; 4]> {
        lerp3(self.colors.get(set)?, tri, bary)
    }
}

#[inline]
fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if l2 > 0.0 && l2.is_finite() {
        let inv = 1.0 / l2.sqrt();
        Some([v[0] * inv, v[1] * inv, v[2] * inv])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{Primitive, Topology};

    fn tri() -> Primitive {
        let mut p = Primitive::new(Topology::Triangles);
        p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        p.normals = Some(vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]);
        p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]];
        p.colors = vec![vec![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]];
        p.tangents = Some(vec![[1.0, 0.0, 0.0, -1.0]; 3]);
        p
    }

    #[test]
    fn interpolates_each_attribute() {
        let p = tri();
        let b = [0.5, 0.25, 0.25];
        let t = [0, 1, 2];
        assert_eq!(p.interpolate_position(t, b), Some([0.25, 0.25, 0.0]));
        assert_eq!(p.interpolate_uv(t, b, 0), Some([0.25, 0.25]));
        assert_eq!(p.interpolate_uv(t, b, 1), None);
        assert_eq!(p.interpolate_color(t, b, 0), Some([0.5, 0.25, 0.25, 1.0]));
        let n = p.interpolate_normal(t, b).unwrap();
        assert!(((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) - 1.0).abs() < 1e-6);
        assert!(n[2] > n[0]);
        assert_eq!(p.interpolate_tangent(t, b), Some([1.0, 0.0, 0.0, -1.0]));
        assert_eq!(p.triangle_normal(t), Some([0.0, 0.0, 1.0]));
    }

    #[test]
    fn out_of_range_is_none() {
        let p = tri();
        assert_eq!(p.interpolate_uv([0, 1, 9], [1.0, 0.0, 0.0], 0), None);
        assert_eq!(p.triangle_normal([0, 0, 0]), None);
    }
}
