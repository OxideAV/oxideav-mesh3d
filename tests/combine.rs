//! Primitive::merge + Mesh::merge_primitives_by_material coverage:
//! pool concatenation, index re-basing, attribute union-with-fill, and
//! material-grouped consolidation.

use oxideav_mesh3d::{Indices, MaterialId, Mesh, Primitive, Topology};

/// A single CCW triangle at the given x-offset, with the given optional
/// attributes toggled.
fn tri(x: f32, with_normals: bool, with_uv: bool) -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2]));
    if with_normals {
        p.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
    }
    if with_uv {
        p.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]];
    }
    p
}

#[test]
fn merge_concatenates_pools_and_rebases_indices() {
    let a = tri(0.0, false, false);
    let b = tri(5.0, false, false);
    let m = a.merge(&b);

    assert_eq!(m.topology, Topology::Triangles);
    assert_eq!(m.positions.len(), 6);
    // b's vertices follow a's.
    assert_eq!(m.positions[3], [5.0, 0.0, 0.0]);

    let tris = m.triangle_indices();
    assert_eq!(tris.len(), 2);
    assert_eq!(tris[0], [0, 1, 2]);
    // b's triangle is shifted by a.positions.len() == 3.
    assert_eq!(tris[1], [3, 4, 5]);
}

#[test]
fn merge_unions_attribute_presence_with_fill() {
    // a has normals, no uvs; b has uvs, no normals. The merge should
    // carry both, filling the missing side with the neutral default.
    let a = tri(0.0, true, false);
    let b = tri(5.0, false, true);
    let m = a.merge(&b);

    let normals = m.normals.expect("normals present");
    assert_eq!(normals.len(), 6);
    // a's real normals, then b's fill (+Z too here, but length proves it
    // was filled to the full pool).
    assert_eq!(normals[0], [0.0, 0.0, 1.0]);
    assert_eq!(normals[3], [0.0, 0.0, 1.0]); // fill row

    assert_eq!(m.uvs.len(), 1);
    assert_eq!(m.uvs[0].len(), 6);
    // a's fill uvs (origin), then b's real uvs.
    assert_eq!(m.uvs[0][0], [0.0, 0.0]); // a fill
    assert_eq!(m.uvs[0][4], [1.0, 0.0]); // b real
}

#[test]
fn merge_color_default_is_opaque_white() {
    let mut a = tri(0.0, false, false);
    a.colors = vec![vec![[0.2, 0.4, 0.6, 1.0]; 3]];
    let b = tri(5.0, false, false);
    let m = a.merge(&b);
    assert_eq!(m.colors.len(), 1);
    assert_eq!(m.colors[0].len(), 6);
    assert_eq!(m.colors[0][0], [0.2, 0.4, 0.6, 1.0]); // a real
    assert_eq!(m.colors[0][3], [1.0, 1.0, 1.0, 1.0]); // b fill = white
}

#[test]
fn merge_drops_out_of_range_triangles() {
    let a = tri(0.0, false, false);
    let mut b = tri(5.0, false, false);
    // Corrupt b's index to reference a nonexistent vertex.
    b.indices = Some(Indices::U32(vec![0, 1, 99]));
    let m = a.merge(&b);
    // Only a's valid triangle survives; b's dangling one is dropped.
    assert_eq!(m.triangle_indices().len(), 1);
    // But b's vertices still join the pool.
    assert_eq!(m.positions.len(), 6);
}

#[test]
fn merge_material_comes_from_self() {
    let mut a = tri(0.0, false, false);
    a.material = Some(MaterialId(7));
    let mut b = tri(5.0, false, false);
    b.material = Some(MaterialId(9));
    let m = a.merge(&b);
    assert_eq!(m.material, Some(MaterialId(7)));
}

#[test]
fn merge_by_material_groups_primitives() {
    let mut mesh = Mesh::new(Some("multi".to_owned()));
    let mat0 = Some(MaterialId(0));
    let mat1 = Some(MaterialId(1));

    let mut p0 = tri(0.0, false, false);
    p0.material = mat0;
    let mut p1 = tri(2.0, false, false);
    p1.material = mat1;
    let mut p2 = tri(4.0, false, false);
    p2.material = mat0; // same material as p0
    mesh.primitives = vec![p0, p1, p2];

    let fused = mesh.merge_primitives_by_material();
    // Two distinct materials → two output primitives.
    assert_eq!(fused.primitives.len(), 2);
    // First group (mat0) fused p0 + p2 → 6 vertices, 2 triangles.
    assert_eq!(fused.primitives[0].material, mat0);
    assert_eq!(fused.primitives[0].positions.len(), 6);
    assert_eq!(fused.primitives[0].triangle_indices().len(), 2);
    // Second group (mat1) is the lone p1.
    assert_eq!(fused.primitives[1].material, mat1);
    assert_eq!(fused.primitives[1].triangle_indices().len(), 1);
    // Mesh metadata preserved.
    assert_eq!(fused.name.as_deref(), Some("multi"));
}

#[test]
fn merge_by_material_keeps_none_group_separate() {
    let mut mesh = Mesh::new(None);
    let mut p0 = tri(0.0, false, false);
    p0.material = Some(MaterialId(0));
    let p1 = tri(2.0, false, false); // material None
    mesh.primitives = vec![p0, p1];

    let fused = mesh.merge_primitives_by_material();
    assert_eq!(fused.primitives.len(), 2);
    assert_eq!(fused.primitives[0].material, Some(MaterialId(0)));
    assert_eq!(fused.primitives[1].material, None);
}

#[test]
fn merge_by_material_on_empty_mesh() {
    let mesh = Mesh::new(Some("empty".to_owned()));
    let fused = mesh.merge_primitives_by_material();
    assert_eq!(fused.primitives.len(), 0);
    assert_eq!(fused.name.as_deref(), Some("empty"));
}

#[test]
fn merge_preserves_total_triangle_area() {
    // Two unit triangles, disjoint in x; the fused primitive's surface
    // area is the sum (merge is geometry-preserving).
    let a = tri(0.0, false, false);
    let b = tri(5.0, false, false);
    let area_sum = a.surface_area() + b.surface_area();
    let merged_area = a.merge(&b).surface_area();
    assert!((merged_area - area_sum).abs() < 1e-6);
}
