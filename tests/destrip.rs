//! Tests for [`Primitive::triangle_indices`] +
//! [`Primitive::to_triangle_list`] — strip/fan → triangle-list
//! de-stripping.
//!
//! De-stripping is the standard OpenGL/glTF primitive-assembly
//! expansion: a `TriangleStrip` of `n` vertices yields `n - 2`
//! triangles with alternating winding, a `TriangleFan` yields `n - 2`
//! triangles sharing the anchor vertex, and `Triangles` is already a
//! flat list. A renderer or a format encoder that only emits flat
//! triangle lists (STL is list-only; OBJ `f` faces are polygons)
//! needs this expansion to consume strip/fan primitives that glTF /
//! FBX carry natively.

use oxideav_mesh3d::{Indices, Primitive, Topology};

fn prim_with_positions(topology: Topology, n: usize) -> Primitive {
    let mut p = Primitive::new(topology);
    p.positions = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect();
    p
}

// ---- Triangles (already a list) ------------------------------------------

#[test]
fn triangles_non_indexed_passthrough() {
    let p = prim_with_positions(Topology::Triangles, 6);
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[0, 1, 2], [3, 4, 5]]);
}

#[test]
fn triangles_drops_incomplete_trailing_vertices() {
    // 7 vertices = 2 complete triangles + 1 leftover.
    let p = prim_with_positions(Topology::Triangles, 7);
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[0, 1, 2], [3, 4, 5]]);
    assert_eq!(tris.len(), p.triangle_count());
}

#[test]
fn triangles_indexed_dereferences_index_buffer() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = (0..4).map(|i| [i as f32, 0.0, 0.0]).collect();
    // Quad as two triangles via index buffer.
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[0, 1, 2], [0, 2, 3]]);
}

#[test]
fn triangles_u32_indices_passthrough() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = (0..5).map(|i| [i as f32, 0.0, 0.0]).collect();
    p.indices = Some(Indices::U32(vec![4, 3, 2, 1, 0, 4]));
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[4, 3, 2], [1, 0, 4]]);
}

// ---- TriangleStrip (alternating winding) ---------------------------------

#[test]
fn strip_four_vertices_two_triangles_alternate_winding() {
    // Strip 0,1,2,3 → tri0 = (0,1,2), tri1 swaps last two → (1,3,2).
    let p = prim_with_positions(Topology::TriangleStrip, 4);
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[0, 1, 2], [1, 3, 2]]);
    assert_eq!(tris.len(), p.triangle_count());
}

#[test]
fn strip_five_vertices_three_triangles() {
    // 0,1,2,3,4:
    //   i=0 even -> (0,1,2)
    //   i=1 odd  -> (1,3,2)
    //   i=2 even -> (2,3,4)
    let p = prim_with_positions(Topology::TriangleStrip, 5);
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[0, 1, 2], [1, 3, 2], [2, 3, 4]]);
}

#[test]
fn strip_six_vertices_four_triangles() {
    let p = prim_with_positions(Topology::TriangleStrip, 6);
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]]);
    assert_eq!(tris.len(), 4);
}

#[test]
fn strip_indexed_dereferences_then_alternates() {
    let mut p = Primitive::new(Topology::TriangleStrip);
    // Index buffer points the strip at vertices 10,11,12,13 (remapped
    // pool); the output must carry the dereferenced vertex indices.
    p.positions = (0..14).map(|i| [i as f32, 0.0, 0.0]).collect();
    p.indices = Some(Indices::U16(vec![10, 11, 12, 13]));
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[10, 11, 12], [11, 13, 12]]);
}

#[test]
fn strip_too_short_yields_empty() {
    assert!(prim_with_positions(Topology::TriangleStrip, 0)
        .triangle_indices()
        .is_empty());
    assert!(prim_with_positions(Topology::TriangleStrip, 1)
        .triangle_indices()
        .is_empty());
    assert!(prim_with_positions(Topology::TriangleStrip, 2)
        .triangle_indices()
        .is_empty());
    assert_eq!(
        prim_with_positions(Topology::TriangleStrip, 3).triangle_indices(),
        vec![[0, 1, 2]]
    );
}

// ---- TriangleFan (shared anchor) -----------------------------------------

#[test]
fn fan_four_vertices_two_triangles_shared_anchor() {
    // 0,1,2,3 → (0,1,2),(0,2,3).
    let p = prim_with_positions(Topology::TriangleFan, 4);
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[0, 1, 2], [0, 2, 3]]);
    assert_eq!(tris.len(), p.triangle_count());
}

#[test]
fn fan_five_vertices_three_triangles() {
    let p = prim_with_positions(Topology::TriangleFan, 5);
    let tris = p.triangle_indices();
    assert_eq!(tris, vec![[0, 1, 2], [0, 2, 3], [0, 3, 4]]);
}

#[test]
fn fan_indexed_uses_first_index_as_anchor() {
    let mut p = Primitive::new(Topology::TriangleFan);
    p.positions = (0..10).map(|i| [i as f32, 0.0, 0.0]).collect();
    p.indices = Some(Indices::U32(vec![7, 2, 5, 9]));
    let tris = p.triangle_indices();
    // Anchor is the *dereferenced* first index (7), not 0.
    assert_eq!(tris, vec![[7, 2, 5], [7, 5, 9]]);
}

#[test]
fn fan_too_short_yields_empty() {
    assert!(prim_with_positions(Topology::TriangleFan, 2)
        .triangle_indices()
        .is_empty());
    assert_eq!(
        prim_with_positions(Topology::TriangleFan, 3).triangle_indices(),
        vec![[0, 1, 2]]
    );
}

// ---- Non-triangle topologies ---------------------------------------------

#[test]
fn lines_yield_empty_triangle_list() {
    let p = prim_with_positions(Topology::Lines, 6);
    assert!(p.triangle_indices().is_empty());
}

#[test]
fn points_yield_empty_triangle_list() {
    let p = prim_with_positions(Topology::Points, 5);
    assert!(p.triangle_indices().is_empty());
}

#[test]
fn line_strip_and_loop_yield_empty() {
    assert!(prim_with_positions(Topology::LineStrip, 4)
        .triangle_indices()
        .is_empty());
    assert!(prim_with_positions(Topology::LineLoop, 4)
        .triangle_indices()
        .is_empty());
}

// ---- to_triangle_list (Primitive materialisation) ------------------------

#[test]
fn to_triangle_list_converts_strip_to_indexed_triangles() {
    let p = prim_with_positions(Topology::TriangleStrip, 4);
    let out = p.to_triangle_list();
    assert_eq!(out.topology, Topology::Triangles);
    // Flat index buffer is the row-major flattening of triangle_indices.
    assert_eq!(out.indices, Some(Indices::U32(vec![0, 1, 2, 1, 3, 2])));
    // Vertex pool is carried verbatim.
    assert_eq!(out.positions, p.positions);
}

#[test]
fn to_triangle_list_self_count_matches_source() {
    let p = prim_with_positions(Topology::TriangleStrip, 8);
    let out = p.to_triangle_list();
    assert_eq!(out.triangle_count(), p.triangle_count());
    assert_eq!(out.triangle_count(), 6);
}

#[test]
fn to_triangle_list_carries_attributes_and_material() {
    use oxideav_mesh3d::MaterialId;
    let mut p = prim_with_positions(Topology::TriangleFan, 4);
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
    p.uvs = vec![vec![[0.0, 0.0]; 4]];
    p.colors = vec![vec![[1.0, 0.0, 0.0, 1.0]; 4]];
    p.material = Some(MaterialId(3));
    let out = p.to_triangle_list();
    assert_eq!(out.normals, p.normals);
    assert_eq!(out.uvs, p.uvs);
    assert_eq!(out.colors, p.colors);
    assert_eq!(out.material, Some(MaterialId(3)));
    assert_eq!(out.indices, Some(Indices::U32(vec![0, 1, 2, 0, 2, 3])));
}

#[test]
fn to_triangle_list_idempotent_on_triangles() {
    // Already-Triangles input: de-stripping is a normalising round-trip
    // (output gains an explicit index buffer; re-running is stable).
    let p = prim_with_positions(Topology::Triangles, 6);
    let once = p.to_triangle_list();
    let twice = once.to_triangle_list();
    assert_eq!(once.topology, twice.topology);
    assert_eq!(once.indices, twice.indices);
    assert_eq!(once.triangle_indices(), twice.triangle_indices());
}

#[test]
fn to_triangle_list_on_lines_yields_empty_index_buffer() {
    let p = prim_with_positions(Topology::Lines, 6);
    let out = p.to_triangle_list();
    assert_eq!(out.topology, Topology::Triangles);
    assert_eq!(out.indices, Some(Indices::U32(vec![])));
    assert_eq!(out.triangle_count(), 0);
    // Positions are still carried (nothing drawn, but the pool survives).
    assert_eq!(out.positions, p.positions);
}

#[test]
fn to_triangle_list_preserves_morph_targets() {
    use oxideav_mesh3d::MorphTarget;
    let mut p = prim_with_positions(Topology::TriangleStrip, 4);
    p.targets.push(MorphTarget {
        position: Some(vec![[0.1, 0.0, 0.0]; 4]),
        normal: None,
        tangent: None,
    });
    let out = p.to_triangle_list();
    // targets are per-vertex-parallel; attribute buffers unchanged so
    // they remain valid.
    assert_eq!(out.targets.len(), 1);
    assert_eq!(out.targets[0].position, p.targets[0].position);
}

// ---- Cross-checks against triangle_count ---------------------------------

#[test]
fn triangle_indices_len_matches_triangle_count_strip() {
    for n in 3..=20 {
        let p = prim_with_positions(Topology::TriangleStrip, n);
        assert_eq!(
            p.triangle_indices().len(),
            p.triangle_count(),
            "strip n={n}"
        );
    }
}

#[test]
fn triangle_indices_len_matches_triangle_count_fan() {
    for n in 3..=20 {
        let p = prim_with_positions(Topology::TriangleFan, n);
        assert_eq!(p.triangle_indices().len(), p.triangle_count(), "fan n={n}");
    }
}

#[test]
fn all_indices_in_range_of_vertex_pool() {
    // De-stripped indices must never reference outside positions.len().
    let n = 12;
    for topo in [Topology::TriangleStrip, Topology::TriangleFan] {
        let p = prim_with_positions(topo, n);
        for t in p.triangle_indices() {
            for v in t {
                assert!((v as usize) < n, "{topo:?} index {v} out of range");
            }
        }
    }
}
