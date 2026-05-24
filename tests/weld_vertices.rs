//! Tests for [`Primitive::weld_vertices`] — coincident-vertex merging
//! (the inverse of attribute explosion / vertex-soup de-duplication).
//!
//! Welding collapses bit-identical rendering vertices into a shared pool
//! and rewrites the index buffer to reference the deduplicated pool. This
//! is the standard compaction a decoder for a non-shared format runs:
//! binary STL stores three fresh vertices per facet with no sharing, so a
//! decoded cube has 36 vertices that weld down to 8 (or 24 when each face
//! carries its own normal). The merge rule is "bit-identical across every
//! attribute stream simultaneously", because one index in an indexed draw
//! call selects one tuple across all streams at once.
//!
//! Truth ladder: glTF 2.0 §3.7.2 mesh.primitive accessor model (one index
//! buffer indexing parallel attribute accessors) + standard
//! indexed-mesh / post-transform-vertex-cache geometry. No format
//! reference; pure connectivity rewrite.

use oxideav_mesh3d::{Indices, MorphTarget, Primitive, Topology};

fn idx_vec(p: &Primitive) -> Vec<u32> {
    match p.indices.as_ref().expect("welded primitive always indexed") {
        Indices::U16(v) => v.iter().map(|&i| i as u32).collect(),
        Indices::U32(v) => v.clone(),
    }
}

// ---- Positions-only dedup -------------------------------------------------

#[test]
fn quad_two_triangles_welds_four_corners() {
    // Two triangles of a quad, written as a 6-vertex soup (corners 1 and
    // 2 are shared along the diagonal).
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [1.0, 1.0, 0.0], // 2
        [0.0, 0.0, 0.0], // 3 == 0
        [1.0, 1.0, 0.0], // 4 == 2
        [0.0, 1.0, 0.0], // 5
    ];
    let w = p.weld_vertices();
    // Four distinct corners survive.
    assert_eq!(w.positions.len(), 4);
    assert_eq!(
        w.positions,
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    );
    // Index buffer reproduces the original two triangles over the pool.
    assert_eq!(idx_vec(&w), vec![0, 1, 2, 0, 2, 3]);
}

#[test]
fn all_distinct_positions_are_a_passthrough_pool() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let w = p.weld_vertices();
    assert_eq!(w.positions, p.positions);
    assert_eq!(idx_vec(&w), vec![0, 1, 2]);
}

#[test]
fn empty_primitive_welds_to_empty_indexed() {
    let p = Primitive::new(Topology::Triangles);
    let w = p.weld_vertices();
    assert!(w.positions.is_empty());
    assert_eq!(idx_vec(&w), Vec::<u32>::new());
    assert!(matches!(w.indices, Some(Indices::U16(_))));
}

#[test]
fn first_seen_order_is_preserved() {
    // The pool is built in first-seen order; later duplicates do not
    // reorder earlier-seen entries.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [9.0, 9.0, 9.0], // 0 -> pool 0
        [1.0, 0.0, 0.0], // 1 -> pool 1
        [9.0, 9.0, 9.0], // 2 == 0 -> pool 0
        [2.0, 0.0, 0.0], // 3 -> pool 2
        [1.0, 0.0, 0.0], // 4 == 1 -> pool 1
        [9.0, 9.0, 9.0], // 5 == 0 -> pool 0
    ];
    let w = p.weld_vertices();
    assert_eq!(
        w.positions,
        vec![[9.0, 9.0, 9.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    );
    assert_eq!(idx_vec(&w), vec![0, 1, 0, 2, 1, 0]);
}

// ---- Attribute-aware splitting --------------------------------------------

#[test]
fn same_position_different_uv_stays_split() {
    // A UV seam: two corners coincide in 3D but carry different UVs, so
    // they are distinct rendering vertices and must NOT merge.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    p.uvs = vec![vec![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]];
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 3);
    assert_eq!(idx_vec(&w), vec![0, 1, 2]);
}

#[test]
fn same_position_same_uv_merges() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    p.uvs = vec![vec![[0.25, 0.75], [0.25, 0.75], [0.5, 0.5]]];
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 2);
    assert_eq!(w.uvs[0], vec![[0.25, 0.75], [0.5, 0.5]]);
    assert_eq!(idx_vec(&w), vec![0, 0, 1]);
}

#[test]
fn same_position_different_normal_stays_split() {
    // Hard edge: shared corner with two face normals (e.g. a cube vertex
    // whose adjacent faces are flat-shaded) stays as two vertices.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 3);
}

#[test]
fn same_position_same_normal_merges() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    p.normals = Some(vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]);
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 2);
    assert_eq!(w.normals.as_ref().unwrap().len(), 2);
}

#[test]
fn different_tangent_stays_split() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    p.tangents = Some(vec![
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0], // differs only in handedness w
        [1.0, 0.0, 0.0, 1.0],
    ]);
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 3);
}

#[test]
fn different_color_stays_split() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    p.colors = vec![vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]];
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 3);
}

#[test]
fn different_joints_or_weights_stays_split() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    p.joints = Some(vec![[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3]]);
    p.weights = Some(vec![
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ]);
    let w = p.weld_vertices();
    // Vertex 0 and 1 share position but differ in joints -> distinct.
    assert_eq!(w.positions.len(), 3);
}

#[test]
fn multiple_uv_sets_all_participate() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    // UV set 0 identical on all three; UV set 1 differs on vertex 2.
    p.uvs = vec![
        vec![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        vec![[0.1, 0.1], [0.1, 0.1], [0.9, 0.9]],
    ];
    let w = p.weld_vertices();
    // Vertices 0,1 merge (both UV sets equal); 2 stays split (set 1).
    assert_eq!(w.positions.len(), 2);
    assert_eq!(idx_vec(&w), vec![0, 0, 1]);
    assert_eq!(w.uvs.len(), 2);
    assert_eq!(w.uvs[1], vec![[0.1, 0.1], [0.9, 0.9]]);
}

// ---- Morph deltas participate in identity ---------------------------------

#[test]
fn same_position_different_morph_delta_stays_split() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    p.targets = vec![MorphTarget {
        position: Some(vec![[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        normal: None,
        tangent: None,
    }];
    let w = p.weld_vertices();
    // Vertices 0 and 1 share base position but have different morph
    // deltas -> distinct.
    assert_eq!(w.positions.len(), 3);
    // The morph buffer is remapped in lockstep with the pool.
    assert_eq!(w.targets.len(), 1);
    assert_eq!(w.targets[0].position.as_ref().unwrap().len(), 3);
}

#[test]
fn morph_delta_buffers_are_gathered_with_the_pool() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [0.0, 0.0, 0.0], // 2 == 0 (and same morph delta below)
    ];
    p.targets = vec![MorphTarget {
        position: Some(vec![[7.0, 0.0, 0.0], [8.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        normal: None,
        tangent: None,
    }];
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 2);
    // Pool morph deltas follow the deduplicated vertices in first-seen
    // order: [7,...] for pool[0], [8,...] for pool[1].
    assert_eq!(
        w.targets[0].position.as_ref().unwrap(),
        &vec![[7.0, 0.0, 0.0], [8.0, 0.0, 0.0]]
    );
    assert_eq!(idx_vec(&w), vec![0, 1, 0]);
}

// ---- Float edge cases -----------------------------------------------------

#[test]
fn negative_zero_merges_with_positive_zero() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [-0.0, -0.0, -0.0], [1.0, 0.0, 0.0]];
    let w = p.weld_vertices();
    // +0.0 and -0.0 are numerically equal -> one pool entry.
    assert_eq!(w.positions.len(), 2);
    assert_eq!(idx_vec(&w), vec![0, 0, 1]);
}

#[test]
fn nan_coordinates_merge_deterministically() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[f32::NAN, 0.0, 0.0], [f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let w = p.weld_vertices();
    // Both NaN vertices canonicalise to the same key -> merge.
    assert_eq!(w.positions.len(), 2);
    assert_eq!(idx_vec(&w), vec![0, 0, 1]);
}

// ---- Indexed input --------------------------------------------------------

#[test]
fn existing_index_buffer_is_remapped_through_dedup() {
    let mut p = Primitive::new(Topology::Triangles);
    // Pool already has duplicate positions; index buffer references them.
    p.positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [0.0, 0.0, 0.0], // 2 == 0
        [0.0, 1.0, 0.0], // 3
    ];
    p.indices = Some(Indices::U16(vec![0, 1, 3, 2, 1, 3]));
    let w = p.weld_vertices();
    // 0 and 2 merge -> three pool entries.
    assert_eq!(w.positions.len(), 3);
    // Index 2 in the original (== pos 0) remaps to pool 0.
    assert_eq!(idx_vec(&w), vec![0, 1, 2, 0, 1, 2]);
}

#[test]
fn out_of_range_index_entry_is_dropped_not_panicked() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 99]));
    let w = p.weld_vertices();
    // The dangling 99 is dropped from the index stream.
    assert_eq!(idx_vec(&w), vec![0, 1, 2, 0, 1]);
}

// ---- Width promotion ------------------------------------------------------

#[test]
fn small_pool_uses_u16_indices() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let w = p.weld_vertices();
    assert!(matches!(w.indices, Some(Indices::U16(_))));
}

#[test]
fn pool_over_65536_promotes_to_u32() {
    // 65 537 distinct positions force a U32 index buffer.
    let mut p = Primitive::new(Topology::Points);
    p.positions = (0..65_537u32).map(|i| [i as f32, 0.0, 0.0]).collect();
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 65_537);
    assert!(matches!(w.indices, Some(Indices::U32(_))));
}

#[test]
fn pool_exactly_65536_stays_u16() {
    // 65 536 distinct entries fit in U16 (max index 65 535).
    let mut p = Primitive::new(Topology::Points);
    p.positions = (0..65_536u32).map(|i| [i as f32, 0.0, 0.0]).collect();
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 65_536);
    assert!(matches!(w.indices, Some(Indices::U16(_))));
}

// ---- Topology + metadata preservation -------------------------------------

#[test]
fn topology_is_preserved() {
    for topo in [
        Topology::Triangles,
        Topology::TriangleStrip,
        Topology::TriangleFan,
        Topology::Lines,
        Topology::LineStrip,
        Topology::LineLoop,
        Topology::Points,
    ] {
        let mut p = Primitive::new(topo);
        p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let w = p.weld_vertices();
        assert_eq!(w.topology, topo, "topology must survive welding");
    }
}

#[test]
fn material_and_extras_carry_over() {
    use oxideav_mesh3d::MaterialId;
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.material = Some(MaterialId(7));
    p.extras.insert("k".to_owned(), serde_json::json!("v"));
    let w = p.weld_vertices();
    assert_eq!(w.material, Some(MaterialId(7)));
    assert_eq!(w.extras.get("k").unwrap(), &serde_json::json!("v"));
}

#[test]
fn strip_topology_index_order_matches_source() {
    // A strip with a repeated vertex: welding compacts the pool but keeps
    // the strip's draw order so triangle_indices is unchanged afterwards.
    let mut p = Primitive::new(Topology::TriangleStrip);
    p.positions = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [0.0, 1.0, 0.0], // 2
        [0.0, 0.0, 0.0], // 3 == 0
    ];
    let before = p.triangle_indices();
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 3); // 0 and 3 merge
                                      // Triangle expansion over the welded pool resolves to the same
                                      // vertex positions as the original.
    let after = w.triangle_indices();
    assert_eq!(before.len(), after.len());
    let pos_before: Vec<_> = before
        .iter()
        .flat_map(|t| t.iter().map(|&i| p.positions[i as usize]))
        .collect();
    let pos_after: Vec<_> = after
        .iter()
        .flat_map(|t| t.iter().map(|&i| w.positions[i as usize]))
        .collect();
    assert_eq!(pos_before, pos_after);
}

// ---- Roundtrip with to_triangle_list (explode <-> weld) -------------------

#[test]
fn weld_then_explode_then_weld_is_stable() {
    // Soup -> weld (compact) -> to_triangle_list keeps indices ->
    // weld again gives the same compact pool.
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let w1 = p.weld_vertices();
    assert_eq!(w1.positions.len(), 4);
    let listed = w1.to_triangle_list(); // still indexed, Triangles
    let w2 = listed.weld_vertices();
    assert_eq!(w2.positions, w1.positions);
    assert_eq!(idx_vec(&w2), idx_vec(&w1));
}

#[test]
fn idempotent_on_already_welded_primitive() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
    let w1 = p.weld_vertices();
    let w2 = w1.weld_vertices();
    assert_eq!(w1.positions, w2.positions);
    assert_eq!(idx_vec(&w1), idx_vec(&w2));
}

#[test]
fn does_not_mutate_self() {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let snapshot = p.positions.clone();
    let _ = p.weld_vertices();
    assert_eq!(p.positions, snapshot);
    assert!(p.indices.is_none());
}

// ---- Realistic decoder scenario: STL cube ---------------------------------

#[test]
fn flat_shaded_cube_soup_welds_to_24_vertices() {
    // A binary-STL-style cube: 12 triangles, 36 vertices, each face's
    // four corners carry that face's flat normal. With normals as part of
    // the identity, the 36 corners weld to 24 (4 per face x 6 faces) —
    // each cube corner is split into 3 because its 3 adjacent faces have
    // different normals.
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +Z face
        (
            [0.0, 0.0, 1.0],
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
        ),
        // -Z face
        (
            [0.0, 0.0, -1.0],
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        ),
        // +X face
        (
            [1.0, 0.0, 0.0],
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
        ),
        // -X face
        (
            [-1.0, 0.0, 0.0],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
        ),
        // +Y face
        (
            [0.0, 1.0, 0.0],
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
        ),
        // -Y face
        (
            [0.0, -1.0, 0.0],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
        ),
    ];
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for (nrm, corners) in faces {
        // Two triangles per quad face: (0,1,2) and (0,2,3).
        for &c in &[
            corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
        ] {
            positions.push(c);
            normals.push(nrm);
        }
    }
    assert_eq!(positions.len(), 36);
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = positions;
    p.normals = Some(normals);
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 24, "4 corners x 6 faces");
    assert_eq!(w.normals.as_ref().unwrap().len(), 24);
    assert_eq!(idx_vec(&w).len(), 36, "12 triangles x 3 indices");
}

#[test]
fn position_only_cube_soup_welds_to_8_corners() {
    // Same cube without normals: identity is position-only, so the 36
    // corners collapse to the 8 distinct cube corners.
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    // 12 triangles by corner index.
    let tris: [[usize; 3]; 12] = [
        [0, 2, 1],
        [0, 3, 2], // -Z
        [4, 5, 6],
        [4, 6, 7], // +Z
        [0, 1, 5],
        [0, 5, 4], // -Y
        [2, 3, 7],
        [2, 7, 6], // +Y
        [1, 2, 6],
        [1, 6, 5], // +X
        [0, 4, 7],
        [0, 7, 3], // -X
    ];
    let mut positions = Vec::new();
    for t in tris {
        for i in t {
            positions.push(corners[i]);
        }
    }
    assert_eq!(positions.len(), 36);
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = positions;
    let w = p.weld_vertices();
    assert_eq!(w.positions.len(), 8, "8 distinct cube corners");
    assert_eq!(idx_vec(&w).len(), 36);
    // Every output index is a valid pool entry.
    assert!(idx_vec(&w).iter().all(|&i| (i as usize) < 8));
}
