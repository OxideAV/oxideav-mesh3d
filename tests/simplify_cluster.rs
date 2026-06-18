//! Tests for `Primitive::simplify_cluster` — uniform-grid vertex
//! clustering decimation (Rossignac–Borrel, 1993).
//!
//! Every vertex snaps to a cell of a cube grid laid over the bounding
//! box; vertices sharing a cell collapse to one averaged representative,
//! and triangles that no longer span three distinct cells are dropped.
//! The result is an indexed `Triangles` proxy with all attributes
//! averaged per cell.

use oxideav_mesh3d::{Indices, MorphTarget, Primitive, Topology};

// --- helpers ---------------------------------------------------------

fn tri(p: Vec<[f32; 3]>) -> Primitive {
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = p;
    prim
}

fn tri_indexed(p: Vec<[f32; 3]>, idx: Vec<u32>) -> Primitive {
    let mut prim = tri(p);
    prim.indices = Some(Indices::U32(idx));
    prim
}

fn idx_vec(p: &Primitive) -> Vec<u32> {
    match &p.indices {
        Some(Indices::U32(v)) => v.clone(),
        Some(Indices::U16(v)) => v.iter().map(|&x| x as u32).collect(),
        None => (0..p.positions.len() as u32).collect(),
    }
}

fn approx(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
    (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
}

// A dense grid sheet on the z=0 plane: (res+1)² vertices, 2·res² tris.
fn grid_sheet(res: usize) -> Primitive {
    let n = res + 1;
    let mut positions = Vec::new();
    for j in 0..n {
        for i in 0..n {
            positions.push([i as f32 / res as f32, j as f32 / res as f32, 0.0]);
        }
    }
    let mut idx = Vec::new();
    for j in 0..res {
        for i in 0..res {
            let a = (j * n + i) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            // two CCW triangles per quad
            idx.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    tri_indexed(positions, idx)
}

// Closed unit tetrahedron (outward CCW).
fn tetrahedron() -> Primitive {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let idx = vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
    tri_indexed(positions, idx)
}

// --- core behaviour --------------------------------------------------

#[test]
fn output_is_always_indexed_triangles() {
    let s = grid_sheet(8).simplify_cluster(4);
    assert_eq!(s.topology, Topology::Triangles);
    assert!(s.indices.is_some());
}

#[test]
fn reduces_triangle_and_vertex_count() {
    let src = grid_sheet(16); // 289 verts, 512 tris
    let s = src.simplify_cluster(4);
    assert!(
        s.positions.len() < src.positions.len(),
        "vertices not reduced: {} -> {}",
        src.positions.len(),
        s.positions.len()
    );
    assert!(
        s.triangle_count() < src.triangle_count(),
        "triangles not reduced: {} -> {}",
        src.triangle_count(),
        s.triangle_count()
    );
    assert!(s.triangle_count() > 0);
}

#[test]
fn finer_grid_keeps_more_detail() {
    let src = grid_sheet(16);
    let coarse = src.simplify_cluster(3).triangle_count();
    let fine = src.simplify_cluster(8).triangle_count();
    assert!(
        fine >= coarse,
        "finer grid should keep at least as many tris: {coarse} vs {fine}"
    );
}

#[test]
fn grid_one_collapses_to_empty() {
    let s = grid_sheet(8).simplify_cluster(1);
    assert!(s.positions.is_empty());
    assert_eq!(s.triangle_count(), 0);
    assert_eq!(s.topology, Topology::Triangles);
}

#[test]
fn grid_zero_clamps_to_one() {
    // grid is clamped to >= 1, so this behaves exactly like grid==1.
    let s = grid_sheet(8).simplify_cluster(0);
    assert!(s.positions.is_empty());
    assert_eq!(s.triangle_count(), 0);
}

#[test]
fn no_orphan_vertices_in_output() {
    let s = grid_sheet(16).simplify_cluster(5);
    let idx = idx_vec(&s);
    let mut used = vec![false; s.positions.len()];
    for &i in &idx {
        used[i as usize] = true;
    }
    assert!(
        used.iter().all(|&u| u),
        "output pool has orphan (unreferenced) vertices"
    );
    // and every index is in range
    assert!(idx.iter().all(|&i| (i as usize) < s.positions.len()));
}

#[test]
fn output_indices_form_valid_triangles() {
    let s = grid_sheet(16).simplify_cluster(6);
    for t in s.triangle_indices() {
        assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
        for &c in &t {
            assert!((c as usize) < s.positions.len());
        }
    }
}

#[test]
fn output_faces_are_deduplicated() {
    let s = grid_sheet(20).simplify_cluster(4);
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for t in s.triangle_indices() {
        let mut key = t;
        key.sort_unstable();
        assert!(seen.insert(key), "duplicate face triple {key:?} emitted");
    }
}

#[test]
fn representative_positions_lie_in_bounds() {
    // The clustered positions are member centroids, so they must stay
    // within the source bounding box.
    let src = grid_sheet(16);
    let bb = src.bounding_box().unwrap();
    let s = src.simplify_cluster(5);
    for p in &s.positions {
        for d in 0..3 {
            assert!(
                p[d] >= bb.min[d] - 1e-5 && p[d] <= bb.max[d] + 1e-5,
                "clustered vertex {p:?} outside bbox"
            );
        }
    }
}

#[test]
fn fine_grid_reproduces_welded_input() {
    // A grid much finer than the vertex spacing isolates every original
    // vertex into its own cell → same vertex count as the welded input.
    let src = grid_sheet(4); // 25 distinct verts, 32 tris
    let s = src.simplify_cluster(1000);
    let welded = src.weld_vertices();
    assert_eq!(s.positions.len(), welded.positions.len());
    assert_eq!(s.triangle_count(), src.triangle_count());
}

// --- attribute preservation ------------------------------------------

#[test]
fn attribute_buffers_match_pool_length_and_average() {
    let mut src = grid_sheet(8);
    let n = src.positions.len();
    // constant normals + one uv set + one colour set + weights/joints
    src.normals = Some(vec![[0.0, 0.0, 1.0]; n]);
    src.tangents = Some(vec![[1.0, 0.0, 0.0, 1.0]; n]);
    src.uvs = vec![(0..n).map(|i| [i as f32, -(i as f32)]).collect()];
    src.colors = vec![vec![[0.25, 0.5, 0.75, 1.0]; n]];
    src.joints = Some(vec![[1, 2, 3, 4]; n]);
    src.weights = Some(vec![[0.4, 0.3, 0.2, 0.1]; n]);

    let s = src.simplify_cluster(4);
    let k = s.positions.len();
    assert!(k > 0);
    assert_eq!(s.normals.as_ref().unwrap().len(), k);
    assert_eq!(s.tangents.as_ref().unwrap().len(), k);
    assert_eq!(s.uvs[0].len(), k);
    assert_eq!(s.colors[0].len(), k);
    assert_eq!(s.joints.as_ref().unwrap().len(), k);
    assert_eq!(s.weights.as_ref().unwrap().len(), k);

    // constant normal averages back to the unit normal
    for nrm in s.normals.as_ref().unwrap() {
        assert!(approx(*nrm, [0.0, 0.0, 1.0], 1e-5));
    }
    // tangent xyz stays unit, handedness preserved (+1)
    for t in s.tangents.as_ref().unwrap() {
        assert!(approx([t[0], t[1], t[2]], [1.0, 0.0, 0.0], 1e-5));
        assert_eq!(t[3], 1.0);
    }
    // constant colour preserved
    for c in &s.colors[0] {
        assert!((c[0] - 0.25).abs() < 1e-5 && (c[3] - 1.0).abs() < 1e-5);
    }
    // joints taken from a member verbatim (categorical, never averaged)
    for j in s.joints.as_ref().unwrap() {
        assert_eq!(*j, [1, 2, 3, 4]);
    }
    // weights re-normalise to sum 1
    for w in s.weights.as_ref().unwrap() {
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "weights sum {sum} != 1");
    }
}

#[test]
fn tangent_handedness_majority_vote() {
    // Three coincident-cell vertices: two -1 vs one +1 → output -1.
    let mut prim = tri(vec![
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ]);
    // one big triangle so the three near-origin verts land in one cell
    prim.indices = Some(Indices::U32(vec![0, 3, 4, 1, 3, 4, 2, 3, 4]));
    prim.tangents = Some(vec![
        [1.0, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ]);
    let s = prim.simplify_cluster(3);
    // locate the cell that merged the three near-origin verts
    let near = s
        .positions
        .iter()
        .position(|p| p[0] < 1.0 && p[1] < 1.0)
        .expect("merged near-origin cell present");
    assert_eq!(s.tangents.as_ref().unwrap()[near][3], -1.0);
}

#[test]
fn morph_targets_roster_and_average_preserved() {
    let mut src = grid_sheet(8);
    let n = src.positions.len();
    src.targets = vec![MorphTarget {
        position: Some(vec![[0.0, 0.0, 0.5]; n]),
        normal: None,
        tangent: None,
    }];
    let s = src.simplify_cluster(4);
    assert_eq!(s.targets.len(), 1);
    let pos = s.targets[0].position.as_ref().unwrap();
    assert_eq!(pos.len(), s.positions.len());
    assert!(s.targets[0].normal.is_none());
    for d in pos {
        assert!(approx(*d, [0.0, 0.0, 0.5], 1e-5));
    }
}

#[test]
fn material_and_extras_carried_over() {
    use oxideav_mesh3d::MaterialId;
    let mut src = grid_sheet(4);
    src.material = Some(MaterialId(7));
    src.extras
        .insert("tag".to_string(), serde_json::json!("proxy"));
    let s = src.simplify_cluster(3);
    assert_eq!(src.material, s.material);
    assert_eq!(s.extras.get("tag"), Some(&serde_json::json!("proxy")));
}

// --- robustness ------------------------------------------------------

#[test]
fn empty_primitive_yields_empty() {
    let s = Primitive::new(Topology::Triangles).simplify_cluster(8);
    assert!(s.positions.is_empty());
    assert_eq!(s.triangle_count(), 0);
    assert_eq!(s.topology, Topology::Triangles);
}

#[test]
fn non_triangle_topology_yields_empty() {
    let mut prim = Primitive::new(Topology::Points);
    prim.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let s = prim.simplify_cluster(8);
    assert!(s.positions.is_empty());
    assert_eq!(s.triangle_count(), 0);
}

#[test]
fn out_of_range_and_degenerate_faces_excluded() {
    // first triangle has an out-of-range corner, second is fine.
    let prim = tri_indexed(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 1, 99, 0, 1, 2],
    );
    let s = prim.simplify_cluster(100);
    // only the valid triangle survives
    assert_eq!(s.triangle_count(), 1);
}

#[test]
fn nan_vertices_do_not_panic_and_are_handled() {
    let prim = tri_indexed(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [f32::NAN, 0.0, 0.0],
        ],
        vec![0, 1, 2, 0, 1, 3],
    );
    // Must not panic, and every emitted index stays in range. The NaN
    // vertex is excluded from the grid frame and clamped into cell 0, so
    // the finite triangle still produces a usable proxy.
    let s = prim.simplify_cluster(8);
    let idx = idx_vec(&s);
    assert!(idx.iter().all(|&i| (i as usize) < s.positions.len()));
    assert!(s.triangle_count() >= 1);
}

#[test]
fn does_not_mutate_self() {
    let src = grid_sheet(6);
    let before_pos = src.positions.clone();
    let before_tris = src.triangle_count();
    let _ = src.simplify_cluster(3);
    assert_eq!(src.positions, before_pos);
    assert_eq!(src.triangle_count(), before_tris);
}

#[test]
fn strip_topology_feeds_through() {
    // 4-vertex strip = 2 triangles, all distinct positions; a fine grid
    // keeps both.
    let mut prim = Primitive::new(Topology::TriangleStrip);
    prim.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let s = prim.simplify_cluster(1000);
    assert_eq!(s.topology, Topology::Triangles);
    assert_eq!(s.triangle_count(), 2);
}

#[test]
fn closed_tetrahedron_fine_grid_survives() {
    // A fine grid keeps the tetra intact (4 verts, 4 faces).
    let s = tetrahedron().simplify_cluster(1000);
    assert_eq!(s.positions.len(), 4);
    assert_eq!(s.triangle_count(), 4);
}
