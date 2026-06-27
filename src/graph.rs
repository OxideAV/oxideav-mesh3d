//! Scene-graph navigation and transform baking.
//!
//! [`Scene3D`] stores its node hierarchy as a forest: a [`roots`] list
//! plus a per-node [`children`] list. Many operations a format crate
//! performs need the *inverse* of that structure — "who is my parent?",
//! "what is my full ancestor chain?", "give me everything beneath this
//! node" — or need the hierarchy collapsed entirely, because the target
//! format (binary STL, Wavefront OBJ) has no concept of nested
//! transforms. This module supplies both.
//!
//! All navigation matches the **first-arrival depth-first** semantics
//! of [`Scene3D::world_node_transforms`]: roots are visited in
//! `roots`-order, children in source order, a node reachable through
//! two parents binds to the first parent encountered, and cycles /
//! out-of-range ids are skipped after their first visit. So the parent
//! map, ancestor chains, and baked transforms are all consistent with
//! the world matrices that method already produces.
//!
//! [`roots`]: Scene3D::roots
//! [`children`]: crate::Node::children

use crate::scene::{Node, NodeId, Scene3D, Transform};

impl Scene3D {
    /// First-parent map over the node forest, indexed by `NodeId.0`.
    ///
    /// `parents()[i]` is `Some(p)` when node `i` is reached as a child
    /// of node `p` in the depth-first walk, or `None` when `i` is a
    /// root (or is unreachable from any root). A node listed under two
    /// parents resolves to the **first** one visited — the same
    /// shared-instance rule [`Scene3D::world_node_transforms`] uses — so
    /// the map is a spanning forest, never ambiguous.
    ///
    /// Cost `O(nodes.len() + total_children)`; one `Vec` allocation.
    pub fn parents(&self) -> Vec<Option<NodeId>> {
        let n = self.nodes.len();
        let mut parent: Vec<Option<NodeId>> = vec![None; n];
        let mut seen = vec![false; n];
        // DFS in the same order world_node_transforms walks.
        let mut stack: Vec<NodeId> = self.roots.iter().rev().copied().collect();
        // Mark valid roots as seen up front so a root that also appears
        // as some node's child still reports None (it is a root first).
        for &r in &self.roots {
            let idx = r.0 as usize;
            if idx < n {
                seen[idx] = true;
            }
        }
        while let Some(nid) = stack.pop() {
            let idx = nid.0 as usize;
            if idx >= n {
                continue;
            }
            for child in self.nodes[idx].children.iter().rev() {
                let cidx = child.0 as usize;
                if cidx >= n || seen[cidx] {
                    continue;
                }
                seen[cidx] = true;
                parent[cidx] = Some(nid);
                stack.push(*child);
            }
        }
        parent
    }

    /// The ancestor chain of `node`, from the owning root down to (but
    /// **excluding**) `node` itself.
    ///
    /// `ancestors(n)` returns `[root, …, parent_of_n]`. An empty vector
    /// means `node` is a root or is unreachable / out of range. The
    /// chain follows the first-parent spanning forest from
    /// [`parents`](Self::parents), so it is unique and cycle-free even
    /// for a malformed graph.
    pub fn ancestors(&self, node: NodeId) -> Vec<NodeId> {
        let parent = self.parents();
        let n = self.nodes.len();
        let mut chain = Vec::new();
        let mut cur = node.0 as usize;
        if cur >= n {
            return chain;
        }
        // Walk up to the root, guarding against any residual cycle.
        let mut guard = 0;
        while let Some(p) = parent.get(cur).copied().flatten() {
            chain.push(p);
            cur = p.0 as usize;
            guard += 1;
            if guard > n {
                break;
            }
        }
        chain.reverse();
        chain
    }

    /// Every node in the subtree rooted at `node`, **including** `node`,
    /// in depth-first first-arrival order.
    ///
    /// Returns an empty vector when `node` is out of range. Each node is
    /// listed once even if the graph routes to it by several paths;
    /// cycles terminate at the first revisit. The first element is
    /// always `node` itself (when in range).
    pub fn descendants(&self, node: NodeId) -> Vec<NodeId> {
        let n = self.nodes.len();
        let mut out = Vec::new();
        if (node.0 as usize) >= n {
            return out;
        }
        let mut seen = vec![false; n];
        let mut stack = vec![node];
        while let Some(nid) = stack.pop() {
            let idx = nid.0 as usize;
            if idx >= n || seen[idx] {
                continue;
            }
            seen[idx] = true;
            out.push(nid);
            for child in self.nodes[idx].children.iter().rev() {
                let cidx = child.0 as usize;
                if cidx < n && !seen[cidx] {
                    stack.push(*child);
                }
            }
        }
        out
    }

    /// `true` when `ancestor` lies on the path from a root down to
    /// `node` (a proper ancestor — a node is **not** its own ancestor).
    pub fn is_ancestor_of(&self, ancestor: NodeId, node: NodeId) -> bool {
        self.ancestors(node).contains(&ancestor)
    }

    /// Bake the transform hierarchy into a **flat** scene.
    ///
    /// Returns a clone of `self` in which every node reachable from a
    /// root carries its full world transform as a
    /// [`Transform::Matrix`], all `children` lists are cleared, and the
    /// reachable nodes become the new [`roots`](Scene3D::roots) in
    /// depth-first first-arrival order. The result draws identically to
    /// the original — each node's world matrix is unchanged — but has
    /// no nested transforms, which is exactly what a hierarchy-free
    /// target format (binary STL, Wavefront OBJ) needs to consume.
    ///
    /// Node *indices are preserved* (the returned scene has the same
    /// `nodes.len()`, and node `i` still describes the same mesh /
    /// camera / light instance), so any external `NodeId` references
    /// stay valid; only each node's `transform` and `children` change,
    /// and `roots` is rewritten. Nodes unreachable from any root keep
    /// their original local transform and an empty child list but are
    /// **not** promoted to roots (they were not drawn before and are
    /// not drawn now). All non-node scene resources (meshes, materials,
    /// animations, …) and scene metadata pass through untouched.
    ///
    /// Does not mutate `self`. A node reached through two parents binds
    /// to the first-parent world matrix, matching
    /// [`world_node_transforms`](Self::world_node_transforms).
    pub fn bake_transforms(&self) -> Scene3D {
        let worlds = self.world_node_transforms();
        let mut out = self.clone();

        // Reachable nodes, in DFS first-arrival order, become the new
        // flat root list. Reuse the same traversal world_node_transforms
        // uses so the ordering is identical.
        let n = self.nodes.len();
        let mut new_roots: Vec<NodeId> = Vec::new();
        let mut seen = vec![false; n];
        let mut stack: Vec<NodeId> = self.roots.iter().rev().copied().collect();
        while let Some(nid) = stack.pop() {
            let idx = nid.0 as usize;
            if idx >= n || seen[idx] {
                continue;
            }
            seen[idx] = true;
            new_roots.push(nid);
            for child in self.nodes[idx].children.iter().rev() {
                let cidx = child.0 as usize;
                if cidx < n && !seen[cidx] {
                    stack.push(*child);
                }
            }
        }

        // Bake: reachable nodes get their world matrix + no children.
        for (idx, node) in out.nodes.iter_mut().enumerate() {
            if let Some(Some(world)) = worlds.get(idx) {
                node.transform = Transform::Matrix(*world);
                node.children.clear();
            }
        }
        out.roots = new_roots;
        out
    }
}

impl Node {
    /// This node's local transform as a 4x4 matrix — a convenience
    /// shorthand for `self.transform.to_matrix()` (the build order is
    /// `T * R * S` for the TRS form).
    pub fn local_matrix(&self) -> [[f32; 4]; 4] {
        self.transform.to_matrix()
    }
}
