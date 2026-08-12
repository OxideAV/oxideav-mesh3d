//! Scene composition: appending one [`Scene3D`] into another with full
//! cross-arena id remapping.
//!
//! Importing several files into one world, instancing a decoded asset
//! into a larger scene, or concatenating a sequence of frames all need
//! the same primitive: take every node, mesh, material, texture,
//! skeleton, skin, animation, camera, light, and audio resource of a
//! *source* scene and splice them onto the end of a *destination*
//! scene's arenas — while rewriting every internal id so the relocated
//! resources still reference each other (and only each other), not
//! whatever happened to sit at the same index in the destination.
//!
//! [`Scene3D::append`] performs that splice in place and returns an
//! [`AppendOffsets`] recording, per arena, the index each source arena
//! started at in the destination — so a caller can locate any relocated
//! resource by adding the offset to its old id.
//!
//! # What gets remapped
//!
//! Every internal reference is shifted by its arena's offset:
//!
//! - **Node** → child `NodeId`s, `mesh`, `camera`, `light`, `skin`,
//!   `audio_emitter` (the id-free `weights` override travels
//!   verbatim).
//! - **Mesh.primitive** → `material`.
//! - **Material** → every texture slot (core + all extensions), via
//!   [`Material::map_texture_ids`].
//! - **Skin** → `skeleton`, `root_node`.
//! - **Skeleton** → every joint `NodeId`.
//! - **Animation channel target** → `node`.
//! - **AudioEmitter** → `source` ([`AudioSourceId`]).
//!
//! The source's [`roots`](Scene3D::roots) are remapped and appended to
//! the destination's roots, so both scenes' forests draw together. The
//! destination's `up_axis` / `front_axis` / `unit` / `extras` metadata
//! is **kept** (the source's orientation metadata is dropped — the
//! caller is responsible for converting coordinate systems before
//! appending a differently-oriented scene, exactly as the model's
//! "store orientation, never implicitly re-project" rule requires).

use crate::scene::{NodeId, Scene3D};

/// Per-arena starting offsets produced by [`Scene3D::append`].
///
/// Each field is the index in the destination scene at which the
/// source scene's corresponding arena began. Old source id `i` of a
/// given kind now lives at `offset + i` in the destination. Captured
/// **before** the append so the values are the pre-append destination
/// arena lengths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AppendOffsets {
    pub nodes: u32,
    pub meshes: u32,
    pub materials: u32,
    pub textures: u32,
    pub skeletons: u32,
    pub skins: u32,
    pub animations: u32,
    pub cameras: u32,
    pub lights: u32,
    pub audio_sources: u32,
    pub audio_emitters: u32,
}

impl Scene3D {
    /// Append every resource of `other` into `self`, remapping all
    /// internal ids so the relocated resources reference each other
    /// correctly, and return the per-arena [`AppendOffsets`].
    ///
    /// `other`'s root forest is remapped and appended to `self.roots`,
    /// so after the call `self` draws both scenes. `self`'s coordinate
    /// metadata (`up_axis` / `front_axis` / `unit` / `extras`) is
    /// preserved; `other`'s is dropped (convert coordinate systems
    /// before appending if they differ — see the [module
    /// docs](crate::compose)). Does not mutate `other`.
    ///
    /// **Material variants** are the one arena merged by *name*
    /// rather than by offset: a variant name is an asset-level
    /// identity, so "Red" declared in both scenes unifies into a
    /// single [`material_variants`](Scene3D::material_variants) entry
    /// (via [`find_or_add_material_variant`](Scene3D::find_or_add_material_variant))
    /// and every relocated
    /// [`variant_mappings`](crate::Primitive::variant_mappings) index
    /// is rewritten through the name map. That's why
    /// [`AppendOffsets`] carries no variant offset — there is no
    /// contiguous relocated block.
    ///
    /// Cost is linear in the size of `other`'s arenas (× the merged
    /// variant-roster length for the name unification).
    pub fn append(&mut self, other: &Scene3D) -> AppendOffsets {
        let off = AppendOffsets {
            nodes: self.nodes.len() as u32,
            meshes: self.meshes.len() as u32,
            materials: self.materials.len() as u32,
            textures: self.textures.len() as u32,
            skeletons: self.skeletons.len() as u32,
            skins: self.skins.len() as u32,
            animations: self.animations.len() as u32,
            cameras: self.cameras.len() as u32,
            lights: self.lights.len() as u32,
            audio_sources: self.audio_sources.len() as u32,
            audio_emitters: self.audio_emitters.len() as u32,
        };

        // --- Nodes: remap every id-bearing field -------------------
        for node in &other.nodes {
            let mut n = node.clone();
            for c in &mut n.children {
                c.0 += off.nodes;
            }
            if let Some(m) = &mut n.mesh {
                m.0 += off.meshes;
            }
            if let Some(c) = &mut n.camera {
                c.0 += off.cameras;
            }
            if let Some(l) = &mut n.light {
                l.0 += off.lights;
            }
            if let Some(s) = &mut n.skin {
                s.0 += off.skins;
            }
            if let Some(e) = &mut n.audio_emitter {
                e.0 += off.audio_emitters;
            }
            self.nodes.push(n);
        }

        // --- Material variants: unify rosters by NAME --------------
        // Variant names are asset-level identities ("Red", "Winter"),
        // not positional indices: appending two scenes that both
        // declare "Red" must yield ONE "Red" entry that a single
        // active-variant switch drives across every primitive. So
        // instead of a flat arena offset, each of `other`'s variant
        // indices maps through find-or-add on its name.
        let variant_map: Vec<u32> = other
            .material_variants
            .iter()
            .map(|name| self.find_or_add_material_variant(name).0)
            .collect();
        // Ids that were already dangling in `other` (validate()
        // territory) are rebased past the merged roster so they stay
        // dangling instead of aliasing a live variant.
        let dangling_base = self.material_variants.len() as u32;

        // --- Meshes: primitive material + variant-mapping refs -----
        for mesh in &other.meshes {
            let mut m = mesh.clone();
            for prim in &mut m.primitives {
                if let Some(mat) = &mut prim.material {
                    mat.0 += off.materials;
                }
                for mapping in &mut prim.variant_mappings {
                    mapping.material.0 += off.materials;
                    for v in &mut mapping.variants {
                        match variant_map.get(v.0 as usize) {
                            Some(&mapped) => v.0 = mapped,
                            None => v.0 = v.0.saturating_add(dangling_base),
                        }
                    }
                }
            }
            self.meshes.push(m);
        }

        // --- Materials: every texture slot -------------------------
        let tex_off = off.textures;
        for material in &other.materials {
            let mut mat = material.clone();
            mat.map_texture_ids(|mut t| {
                t.0 += tex_off;
                t
            });
            self.materials.push(mat);
        }

        // --- Textures: no internal ids -----------------------------
        self.textures.extend(other.textures.iter().cloned());

        // --- Skeletons: joint node refs ----------------------------
        for skel in &other.skeletons {
            let mut s = skel.clone();
            for j in &mut s.joints {
                j.0 += off.nodes;
            }
            self.skeletons.push(s);
        }

        // --- Skins: skeleton + root node ---------------------------
        for skin in &other.skins {
            let mut s = *skin;
            s.skeleton.0 += off.skeletons;
            if let Some(r) = &mut s.root_node {
                r.0 += off.nodes;
            }
            self.skins.push(s);
        }

        // --- Animations: channel target nodes ----------------------
        for anim in &other.animations {
            let mut a = anim.clone();
            for ch in &mut a.channels {
                ch.target.node.0 += off.nodes;
            }
            self.animations.push(a);
        }

        // --- Cameras / lights: no internal ids ---------------------
        self.cameras.extend(other.cameras.iter().cloned());
        self.lights.extend(other.lights.iter().cloned());

        // --- Audio sources: no internal ids ------------------------
        self.audio_sources
            .extend(other.audio_sources.iter().cloned());

        // --- Audio emitters: source ref ----------------------------
        for em in &other.audio_emitters {
            let mut e = em.clone();
            e.source.0 += off.audio_sources;
            self.audio_emitters.push(e);
        }

        // --- Roots: remap + append ---------------------------------
        for r in &other.roots {
            self.roots.push(NodeId(r.0 + off.nodes));
        }

        off
    }
}
