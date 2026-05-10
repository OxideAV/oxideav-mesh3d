//! Round-2 surface coverage:
//!
//! * `AssetSource` trait round-trip (open + read_to_end)
//! * `InMemoryAsset` defaults (no `raw_storage` pass-through)
//! * Custom `AssetSource` impl exposing a `RawStorage` block —
//!   exercises the USDZ → USDZ pass-through detection path.
//! * `ImageData::Source` round-trips through the `Texture` struct.
//! * `AudioSource` builder + `Scene3D::add_audio_source` returns
//!   sequential `AudioSourceId`s.
//! * `AudioEmitter` defaults (gain 1.0, no loop / auto-play, no
//!   spatial block).
//! * `SpatialAudio::default()` documented values.
//! * `Node::audio_emitter` resolves through `Scene3D::audio_emitter`.

use std::io::{Cursor, Read, Result as IoResult};
use std::sync::Arc;

use oxideav_mesh3d::{
    asset::ReadSeek, AssetSource, AudioData, AudioEmitter, AudioSource, AudioSourceId, AuralMode,
    DistanceModel, ImageData, InMemoryAsset, Node, RawStorage, Scene3D, SpatialAudio, Texture,
};

// ─────────────────────────── AssetSource ────────────────────────────

#[test]
fn in_memory_asset_round_trips_via_open() {
    let payload = b"hello mesh3d round 2";
    let asset = InMemoryAsset::new(
        Some("application/octet-stream".to_string()),
        payload.to_vec(),
    );
    assert_eq!(asset.mime(), Some("application/octet-stream"));
    assert_eq!(asset.size_hint(), Some(payload.len() as u64));
    // Default raw_storage is None — InMemoryAsset has no scheme.
    assert!(asset.raw_storage().is_none());

    let mut reader = asset.open().expect("open");
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).expect("read");
    assert_eq!(buf, payload);
}

#[test]
fn in_memory_asset_open_yields_independent_cursors() {
    // Two `open()` calls must give two independently-positionable
    // readers; advancing one mustn't move the other.
    let asset = InMemoryAsset::new(None, b"abcdef".to_vec());
    let mut a = asset.open().expect("a");
    let mut b = asset.open().expect("b");
    let mut head = [0u8; 3];
    a.read_exact(&mut head).expect("a head");
    assert_eq!(&head, b"abc");
    let mut whole = Vec::new();
    b.read_to_end(&mut whole).expect("b whole");
    assert_eq!(whole, b"abcdef");
}

#[test]
fn arc_dyn_asset_source_object_safe() {
    // The trait MUST be object-safe so callers can store
    // `Arc<dyn AssetSource>` in arena slots.
    let asset: Arc<dyn AssetSource> = Arc::new(InMemoryAsset::new(None, vec![1, 2, 3]));
    assert_eq!(asset.size_hint(), Some(3));
    let mut buf = Vec::new();
    asset.open().unwrap().read_to_end(&mut buf).unwrap();
    assert_eq!(buf, vec![1, 2, 3]);
}

// ───────────────────── Custom RawStorage impl ──────────────────────

/// Stand-in for "the bytes came from a ZIP entry stored under
/// deflate compression". The format crate (a hypothetical
/// oxideav-usdz reader) would expose its own AssetSource impl just
/// like this so a USDZ writer can detect the matching scheme and
/// pass the deflated payload through verbatim.
#[derive(Debug)]
struct ZipDeflateAsset {
    mime: Option<String>,
    deflated: Vec<u8>,
    uncompressed_len: u64,
}

impl AssetSource for ZipDeflateAsset {
    fn mime(&self) -> Option<&str> {
        self.mime.as_deref()
    }
    fn size_hint(&self) -> Option<u64> {
        Some(self.uncompressed_len)
    }
    fn open(&self) -> IoResult<Box<dyn ReadSeek + Send>> {
        // In a real impl this would inflate and return the inflated
        // stream; for the test we just expose the deflated bytes
        // verbatim — the test only cares about the raw_storage() path.
        Ok(Box::new(Cursor::new(self.deflated.clone())))
    }
    fn raw_storage(&self) -> Option<RawStorage<'_>> {
        Some(RawStorage {
            scheme: "zip-deflate",
            bytes: &self.deflated,
            uncompressed_size: Some(self.uncompressed_len),
        })
    }
}

#[test]
fn custom_asset_exposes_raw_storage_for_passthrough() {
    let asset = ZipDeflateAsset {
        mime: Some("image/png".into()),
        deflated: vec![0x78, 0x9c, 0x01, 0x02, 0x03],
        uncompressed_len: 1024,
    };
    let rs = asset.raw_storage().expect("raw_storage");
    assert_eq!(rs.scheme, "zip-deflate");
    assert_eq!(rs.bytes, &[0x78, 0x9c, 0x01, 0x02, 0x03]);
    assert_eq!(rs.uncompressed_size, Some(1024));

    // Caller-side dispatch the way a USDZ → USDZ writer would do it:
    let asset_dyn: Arc<dyn AssetSource> = Arc::new(asset);
    match asset_dyn.raw_storage() {
        Some(rs) if rs.scheme == "zip-deflate" => {
            // Pass-through path — copy `rs.bytes` straight into the
            // output ZIP without inflating.
            assert!(!rs.bytes.is_empty());
        }
        _ => panic!("expected zip-deflate raw_storage to be detected"),
    }
}

// ─────────────────────── ImageData::Source ─────────────────────────

#[test]
fn image_data_source_via_texture_round_trip() {
    let asset: Arc<dyn AssetSource> = Arc::new(InMemoryAsset::new(
        Some("image/png".into()),
        vec![0x89, 0x50, 0x4e, 0x47],
    ));
    let tex = Texture::from_source(Arc::clone(&asset));
    match &tex.image {
        ImageData::Source(s) => {
            let mut buf = Vec::new();
            s.open().unwrap().read_to_end(&mut buf).unwrap();
            assert_eq!(buf, vec![0x89, 0x50, 0x4e, 0x47]);
            assert_eq!(s.mime(), Some("image/png"));
        }
        _ => panic!("expected ImageData::Source"),
    }
}

#[test]
fn texture_from_encoded_now_wraps_in_in_memory_asset() {
    // The convenience constructor's signature is unchanged across
    // round 1 → round 2; only the internal representation moves
    // from `Encoded { mime, bytes }` to `Source(Arc<InMemoryAsset>)`.
    let tex = Texture::from_encoded("image/jpeg", vec![0xff, 0xd8, 0xff, 0xe0]);
    match &tex.image {
        ImageData::Source(s) => {
            assert_eq!(s.mime(), Some("image/jpeg"));
            assert_eq!(s.size_hint(), Some(4));
        }
        _ => panic!("expected ImageData::Source after migration"),
    }
}

// ───────────────────── AudioSource + Scene3D ───────────────────────

#[test]
fn add_audio_source_issues_sequential_ids() {
    let mut s = Scene3D::new();
    let bg = AudioSource::from_uri("file://background.ogg").with_name("bg");
    let sfx = AudioSource::from_uri("file://sfx.ogg").with_name("sfx");
    let id_bg = s.add_audio_source(bg);
    let id_sfx = s.add_audio_source(sfx);
    assert_eq!(id_bg, AudioSourceId(0));
    assert_eq!(id_sfx, AudioSourceId(1));
    assert_eq!(s.audio_source(id_bg).unwrap().name.as_deref(), Some("bg"));
    assert_eq!(s.audio_source(id_sfx).unwrap().name.as_deref(), Some("sfx"));
}

#[test]
fn audio_source_from_source_carries_asset() {
    let asset: Arc<dyn AssetSource> = Arc::new(InMemoryAsset::new(
        Some("audio/ogg".into()),
        vec![0x4f, 0x67, 0x67, 0x53],
    ));
    let src = AudioSource::from_source(Arc::clone(&asset)).with_name("clip");
    assert_eq!(src.name.as_deref(), Some("clip"));
    match &src.data {
        AudioData::Source(s) => {
            assert_eq!(s.mime(), Some("audio/ogg"));
            assert_eq!(s.size_hint(), Some(4));
        }
        _ => panic!("expected AudioData::Source"),
    }
}

// ───────────────────── AudioEmitter defaults ────────────────────────

#[test]
fn audio_emitter_defaults_are_documented_values() {
    let e = AudioEmitter::new(AudioSourceId(0));
    assert_eq!(e.source, AudioSourceId(0));
    assert!((e.gain - 1.0).abs() < f32::EPSILON);
    assert!(!e.looping);
    assert!(!e.auto_play);
    assert!(e.spatial.is_none());
    assert!(e.name.is_none());
}

#[test]
fn spatial_audio_defaults_match_module_docs() {
    let s = SpatialAudio::default();
    assert_eq!(s.aural_mode, AuralMode::SpatialNonAcoustic);
    let two_pi = std::f32::consts::TAU;
    assert!((s.cone_inner_angle - two_pi).abs() < f32::EPSILON);
    assert!((s.cone_outer_angle - two_pi).abs() < f32::EPSILON);
    assert!(s.cone_outer_gain.abs() < f32::EPSILON);
    assert!((s.min_distance - 1.0).abs() < f32::EPSILON);
    assert!((s.max_distance - 10000.0).abs() < f32::EPSILON);
    assert!((s.rolloff_factor - 1.0).abs() < f32::EPSILON);
    assert_eq!(s.distance_model, DistanceModel::Inverse);
}

#[test]
fn audio_emitter_with_spatial_round_trips() {
    let mut e = AudioEmitter::new(AudioSourceId(0)).with_spatial(SpatialAudio {
        aural_mode: AuralMode::SpatialAcoustic,
        distance_model: DistanceModel::Linear,
        max_distance: 50.0,
        ..SpatialAudio::default()
    });
    e.gain = 0.75;
    e.looping = true;
    e.auto_play = true;
    let sp = e.spatial.unwrap();
    assert_eq!(sp.aural_mode, AuralMode::SpatialAcoustic);
    assert_eq!(sp.distance_model, DistanceModel::Linear);
    assert!((sp.max_distance - 50.0).abs() < f32::EPSILON);
    assert!(e.looping);
    assert!(e.auto_play);
}

// ───────────────── Node::audio_emitter resolution ──────────────────

#[test]
fn node_audio_emitter_resolves_via_scene_helpers() {
    let mut s = Scene3D::new();
    let src_id = s.add_audio_source(AudioSource::from_uri("file://chime.wav"));
    let emitter_id = s.add_audio_emitter(
        AudioEmitter::new(src_id)
            .with_name("chime-emitter")
            .with_spatial(SpatialAudio::default()),
    );
    let node_id = s.add_node(Node::new().with_audio_emitter(emitter_id));
    s.add_root(node_id);

    let node = s.node(node_id).unwrap();
    assert_eq!(node.audio_emitter, Some(emitter_id));
    let emitter = s.audio_emitter(node.audio_emitter.unwrap()).unwrap();
    assert_eq!(emitter.name.as_deref(), Some("chime-emitter"));
    assert_eq!(emitter.source, src_id);
    let src = s.audio_source(emitter.source).unwrap();
    match &src.data {
        AudioData::External { uri, .. } => assert_eq!(uri, "file://chime.wav"),
        _ => panic!("expected External"),
    }
}

#[test]
fn fresh_scene_has_empty_audio_arenas() {
    let s = Scene3D::new();
    assert!(s.audio_sources.is_empty());
    assert!(s.audio_emitters.is_empty());
    let n = Node::new();
    assert!(n.audio_emitter.is_none());
}
