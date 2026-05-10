//! Audio assets, emitters, and spatial-audio metadata.
//!
//! Audio is a first-class part of every modern 3D scene format that
//! the type model targets: USD/USDZ carries `UsdMediaSpatialAudio`
//! prims, glTF gained the `KHR_audio_emitter` extension, FBX has
//! `FbxAudioClip`, and modern Blender exports follow suit. Round 1
//! deliberately punted on this — round 2 lands the typed surface
//! before the v0.1 publish locks the API in.
//!
//! ## Shape
//!
//! * [`AudioSource`] is the asset (the bytes — analogous to
//!   [`Texture`](crate::Texture)).
//! * [`AudioEmitter`] is the in-scene instance — references one
//!   [`AudioSource`] plus playback / spatialisation parameters
//!   (analogous to a [`Light`](crate::Light) reference on a
//!   [`Node`](crate::Node)).
//! * [`SpatialAudio`] is the optional positional-rendering payload
//!   (cone, distance model, rolloff). When `None` the emitter is a
//!   global non-positional background source.
//!
//! Formats that don't carry audio (STL, OBJ) trivially leave the
//! `audio_sources` / `audio_emitters` arrays empty and the new
//! `Node::audio_emitter` field `None`.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::asset::AssetSource;

/// Decoded or referenced audio payload.
///
/// Mirrors [`ImageData`](crate::ImageData)'s three-shape design: an
/// in-process decoded buffer ([`AudioFrame`](oxideav_core::AudioFrame)
/// when the `registry` feature is on), a lazy [`AssetSource`] that
/// can be opened on demand, or an unresolved external URI.
#[derive(Clone, Debug)]
pub enum AudioData {
    /// Decoded PCM samples. Only available with the default-on
    /// `registry` feature, which provides
    /// [`oxideav_core::AudioFrame`]. Format-level metadata (sample
    /// rate, channel count, sample format) lives on the consumer's
    /// stream-parameters side-table — `AudioFrame` itself is
    /// intentionally lightweight.
    #[cfg(feature = "registry")]
    Embedded(oxideav_core::AudioFrame),
    /// Lazy reference. The wrapped [`AssetSource`] supplies bytes on
    /// demand and may expose `raw_storage()` for zero-copy
    /// pass-through when the writer's container scheme matches.
    Source(Arc<dyn AssetSource>),
    /// URI to fetch and decode lazily — the consumer-side fetcher
    /// resolves it. `mime` is a hint when known.
    External { uri: String, mime: Option<String> },
}

/// One audio asset owned by the [`Scene3D`](crate::Scene3D).
///
/// Held in an arena addressed by [`AudioSourceId`]; emitters
/// reference it by id so the same asset can drive multiple
/// emitters without duplication.
#[derive(Clone)]
pub struct AudioSource {
    pub name: Option<String>,
    pub data: AudioData,
    /// Round-trip side-channel for format-specific metadata that
    /// doesn't fit the typed shape (USD `mediaOffset` / `mediaStart`,
    /// FBX clip-rate hints, etc.).
    pub extras: HashMap<String, serde_json::Value>,
}

impl AudioSource {
    /// Empty audio source pointing at an external URI.
    pub fn from_uri(uri: impl Into<String>) -> Self {
        Self {
            name: None,
            data: AudioData::External {
                uri: uri.into(),
                mime: None,
            },
            extras: HashMap::new(),
        }
    }

    /// Build an audio source from any [`AssetSource`] implementor.
    pub fn from_source(source: Arc<dyn AssetSource>) -> Self {
        Self {
            name: None,
            data: AudioData::Source(source),
            extras: HashMap::new(),
        }
    }

    /// Builder-style name setter.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl fmt::Debug for AudioSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioSource")
            .field("name", &self.name)
            .field("data", &self.data)
            .field("extras", &self.extras)
            .finish()
    }
}

/// In-scene instance of an [`AudioSource`].
///
/// Defaults are tuned for "drop a sound on a node and have it just
/// play" usability: gain 1.0, no looping, no auto-play, no spatial
/// localisation. Format crates override per-field as the file
/// directs.
#[derive(Clone, Debug)]
pub struct AudioEmitter {
    pub name: Option<String>,
    pub source: AudioSourceId,
    /// Linear gain multiplier; default `1.0`. `0.0` mutes the
    /// emitter without disconnecting it from the scene-graph (handy
    /// for animation-driven mute toggles).
    pub gain: f32,
    /// Loop the source once it ends. Default `false`.
    pub looping: bool,
    /// Begin playing as soon as the scene is loaded. Default `false`
    /// — most engines prefer to script playback start themselves.
    pub auto_play: bool,
    /// `None` ⇒ global / non-positional background source.
    /// `Some` ⇒ positional source rendered at the owning node's
    /// world transform with the supplied attenuation parameters.
    pub spatial: Option<SpatialAudio>,
    /// Round-trip side-channel for format-specific knobs.
    pub extras: HashMap<String, serde_json::Value>,
}

impl AudioEmitter {
    /// Construct a non-spatial emitter with the documented defaults
    /// (gain 1.0, no loop, no auto-play, global / non-positional).
    pub fn new(source: AudioSourceId) -> Self {
        Self {
            name: None,
            source,
            gain: 1.0,
            looping: false,
            auto_play: false,
            spatial: None,
            extras: HashMap::new(),
        }
    }

    /// Builder-style name setter.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Builder-style spatial-audio attachment.
    pub fn with_spatial(mut self, spatial: SpatialAudio) -> Self {
        self.spatial = Some(spatial);
        self
    }
}

/// Positional-audio rendering parameters.
///
/// All angles are in radians. Distances are in scene units (typically
/// metres — see [`Scene3D::unit`](crate::Scene3D::unit)). The
/// defaults below match the OpenAL / WebAudio "omni-directional
/// inverse-square fall-off" convention that USD and glTF both align
/// with.
#[derive(Clone, Copy, Debug)]
pub struct SpatialAudio {
    pub aural_mode: AuralMode,
    /// Half-angle of the inner full-gain cone, in radians. Default
    /// `2π` (omnidirectional).
    pub cone_inner_angle: f32,
    /// Half-angle of the outer attenuation cone, in radians. Default
    /// `2π` (omnidirectional).
    pub cone_outer_angle: f32,
    /// Gain multiplier applied OUTSIDE `cone_outer_angle`. `[0,1]`;
    /// default `0.0`.
    pub cone_outer_gain: f32,
    /// Distance at which gain reaches 1.0 in the inverse / linear
    /// models. Metres; default `1.0`.
    pub min_distance: f32,
    /// Distance beyond which the attenuated gain is clamped /
    /// silenced. Metres; default `10000.0`.
    pub max_distance: f32,
    /// Per-model rolloff multiplier. Default `1.0`. See
    /// [`DistanceModel`] for the per-model formula.
    pub rolloff_factor: f32,
    pub distance_model: DistanceModel,
}

impl Default for SpatialAudio {
    fn default() -> Self {
        let two_pi = std::f32::consts::TAU;
        Self {
            aural_mode: AuralMode::SpatialNonAcoustic,
            cone_inner_angle: two_pi,
            cone_outer_angle: two_pi,
            cone_outer_gain: 0.0,
            min_distance: 1.0,
            max_distance: 10000.0,
            rolloff_factor: 1.0,
            distance_model: DistanceModel::Inverse,
        }
    }
}

impl SpatialAudio {
    /// Spatial-audio block with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Acoustic-modelling regime the renderer should apply.
///
/// The type model just carries the flag — actually applying HRTF /
/// reverb / occlusion is the consumer's job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuralMode {
    /// USD `UsdMediaSpatialAudio.auralMode` default — non-acoustic
    /// spatialisation: panning + distance attenuation only.
    SpatialNonAcoustic,
    /// Full acoustic modelling: HRTF, occlusion, reverb. Used by
    /// engines that opt into a physics-based audio pipeline.
    SpatialAcoustic,
}

/// Distance-attenuation curve for [`SpatialAudio`].
///
/// Names match WebAudio + OpenAL. The renderer applies the curve
/// using `min_distance` / `max_distance` / `rolloff_factor` from
/// the parent [`SpatialAudio`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceModel {
    /// `gain = 1 - rolloff * ((d - min) / (max - min))`, clamped to `[0, 1]`.
    Linear,
    /// `gain = min / (min + rolloff * (max(d, min) - min))`. Default in
    /// WebAudio + OpenAL.
    Inverse,
    /// `gain = (max(d, min) / min) ^ -rolloff`.
    Exponential,
}

/// Index into [`Scene3D::audio_sources`](crate::Scene3D::audio_sources).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AudioSourceId(pub u32);

/// Index into [`Scene3D::audio_emitters`](crate::Scene3D::audio_emitters).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AudioEmitterId(pub u32);
