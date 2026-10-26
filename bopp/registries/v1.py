# AUTO-GENERATED: Do not edit manually.

from ..models.v1.annotation import Annotation
from ..models.v1.confidence.agreement import ConfidenceByInterAnnotatorAgreement
from ..models.v1.confidence.likelihood import LikelihoodConfidence
from ..models.v1.confidence.variance import VarianceConfidence
from ..models.v1.extent.cartesian import CartesianExtent
from ..models.v1.extent.cartesian_time import CartesianTimeExtent
from ..models.v1.extent.cartesian_time_interval import CartesianTimeIntervalExtent
from ..models.v1.extent.midi_interval import MidiInterval
from ..models.v1.extent.midi_tick import MidiTicks
from ..models.v1.extent.pixel_box import PixelBoxExtent
from ..models.v1.extent.pixels import Pixels
from ..models.v1.extent.score_interval import ScoreInterval
from ..models.v1.extent.score_interval_float import ScoreIntervalFloat
from ..models.v1.extent.score_quarter import ScoreQuarterNotes
from ..models.v1.extent.score_quarter_float import ScoreQuarterNotesFloat
from ..models.v1.extent.spherical import SphericalExtent
from ..models.v1.extent.spherical_time import SphericalTimeExtent
from ..models.v1.extent.spherical_time_interval import SphericalTimeIntervalExtent
from ..models.v1.extent.text import Text
from ..models.v1.extent.text_interval import TextInterval
from ..models.v1.extent.time_frequency_box import TimeFrequencyBoxExtent
from ..models.v1.extent.time_interval import TimeIntervalExtent
from ..models.v1.extent.time_uniform import UniformTimeExtent
from ..models.v1.extent.times import Times
from ..models.v1.grouping import Grouping
from ..models.v1.metadata.algorithm import AlgorithmAnnotationMetadata
from ..models.v1.metadata.crowd import CrowdSourcedAnnotationMetadata
from ..models.v1.metadata.derived import DerivedAnnotationMetadata
from ..models.v1.metadata.human import HumanAnnotationMetadata
from ..models.v1.metadata.other import AnnotationMetadataOther
from ..models.v1.metadata.sensor import SensorAnnotationMetadata
from ..models.v1.payload.beat import BeatPositionPayload
from ..models.v1.payload.chord import ChordPayload
from ..models.v1.payload.ext import ExtensionPayload
from ..models.v1.payload.frequency_hz import FrequencyHzPayload
from ..models.v1.payload.key_mode import KeyModePayload
from ..models.v1.payload.lyrics import LyricsPayload
from ..models.v1.payload.marker import MarkerPayload
from ..models.v1.payload.mood_thayer import MoodThayerPayload
from ..models.v1.payload.note import NotePayload
from ..models.v1.payload.note_midi import NoteMidiPayload
from ..models.v1.payload.object import ObjectPayload
from ..models.v1.payload.relation import RelationPayload
from ..models.v1.payload.tag import TagPayload
from ..models.v1.payload.tempo import TempoPayload

ANNOTATION_CLASS = Annotation

GROUPING_CLASS = Grouping

CONFIDENCE_TYPE_REGISTRY = {
    "agreement": ConfidenceByInterAnnotatorAgreement,
    "likelihood": LikelihoodConfidence,
    "variance": VarianceConfidence,
}

EXTENT_TYPE_REGISTRY = {
    "cartesian": CartesianExtent,
    "cartesian_time": CartesianTimeExtent,
    "cartesian_time_interval": CartesianTimeIntervalExtent,
    "midi_interval": MidiInterval,
    "midi_tick": MidiTicks,
    "pixel_box": PixelBoxExtent,
    "pixels": Pixels,
    "score_interval": ScoreInterval,
    "score_interval_float": ScoreIntervalFloat,
    "score_quarter": ScoreQuarterNotes,
    "score_quarter_float": ScoreQuarterNotesFloat,
    "spherical": SphericalExtent,
    "spherical_time": SphericalTimeExtent,
    "spherical_time_interval": SphericalTimeIntervalExtent,
    "text": Text,
    "text_interval": TextInterval,
    "time": Times,
    "time_frequency_box": TimeFrequencyBoxExtent,
    "time_interval": TimeIntervalExtent,
    "time_uniform": UniformTimeExtent,
}

METADATA_TYPE_REGISTRY = {
    "algorithm": AlgorithmAnnotationMetadata,
    "crowd": CrowdSourcedAnnotationMetadata,
    "derived": DerivedAnnotationMetadata,
    "human": HumanAnnotationMetadata,
    "other": AnnotationMetadataOther,
    "sensor": SensorAnnotationMetadata,
}

PAYLOAD_TYPE_REGISTRY = {
    "beat": BeatPositionPayload,
    "chord": ChordPayload,
    "ext": ExtensionPayload,
    "frequency_hz": FrequencyHzPayload,
    "key_mode": KeyModePayload,
    "lyrics": LyricsPayload,
    "marker": MarkerPayload,
    "mood_thayer": MoodThayerPayload,
    "note": NotePayload,
    "note_midi": NoteMidiPayload,
    "object": ObjectPayload,
    "relation": RelationPayload,
    "tag": TagPayload,
    "tempo": TempoPayload,
}

COMPLEX_FIELDS_REGISTRY: dict[str, dict[str, list[str]]] = {
    "extent_type": {
        "score_interval": ["quarter", "duration"],
        "score_quarter": ["quarter"],
    },
    "metadata_type": {
        "algorithm": ["parameters"],
        "derived": ["parameters"],
        "sensor": ["settings"],
    },
    "payload_type": {
        "ext": ["value"],
        "object": ["value"],
    },
}

STRING_FIELDS_REGISTRY: dict[str, dict[str, list[str]]] = {
    "metadata_type": {
        "algorithm": ["algorithm_id", "version"],
        "crowd": ["platform", "consensus"],
        "derived": ["transform", "software", "description"],
        "human": ["annotator_id", "tool"],
        "other": ["notes"],
        "sensor": ["device"],
    },
    "payload_type": {
        "chord": ["value"],
        "ext": ["ext_schema"],
        "key_mode": ["value"],
        "lyrics": ["value", "language"],
        "marker": ["description"],
        "note": ["value"],
        "object": ["description"],
        "relation": ["performance_id", "value", "relation"],
        "tag": ["value"],
    },
}

__all__ = [
    "ANNOTATION_CLASS",
    "COMPLEX_FIELDS_REGISTRY",
    "CONFIDENCE_TYPE_REGISTRY",
    "EXTENT_TYPE_REGISTRY",
    "GROUPING_CLASS",
    "METADATA_TYPE_REGISTRY",
    "PAYLOAD_TYPE_REGISTRY",
    "STRING_FIELDS_REGISTRY",
]
