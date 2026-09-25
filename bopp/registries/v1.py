# AUTO-GENERATED: Do not edit manually.

from ..models.v1.confidence.likelihood import LikelihoodConfidence
from ..models.v1.confidence.variance import VarianceConfidence
from ..models.v1.extent.time_frequency_box import TimeFrequencyBoxExtent
from ..models.v1.extent.time_interval import TimeIntervalExtent
from ..models.v1.extent.times import Times
from ..models.v1.metadata.algorithm import AlgorithmAnnotationMetadata
from ..models.v1.metadata.crowd import CrowdSourcedAnnotationMetadata
from ..models.v1.metadata.derived import DerivedAnnotationMetadata
from ..models.v1.metadata.human import HumanAnnotationMetadata
from ..models.v1.metadata.other import AnnotationMetadataOther
from ..models.v1.metadata.sensor import SensorAnnotationMetadata
from ..models.v1.payload.beat import BeatPositionPayload
from ..models.v1.payload.chord import ChordPayload
from ..models.v1.payload.key_mode import KeyModePayload
from ..models.v1.payload.mood_thayer import MoodThayerPayload
from ..models.v1.payload.note_hz import NoteHzPayload
from ..models.v1.payload.note_midi import NoteMidiPayload
from ..models.v1.payload.onset import OnsetPayload
from ..models.v1.payload.pattern_jku import PatternJkuPayload
from ..models.v1.payload.pitch_class import PitchClassPayload
from ..models.v1.payload.pitch_contour_hz import PitchContourPayload
from ..models.v1.payload.segment_multi import MultiSegmentPayload
from ..models.v1.payload.segment_open import SegmentOpenPayload
from ..models.v1.payload.tag_open import TagOpenPayload
from ..models.v1.payload.tempo import TempoPayload

CONFIDENCE_TYPE_REGISTRY = {
    'likelihood': LikelihoodConfidence,
    'variance': VarianceConfidence,
}

EXTENT_TYPE_REGISTRY = {
    'time': Times,
    'time_frequency_box': TimeFrequencyBoxExtent,
    'time_interval': TimeIntervalExtent,
}

METADATA_TYPE_REGISTRY = {
    'algorithm': AlgorithmAnnotationMetadata,
    'crowd': CrowdSourcedAnnotationMetadata,
    'derived': DerivedAnnotationMetadata,
    'human': HumanAnnotationMetadata,
    'other': AnnotationMetadataOther,
    'sensor': SensorAnnotationMetadata,
}

PAYLOAD_TYPE_REGISTRY = {
    'beat': BeatPositionPayload,
    'chord': ChordPayload,
    'key_mode': KeyModePayload,
    'mood_thayer': MoodThayerPayload,
    'multi_segment': MultiSegmentPayload,
    'note_hz': NoteHzPayload,
    'note_midi': NoteMidiPayload,
    'onset': OnsetPayload,
    'pattern_jku': PatternJkuPayload,
    'pitch_class': PitchClassPayload,
    'pitch_contour': PitchContourPayload,
    'segment_open': SegmentOpenPayload,
    'tag_open': TagOpenPayload,
    'tempo': TempoPayload,
}

