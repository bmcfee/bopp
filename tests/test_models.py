import pyarrow as pa
import pytest
import msgspec

from bopp.models.v1.annotation import Annotation
from bopp.models.v1.core import CoreAnnotationMetadata
from bopp.models.v1.extent.times import Timestamps
from bopp.models.v1.payload.onset import OnsetPayload


def test_annotation_instantiation():
    ann = Annotation(
        annotation_info=CoreAnnotationMetadata(
            media_id="test_media",
            annotator_id="test_annotator",
        ),
        extent=Timestamps(time=pa.array([0.1, 0.5, 1.2])),
        payload=OnsetPayload(values=pa.array([True, True, True])),
    )
    assert ann.annotation_info.media_id == "test_media"
    assert len(ann.extent.time) == 3
    assert len(ann.payload.values) == 3


def test_annotation_length_mismatch_raises_validation_error():
    with pytest.raises(msgspec.ValidationError):
        Annotation(
            annotation_info=CoreAnnotationMetadata(
                media_id="test_media",
                annotator_id="test_annotator",
            ),
            extent=Timestamps(time=pa.array([0.1, 0.5])),
            payload=OnsetPayload(values=pa.array([True, True, True])),
        )
