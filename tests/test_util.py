import pyarrow as pa
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.core import CoreAnnotationMetadata
from bopp.models.v1.extent.times import Timestamps
from bopp.models.v1.payload.onset import OnsetPayload
from bopp.util import extract_header, from_dataframe, to_dataframe


def test_dataframe_roundtrip():
    ann = Annotation(
        annotation_info=CoreAnnotationMetadata(
            media_id="test_media",
            annotator_id="test_annotator",
        ),
        extent=Timestamps(time=pa.array([0.1, 0.5])),
        payload=OnsetPayload(values=pa.array([True, False])),
    )

    df = to_dataframe(ann)
    reconstituted = from_dataframe(df)

    assert reconstituted.annotation_info.media_id == "test_media"
    assert reconstituted.extent.time.to_pylist() == [0.1, 0.5]


def test_extract_header():
    ann = Annotation(
        annotation_info=CoreAnnotationMetadata(
            media_id="test_media",
            annotator_id="test_annotator",
        ),
        extent=Timestamps(time=pa.array([0.1])),
        payload=OnsetPayload(values=pa.array([True])),
    )

    header = extract_header(ann)
    assert header["annotation_info"]["media_id"] == "test_media"
    assert header["annotation_info"]["annotator_id"] == "test_annotator"
