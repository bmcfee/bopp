import pyarrow as pa
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_json,
    save_bopp_msgpack,
    to_csv,
)
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.core import CoreAnnotationMetadata
from bopp.models.v1.extent.times import Timestamps
from bopp.models.v1.payload.onset import OnsetPayload


def create_sample_annotation() -> Annotation:
    return Annotation(
        annotation_info=CoreAnnotationMetadata(
            media_id="test_media_123",
            annotator_id="annotator_456",
        ),
        extent=Timestamps(time=pa.array([0.0, 1.0, 2.0])),
        payload=OnsetPayload(values=pa.array([True, False, True])),
    )


def test_json_roundtrip(tmp_path):
    ann = create_sample_annotation()
    file_path = str(tmp_path / "annotation.json")

    save_bopp_json(ann, file_path)
    loaded_ann = load_bopp_json(file_path)

    assert loaded_ann.annotation_info.media_id == ann.annotation_info.media_id
    assert loaded_ann.extent.time.to_pylist() == [0.0, 1.0, 2.0]


def test_msgpack_roundtrip(tmp_path):
    ann = create_sample_annotation()
    file_path = str(tmp_path / "annotation.msgpack")

    save_bopp_msgpack(ann, file_path)
    loaded_ann = load_bopp_msgpack(file_path)

    assert loaded_ann.annotation_info.media_id == ann.annotation_info.media_id
    assert loaded_ann.extent.time.to_pylist() == [0.0, 1.0, 2.0]


def test_csv_roundtrip(tmp_path):
    ann = create_sample_annotation()
    file_path = tmp_path / "annotation.csv"

    to_csv(ann, file_path)
    loaded_ann = load_bopp_csv(file_path)

    assert loaded_ann.annotation_info.media_id == ann.annotation_info.media_id
    assert loaded_ann.extent.time.to_pylist() == [0.0, 1.0, 2.0]
