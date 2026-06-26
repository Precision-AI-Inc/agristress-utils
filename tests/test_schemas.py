"""Unit tests for schemas.dataset and schemas.retrieval."""

from precisionai.agribench.utils.schemas.dataset import (
    ImageAttributes,
    ImageRecord,
    InstanceRecord,
)
from precisionai.agribench.utils.schemas.retrieval import (
    I2IPayload,
    MetadataGroupDict,
    P2IPayload,
    P2PPayload,
)

# ── ImageAttributes ────────────────────────────────────────────────────────────


def test_image_attributes_required_fields() -> None:
    attrs = ImageAttributes(
        plants=["Crop | Soybean"],
        time_period="noon",
        camera_model="HB-25000SBC",
        camera_source="drone",
        collecting_date="2022-06-22",
    )
    assert attrs.plants == ["Crop | Soybean"]
    assert attrs.camera_model == "HB-25000SBC"


def test_image_attributes_nullable_fields_accept_none() -> None:
    attrs = ImageAttributes(
        plants=[],
        time_period=None,
        camera_model="",
        camera_source="",
        collecting_date=None,
    )
    assert attrs.time_period is None
    assert attrs.collecting_date is None


def test_image_attributes_empty_plants_list() -> None:
    attrs = ImageAttributes(
        plants=[],
        time_period=None,
        camera_model="cam",
        camera_source="drone",
        collecting_date=None,
    )
    assert attrs.plants == []


# ── ImageRecord ────────────────────────────────────────────────────────────────


def _make_attrs() -> ImageAttributes:
    return ImageAttributes(
        plants=["Crop | Corn"],
        time_period=None,
        camera_model="DJI",
        camera_source="drone",
        collecting_date=None,
    )


def test_image_record_fields() -> None:
    record = ImageRecord(
        image_id="images/A1/pai-abc.png",
        cluster="A1",
        attributes=_make_attrs(),
        instances=["instances/A1/pai-abc-1.png"],
    )
    assert record.image_id == "images/A1/pai-abc.png"
    assert record.cluster == "A1"
    assert len(record.instances) == 1


def test_image_record_embedding_defaults_to_none() -> None:
    record = ImageRecord(
        image_id="images/A1/pai-abc.png",
        cluster="A1",
        attributes=_make_attrs(),
        instances=[],
    )
    assert record.embedding is None


def test_image_record_embedding_populated() -> None:
    record = ImageRecord(
        image_id="images/A1/pai-abc.png",
        cluster="A1",
        attributes=_make_attrs(),
        instances=[],
        embedding=[0.1, 0.2, 0.7],
    )
    assert record.embedding == [0.1, 0.2, 0.7]


# ── InstanceRecord ─────────────────────────────────────────────────────────────


def test_instance_record_fields() -> None:
    record = InstanceRecord(
        instance_id="instances/A1/pai-abc-1.png",
        label="Crop | Soybean",
        parent_image="images/A1/pai-abc.png",
    )
    assert record.instance_id == "instances/A1/pai-abc-1.png"
    assert record.label == "Crop | Soybean"
    assert record.parent_image == "images/A1/pai-abc.png"


def test_instance_record_embedding_defaults_to_none() -> None:
    record = InstanceRecord(
        instance_id="instances/A1/pai-abc-1.png",
        label="Crop | Soybean",
        parent_image="images/A1/pai-abc.png",
    )
    assert record.embedding is None


def test_instance_record_embedding_populated() -> None:
    record = InstanceRecord(
        instance_id="instances/A1/pai-abc-1.png",
        label="Crop | Soybean",
        parent_image="images/A1/pai-abc.png",
        embedding=[0.5, 0.5],
    )
    assert record.embedding == [0.5, 0.5]


# ── TypedDict shapes ───────────────────────────────────────────────────────────


def test_metadata_group_dict_shape() -> None:
    group: MetadataGroupDict = {
        "images": ["images/A1/pai-abc.png"],
        "class_name": "A1",
        "attributes": {"plants": ["Crop | Soybean"]},
    }
    assert group["class_name"] == "A1"
    assert group["images"] == ["images/A1/pai-abc.png"]


def test_i2i_payload_shape() -> None:
    payload: I2IPayload = {
        "embeddings": {"images/A1/pai-abc.png": [0.1, 0.9]},
        "metadata": None,
        "dataset_root": None,
        "k_values": [5, 10, 20],
        "sample_pairs": 1_000_000,
    }
    assert "embeddings" in payload
    assert payload["metadata"] is None


def test_p2i_payload_shape() -> None:
    payload: P2IPayload = {
        "embeddings": {},
        "instance_to_image": {"images/A1/pai-abc.png": ["instances/A1/pai-abc-1.png"]},
        "dataset_root": "/data",
        "k_values": [5],
        "sample_pairs": None,
    }
    assert "instance_to_image" in payload
    assert payload["dataset_root"] == "/data"


def test_p2p_payload_shape() -> None:
    payload: P2PPayload = {
        "embeddings": {},
        "instance_labels": {"instances/A1/pai-abc-1.png": "Crop | Soybean"},
        "k_values": [5, 10],
        "sample_pairs": None,
    }
    assert payload["instance_labels"]["instances/A1/pai-abc-1.png"] == "Crop | Soybean"
