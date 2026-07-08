# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dataset.loader.AgriStressDataset."""

import json
from pathlib import Path

import pytest

from precisionai.agristress.utils.dataset.loader import AgriStressDataset
from precisionai.agristress.utils.schemas.dataset import ImageRecord, InstanceRecord

# ── Init ──────────────────────────────────────────────────────────────────────


def test_init_loads_all_manifests(dataset: AgriStressDataset) -> None:
    assert len(dataset._i2i) == 2
    assert len(dataset._p2i) == 2
    assert len(dataset._p2p) == 3


def test_init_builds_image_to_cluster_index(dataset: AgriStressDataset) -> None:
    assert dataset._image_to_cluster["images/A1/pai-abc.png"] == "A1"
    assert dataset._image_to_cluster["images/B1/pai-ghi.png"] == "B1"


def test_init_builds_instance_to_parent_index(dataset: AgriStressDataset) -> None:
    assert dataset._instance_to_parent["instances/A1/pai-abc-1.png"] == "images/A1/pai-abc.png"


def test_init_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"image2image\.json"):
        AgriStressDataset(tmp_path)


# ── image_metadata ─────────────────────────────────────────────────────────────


def test_image_metadata_returns_record(dataset: AgriStressDataset) -> None:
    records = dataset.image_metadata(["images/A1/pai-abc.png"])
    assert len(records) == 1
    assert isinstance(records[0], ImageRecord)


def test_image_metadata_correct_fields(dataset: AgriStressDataset) -> None:
    record = dataset.image_metadata(["images/A1/pai-abc.png"])[0]
    assert record.image_id == "images/A1/pai-abc.png"
    assert record.cluster == "A1"
    assert record.attributes.camera_source == "drone"
    assert record.attributes.time_period == "noon"
    assert "instances/A1/pai-abc-1.png" in record.instances


def test_image_metadata_no_embeddings_gives_none(dataset: AgriStressDataset) -> None:
    record = dataset.image_metadata(["images/A1/pai-abc.png"])[0]
    assert record.embedding is None


def test_image_metadata_with_embeddings_populates_field(dataset: AgriStressDataset) -> None:
    emb = {"images/A1/pai-abc.png": [0.1, 0.9]}
    record = dataset.image_metadata(["images/A1/pai-abc.png"], embeddings=emb)[0]
    assert record.embedding == [0.1, 0.9]


def test_image_metadata_multiple_ids(dataset: AgriStressDataset) -> None:
    records = dataset.image_metadata(["images/A1/pai-abc.png", "images/A1/pai-def.png"])
    assert len(records) == 2
    assert records[0].image_id == "images/A1/pai-abc.png"
    assert records[1].image_id == "images/A1/pai-def.png"


def test_image_metadata_unknown_id_raises(dataset: AgriStressDataset) -> None:
    with pytest.raises(KeyError, match="not found"):
        dataset.image_metadata(["images/Z9/unknown.png"])


def test_image_metadata_partial_embeddings_raises(dataset: AgriStressDataset) -> None:
    emb = {"images/A1/pai-abc.png": [0.1, 0.9]}
    with pytest.raises(ValueError, match="Embedding mismatch"):
        dataset.image_metadata(
            ["images/A1/pai-abc.png", "images/A1/pai-def.png"],
            embeddings=emb,
        )


def test_image_metadata_all_embeddings_no_error(dataset: AgriStressDataset) -> None:
    emb = {
        "images/A1/pai-abc.png": [0.1, 0.9],
        "images/A1/pai-def.png": [0.2, 0.8],
    }
    records = dataset.image_metadata(
        ["images/A1/pai-abc.png", "images/A1/pai-def.png"],
        embeddings=emb,
    )
    assert all(r.embedding is not None for r in records)


# ── instance_metadata ──────────────────────────────────────────────────────────


def test_instance_metadata_returns_record(dataset: AgriStressDataset) -> None:
    records = dataset.instance_metadata(["instances/A1/pai-abc-1.png"])
    assert len(records) == 1
    assert isinstance(records[0], InstanceRecord)


def test_instance_metadata_correct_fields(dataset: AgriStressDataset) -> None:
    record = dataset.instance_metadata(["instances/A1/pai-abc-1.png"])[0]
    assert record.instance_id == "instances/A1/pai-abc-1.png"
    assert record.label == "Crop | Soybean"
    assert record.parent_image == "images/A1/pai-abc.png"


def test_instance_metadata_no_embeddings_gives_none(dataset: AgriStressDataset) -> None:
    record = dataset.instance_metadata(["instances/A1/pai-abc-1.png"])[0]
    assert record.embedding is None


def test_instance_metadata_with_embeddings_populates_field(dataset: AgriStressDataset) -> None:
    emb = {"instances/A1/pai-abc-1.png": [0.5, 0.5]}
    record = dataset.instance_metadata(["instances/A1/pai-abc-1.png"], embeddings=emb)[0]
    assert record.embedding == [0.5, 0.5]


def test_instance_metadata_unknown_id_raises(dataset: AgriStressDataset) -> None:
    with pytest.raises(KeyError, match="not found"):
        dataset.instance_metadata(["instances/Z9/unknown.png"])


def test_instance_metadata_partial_embeddings_raises(dataset: AgriStressDataset) -> None:
    emb = {"instances/A1/pai-abc-1.png": [0.5, 0.5]}
    with pytest.raises(ValueError, match="Embedding mismatch"):
        dataset.instance_metadata(
            ["instances/A1/pai-abc-1.png", "instances/A1/pai-abc-2.png"],
            embeddings=emb,
        )


def test_instance_metadata_orphan_raises(dataset_root: Path) -> None:
    # instance exists in plant2plant.json but has no entry in plant2image.json
    p2p_path = dataset_root / "plant2plant.json"
    raw = json.loads(p2p_path.read_text(encoding="utf-8"))
    raw["instance_labels"]["instances/A1/orphan.png"] = "Crop | Soybean"
    p2p_path.write_text(json.dumps(raw), encoding="utf-8")

    ds = AgriStressDataset(dataset_root)
    with pytest.raises(KeyError, match="no parent image"):
        ds.instance_metadata(["instances/A1/orphan.png"])


# ── build_i2i_payload ──────────────────────────────────────────────────────────


def test_build_i2i_payload_structure(dataset: AgriStressDataset) -> None:
    embeddings = {"images/A1/pai-abc.png": [0.1, 0.9]}
    payload = dataset.build_i2i_payload(embeddings)
    assert payload["embeddings"] is embeddings
    assert payload["k_values"] == [5, 10, 20]
    assert payload["sample_pairs"] == 1_000_000
    assert payload["dataset_root"] is None


def test_build_i2i_payload_metadata_present(dataset: AgriStressDataset) -> None:
    embeddings = {"images/A1/pai-abc.png": [0.1, 0.9]}
    payload = dataset.build_i2i_payload(embeddings)
    assert payload["metadata"] is not None
    assert "A1" in payload["metadata"]
    assert payload["metadata"]["A1"]["class_name"] == "A1"
    assert "images/A1/pai-abc.png" in payload["metadata"]["A1"]["images"]


def test_build_i2i_payload_filters_unmatched_clusters(dataset: AgriStressDataset) -> None:
    embeddings = {"images/A1/pai-abc.png": [0.1, 0.9]}
    payload = dataset.build_i2i_payload(embeddings)
    assert "B1" not in (payload["metadata"] or {})


def test_build_i2i_payload_no_matches_gives_none_metadata(dataset: AgriStressDataset) -> None:
    payload = dataset.build_i2i_payload({"nonexistent/path.png": [0.5]})
    assert payload["metadata"] is None


def test_build_i2i_payload_custom_options(dataset: AgriStressDataset) -> None:
    embeddings = {"images/A1/pai-abc.png": [0.1]}
    payload = dataset.build_i2i_payload(embeddings, dataset_root="/data", k_values=[1, 5], sample_pairs=None)
    assert payload["dataset_root"] == "/data"
    assert payload["k_values"] == [1, 5]
    assert payload["sample_pairs"] is None


# ── build_p2i_payload ──────────────────────────────────────────────────────────


def test_build_p2i_payload_structure(dataset: AgriStressDataset) -> None:
    embeddings = {
        "images/A1/pai-abc.png": [0.1, 0.9],
        "instances/A1/pai-abc-1.png": [0.3, 0.7],
        "instances/A1/pai-abc-2.png": [0.4, 0.6],
    }
    payload = dataset.build_p2i_payload(embeddings)
    assert "images/A1/pai-abc.png" in payload["instance_to_image"]
    assert "instances/A1/pai-abc-1.png" in payload["instance_to_image"]["images/A1/pai-abc.png"]


def test_build_p2i_payload_filters_missing_parent(dataset: AgriStressDataset) -> None:
    embeddings = {"instances/A1/pai-abc-1.png": [0.5, 0.5]}
    payload = dataset.build_p2i_payload(embeddings)
    assert payload["instance_to_image"] == {}


def test_build_p2i_payload_filters_missing_instances(dataset: AgriStressDataset) -> None:
    embeddings = {"images/A1/pai-abc.png": [0.1, 0.9]}
    payload = dataset.build_p2i_payload(embeddings)
    assert payload["instance_to_image"] == {}


def test_build_p2i_payload_partial_instances(dataset: AgriStressDataset) -> None:
    embeddings = {
        "images/A1/pai-abc.png": [0.1, 0.9],
        "instances/A1/pai-abc-1.png": [0.3, 0.7],
    }
    payload = dataset.build_p2i_payload(embeddings)
    assert payload["instance_to_image"]["images/A1/pai-abc.png"] == ["instances/A1/pai-abc-1.png"]


# ── build_p2p_payload ──────────────────────────────────────────────────────────


def test_build_p2p_payload_structure(dataset: AgriStressDataset) -> None:
    embeddings = {
        "instances/A1/pai-abc-1.png": [0.1, 0.9],
        "instances/A1/pai-def-1.png": [0.3, 0.7],
    }
    payload = dataset.build_p2p_payload(embeddings)
    assert payload["instance_labels"]["instances/A1/pai-abc-1.png"] == "Crop | Soybean"
    assert payload["instance_labels"]["instances/A1/pai-def-1.png"] == "Weed | Water Hemp"


def test_build_p2p_payload_filters_unknown_instances(dataset: AgriStressDataset) -> None:
    payload = dataset.build_p2p_payload({"nonexistent.png": [0.5]})
    assert payload["instance_labels"] == {}


def test_build_p2p_payload_custom_k_values(dataset: AgriStressDataset) -> None:
    payload = dataset.build_p2p_payload({"instances/A1/pai-abc-1.png": [0.1]}, k_values=[1, 3])
    assert payload["k_values"] == [1, 3]


def test_build_p2p_payload_sample_pairs_none(dataset: AgriStressDataset) -> None:
    payload = dataset.build_p2p_payload({"instances/A1/pai-abc-1.png": [0.1]}, sample_pairs=None)
    assert payload["sample_pairs"] is None


# ── list_clusters ──────────────────────────────────────────────────────────────


def test_list_clusters_returns_sorted(dataset: AgriStressDataset) -> None:
    assert dataset.list_clusters() == ["A1", "B1"]


def test_list_clusters_all_present(dataset: AgriStressDataset) -> None:
    clusters = dataset.list_clusters()
    assert "A1" in clusters
    assert "B1" in clusters


# ── list_image_ids ─────────────────────────────────────────────────────────────


def test_list_image_ids_all(dataset: AgriStressDataset) -> None:
    ids = dataset.list_image_ids()
    assert len(ids) == 3
    assert "images/A1/pai-abc.png" in ids
    assert "images/B1/pai-ghi.png" in ids


def test_list_image_ids_by_cluster(dataset: AgriStressDataset) -> None:
    ids = dataset.list_image_ids(cluster="A1")
    assert set(ids) == {"images/A1/pai-abc.png", "images/A1/pai-def.png"}


def test_list_image_ids_single_image_cluster(dataset: AgriStressDataset) -> None:
    ids = dataset.list_image_ids(cluster="B1")
    assert ids == ["images/B1/pai-ghi.png"]


def test_list_image_ids_unknown_cluster_raises(dataset: AgriStressDataset) -> None:
    with pytest.raises(KeyError, match="Z9"):
        dataset.list_image_ids(cluster="Z9")


# ── list_instance_ids ──────────────────────────────────────────────────────────


def test_list_instance_ids_all(dataset: AgriStressDataset) -> None:
    ids = dataset.list_instance_ids()
    assert len(ids) == 3
    assert "instances/A1/pai-abc-1.png" in ids


def test_list_instance_ids_by_label(dataset: AgriStressDataset) -> None:
    ids = dataset.list_instance_ids(label="Crop | Soybean")
    assert set(ids) == {"instances/A1/pai-abc-1.png", "instances/A1/pai-abc-2.png"}


def test_list_instance_ids_single_match(dataset: AgriStressDataset) -> None:
    ids = dataset.list_instance_ids(label="Weed | Water Hemp")
    assert ids == ["instances/A1/pai-def-1.png"]


def test_list_instance_ids_unknown_label_returns_empty(dataset: AgriStressDataset) -> None:
    assert dataset.list_instance_ids(label="Unknown | Species") == []


# ── filter_images_by_attribute ─────────────────────────────────────────────────


def test_filter_images_by_attribute_scalar(dataset: AgriStressDataset) -> None:
    ids = dataset.filter_images_by_attribute(camera_source="drone")
    assert set(ids) == {"images/A1/pai-abc.png", "images/A1/pai-def.png"}


def test_filter_images_by_attribute_list_value(dataset: AgriStressDataset) -> None:
    ids = dataset.filter_images_by_attribute(plants="Crop | Soybean")
    assert set(ids) == {"images/A1/pai-abc.png", "images/A1/pai-def.png"}


def test_filter_images_by_attribute_no_match(dataset: AgriStressDataset) -> None:
    ids = dataset.filter_images_by_attribute(camera_source="satellite")
    assert ids == []


def test_filter_images_by_attribute_multi_attr(dataset: AgriStressDataset) -> None:
    ids = dataset.filter_images_by_attribute(camera_source="drone", time_period="noon")
    assert set(ids) == {"images/A1/pai-abc.png", "images/A1/pai-def.png"}


def test_filter_images_by_attribute_multi_attr_no_match(dataset: AgriStressDataset) -> None:
    ids = dataset.filter_images_by_attribute(camera_source="drone", time_period="night")
    assert ids == []


def test_filter_images_by_attribute_none_value(dataset: AgriStressDataset) -> None:
    ids = dataset.filter_images_by_attribute(time_period=None)
    assert set(ids) == {"images/B1/pai-ghi.png"}


# ── resolve_paths ──────────────────────────────────────────────────────────────


def test_resolve_paths_returns_absolute(dataset: AgriStressDataset, dataset_root: Path) -> None:
    paths = dataset.resolve_paths(["images/A1/pai-abc.png"])
    assert "images/A1/pai-abc.png" in paths
    assert paths["images/A1/pai-abc.png"].is_absolute()


def test_resolve_paths_correct_location(dataset: AgriStressDataset, dataset_root: Path) -> None:
    paths = dataset.resolve_paths(["images/A1/pai-abc.png"])
    assert paths["images/A1/pai-abc.png"] == (dataset_root / "images/A1/pai-abc.png").resolve()


def test_resolve_paths_multiple_ids(dataset: AgriStressDataset) -> None:
    ids = ["images/A1/pai-abc.png", "instances/A1/pai-abc-1.png"]
    paths = dataset.resolve_paths(ids)
    assert set(paths.keys()) == set(ids)


def test_resolve_paths_empty_list(dataset: AgriStressDataset) -> None:
    assert dataset.resolve_paths([]) == {}
