# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the test suite."""

import json
from pathlib import Path

import pytest

from precisionai.agribench.utils.dataset.loader import AgriBenchDataset

_I2I: dict = {
    "metadata": {
        "A1": {
            "class_name": "A1",
            "images": ["images/A1/pai-abc.png", "images/A1/pai-def.png"],
            "attributes": {
                "plants": ["Crop | Soybean"],
                "time_period": "noon",
                "camera_model": "HB-25000SBC",
                "camera_source": "drone",
                "collecting_date": "2022-06-22",
            },
        },
        "B1": {
            "class_name": "B1",
            "images": ["images/B1/pai-ghi.png"],
            "attributes": {
                "plants": ["Weed | Water Hemp"],
                "time_period": None,
                "camera_model": "nikon d610",
                "camera_source": "ground",
                "collecting_date": None,
            },
        },
    }
}

_P2I: dict = {
    "instance_to_image": {
        "images/A1/pai-abc.png": [
            "instances/A1/pai-abc-1.png",
            "instances/A1/pai-abc-2.png",
        ],
        "images/A1/pai-def.png": [
            "instances/A1/pai-def-1.png",
        ],
    }
}

_P2P: dict = {
    "instance_labels": {
        "instances/A1/pai-abc-1.png": "Crop | Soybean",
        "instances/A1/pai-abc-2.png": "Crop | Soybean",
        "instances/A1/pai-def-1.png": "Weed | Water Hemp",
    }
}


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Return a temporary directory containing the fixture dataset manifests.

    Mirrors the expected layout: manifests and ``images/`` / ``instances/``
    directories sit directly under the dataset root.
    """
    (tmp_path / "image2image.json").write_text(json.dumps(_I2I))
    (tmp_path / "plant2image.json").write_text(json.dumps(_P2I))
    (tmp_path / "plant2plant.json").write_text(json.dumps(_P2P))
    return tmp_path


@pytest.fixture
def dataset(dataset_root: Path) -> AgriBenchDataset:
    """Return an AgriBenchDataset loaded from the fixture manifests."""
    return AgriBenchDataset(dataset_root)
