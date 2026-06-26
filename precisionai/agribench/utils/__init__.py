# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""PAI Agribench Utils — dataset utilities for AgriBench.

Package structure
-----------------
- ``precisionai.agribench.utils.dataset``  — HuggingFace download and in-memory loader
- ``precisionai.agribench.utils.schemas``  — Pydantic record models and retrieval payload TypedDicts
"""

from precisionai.agribench.utils.dataset.download import download_dataset
from precisionai.agribench.utils.dataset.loader import AgriBenchDataset
from precisionai.agribench.utils.embeddings import normalize_embeddings, validate_embeddings

__all__ = [
    "AgriBenchDataset",
    "download_dataset",
    "normalize_embeddings",
    "validate_embeddings",
]
