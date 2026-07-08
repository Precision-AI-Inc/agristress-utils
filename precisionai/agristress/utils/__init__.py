# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""PAI AgriStress Utils — dataset utilities for AgriStress-500.

Package structure
-----------------
- ``precisionai.agristress.utils.dataset``  — HuggingFace download and in-memory loader
- ``precisionai.agristress.utils.schemas``  — Pydantic record models and retrieval payload TypedDicts
"""

from precisionai.agristress.utils.dataset.download import download_dataset
from precisionai.agristress.utils.dataset.loader import AgriStressDataset
from precisionai.agristress.utils.embeddings import normalize_embeddings, validate_embeddings

__all__ = [
    "AgriStressDataset",
    "download_dataset",
    "normalize_embeddings",
    "validate_embeddings",
]
