# ======================================================================
#  CONFIDENTIAL — © Precision AI. All Rights Reserved.
#
#  This source code and any accompanying documentation contain
#  confidential and proprietary information of Precision AI.
#
#  Unauthorized reproduction, disclosure, modification, or distribution
#  of this material is strictly prohibited and will be prosecuted to the
#  fullest extent of the law.
# ======================================================================
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
