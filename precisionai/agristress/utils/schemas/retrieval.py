# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""TypedDicts for retrieval payload assembly.

Each type mirrors the corresponding request schema in
``precisionai.agrieval.emb.schemas.evaluate`` so payloads can be posted
directly to the embedding evaluator without further transformation.
"""

from typing import Any, TypedDict


class MetadataGroupDict(TypedDict):
    """Payload form of a single metadata similarity group.

    Mirrors ``MetadataGroup`` in ``precisionai.agrieval.emb.schemas.evaluate``.

    Attributes
    ----------
    images : list[str]
        Exact embedding path keys belonging to this group.
    class_name : str
        Cluster identifier (e.g. ``"A1"``). Leading letters form the L1 label.
    attributes : dict[str, Any]
        Optional cluster attributes (plants, time_period, camera_source, …).
    """

    images: list[str]
    class_name: str
    attributes: dict[str, Any]


class I2IPayload(TypedDict):
    """Payload for ``POST /v1/embeddings/evaluate/image2image``.

    Mirrors ``EmbeddingEvaluateRequest`` in
    ``precisionai.agrieval.emb.schemas.evaluate``.

    Attributes
    ----------
    embeddings : dict[str, list[float]]
        Image path → L2-normalized float32 embedding.
    metadata : dict[str, MetadataGroupDict] or None
        Cluster similarity groups derived from ``image2image.json``.
        ``None`` when no embedding keys matched any cluster.
    dataset_root : str or None
        Dataset root prefix forwarded to the evaluator for label extraction.
    k_values : list[int]
        K cutoffs for nearest-neighbour metrics.
    sample_pairs : int or None
        Max random pairs for global pairwise statistics.
    """

    embeddings: dict[str, list[float]]
    metadata: dict[str, MetadataGroupDict] | None
    dataset_root: str | None
    k_values: list[int]
    sample_pairs: int | None


class P2IPayload(TypedDict):
    """Payload for ``POST /v1/embeddings/evaluate/plant2image``.

    Mirrors ``Plant2ImageRequest`` in
    ``precisionai.agrieval.emb.schemas.evaluate``.

    Attributes
    ----------
    embeddings : dict[str, list[float]]
        Combined image and instance path → L2-normalized embedding.
    instance_to_image : dict[str, list[str]]
        Parent image path → list of instance crop paths, filtered to paths
        present in *embeddings*.
    dataset_root : str or None
        Dataset root prefix forwarded to the evaluator.
    k_values : list[int]
        K cutoffs for nearest-neighbour metrics.
    sample_pairs : int or None
        Max random pairs for global pairwise statistics.
    """

    embeddings: dict[str, list[float]]
    instance_to_image: dict[str, list[str]]
    dataset_root: str | None
    k_values: list[int]
    sample_pairs: int | None


class P2PPayload(TypedDict):
    """Payload for ``POST /v1/embeddings/evaluate/plant2plant``.

    Mirrors ``Plant2PlantRequest`` in
    ``precisionai.agrieval.emb.schemas.evaluate``.

    Attributes
    ----------
    embeddings : dict[str, list[float]]
        Instance path → L2-normalized embedding.
    instance_labels : dict[str, str]
        Instance path → plant class label, filtered to paths present in
        *embeddings*.
    k_values : list[int]
        K cutoffs for nearest-neighbour metrics.
    sample_pairs : int or None
        Max random pairs for global pairwise statistics.
    """

    embeddings: dict[str, list[float]]
    instance_labels: dict[str, str]
    k_values: list[int]
    sample_pairs: int | None
