# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Embedding validation and normalisation utilities.

Provides pre-flight checks and L2 normalisation so that embedding vectors
satisfy the contract required by the ``pai-agrieval-emb`` embedding evaluator
before payloads are posted.
"""

import math

_NORM_TOLERANCE: float = 1e-3


def validate_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Validate that all embedding vectors satisfy the evaluator's L2-norm contract.

    Checks that every vector is non-empty, shares the same dimension, contains
    only finite values, and is L2-normalized to within the tolerance required by
    the embedding evaluator (‖v‖₂ = 1.0 ± 1e-3).

    Parameters
    ----------
    embeddings : dict[str, list[float]]
        Mapping of path → embedding vector to validate.

    Raises
    ------
    ValueError
        If any vector is empty, has an inconsistent dimension, contains
        non-finite values, or deviates from unit norm beyond the tolerance.

    Examples
    --------
    >>> validate_embeddings({"images/A1/img.png": [0.6, 0.8]})
    """
    if not embeddings:
        return

    dims = {len(vec) for vec in embeddings.values()}
    if len(dims) > 1:
        raise ValueError(f"All embeddings must have the same dimension. Found: {sorted(dims)}")
    dim = next(iter(dims))
    if dim == 0:
        raise ValueError("Embedding vectors must not be empty.")

    for path, vec in embeddings.items():
        norm = math.sqrt(sum(x * x for x in vec))
        if math.isnan(norm) or math.isinf(norm):
            raise ValueError(f"Embedding for '{path}' contains non-finite values (NaN or inf).")
        if abs(norm - 1.0) > _NORM_TOLERANCE:
            raise ValueError(
                f"Embedding for '{path}' is not L2-normalized "
                f"(‖v‖₂ = {norm:.6f}, expected 1.0 ± {_NORM_TOLERANCE}). "
                "Call normalize_embeddings() before passing to the evaluator."
            )


def normalize_embeddings(embeddings: dict[str, list[float]]) -> dict[str, list[float]]:
    """Return a copy of *embeddings* with every vector L2-normalized.

    Each vector is divided by its Euclidean norm so that ‖v‖₂ = 1.0. The
    input dict is not mutated.

    Parameters
    ----------
    embeddings : dict[str, list[float]]
        Mapping of path → raw embedding vector.

    Returns
    -------
    dict[str, list[float]]
        New mapping with the same keys and L2-normalized vectors.

    Raises
    ------
    ValueError
        If any vector has zero magnitude and therefore cannot be normalized.

    Examples
    --------
    >>> normed = normalize_embeddings({"images/A1/img.png": [3.0, 4.0]})
    >>> normed["images/A1/img.png"]
    [0.6, 0.8]
    """
    result: dict[str, list[float]] = {}
    for path, vec in embeddings.items():
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-12:
            raise ValueError(f"Cannot normalize zero vector for '{path}'.")
        result[path] = [x / norm for x in vec]
    return result
