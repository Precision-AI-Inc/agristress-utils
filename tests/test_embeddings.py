# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for embeddings.validate_embeddings and normalize_embeddings."""

import math

import pytest

from precisionai.agristress.utils.embeddings import normalize_embeddings, validate_embeddings

# ── validate_embeddings ────────────────────────────────────────────────────────


def test_validate_embeddings_unit_norm_passes() -> None:
    validate_embeddings({"a.png": [0.6, 0.8]})


def test_validate_embeddings_empty_dict_passes() -> None:
    validate_embeddings({})


def test_validate_embeddings_not_normalized_raises() -> None:
    with pytest.raises(ValueError, match="not L2-normalized"):
        validate_embeddings({"a.png": [3.0, 4.0]})


def test_validate_embeddings_error_includes_norm() -> None:
    with pytest.raises(ValueError, match=r"‖v‖₂ = 5\."):
        validate_embeddings({"a.png": [3.0, 4.0]})


def test_validate_embeddings_nan_raises() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_embeddings({"a.png": [float("nan"), 0.0]})


def test_validate_embeddings_inf_raises() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_embeddings({"a.png": [float("inf"), 0.0]})


def test_validate_embeddings_empty_vector_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_embeddings({"a.png": []})


def test_validate_embeddings_inconsistent_dims_raises() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        validate_embeddings({"a.png": [0.6, 0.8], "b.png": [1.0]})


def test_validate_embeddings_multiple_unit_vectors_passes() -> None:
    validate_embeddings({"a.png": [1.0, 0.0], "b.png": [0.0, 1.0]})


def test_validate_embeddings_within_tolerance_passes() -> None:
    # Norm 1.0009 is within the 1e-3 tolerance.
    scale = 1.0009
    validate_embeddings({"a.png": [scale, 0.0]})


# ── normalize_embeddings ───────────────────────────────────────────────────────


def test_normalize_embeddings_produces_unit_norm() -> None:
    result = normalize_embeddings({"a.png": [3.0, 4.0]})
    norm = math.sqrt(sum(x * x for x in result["a.png"]))
    assert abs(norm - 1.0) < 1e-6


def test_normalize_embeddings_correct_values() -> None:
    result = normalize_embeddings({"a.png": [3.0, 4.0]})
    assert abs(result["a.png"][0] - 0.6) < 1e-6
    assert abs(result["a.png"][1] - 0.8) < 1e-6


def test_normalize_embeddings_does_not_mutate_input() -> None:
    original = {"a.png": [3.0, 4.0]}
    result = normalize_embeddings(original)
    assert result is not original
    assert original["a.png"] == [3.0, 4.0]


def test_normalize_embeddings_already_unit_norm() -> None:
    result = normalize_embeddings({"a.png": [0.6, 0.8]})
    norm = math.sqrt(sum(x * x for x in result["a.png"]))
    assert abs(norm - 1.0) < 1e-6


def test_normalize_embeddings_multiple_keys() -> None:
    emb = {"a.png": [3.0, 4.0], "b.png": [1.0, 0.0]}
    result = normalize_embeddings(emb)
    assert set(result.keys()) == {"a.png", "b.png"}
    assert abs(result["b.png"][0] - 1.0) < 1e-6


def test_normalize_embeddings_zero_vector_raises() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        normalize_embeddings({"a.png": [0.0, 0.0]})


def test_normalize_embeddings_empty_dict() -> None:
    assert normalize_embeddings({}) == {}
