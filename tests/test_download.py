# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dataset.download."""

from pathlib import Path
from unittest.mock import call, patch

import pytest

import precisionai.agribench.utils.dataset.download as dl_module
from precisionai.agribench.utils.dataset.download import _DEFAULT_REPO_ID, download_dataset


def test_download_calls_snapshot_with_correct_args(tmp_path: Path) -> None:
    with patch("precisionai.agribench.utils.dataset.download.snapshot_download") as mock_dl:
        download_dataset(tmp_path, repo_id="org/repo")
    assert mock_dl.call_args == call(
        repo_id="org/repo",
        repo_type="dataset",
        local_dir=str(tmp_path),
        ignore_patterns=["internal/*"],
    )


def test_download_uses_default_repo_id(tmp_path: Path) -> None:
    with patch("precisionai.agribench.utils.dataset.download.snapshot_download") as mock_dl:
        download_dataset(tmp_path)
    assert mock_dl.call_args.kwargs["repo_id"] == _DEFAULT_REPO_ID


def test_download_accepts_string_path(tmp_path: Path) -> None:
    with patch("precisionai.agribench.utils.dataset.download.snapshot_download"):
        result = download_dataset(str(tmp_path))
    assert isinstance(result, Path)


def test_download_returns_resolved_absolute_path(tmp_path: Path) -> None:
    with patch("precisionai.agribench.utils.dataset.download.snapshot_download"):
        result = download_dataset(tmp_path)
    assert result.is_absolute()


def test_download_raises_when_hf_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl_module, "snapshot_download", None)
    with pytest.raises(ImportError, match="huggingface_hub"):
        download_dataset(tmp_path)


def test_download_custom_repo_type(tmp_path: Path) -> None:
    with patch("precisionai.agribench.utils.dataset.download.snapshot_download") as mock_dl:
        download_dataset(tmp_path, repo_id="org/repo", repo_type="model")
    assert mock_dl.call_args.kwargs["repo_type"] == "model"
