# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Hub dataset download utility."""

from pathlib import Path

try:
    from huggingface_hub import snapshot_download  # type: ignore[import]
except ImportError:
    snapshot_download = None  # type: ignore[assignment]

_DEFAULT_REPO_ID = "precisionaiinc/AgriBench"


def download_dataset(
    local_dir: str | Path,
    *,
    repo_id: str = _DEFAULT_REPO_ID,
    repo_type: str = "dataset",
) -> Path:
    """Download the AgriBench dataset from HuggingFace Hub.

    Downloads the full public dataset in a single call. Partial downloads
    (individual clusters or subsets) are not supported. The ``internal/``
    directory is excluded automatically and will not be present in the
    downloaded dataset.

    Parameters
    ----------
    local_dir : str or Path
        Destination directory for the downloaded dataset files.
    repo_id : str, optional
        HuggingFace repository ID.
        Defaults to ``"precisionaiinc/AgriBench"``.
    repo_type : str, optional
        HuggingFace repository type. Defaults to ``"dataset"``.

    Returns
    -------
    Path
        Resolved absolute path to the downloaded dataset root.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    """
    if snapshot_download is None:
        raise ImportError("huggingface_hub is required: pip install huggingface-hub") from None

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(Path(local_dir)),
        ignore_patterns=["internal/*"],
    )
    return Path(local_dir).resolve()
