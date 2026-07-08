# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Validate a locally downloaded AgriStress-500 dataset.

Checks that all three JSON manifests are present and parseable, that file
counts match the AgriStress-500 V1.0.0 specification, that every image and
instance file referenced in the manifests exists on disk, and that the
manifests are internally consistent with one another.

Exit code is 0 when all checks pass, 1 when any check fails.

Run from the project root::

    python examples/validate_dataset.py --dataset /path/to/agristress
"""

import argparse
import sys
from pathlib import Path

from precisionai.agristress.utils import AgriStressDataset

# Expected counts for AgriStress-500 V1.0.0.
_EXPECTED_CLUSTERS: int = 67
_EXPECTED_IMAGES: int = 489
_EXPECTED_INSTANCES: int = 5_933

# Truncate long file lists in error output.
_MAX_LISTED: int = 10


def _ok(msg: str) -> None:
    print(f"  [ OK ]  {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL]  {msg}")


def _info(msg: str) -> None:
    print(f"          {msg}")


def _check_manifests(root: Path) -> AgriStressDataset | None:
    print("Manifests")
    try:
        ds = AgriStressDataset(root)
        _ok("image2image.json, plant2image.json, plant2plant.json -- all present and parseable")
        return ds
    except FileNotFoundError as err:
        _fail(str(err))
        print("\nValidation FAILED -- cannot continue without manifests.")
        return None
    except ValueError as err:
        _fail(f"Manifest parse error: {err}")
        print("\nValidation FAILED -- cannot continue without manifests.")
        return None


def _check_counts(ds: AgriStressDataset) -> tuple[bool, list[str], list[str]]:
    print("\nCounts (expected for AgriStress-500 V1.0.0)")
    passed = True
    images = ds.list_image_ids()
    instances = ds.list_instance_ids()

    for label, expected, actual in [
        ("L2 clusters", _EXPECTED_CLUSTERS, len(ds.list_clusters())),
        ("source images", _EXPECTED_IMAGES, len(images)),
        ("instance crops", _EXPECTED_INSTANCES, len(instances)),
    ]:
        if actual == expected:
            _ok(f"{label}: {actual:,}")
        else:
            _fail(f"{label}: expected {expected:,}, found {actual:,}")
            passed = False

    return passed, images, instances


def _check_missing(label: str, ids: list[str], paths: dict[str, Path]) -> bool:
    missing = [rel for rel, abs_path in paths.items() if not abs_path.exists()]
    if missing:
        _fail(f"{len(missing):,} file(s) missing from disk")
        for f in missing[:_MAX_LISTED]:
            _info(f)
        if len(missing) > _MAX_LISTED:
            _info(f"... and {len(missing) - _MAX_LISTED} more")
        return False
    _ok(f"All {len(ids):,} {label} present")
    return True


def _check_files(ds: AgriStressDataset, images: list[str], instances: list[str]) -> bool:
    print(f"\nImage files ({len(images):,})")
    img_ok = _check_missing("image files", images, ds.resolve_paths(images))
    print(f"\nInstance files ({len(instances):,})")
    inst_ok = _check_missing("instance files", instances, ds.resolve_paths(instances))
    return img_ok and inst_ok


def _check_consistency(ds: AgriStressDataset) -> bool:
    print("\nCross-manifest consistency")
    passed = True

    all_image_set = set(ds._image_to_cluster.keys())
    p2i_parents = set(ds._p2i.keys())
    orphan_parents = p2i_parents - all_image_set
    if orphan_parents:
        _fail(f"plant2image.json references {len(orphan_parents)} parent image(s) not present in image2image.json")
        for p in sorted(orphan_parents)[:_MAX_LISTED]:
            _info(p)
        passed = False
    else:
        _ok("All plant2image.json parent images appear in image2image.json")

    all_instance_set = set(ds._instance_to_parent.keys())
    p2p_instances = set(ds._p2p.keys())
    orphan_instances = p2p_instances - all_instance_set
    if orphan_instances:
        _fail(f"plant2plant.json references {len(orphan_instances)} instance(s) not present in plant2image.json")
        for inst in sorted(orphan_instances)[:_MAX_LISTED]:
            _info(inst)
        passed = False
    else:
        _ok("All plant2plant.json instances appear in plant2image.json")

    unlabeled = all_instance_set - p2p_instances
    if unlabeled:
        _fail(f"{len(unlabeled)} instance(s) in plant2image.json have no label in plant2plant.json")
        for inst in sorted(unlabeled)[:_MAX_LISTED]:
            _info(inst)
        passed = False
    else:
        _ok("All plant2image.json instances have a label in plant2plant.json")

    return passed


def validate(dataset_dir: str | Path) -> bool:
    """Validate the AgriStress-500 dataset at *dataset_dir*.

    Parameters
    ----------
    dataset_dir : str or Path
        Directory containing ``image2image.json``, ``plant2image.json``,
        ``plant2plant.json``, ``images/``, and ``instances/``.

    Returns
    -------
    bool
        ``True`` if every check passed, ``False`` otherwise.
    """
    root = Path(dataset_dir).resolve()
    print(f"Validating AgriStress-500 dataset at: {root}\n")

    ds = _check_manifests(root)
    if ds is None:
        return False

    passed, images, instances = _check_counts(ds)
    passed = _check_files(ds, images, instances) and passed
    passed = _check_consistency(ds) and passed

    print()
    if passed:
        print("Dataset is valid.")
    else:
        print("Validation FAILED -- see errors above.")

    return passed


def main() -> None:
    """Parse arguments and run dataset validation."""
    parser = argparse.ArgumentParser(description="Validate a locally downloaded AgriStress-500 V1.0.0 dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset directory (e.g. /path/to/agristress).",
    )
    args = parser.parse_args()

    if not validate(args.dataset):
        sys.exit(1)


if __name__ == "__main__":
    main()
