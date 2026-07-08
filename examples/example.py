# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Standalone example: load the dataset and build evaluator payloads.

Demonstrates the full workflow — enumerating IDs, resolving file paths,
normalising embeddings, and assembling all three retrieval payloads for use
with the agrieval embedding evaluator.

Run from the project root::

    python examples/example.py --dataset /path/to/agristress
"""

import argparse
import math

from precisionai.agristress.utils import (
    AgriStressDataset,
    normalize_embeddings,
    validate_embeddings,
)


def _stub(ids: list[str], dim: int = 512) -> dict[str, list[float]]:
    """Build unit-norm stub vectors — replace with real model output in production."""
    return {id_: [1.0] + [0.0] * (dim - 1) for id_ in ids}


def main() -> None:
    """Run the standalone dataset usage example."""
    parser = argparse.ArgumentParser(description="PAI AgriStress Utils — usage example")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset directory (e.g. /path/to/agristress).",
    )
    args = parser.parse_args()

    ds = AgriStressDataset(args.dataset)

    # ── Enumerate the dataset ─────────────────────────────────────────────────
    clusters = ds.list_clusters()
    all_image_ids = ds.list_image_ids()
    all_instance_ids = ds.list_instance_ids()
    print(f"Dataset  : {len(clusters)} clusters | {len(all_image_ids)} images | {len(all_instance_ids)} instances")

    # Filter by attribute or label.
    drone_ids = ds.filter_images_by_attribute(camera_source="drone")
    soy_ids = ds.list_instance_ids(label="Crop | Soybean")
    print(f"Filtered : {len(drone_ids)} drone images | {len(soy_ids)} soybean instances")

    # ── Resolve IDs to absolute paths for image loading ───────────────────────
    sample_image_ids = all_image_ids[:2]
    paths = ds.resolve_paths(sample_image_ids)
    for rel_id, abs_path in paths.items():
        print(f"Resolved : {rel_id} -> {abs_path}  exists={abs_path.exists()}")

    # ── Metadata lookup ───────────────────────────────────────────────────────
    image_records = ds.image_metadata(sample_image_ids)
    for rec in image_records:
        print(
            f"Image    : {rec.image_id}  cluster={rec.cluster}  "
            f"camera={rec.attributes.camera_source}  instances={len(rec.instances)}"
        )

    sample_instance_ids = all_instance_ids[:2]
    instance_records = ds.instance_metadata(sample_instance_ids)
    for rec in instance_records:
        print(f"Instance : {rec.instance_id}  label={rec.label}  parent={rec.parent_image}")

    # ── Embedding normalisation ───────────────────────────────────────────────
    # In real usage replace _stub() with your model's encode() output.
    raw_image_emb = _stub(sample_image_ids)
    normed_image_emb = normalize_embeddings(raw_image_emb)
    validate_embeddings(normed_image_emb)
    norm = math.sqrt(sum(x * x for x in next(iter(normed_image_emb.values()))))
    print(f"\nNorm     : {norm:.6f} (should be 1.0)")

    # ── Route 1: Image -> Image (I2I) ──────────────────────────────────────────
    # One embedding per source image. Clusters from image2image.json are used
    # as ground-truth similarity groups.
    i2i = ds.build_i2i_payload(normed_image_emb)
    print(
        f"\nI2I      : {len(i2i['embeddings'])} embeddings | "
        f"{len(i2i['metadata'] or {})} cluster groups | "
        f"k={i2i['k_values']}"
    )
    # POST to: http://localhost:8000/v1/embeddings/evaluate/image2image

    # ── Route 2: Plant -> Image (P2I) ──────────────────────────────────────────
    # Embeddings must include BOTH the parent source images AND their instance
    # crops. The payload wires each parent to its crops as explicit positives.
    #
    # Collect instances belonging to the sampled parent images.
    instance_ids_for_parents = [inst for rec in image_records for inst in rec.instances]
    combined_ids = sample_image_ids + instance_ids_for_parents
    raw_combined = _stub(combined_ids)
    normed_combined = normalize_embeddings(raw_combined)

    p2i = ds.build_p2i_payload(normed_combined)
    group_sizes = {parent: len(insts) for parent, insts in p2i["instance_to_image"].items()}
    print(f"\nP2I      : {len(p2i['embeddings'])} embeddings | {len(p2i['instance_to_image'])} parent->instance groups")
    for parent, n_insts in group_sizes.items():
        print(f"           {parent} -> {n_insts} instances")
    # POST to: http://localhost:8000/v1/embeddings/evaluate/plant2image

    # ── Route 3: Plant -> Plant (P2P) ──────────────────────────────────────────
    # One embedding per instance crop. Class labels from plant2plant.json are
    # used as ground-truth similarity groups.
    raw_inst_emb = _stub(sample_instance_ids)
    normed_inst_emb = normalize_embeddings(raw_inst_emb)

    p2p = ds.build_p2p_payload(normed_inst_emb)
    print(
        f"\nP2P      : {len(p2p['embeddings'])} embeddings | "
        f"{len(p2p['instance_labels'])} labeled instances | "
        f"k={p2p['k_values']}"
    )
    for inst, label in p2p["instance_labels"].items():
        print(f"           {inst} -> {label}")
    # POST to: http://localhost:8000/v1/embeddings/evaluate/plant2plant


if __name__ == "__main__":
    main()
