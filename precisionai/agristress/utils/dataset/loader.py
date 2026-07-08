# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""AgriStress-500 dataset loader."""

import json
from pathlib import Path
from typing import Any

from precisionai.agristress.utils.schemas.dataset import (
    ImageAttributes,
    ImageRecord,
    InstanceRecord,
)
from precisionai.agristress.utils.schemas.retrieval import (
    I2IPayload,
    MetadataGroupDict,
    P2IPayload,
    P2PPayload,
)


class AgriStressDataset:
    """In-memory view of the AgriStress-500 dataset.

    Loads the three JSON manifests once on construction. All subsequent
    lookups and payload builders are pure in-memory operations with no
    further file I/O.

    Parameters
    ----------
    root : str or Path
        Directory containing ``image2image.json``, ``plant2image.json``,
        ``plant2plant.json``, ``images/``, and ``instances/``.

    Raises
    ------
    FileNotFoundError
        If any required JSON manifest is absent from *root*.

    Examples
    --------
    >>> ds = AgriStressDataset("/path/to/agristress")
    >>> records = ds.image_metadata(["images/A1/pai-FvearqMK.png"])
    >>> payload = ds.build_i2i_payload({"images/A1/pai-FvearqMK.png": [...]})
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

        raw_i2i = self._load_json("image2image.json")
        raw_p2i = self._load_json("plant2image.json")
        raw_p2p = self._load_json("plant2plant.json")

        self._i2i: dict[str, Any] = raw_i2i["metadata"]
        self._p2i: dict[str, list[str]] = raw_p2i["instance_to_image"]
        self._p2p: dict[str, str] = raw_p2p["instance_labels"]

        self._image_to_cluster: dict[str, str] = {
            img: cluster for cluster, group in self._i2i.items() for img in group["images"]
        }
        self._instance_to_parent: dict[str, str] = {
            instance: parent for parent, instances in self._p2i.items() for instance in instances
        }

    def _load_json(self, relative_path: str) -> dict[str, Any]:
        path = self._root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {path}")
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]

    # ── Embedding validation ───────────────────────────────────────────────────

    @staticmethod
    def _validate_embeddings_coverage(
        ids: list[str],
        embeddings: dict[str, list[float]] | None,
        id_label: str,
    ) -> None:
        """Enforce the all-or-nothing embedding contract.

        Parameters
        ----------
        ids : list[str]
            The requested IDs.
        embeddings : dict[str, list[float]] or None
            Caller-supplied embeddings, or ``None`` when no embeddings are
            expected.
        id_label : str
            Human-readable label for the ID type used in error messages
            (e.g. ``"image"`` or ``"instance"``).

        Raises
        ------
        ValueError
            If *embeddings* is provided but does not contain every requested ID.
        """
        if embeddings is None:
            return
        missing = [id_ for id_ in ids if id_ not in embeddings]
        if missing:
            raise ValueError(
                f"Embedding mismatch: embeddings were supplied but "
                f"{len(missing)} {id_label} ID(s) have no corresponding "
                f"embedding. Either supply embeddings for all IDs or omit "
                f"embeddings entirely. Missing: {missing!r}"
            )

    # ── Metadata lookups ───────────────────────────────────────────────────────

    def image_metadata(
        self,
        image_ids: list[str],
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[ImageRecord]:
        """Return metadata records for the given source image IDs.

        Parameters
        ----------
        image_ids : list[str]
            Relative image paths, e.g. ``"images/A1/pai-FvearqMK.png"``.
        embeddings : dict[str, list[float]] or None, optional
            When provided, every ID in *image_ids* must have a corresponding
            key in *embeddings*. Passing embeddings for only a subset raises
            ``ValueError``. Omit (or pass ``None``) to retrieve metadata only.

        Returns
        -------
        list[ImageRecord]
            One record per requested ID, in the same order. ``ImageRecord.embedding``
            is populated when *embeddings* is supplied, ``None`` otherwise.

        Raises
        ------
        KeyError
            If an image ID is not found in the dataset.
        ValueError
            If *embeddings* is provided but does not cover all requested IDs.
        """
        self._validate_embeddings_coverage(image_ids, embeddings, "image")

        records: list[ImageRecord] = []
        for image_id in image_ids:
            cluster = self._image_to_cluster.get(image_id)
            if cluster is None:
                raise KeyError(f"Image ID not found in dataset: {image_id!r}")
            group = self._i2i[cluster]
            attrs = group["attributes"]
            records.append(
                ImageRecord(
                    image_id=image_id,
                    cluster=cluster,
                    attributes=ImageAttributes(
                        plants=attrs.get("plants", []),
                        time_period=attrs.get("time_period"),
                        camera_model=attrs.get("camera_model", ""),
                        camera_source=attrs.get("camera_source", ""),
                        collecting_date=attrs.get("collecting_date"),
                    ),
                    instances=self._p2i.get(image_id, []),
                    embedding=embeddings[image_id] if embeddings is not None else None,
                )
            )
        return records

    def instance_metadata(
        self,
        instance_ids: list[str],
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[InstanceRecord]:
        """Return metadata records for the given instance crop IDs.

        Parameters
        ----------
        instance_ids : list[str]
            Relative instance crop paths,
            e.g. ``"instances/A2/pai-wSqJ3Pkp-1.png"``.
        embeddings : dict[str, list[float]] or None, optional
            When provided, every ID in *instance_ids* must have a corresponding
            key in *embeddings*. Passing embeddings for only a subset raises
            ``ValueError``. Omit (or pass ``None``) to retrieve metadata only.

        Returns
        -------
        list[InstanceRecord]
            One record per requested ID, in the same order.
            ``InstanceRecord.embedding`` is populated when *embeddings* is
            supplied, ``None`` otherwise.

        Raises
        ------
        KeyError
            If an instance ID is not found in the dataset.
        ValueError
            If *embeddings* is provided but does not cover all requested IDs.
        """
        self._validate_embeddings_coverage(instance_ids, embeddings, "instance")

        records: list[InstanceRecord] = []
        for instance_id in instance_ids:
            label = self._p2p.get(instance_id)
            if label is None:
                raise KeyError(f"Instance ID not found in dataset: {instance_id!r}")
            parent = self._instance_to_parent.get(instance_id)
            if parent is None:
                raise KeyError(f"Instance has no parent image: {instance_id!r}")
            records.append(
                InstanceRecord(
                    instance_id=instance_id,
                    label=label,
                    parent_image=parent,
                    embedding=embeddings[instance_id] if embeddings is not None else None,
                )
            )
        return records

    # ── Enumeration and filtering ──────────────────────────────────────────────

    def list_clusters(self) -> list[str]:
        """Return all L2 cluster names in alphabetical order.

        Returns
        -------
        list[str]
            Sorted list of L2 cluster identifiers (e.g. ``["A1", "A2", "B1", …]``).

        Examples
        --------
        >>> ds.list_clusters()[:3]
        ['A1', 'A2', 'A3']
        """
        return sorted(self._i2i.keys())

    def list_image_ids(self, cluster: str | None = None) -> list[str]:
        """Return all image IDs in the dataset, optionally filtered by cluster.

        Parameters
        ----------
        cluster : str or None, optional
            L2 cluster name (e.g. ``"A1"``). When provided, only images in
            that cluster are returned.

        Returns
        -------
        list[str]
            Relative image paths as they appear in the manifests
            (e.g. ``"images/A1/pai-FvearqMK.png"``).

        Raises
        ------
        KeyError
            If *cluster* is specified but does not exist in the dataset.

        Examples
        --------
        >>> all_ids = ds.list_image_ids()
        >>> a1_ids = ds.list_image_ids(cluster="A1")
        """
        if cluster is not None:
            group = self._i2i.get(cluster)
            if group is None:
                raise KeyError(f"Cluster not found in dataset: {cluster!r}")
            return list(group["images"])
        return list(self._image_to_cluster.keys())

    def list_instance_ids(self, label: str | None = None) -> list[str]:
        """Return all instance IDs in the dataset, optionally filtered by label.

        Parameters
        ----------
        label : str or None, optional
            Exact plant class label (e.g. ``"Crop | Soybean"``). When
            provided, only instances with that label are returned. An
            unrecognized label returns an empty list.

        Returns
        -------
        list[str]
            Relative instance crop paths as they appear in the manifests
            (e.g. ``"instances/A1/pai-FvearqMK-1.png"``).

        Examples
        --------
        >>> all_ids = ds.list_instance_ids()
        >>> soy_ids = ds.list_instance_ids(label="Crop | Soybean")
        """
        if label is not None:
            return [inst for inst, lbl in self._p2p.items() if lbl == label]
        return list(self._p2p.keys())

    def filter_images_by_attribute(self, **kwargs: Any) -> list[str]:
        """Return image IDs from clusters that match all specified attributes.

        Attribute values are compared per-type: for scalar JSON attributes
        (``time_period``, ``camera_model``, ``camera_source``,
        ``collecting_date``) the filter value must equal the attribute exactly,
        including ``None``. For list attributes (``plants``) a string filter
        value must be present somewhere in the list.

        Parameters
        ----------
        **kwargs : Any
            Attribute name → expected value pairs. All conditions must be
            satisfied (logical AND). Unrecognized attribute names silently
            produce no matches.

        Returns
        -------
        list[str]
            Relative image paths for all images whose cluster matches every
            filter condition.

        Examples
        --------
        >>> drone_ids = ds.filter_images_by_attribute(camera_source="drone")
        >>> noon_drone = ds.filter_images_by_attribute(camera_source="drone", time_period="noon")
        >>> soy_ids = ds.filter_images_by_attribute(plants="Crop | Soybean")
        """
        result: list[str] = []
        for group in self._i2i.values():
            attrs = group.get("attributes", {})
            if all(self._attr_matches(attrs.get(key), value) for key, value in kwargs.items()):
                result.extend(group["images"])
        return result

    @staticmethod
    def _attr_matches(attr_val: Any, filter_val: Any) -> bool:
        if isinstance(attr_val, list):
            return filter_val in attr_val
        return attr_val == filter_val  # type: ignore[no-any-return]

    def resolve_paths(self, ids: list[str]) -> dict[str, Path]:
        """Map relative dataset IDs to absolute file paths on disk.

        Resolves each ID against the dataset root supplied at construction.
        Both image and instance IDs are accepted. The returned paths are not
        validated for existence — call ``Path.exists()`` if needed.

        Parameters
        ----------
        ids : list[str]
            Relative paths as returned by ``list_image_ids()``,
            ``list_instance_ids()``, or the manifest keys directly
            (e.g. ``"images/A1/pai-FvearqMK.png"``).

        Returns
        -------
        dict[str, Path]
            Mapping of the original relative ID to its resolved absolute
            ``Path`` under the dataset root.

        Examples
        --------
        >>> paths = ds.resolve_paths(ds.list_image_ids(cluster="A1"))
        >>> for rel_id, abs_path in paths.items():
        ...     img = Image.open(abs_path)
        """
        return {id_: (self._root / id_).resolve() for id_ in ids}

    # ── Retrieval payload builders ─────────────────────────────────────────────

    def build_i2i_payload(
        self,
        embeddings: dict[str, list[float]],
        *,
        dataset_root: str | None = None,
        k_values: list[int] | None = None,
        sample_pairs: int | None = 1_000_000,
    ) -> I2IPayload:
        """Assemble a payload for the Image→Image evaluator endpoint.

        Builds the ``metadata`` field from ``image2image.json``, including only
        clusters that have at least one image present in *embeddings*.

        Parameters
        ----------
        embeddings : dict[str, list[float]]
            Image path → L2-normalized embedding. Keys must be relative paths
            matching the dataset (e.g. ``"images/A1/pai-FvearqMK.png"``).
        dataset_root : str or None, optional
            Dataset root prefix forwarded to the evaluator for path-based label
            extraction.
        k_values : list[int] or None, optional
            K cutoffs for nearest-neighbour metrics. Defaults to
            ``[5, 10, 20]``.
        sample_pairs : int or None, optional
            Max random pairs for global pairwise statistics. ``None`` computes
            exact pairwise stats (slow for large N).

        Returns
        -------
        I2IPayload
            Ready-to-POST payload matching ``EmbeddingEvaluateRequest``.
            ``metadata`` is ``None`` when no embedding key matches any cluster.
        """
        _k = [5, 10, 20] if k_values is None else k_values
        embedding_keys = set(embeddings)

        metadata: dict[str, MetadataGroupDict] = {}
        for cluster, group in self._i2i.items():
            matched = [img for img in group["images"] if img in embedding_keys]
            if not matched:
                continue
            metadata[cluster] = MetadataGroupDict(
                images=matched,
                class_name=group["class_name"],
                attributes=group.get("attributes", {}),
            )

        return I2IPayload(
            embeddings=embeddings,
            metadata=metadata if metadata else None,
            dataset_root=dataset_root,
            k_values=_k,
            sample_pairs=sample_pairs,
        )

    def build_p2i_payload(
        self,
        embeddings: dict[str, list[float]],
        *,
        dataset_root: str | None = None,
        k_values: list[int] | None = None,
        sample_pairs: int | None = 1_000_000,
    ) -> P2IPayload:
        """Assemble a payload for the Plant→Image evaluator endpoint.

        Parameters
        ----------
        embeddings : dict[str, list[float]]
            Combined image and instance path → L2-normalized embedding. Must
            include embeddings for both parent images and their instance crops,
            as required by ``Plant2ImageRequest``.
        dataset_root : str or None, optional
            Dataset root prefix forwarded to the evaluator.
        k_values : list[int] or None, optional
            K cutoffs for nearest-neighbour metrics. Defaults to
            ``[5, 10, 20]``.
        sample_pairs : int or None, optional
            Max random pairs for global pairwise statistics.

        Returns
        -------
        P2IPayload
            Ready-to-POST payload matching ``Plant2ImageRequest``.
            ``instance_to_image`` contains only entries whose parent image and
            at least one instance crop are both present in *embeddings*.
        """
        _k = [5, 10, 20] if k_values is None else k_values
        embedding_keys = set(embeddings)

        instance_to_image: dict[str, list[str]] = {}
        for parent, instances in self._p2i.items():
            if parent not in embedding_keys:
                continue
            matched = [inst for inst in instances if inst in embedding_keys]
            if matched:
                instance_to_image[parent] = matched

        return P2IPayload(
            embeddings=embeddings,
            instance_to_image=instance_to_image,
            dataset_root=dataset_root,
            k_values=_k,
            sample_pairs=sample_pairs,
        )

    def build_p2p_payload(
        self,
        embeddings: dict[str, list[float]],
        *,
        k_values: list[int] | None = None,
        sample_pairs: int | None = 1_000_000,
    ) -> P2PPayload:
        """Assemble a payload for the Plant→Plant evaluator endpoint.

        Parameters
        ----------
        embeddings : dict[str, list[float]]
            Instance path → L2-normalized embedding. Keys must be relative
            instance crop paths (e.g. ``"instances/A2/pai-abc-1.png"``).
        k_values : list[int] or None, optional
            K cutoffs for nearest-neighbour metrics. Defaults to
            ``[5, 10, 20]``.
        sample_pairs : int or None, optional
            Max random pairs for global pairwise statistics.

        Returns
        -------
        P2PPayload
            Ready-to-POST payload matching ``Plant2PlantRequest``.
            ``instance_labels`` contains only entries present in *embeddings*.
        """
        _k = [5, 10, 20] if k_values is None else k_values
        embedding_keys = set(embeddings)

        instance_labels = {inst: label for inst, label in self._p2p.items() if inst in embedding_keys}

        return P2PPayload(
            embeddings=embeddings,
            instance_labels=instance_labels,
            k_values=_k,
            sample_pairs=sample_pairs,
        )
