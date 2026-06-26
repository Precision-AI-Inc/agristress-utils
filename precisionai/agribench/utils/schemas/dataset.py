"""Pydantic models for AgriBench dataset records."""

from pydantic import BaseModel


class ImageAttributes(BaseModel):
    """Cluster-level attributes associated with a source image.

    Attributes
    ----------
    plants : list[str]
        Plant types present in the cluster (e.g. ``["Crop | Soybean", "Weed | Weed"]``).
    time_period : str or None
        Dominant lighting period (e.g. ``"ss_sr"``, ``"noon"``).
    camera_model : str
        Camera model identifier (e.g. ``"HB-25000SBC"``).
    camera_source : str
        Capture method (e.g. ``"drone"``, ``"ground"``).
    collecting_date : str or None
        ISO 8601 collection date, or ``None`` for older batches.
    """

    plants: list[str]
    time_period: str | None
    camera_model: str
    camera_source: str
    collecting_date: str | None


class ImageRecord(BaseModel):
    """Metadata for a single source image in the dataset.

    Attributes
    ----------
    image_id : str
        Relative path to the image (e.g. ``"images/A1/pai-FvearqMK.png"``).
    cluster : str
        L2 cluster identifier (e.g. ``"A1"``).
    attributes : ImageAttributes
        Cluster-level metadata attributes.
    instances : list[str]
        Relative paths to instance crops extracted from this image.
    embedding : list[float] or None
        L2-normalised embedding vector, present only when the caller supplied
        embeddings to the lookup method.
    """

    image_id: str
    cluster: str
    attributes: ImageAttributes
    instances: list[str]
    embedding: list[float] | None = None


class InstanceRecord(BaseModel):
    """Metadata for a single instance crop in the dataset.

    Attributes
    ----------
    instance_id : str
        Relative path to the instance crop
        (e.g. ``"instances/A2/pai-wSqJ3Pkp-1.png"``).
    label : str
        Plant class label (e.g. ``"Crop | Soybean"``).
    parent_image : str
        Relative path to the source image this crop was extracted from.
    embedding : list[float] or None
        L2-normalised embedding vector, present only when the caller supplied
        embeddings to the lookup method.
    """

    instance_id: str
    label: str
    parent_image: str
    embedding: list[float] | None = None
