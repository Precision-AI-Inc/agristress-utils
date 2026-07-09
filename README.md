<p align="center">
  <img src="https://raw.githubusercontent.com/Precision-AI-Inc/agristress-utils/main/docs/assets/logo.png" alt="Precision AI Logo" width="160"/>
</p>

# PAI AgriStress Utils

Dataset utilities for AgriStress-500 — download the dataset from HuggingFace, look up image and instance metadata, and assemble ready-to-POST payloads for the [`agrieval`](https://github.com/Precision-AI-Inc/agrieval) embedding evaluator.

[![PyPI](https://img.shields.io/pypi/v/precisionai-agristress-utils.svg?include_prereleases)](https://pypi.org/project/precisionai-agristress-utils/)
[![Python](https://img.shields.io/pypi/pyversions/precisionai-agristress-utils.svg?include_prereleases)](https://pypi.org/project/precisionai-agristress-utils/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Precision-AI-Inc/agristress-utils/blob/main/LICENSE.md)

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install precisionai-agristress-utils

# Full development (tests, pre-commit, type checking)
pip install -e ".[dev]"
```

---

## Quick start

```python
from precisionai.agristress.utils import AgriStressDataset, download_dataset

download_dataset("/path/to/agristress", repo_id="precisionaiinc/AgriStress-500")
ds = AgriStressDataset("/path/to/agristress")
```

See [`examples/`](https://github.com/Precision-AI-Inc/agristress-utils/blob/main/examples/) for complete walkthroughs of all three evaluator routes and dataset validation.

---

## Downloading the dataset

```python
from precisionai.agristress.utils import download_dataset

download_dataset("/path/to/agristress", repo_id="precisionaiinc/AgriStress-500")
```

Single-shot only — partial downloads are not supported. The `internal/` directory is always excluded. Returns the resolved local `Path`.

After downloading, the directory layout is:

```
/path/to/agristress/
  images/          # source images, one subdirectory per L2 cluster
  instances/       # instance crops, one subdirectory per annotated cluster
  image2image.json
  plant2image.json
  plant2plant.json
  README.md
```

Pass that directory directly to `AgriStressDataset`:

```python
ds = AgriStressDataset("/path/to/agristress")
```

---

## Enumeration and filtering

Before embedding images you need to know what IDs exist. All IDs are relative paths that match the dataset layout exactly (e.g. `"images/A1/pai-FvearqMK.png"`).

```python
# All clusters, images, and instances
clusters      = ds.list_clusters()                        # sorted list of L2 cluster names
all_images    = ds.list_image_ids()                       # all image paths
all_instances = ds.list_instance_ids()                    # all instance paths

# Narrow by cluster or label
a1_images     = ds.list_image_ids(cluster="A1")
soy_instances = ds.list_instance_ids(label="Crop | Soybean")

# Filter by cluster attributes — scalar or list values
drone_images  = ds.filter_images_by_attribute(camera_source="drone")
noon_drone    = ds.filter_images_by_attribute(camera_source="drone", time_period="noon")
soy_clusters  = ds.filter_images_by_attribute(plants="Crop | Soybean")
```

---

## Path resolution

Map relative IDs to absolute file paths so you can open them with your image loader:

```python
paths = ds.resolve_paths(ds.list_image_ids(cluster="A1"))
# {"images/A1/pai-FvearqMK.png": Path("/path/to/agristress/images/A1/pai-FvearqMK.png"), ...}

from PIL import Image
for abs_path in paths.values():
    img = Image.open(abs_path)
    # … run embedding model
```

Both image and instance IDs are accepted. The returned paths are resolved but not checked for existence.

---

## Embedding utilities

The `agrieval` evaluator requires every vector to be **L2-normalized** (‖v‖₂ = 1.0 ± 1e-3). Use these utilities to normalize and validate before building payloads:

```python
from precisionai.agristress.utils import normalize_embeddings, validate_embeddings

# Normalize raw model output
raw = {"images/A1/pai-FvearqMK.png": model.encode(...)}  # unnormalized
normed = normalize_embeddings(raw)                        # returns new dict, does not mutate

# Validate before posting (raises ValueError if any vector deviates)
validate_embeddings(normed)
```

`normalize_embeddings` raises `ValueError` for zero vectors. `validate_embeddings` raises `ValueError` with the offending path and its actual norm if any vector fails the check.

---

## Metadata lookups

IDs are relative paths matching the dataset layout (e.g. `"images/A1/pai-FvearqMK.png"`).

### Image metadata

```python
records = ds.image_metadata([
    "images/A1/pai-FvearqMK.png",
    "images/A1/pai-XYZ123.png",
])
for r in records:
    print(r.image_id, r.cluster, r.attributes.plants, r.instances)
```

Optionally attach embeddings. Embeddings are **all-or-nothing** — every requested ID must have an entry in the dict, or `ValueError` is raised:

```python
embeddings = {
    "images/A1/pai-FvearqMK.png": [...],
    "images/A1/pai-XYZ123.png":   [...],
}
records = ds.image_metadata(list(embeddings), embeddings=embeddings)
print(records[0].embedding)   # [...] populated
```

### Instance metadata

```python
records = ds.instance_metadata([
    "instances/A1/pai-FvearqMK-1.png",
])
for r in records:
    print(r.instance_id, r.label, r.parent_image)
```

---

## Retrieval payload builders

Assemble the exact request body expected by `agrieval`. All embedding vectors must be **L2-normalized** flat `list[float]`.

| Method | Evaluator endpoint |
|---|---|
| `build_i2i_payload(embeddings)` | `POST /v1/embeddings/evaluate/image2image` |
| `build_p2i_payload(embeddings)` | `POST /v1/embeddings/evaluate/plant2image` |
| `build_p2p_payload(embeddings)` | `POST /v1/embeddings/evaluate/plant2plant` |

```python
# Image → Image
i2i = ds.build_i2i_payload(image_embeddings, k_values=[5, 10, 20])

# Plant → Image  (embeddings must include BOTH image and instance paths)
p2i = ds.build_p2i_payload(combined_embeddings)

# Plant → Plant  (instance paths only)
p2p = ds.build_p2p_payload(instance_embeddings, sample_pairs=None)
```

Embedding keys not present in the dataset manifests are ignored. Clusters or instance groups with no matching embedding are dropped from the payload automatically.

---

## Project layout

```
precisionai/agristress/utils/
  dataset/
    download.py      # HuggingFace single-shot download
    loader.py        # AgriStressDataset — metadata lookups + payload builders
  schemas/
    dataset.py       # ImageRecord, InstanceRecord, ImageAttributes  (Pydantic)
    retrieval.py     # I2IPayload, P2IPayload, P2PPayload  (TypedDicts)
docs/                # Sphinx (HTML + LaTeX/PDF)
tests/               # Mirrors package structure
examples/            # Standalone runnable scripts
```

---

## Development

See [CONTRIBUTING.md](https://github.com/Precision-AI-Inc/agristress-utils/blob/main/CONTRIBUTING.md) for setup, branching, and PR guidelines, and [CODE_OF_CONDUCT.md](https://github.com/Precision-AI-Inc/agristress-utils/blob/main/CODE_OF_CONDUCT.md) for community expectations. [CLAUDE.md](https://github.com/Precision-AI-Inc/agristress-utils/blob/main/CLAUDE.md) documents the code style and conventions enforced in this repo.

```bash
# Run all tests with coverage report
python -m pytest

# Run a specific file
pytest tests/test_loader.py

# Run pre-commit hooks manually
pre-commit run --all-files
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE.md](https://github.com/Precision-AI-Inc/agristress-utils/blob/main/LICENSE.md).

See [CHANGELOG.md](https://github.com/Precision-AI-Inc/agristress-utils/blob/main/CHANGELOG.md) for release history.
