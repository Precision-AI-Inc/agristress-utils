<p align="center">
  <img src="docs/assets/logo.png" alt="Precision AI Logo" width="120"/>
</p>

# PAI Agribench Utils

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

Dataset utilities for AgriBench — download the dataset from HuggingFace, look up image and instance metadata, and assemble ready-to-POST payloads for the [`pai-agrieval-emb`](https://github.com/Precision-AI-Inc/pai-agrieval-emb) embedding evaluator.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install pai-agribench-utils

# Full development (tests, pre-commit, type checking)
pip install -e ".[dev]"
pre-commit install
```

---

## Quickstart

```python
from precisionai.agribench.utils import AgriBenchDataset, download_dataset

download_dataset("/path/to/agribench", repo_id="precisionaiinc/AgriBench")
ds = AgriBenchDataset("/path/to/agribench")
```

See [`examples/`](examples/) for complete walkthroughs of all three evaluator routes and dataset validation.

---

## Downloading the dataset

```python
from precisionai.agribench.utils import download_dataset

download_dataset("/path/to/agribench", repo_id="precisionaiinc/AgriBench")
```

Single-shot only — partial downloads are not supported. The `internal/` directory is always excluded. Returns the resolved local `Path`.

After downloading, the directory layout is:

```
/path/to/agribench/
  images/          # source images, one subdirectory per L2 cluster
  instances/       # instance crops, one subdirectory per annotated cluster
  image2image.json
  plant2image.json
  plant2plant.json
  README.md
```

Pass that directory directly to `AgriBenchDataset`:

```python
ds = AgriBenchDataset("/path/to/agribench")
```

---

## Enumeration and filtering

Before embedding images you need to know what IDs exist. All IDs are relative paths that match the dataset layout exactly (e.g. `"images/A1/pai-FvearqMK.png"`).

```python
# All clusters, images, and instances
clusters     = ds.list_clusters()                        # sorted list of L2 cluster names
all_images   = ds.list_image_ids()                       # all 489 image paths
all_instances = ds.list_instance_ids()                   # all 5 933 instance paths

# Narrow by cluster or label
a1_images = ds.list_image_ids(cluster="A1")
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
# {"images/A1/pai-FvearqMK.png": Path("/path/to/agribench/images/A1/pai-FvearqMK.png"), ...}

from PIL import Image
for abs_path in paths.values():
    img = Image.open(abs_path)
    # … run embedding model
```

Both image and instance IDs are accepted. The returned paths are resolved but not checked for existence.

---

## Embedding utilities

The `pai-agrieval-emb` evaluator requires every vector to be **L2-normalised** (‖v‖₂ = 1.0 ± 1e-3). Use these utilities to normalise and validate before building payloads:

```python
from precisionai.agribench.utils import normalize_embeddings, validate_embeddings

# Normalise raw model output
raw = {"images/A1/pai-FvearqMK.png": model.encode(...)}  # unnormalised
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

Assemble the exact request body expected by `pai-agrieval-emb`. All embedding vectors must be **L2-normalised** flat `list[float]`.

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

## Standalone example

```bash
python examples/example.py --dataset /path/to/agribench
```

---

## Testing

```bash
# Run all tests with coverage report
python -m pytest

# Run a specific file
pytest tests/test_loader.py

# Run tests matching a keyword
pytest -k "p2p"
```

Coverage must remain at or above **90%** — enforced by pytest and the pre-commit hook.

---

## Pre-commit hooks

| Hook | What it checks |
|---|---|
| File hygiene | Large files (> 1 MB), trailing whitespace, merge conflicts, private keys, debug statements |
| `ruff` | Linting and import sorting (auto-fix) |
| `ruff-format` | Code formatting (auto-fix) |
| `pyright` | Static type checking |
| `pytest` | Full test suite with ≥ 90% coverage |

```bash
pre-commit run --all-files
```

---

## Project layout

```
precisionai/agribench/utils/
  dataset/
    download.py      # HuggingFace single-shot download
    loader.py        # AgriBenchDataset — metadata lookups + payload builders
  schemas/
    dataset.py       # ImageRecord, InstanceRecord, ImageAttributes  (Pydantic)
    retrieval.py     # I2IPayload, P2IPayload, P2PPayload  (TypedDicts)
docs/                # Sphinx (HTML + LaTeX/PDF)
tests/               # Mirrors package structure
examples/            # Standalone runnable scripts
```

See [CLAUDE.md](CLAUDE.md) for the full coding standard.

---

## Documentation

```bash
cd docs
make html       # HTML docs → docs/_build/html/index.html
make latexpdf   # PDF      → docs/_build/latex/documentation.pdf
make clean      # Remove build artefacts
```

Dependencies: `pip install -r docs/requirements.txt`

---

## Security

To report a security vulnerability, see [SECURITY.md](SECURITY.md). Do not open a public issue.

---

## License

[Apache 2.0](LICENSE.md) © Precision AI, Inc.
