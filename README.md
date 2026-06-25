# PAI Agricultural Project Template

[![Coverage](.github/badges/coverage.svg)](https://github.com/Precision-AI-Inc/pai-ag-template/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

> **Template repository.** Replace every occurrence of `myproject` / `pai-myproject` / `PAI MyProject` with your actual project name. Update the badge URLs above to point to your repository on GitHub.

This repository is the canonical starting point for new Precision AI Python projects. It ships with:

- A working `pai/myproject/` package demonstrating the full layer stack: metrics → services → schemas → API (FastAPI included as one example; remove or replace it for non-API projects)
- **Pre-commit hooks** that enforce ruff lint/format, pyright type checking, and ≥ 90% test coverage on every commit
- **GitHub Actions** CI that mirrors the pre-commit gate
- **Sphinx** documentation scaffolding (HTML + LaTeX/PDF)
- A standalone **example script** and a complete **test suite** to validate the template is functional out of the box

Coding standards, naming conventions, and tooling configuration are governed by [CLAUDE.md](CLAUDE.md). All new code in any PAI project must conform to those standards.

---

## Quickstart — rename the template

Global find-and-replace in this order (`myproject` last to avoid partial matches):

| Find | Replace with |
|---|---|
| `pai-ag-template` | your GitHub repository name |
| `pai-myproject` | your distribution name (kebab-case, e.g. `pai-ag-emb`) |
| `PAI MyProject` | your human-readable project name |
| `myproject` | your project namespace (snake_case, e.g. `ag_emb`) |

Then:

1. Update `pyproject.toml` `description` and `dependencies` for your domain.
2. Replace `pai/myproject/` business logic with your domain logic.
3. Update `docs/conf.py` header string and `docs/index.rst` / `docs/modules.rst` autodoc references.
4. Remove the **Using this template** section from `CLAUDE.md`.

See [CLAUDE.md](CLAUDE.md) for the full step-by-step template setup guide.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Runtime only
pip install -r requirements.txt

# Full development (tests, pre-commit, type checking)
pip install -e ".[dev]"
pre-commit install
```

---

## Running the API server

```bash
# Default: host 0.0.0.0, port 8000
pai-myproject

# Or explicitly
python -m pai.myproject.api.app --host 0.0.0.0 --port 8000 --no-reload

# Interactive docs
open http://localhost:8000/docs
```

### `POST /v1/hello`

```json
{
  "name": "Precision AI",
  "shout": false
}
```

Response:

```json
{
  "greeting": "Hello, Precision AI!",
  "word_count": 3,
  "char_count": 20
}
```

---

## Python SDK usage

```python
from pai.myproject.services.hello import say_hello, describe
from pai.myproject.metrics.compute import word_count, char_count

# Call the service layer directly (no HTTP)
greeting = say_hello("Precision AI")
print(greeting)                  # Hello, Precision AI!

result = describe("Precision AI", shout=True)
print(result["greeting"])        # HELLO, PRECISION AI!
print(result["word_count"])      # 3
print(result["char_count"])      # 21
```

---

## Standalone example script

```bash
python examples/example.py
python examples/example.py --name "Precision AI" --shout
```

---

## Testing

```bash
# Run all tests with coverage report
python -m pytest

# Run a specific module
pytest tests/test_hello.py

# Run tests matching a keyword
pytest -k "shout"
```

Coverage must remain at or above **90%** — enforced by pytest and the pre-commit hook.

---

## Pre-commit hooks

| Hook | What it checks |
|---|---|
| File hygiene | Large files (> 1 000 KB), trailing whitespace, merge conflicts, private keys, debug statements |
| `ruff` | Linting and import sorting (auto-fix) |
| `ruff-format` | Code formatting (auto-fix) |
| `pyright` | Static type checking |
| `pytest` | Full test suite with ≥ 90% coverage |

Run all hooks manually without committing:

```bash
pre-commit run --all-files
```

---

## Project layout

```
pai/myproject/
  api/
    routes/         # FastAPI route handlers (thin — delegate to services)
    config.py       # Environment-variable configuration only
    app.py          # FastAPI app factory + CLI entry point
  schemas/          # Pydantic v2 request/response models
  services/         # Business logic
  metrics/          # Pure computation modules (no I/O)
    __init__.py     # Re-exports only — no logic
docs/               # Sphinx (HTML + LaTeX/PDF)
tests/              # Mirrors package structure
examples/           # Standalone runnable scripts
```

See [CLAUDE.md](CLAUDE.md) for the full coding standard covering imports, docstrings, type hints, testing, and what to avoid.

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

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PAI_APP_ENV` | `development` | Application environment (`development`, `staging`, `production`) |

Add project-specific variables to `pai/myproject/api/config.py` following the same pattern.

---

## Security

To report a security vulnerability, see [SECURITY.md](SECURITY.md). Do not open a public issue.

---

## License

[Apache 2.0](LICENSE.md) © Precision AI, Inc.
