# GitHub Actions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CI (ruff + pytest on push/PR) and automated release (on v* tag) GitHub Actions workflows.

**Architecture:** Two workflow files in `.github/workflows/`. CI installs deps via `uv`, lints with `ruff`, and runs `pytest`. Release workflow uses `gh` CLI to create a GitHub release with auto-generated notes when a version tag is pushed.

**Tech Stack:** GitHub Actions, uv, ruff, pytest, gh CLI

---

### Task 1: Add ruff to dev dependencies and configure it

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add ruff to dev dependencies**

`pyproject.toml` already has `[dependency-groups]` with `dev`. Add ruff:

```toml
[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "ruff>=0.9",
]
```

**Step 2: Add ruff config targeting Python 3.11**

Append to `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"

[tool.ruff.lint]
select = ["F"]
```

(Selecting only `F` — Pyflakes rules — matches what was scoped in design: unused imports, unused variables, f-string issues.)

**Step 3: Sync dependencies**

```bash
uv sync --dev
```

Expected: resolves without error, ruff added to `.venv`.

**Step 4: Verify ruff finds the known 18 violations**

```bash
.venv/bin/ruff check src/ tests/
```

Expected: `Found 18 errors.`

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add ruff to dev dependencies"
```

---

### Task 2: Fix all ruff violations

**Files:**
- Modify: `src/topo_shadow_box/core/models.py`
- Modify: `src/topo_shadow_box/exporters/openscad.py`
- Modify: `src/topo_shadow_box/tools/generate.py`
- Modify: `tests/test_area_tools.py`
- Modify: `tests/test_elevation.py`
- Modify: `tests/test_exporters.py`
- Modify: `tests/test_generate_progress.py`
- Modify: `tests/test_mesh.py`
- Modify: `tests/test_shape_clipper.py`
- Modify: `tests/test_state_resource.py`

**Step 1: Auto-fix all violations**

```bash
.venv/bin/ruff check src/ tests/ --fix --unsafe-fixes
```

Expected: `Found 18 errors, fixed 18.` (or similar — all errors resolved)

**Step 2: Verify ruff is clean**

```bash
.venv/bin/ruff check src/ tests/
```

Expected: no output, exit code 0.

**Step 3: Run tests to confirm nothing broke**

```bash
.venv/bin/pytest
```

Expected: all tests pass (same count as before).

**Step 4: Commit**

```bash
git add src/ tests/
git commit -m "fix: resolve all ruff lint violations"
```

---

### Task 3: Create CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create the workflows directory and ci.yml**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Set up Python 3.11
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --dev

      - name: Lint with ruff
        run: uv run ruff check src/ tests/

      - name: Run tests
        run: uv run pytest
```

**Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add CI workflow with ruff and pytest"
```

---

### Task 4: Create release workflow

**Files:**
- Create: `.github/workflows/release.yml`

**Step 1: Create release.yml**

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - name: Create GitHub Release
        run: gh release create "${{ github.ref_name }}" --generate-notes
        env:
          GH_TOKEN: ${{ github.token }}
```

**Step 2: Commit and push**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add release workflow for v* tags"
git push
```

**Step 3: Verify CI passes on GitHub**

Go to https://github.com/huslage/topo-shadow-box/actions and confirm the CI workflow run on `main` passes.
