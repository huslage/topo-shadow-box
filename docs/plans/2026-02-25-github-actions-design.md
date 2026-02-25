# GitHub Actions Design

**Date:** 2026-02-25

## Goal

Add CI/CD workflows to automate testing, linting, and releases.

## Workflows

### `ci.yml`

Triggers on push to `main` and all pull requests.

Steps:
1. Checkout code
2. Install `uv`
3. `uv sync --dev`
4. `ruff check src/ tests/`
5. `pytest`

Python version: 3.11 (project minimum)

### `release.yml`

Triggers on `v*` tag push.

Steps:
1. Checkout code
2. Create GitHub release with auto-generated notes via `gh release create`

## Ruff Setup

- Add `ruff` to `[dependency-groups.dev]` in `pyproject.toml`
- Add `[tool.ruff]` config in `pyproject.toml` targeting Python 3.11
- Fix any existing violations before enabling CI lint gate

## Non-Goals

- No PyPI publishing
- No matrix testing across multiple Python versions
