# Releasing

Releases are created via GitHub Actions. The workflow bumps the version in all required files, commits, tags, and publishes the GitHub Release in one step.

## Cut a release

```bash
gh workflow run release.yml -f version=1.2.0
```

Or from the GitHub UI: **Actions → Release → Run workflow**, enter the version.

The version must be in `X.Y.Z` format. The workflow will:

1. Update `pyproject.toml`, `.claude-plugin/plugin.json`, and `.claude-plugin/marketplace.json`
2. Commit the changes to `main`
3. Tag the commit as `vX.Y.Z`
4. Create a GitHub Release with auto-generated notes

## What the version bump affects

| File | Field |
|------|-------|
| `pyproject.toml` | `version` |
| `.claude-plugin/plugin.json` | `version` |
| `.claude-plugin/marketplace.json` | `plugins[].version` |

The `plugin.json` version is what Claude Code uses to detect updates. Users with the plugin installed will be offered the update automatically when the version changes.
