# Releasing

Releases are built from Git tags and publish cross-platform Go binaries.

## Cut a release

Create and push a semantic version tag:

```bash
git tag v1.2.0
git push origin v1.2.0
```

Or run the Release workflow manually in GitHub Actions and provide `1.2.0`.

## What the Release workflow does

1. Builds `topo-shadow-box` for:
   - macOS (`amd64`, `arm64`)
   - Linux (`amd64`, `arm64`)
   - Windows (`amd64`)
2. Packages archives (`.tar.gz` on Unix, `.zip` on Windows)
3. Publishes/updates the GitHub Release for tag `vX.Y.Z`
4. Uploads all binary archives as release assets

## Version metadata

If you're incrementing plugin metadata for marketplace visibility, update:

- `.claude-plugin/plugin.json` -> `version`
- `.claude-plugin/marketplace.json` -> `plugins[].version`
