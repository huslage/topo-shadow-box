# Development

```bash
# Install dev dependencies
uv sync

# Run all tests
.venv/bin/pytest

# Run a specific test file
.venv/bin/pytest tests/test_mesh.py

# Run a specific test
.venv/bin/pytest tests/test_mesh.py::TestCreateRoadStrip::test_basic_strip
```
