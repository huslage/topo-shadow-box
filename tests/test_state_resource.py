"""Tests for state://session MCP resource."""
import json
import pytest


def test_state_resource_returns_valid_json():
    """state://session resource should return JSON matching state.summary()."""
    from topo_shadow_box.server import mcp
    from topo_shadow_box.state import state

    resources = {str(r.uri): r for r in mcp._resource_manager._resources.values()}
    assert "state://session" in resources, (
        f"state://session not registered. Registered: {list(resources.keys())}"
    )


def test_state_resource_content_matches_summary():
    """Resource content should return valid JSON with expected keys."""
    from topo_shadow_box.server import mcp
    from topo_shadow_box.state import state

    resources = {str(r.uri): r for r in mcp._resource_manager._resources.values()}
    resource = resources.get("state://session")
    assert resource is not None

    # Call the resource function synchronously (it's not async)
    result = resource.fn()
    parsed = json.loads(result)
    expected = state.summary()
    assert parsed.keys() == expected.keys()
