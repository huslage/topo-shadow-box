"""Validate .claude-plugin/ manifest files are well-formed and have required fields."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
PLUGIN = ROOT / ".claude-plugin" / "plugin.json"


def test_marketplace_json_exists():
    assert MARKETPLACE.exists(), ".claude-plugin/marketplace.json is missing"


def test_marketplace_json_is_valid():
    data = json.loads(MARKETPLACE.read_text())
    assert "name" in data
    assert "owner" in data
    assert "name" in data["owner"]
    assert "plugins" in data
    assert len(data["plugins"]) > 0


def test_marketplace_plugin_entry_has_required_fields():
    data = json.loads(MARKETPLACE.read_text())
    for plugin in data["plugins"]:
        assert "name" in plugin, f"plugin entry missing 'name': {plugin}"
        assert "source" in plugin, f"plugin entry missing 'source': {plugin}"


def test_plugin_json_exists():
    assert PLUGIN.exists(), ".claude-plugin/plugin.json is missing"


def test_plugin_json_is_valid():
    data = json.loads(PLUGIN.read_text())
    assert "name" in data
    assert "version" in data
    assert "mcpServers" in data
    assert len(data["mcpServers"]) > 0


def test_plugin_mcp_server_has_command():
    data = json.loads(PLUGIN.read_text())
    for server_name, server in data["mcpServers"].items():
        assert "command" in server, f"mcpServer {server_name!r} missing 'command'"
        assert "args" in server, f"mcpServer {server_name!r} missing 'args'"
