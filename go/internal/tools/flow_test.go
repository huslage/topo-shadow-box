package tools_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/huslage/topo-shadow-box/internal/session"
	"github.com/huslage/topo-shadow-box/internal/tools"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func callTool(t *testing.T, srv *server.MCPServer, name string, args map[string]any) *mcp.CallToolResult {
	t.Helper()
	tool := srv.GetTool(name)
	if tool == nil {
		t.Fatalf("tool %q not found", name)
	}
	res, err := tool.Handler(context.Background(), mcp.CallToolRequest{Params: mcp.CallToolParams{Name: name, Arguments: args}})
	if err != nil {
		t.Fatalf("tool %q returned protocol error: %v", name, err)
	}
	if res == nil {
		t.Fatalf("tool %q returned nil result", name)
	}
	return res
}

func toolText(res *mcp.CallToolResult) string {
	parts := make([]string, 0, len(res.Content))
	for _, c := range res.Content {
		parts = append(parts, mcp.GetTextFromContent(c))
	}
	return strings.Join(parts, "\n")
}

func TestToolFlowGenerateAndExport(t *testing.T) {
	s := session.New()
	srv := server.NewMCPServer("topo-shadow-box", "test")
	tools.RegisterAll(srv, s)

	callTool(t, srv, "set_area_from_coordinates", map[string]any{"lat": 35.99, "lon": -78.9, "radius_m": 500.0})
	if !s.Config.Bounds.IsSet {
		t.Fatal("bounds should be set")
	}

	// Seed elevation to avoid network-dependent fetch in unit tests.
	s.FetchedData.Elevation = &session.ElevationData{
		Grid:         [][]float64{{100, 110, 120}, {90, 100, 110}, {80, 90, 100}},
		Lats:         []float64{s.Config.Bounds.North, (s.Config.Bounds.North + s.Config.Bounds.South) / 2, s.Config.Bounds.South},
		Lons:         []float64{s.Config.Bounds.West, (s.Config.Bounds.East + s.Config.Bounds.West) / 2, s.Config.Bounds.East},
		Resolution:   3,
		MinElevation: 80,
		MaxElevation: 120,
		IsSet:        true,
	}

	res := callTool(t, srv, "generate_model", nil)
	if res.IsError {
		t.Fatalf("generate_model should succeed, got: %s", toolText(res))
	}
	if s.Results.TerrainMesh == nil {
		t.Fatal("terrain mesh expected after generate_model")
	}

	mapRes := callTool(t, srv, "generate_map_insert", map[string]any{"format": "plate"})
	if mapRes.IsError {
		t.Fatalf("generate_map_insert should succeed, got: %s", toolText(mapRes))
	}
	if s.Results.MapInsertMesh == nil {
		t.Fatal("map insert mesh expected")
	}

	out := filepath.Join(t.TempDir(), "model.3mf")
	exp := callTool(t, srv, "export_3mf", map[string]any{"output_path": out})
	if exp.IsError {
		t.Fatalf("export_3mf should succeed, got: %s", toolText(exp))
	}
	if _, err := os.Stat(out); err != nil {
		t.Fatalf("expected output file, stat error: %v", err)
	}
}

func TestToolFlowGeocodeSelection(t *testing.T) {
	s := session.New()
	srv := server.NewMCPServer("topo-shadow-box", "test")
	tools.RegisterAll(srv, s)

	s.Config.PendingGeocodeCandidates = []session.GeocodeCandidate{
		{DisplayName: "A", BboxNorth: 36, BboxSouth: 35, BboxEast: -78, BboxWest: -79},
		{DisplayName: "B", BboxNorth: 37, BboxSouth: 36, BboxEast: -77, BboxWest: -78},
	}

	res := callTool(t, srv, "select_geocode_result", map[string]any{"number": 2})
	if res.IsError {
		t.Fatalf("select_geocode_result should succeed, got: %s", toolText(res))
	}
	if s.Config.Bounds.North != 37 {
		t.Fatalf("expected selected candidate bounds, got %+v", s.Config.Bounds)
	}
}

func TestToolFlowGeocodePlaceMultipleReturnsError(t *testing.T) {
	s := session.New()
	srv := server.NewMCPServer("topo-shadow-box", "test")
	tools.RegisterAll(srv, s)

	s.Config.PendingGeocodeCandidates = []session.GeocodeCandidate{
		{DisplayName: "A", BboxNorth: 36, BboxSouth: 35, BboxEast: -78, BboxWest: -79},
	}
	res := callTool(t, srv, "select_geocode_result", map[string]any{"number": 99})
	if !res.IsError {
		t.Fatalf("expected select_geocode_result invalid index to return error, got: %s", toolText(res))
	}
}

func TestToolFlowSaveLoadSession(t *testing.T) {
	s := session.New()
	srv := server.NewMCPServer("topo-shadow-box", "test")
	tools.RegisterAll(srv, s)

	s.Config.Bounds = session.Bounds{North: 36, South: 35, East: -82, West: -83, IsSet: true}
	s.Config.ModelParams.WidthMM = 150

	p := filepath.Join(t.TempDir(), "session.json")
	save := callTool(t, srv, "save_session", map[string]any{"path": p})
	if save.IsError {
		t.Fatalf("save_session failed: %s", toolText(save))
	}

	s.Config.Bounds = session.Bounds{}
	s.Config.ModelParams.WidthMM = 200

	load := callTool(t, srv, "load_session", map[string]any{"path": p})
	if load.IsError {
		t.Fatalf("load_session failed: %s", toolText(load))
	}
	if !s.Config.Bounds.IsSet || s.Config.ModelParams.WidthMM != 150 {
		t.Fatalf("session not restored correctly: bounds=%+v width=%v", s.Config.Bounds, s.Config.ModelParams.WidthMM)
	}
}
