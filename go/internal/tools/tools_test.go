package tools_test

import (
	"testing"

	"github.com/huslage/topo-shadow-box/internal/session"
	"github.com/huslage/topo-shadow-box/internal/tools"
	"github.com/mark3labs/mcp-go/server"
)

func TestRegisterAllRegistersTools(t *testing.T) {
	s := session.New()
	srv := server.NewMCPServer("topo-shadow-box", "test")
	tools.RegisterAll(srv, s)

	mustHave := []string{
		"get_status",
		"set_area_from_coordinates",
		"set_area_from_gpx",
		"geocode_place",
		"select_geocode_result",
		"validate_area",
		"fetch_elevation",
		"fetch_features",
		"set_model_params",
		"set_colors",
		"generate_model",
		"generate_map_insert",
		"preview",
		"export_3mf",
		"export_openscad",
		"export_svg",
		"save_session",
		"load_session",
	}
	for _, name := range mustHave {
		if srv.GetTool(name) == nil {
			t.Fatalf("expected tool %q to be registered", name)
		}
	}
}
