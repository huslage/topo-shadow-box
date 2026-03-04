package session_test

import (
	"testing"
	"github.com/huslage/topo-shadow-box/internal/session"
)

func TestNewSession(t *testing.T) {
	s := session.New()
	if s.Config.ModelParams.WidthMM != 200 {
		t.Fatalf("expected default width 200, got %v", s.Config.ModelParams.WidthMM)
	}
	if s.FetchedData.Elevation != nil {
		t.Fatal("elevation should be nil on new session")
	}
	if s.Results.TerrainMesh != nil {
		t.Fatal("terrain mesh should be nil on new session")
	}
}

func TestSessionClearDownstream(t *testing.T) {
	s := session.New()
	s.FetchedData.Elevation = &session.ElevationData{IsSet: true}
	s.Results.TerrainMesh = &session.Mesh{Name: "terrain"}
	s.ClearDownstream()
	if s.FetchedData.Elevation != nil {
		t.Fatal("elevation should be nil after ClearDownstream")
	}
	if s.Results.TerrainMesh != nil {
		t.Fatal("terrain mesh should be nil after ClearDownstream")
	}
}
