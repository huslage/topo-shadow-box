package session_test

import (
	"github.com/huslage/topo-shadow-box/internal/session"
	"path/filepath"
	"testing"
)

func TestSaveAndLoadSession(t *testing.T) {
	s := session.New()
	s.Config.Bounds = session.Bounds{North: 36, South: 35, East: -82, West: -83, IsSet: true}
	s.Config.ModelParams.WidthMM = 150
	// Elevation and meshes should NOT be persisted
	s.FetchedData.Elevation = &session.ElevationData{IsSet: true, Resolution: 200}

	tmp := filepath.Join(t.TempDir(), "session.json")
	if err := session.SaveSession(s, tmp); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	s2 := session.New()
	if err := session.LoadSession(s2, tmp); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	if s2.Config.Bounds.North != 36 {
		t.Fatalf("bounds not restored")
	}
	if s2.Config.ModelParams.WidthMM != 150 {
		t.Fatalf("model params not restored")
	}
	if s2.FetchedData.Elevation != nil {
		t.Fatal("elevation should not be restored from session file")
	}
}
