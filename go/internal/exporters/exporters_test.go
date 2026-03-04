package exporters_test

import (
	"archive/zip"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/huslage/topo-shadow-box/internal/exporters"
	"github.com/huslage/topo-shadow-box/internal/session"
)

func seededSession() *session.Session {
	s := session.New()
	s.Config.Bounds = session.Bounds{North: 36, South: 35, East: -82, West: -83, IsSet: true}
	s.Results.TerrainMesh = &session.Mesh{
		Name: "terrain", FeatureType: "terrain",
		Vertices: [][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
		Faces:    [][3]int{{0, 1, 2}},
	}
	return s
}

func TestExport3MF(t *testing.T) {
	s := seededSession()
	out := filepath.Join(t.TempDir(), "test.3mf")
	if err := exporters.Export3MF(s, out); err != nil {
		t.Fatalf("export 3mf failed: %v", err)
	}
	r, err := zip.OpenReader(out)
	if err != nil {
		t.Fatalf("not zip: %v", err)
	}
	defer r.Close()
	found := false
	for _, f := range r.File {
		if f.Name == "3D/3dmodel.model" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("missing 3D/3dmodel.model")
	}
}

func TestExportOpenSCAD(t *testing.T) {
	s := seededSession()
	out := filepath.Join(t.TempDir(), "test.scad")
	if err := exporters.ExportOpenSCAD(s, out); err != nil {
		t.Fatalf("export openscad failed: %v", err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read openscad: %v", err)
	}
	if !strings.Contains(string(data), "polyhedron") {
		t.Fatal("expected polyhedron in output")
	}
}

func TestExportSVG(t *testing.T) {
	s := seededSession()
	out := filepath.Join(t.TempDir(), "test.svg")
	if err := exporters.ExportSVG(s, out); err != nil {
		t.Fatalf("export svg failed: %v", err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read svg: %v", err)
	}
	if !strings.Contains(string(data), "<svg") {
		t.Fatal("expected svg output")
	}
}
