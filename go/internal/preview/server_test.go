package preview_test

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/huslage/topo-shadow-box/internal/preview"
	"github.com/huslage/topo-shadow-box/internal/session"
)

func TestStartOrUpdateServesViewerAndData(t *testing.T) {
	s := session.New()
	s.Results.TerrainMesh = &session.Mesh{
		Name: "Terrain", FeatureType: "terrain",
		Vertices: [][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
		Faces:    [][3]int{{0, 1, 2}},
	}
	url, err := preview.StartOrUpdate(s, 3333)
	if err != nil {
		t.Fatalf("start preview failed: %v", err)
	}
	if url == "" {
		t.Fatal("expected url")
	}

	deadline := time.Now().Add(3 * time.Second)
	for {
		res, err := http.Get("http://localhost:3333/")
		if err == nil && res.StatusCode == http.StatusOK {
			body, _ := io.ReadAll(res.Body)
			_ = res.Body.Close()
			if !strings.Contains(string(body), "Topo Shadow Box Preview") {
				t.Fatalf("unexpected viewer html")
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("preview server did not become ready")
		}
		time.Sleep(60 * time.Millisecond)
	}

	res, err := http.Get("http://localhost:3333/data")
	if err != nil {
		t.Fatalf("fetch /data failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", res.StatusCode)
	}
	var payload struct {
		Meshes []any `json:"meshes"`
	}
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		t.Fatalf("decode /data failed: %v", err)
	}
	if len(payload.Meshes) == 0 {
		t.Fatal("expected at least one mesh in payload")
	}
}
