package core_test

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/huslage/topo-shadow-box/internal/core"
)

type mockOSMClient struct{ body string }

func (m *mockOSMClient) Get(url string) (*http.Response, error) { panic("use Post") }
func (m *mockOSMClient) Post(url, contentType string, body io.Reader) (*http.Response, error) {
	return &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(strings.NewReader(m.body)),
	}, nil
}

const mockOSMResponse = `{"elements":[
  {"type":"way","id":1,"tags":{"highway":"primary","name":"Main St"},
   "geometry":[{"lat":35.5,"lon":-82.5},{"lat":35.6,"lon":-82.4}]}
]}`

func TestFetchOSMRoads(t *testing.T) {
	client := &mockOSMClient{body: mockOSMResponse}
	features, err := core.FetchOSMFeatures(nil, client, 36, 35, -82, -83, []string{"roads"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(features.Roads) != 1 {
		t.Fatalf("expected 1 road, got %d", len(features.Roads))
	}
	if features.Roads[0].Name != "Main St" {
		t.Fatalf("expected 'Main St', got %q", features.Roads[0].Name)
	}
}

const mockOSMWaterResponse = `{"elements":[
  {"type":"way","id":2,"tags":{"waterway":"river","name":"Test River"},
   "geometry":[{"lat":35.5,"lon":-82.5},{"lat":35.6,"lon":-82.4}]}
]}`

func TestFetchOSMWater(t *testing.T) {
	client := &mockOSMClient{body: mockOSMWaterResponse}
	features, err := core.FetchOSMFeatures(nil, client, 36, 35, -82, -83, []string{"water"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(features.Water) != 1 {
		t.Fatalf("expected 1 water feature, got %d", len(features.Water))
	}
	if features.Water[0].Name != "Test River" {
		t.Fatalf("expected 'Test River', got %q", features.Water[0].Name)
	}
}

const mockOSMBuildingResponse = `{"elements":[
  {"type":"way","id":3,"tags":{"building":"yes","name":"Test Building","height":"12"},
   "geometry":[{"lat":35.5,"lon":-82.5},{"lat":35.5,"lon":-82.4},{"lat":35.6,"lon":-82.4},{"lat":35.5,"lon":-82.5}]}
]}`

func TestFetchOSMBuildings(t *testing.T) {
	client := &mockOSMClient{body: mockOSMBuildingResponse}
	features, err := core.FetchOSMFeatures(nil, client, 36, 35, -82, -83, []string{"buildings"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(features.Buildings) != 1 {
		t.Fatalf("expected 1 building, got %d", len(features.Buildings))
	}
	if features.Buildings[0].Height != 12.0 {
		t.Fatalf("expected height 12.0, got %v", features.Buildings[0].Height)
	}
}
