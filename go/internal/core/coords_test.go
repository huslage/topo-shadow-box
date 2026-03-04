package core_test

import (
	"math"
	"testing"

	"github.com/huslage/topo-shadow-box/internal/core"
	"github.com/huslage/topo-shadow-box/internal/session"
)

func TestGeoToModel(t *testing.T) {
	b := session.Bounds{North: 36.0, South: 35.0, East: -82.0, West: -83.0, IsSet: true}
	tr := core.NewGeoToModelTransform(b, 200.0)
	// NW corner should map to (0, 0)
	x2, z2 := tr.GeoToModel(36.0, -83.0)
	if math.Abs(x2) > 0.01 {
		t.Fatalf("NW corner x should be ~0, got %v", x2)
	}
	if math.Abs(z2) > 0.01 {
		t.Fatalf("NW corner z should be ~0, got %v", z2)
	}
}

func TestAddPaddingToBounds(t *testing.T) {
	b := session.Bounds{North: 36.0, South: 36.0, East: -82.0, West: -82.0}
	padded := core.AddPaddingToBounds(b, 1000.0, true)
	if !padded.IsSet {
		t.Fatal("padded bounds should be set")
	}
	if padded.North <= 36.0 || padded.South >= 36.0 {
		t.Fatal("padding should expand bounds")
	}
}
