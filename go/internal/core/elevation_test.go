package core_test

import (
	"testing"

	"github.com/huslage/topo-shadow-box/internal/core"
)

func TestLatLonToTile(t *testing.T) {
	// Known tile: zoom 10, Portland, OR roughly at (163, 357)
	x, y := core.LatLonToTile(45.5, -122.7, 10)
	if x < 160 || x > 170 {
		t.Fatalf("unexpected tile x: %d", x)
	}
	_ = y
}

func TestDecodeTerrariumPixel(t *testing.T) {
	// R=0, G=0, B=0 should decode to -32768m
	elev := core.DecodeTerrariumPixel(0, 0, 0)
	if elev != -32768.0 {
		t.Fatalf("expected -32768, got %v", elev)
	}
	// R=128, G=0, B=0 => 128*256 - 32768 = 0m
	elev2 := core.DecodeTerrariumPixel(128, 0, 0)
	if elev2 != 0.0 {
		t.Fatalf("expected 0, got %v", elev2)
	}
}

func TestPickZoom(t *testing.T) {
	z := core.PickZoom(36, 35, -82, -83) // ~1 degree span
	if z < 10 || z > 12 {
		t.Fatalf("unexpected zoom %d for ~1deg area", z)
	}
}

func TestBilinearInterp(t *testing.T) {
	// 2x2 grid
	grid := [][]float64{{0, 1}, {2, 3}}
	lats := []float64{0, 1}
	lons := []float64{0, 1}
	// Midpoint should be average = 1.5
	v := core.BilinearInterp(grid, lats, lons, 0.5, 0.5)
	if v < 1.4 || v > 1.6 {
		t.Fatalf("expected ~1.5, got %v", v)
	}
}
