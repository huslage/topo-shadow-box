package core_test

import (
	"strings"
	"testing"

	"github.com/huslage/topo-shadow-box/internal/core"
)

func TestParseGPX(t *testing.T) {
	data := `<?xml version="1.0"?><gpx version="1.1" creator="test"><trk><name>Ride</name><trkseg><trkpt lat="35.0" lon="-78.0"><ele>10</ele></trkpt><trkpt lat="35.1" lon="-78.1"><ele>20</ele></trkpt></trkseg></trk></gpx>`
	tracks, waypoints, bounds, err := core.ParseGPX(strings.NewReader(data))
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(tracks) != 1 {
		t.Fatalf("expected 1 track, got %d", len(tracks))
	}
	if len(waypoints) != 0 {
		t.Fatalf("expected 0 waypoints, got %d", len(waypoints))
	}
	if !bounds.IsSet {
		t.Fatal("expected bounds to be set")
	}
	if bounds.North <= bounds.South {
		t.Fatalf("invalid bounds: %+v", bounds)
	}
}
