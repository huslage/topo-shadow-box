package core_test

import (
	"context"
	"testing"

	"github.com/huslage/topo-shadow-box/internal/core"
	"github.com/huslage/topo-shadow-box/internal/session"
)

func TestSetAreaFromGeocodeCandidate(t *testing.T) {
	s := session.New()
	c := session.GeocodeCandidate{DisplayName: "x", BboxNorth: 36, BboxSouth: 35, BboxEast: -78, BboxWest: -79}
	if err := core.SetAreaFromGeocodeCandidate(context.Background(), s, c); err != nil {
		t.Fatalf("set area from candidate failed: %v", err)
	}
	if !s.Config.Bounds.IsSet {
		t.Fatal("expected bounds set")
	}
}
