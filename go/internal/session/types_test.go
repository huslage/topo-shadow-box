package session_test

import (
	"testing"
	"github.com/huslage/topo-shadow-box/internal/session"
)

func TestBoundsValidation(t *testing.T) {
	b := session.Bounds{North: 36.0, South: 35.0, East: -82.0, West: -83.0, IsSet: true}
	if err := b.Validate(); err != nil {
		t.Fatalf("valid bounds failed: %v", err)
	}
}

func TestBoundsInvalidNorthLtSouth(t *testing.T) {
	b := session.Bounds{North: 34.0, South: 35.0, East: -82.0, West: -83.0, IsSet: true}
	if err := b.Validate(); err == nil {
		t.Fatal("expected error for north < south")
	}
}

func TestColorsValidHex(t *testing.T) {
	c := session.DefaultColors()
	if err := c.Validate(); err != nil {
		t.Fatalf("default colors invalid: %v", err)
	}
}

func TestColorsInvalidHex(t *testing.T) {
	c := session.DefaultColors()
	c.Terrain = "notahex"
	if err := c.Validate(); err == nil {
		t.Fatal("expected error for invalid hex")
	}
}
