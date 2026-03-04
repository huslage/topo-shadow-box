package core_test

import (
	"testing"

	"github.com/huslage/topo-shadow-box/internal/core"
)

func TestCircleClipperContains(t *testing.T) {
	c := core.NewCircleClipper(0, 0, 10)
	if !c.IsInside(0, 0) {
		t.Fatal("center should be inside")
	}
	if !c.IsInside(9, 0) {
		t.Fatal("point within radius should be inside")
	}
	if c.IsInside(11, 0) {
		t.Fatal("point outside radius should not be inside")
	}
}

func TestCircleClipperClipLinestring(t *testing.T) {
	c := core.NewCircleClipper(0, 0, 10)
	pts := [][2]float64{{-20, 0}, {0, 0}, {20, 0}}
	segs := c.ClipLinestring(pts)
	if len(segs) == 0 {
		t.Fatal("expected at least one segment")
	}
}

func TestSquareClipperContains(t *testing.T) {
	c := core.NewSquareClipper(0, 0, 10)
	if !c.IsInside(5, 5) {
		t.Fatal("should be inside")
	}
	if c.IsInside(15, 0) {
		t.Fatal("should be outside")
	}
}

func TestHexagonClipperContains(t *testing.T) {
	c := core.NewHexagonClipper(0, 0, 10)
	if !c.IsInside(0, 0) {
		t.Fatal("center should be inside")
	}
	if c.IsInside(15, 0) {
		t.Fatal("far point should be outside")
	}
}

func TestSquareClipperClipLinestring(t *testing.T) {
	c := core.NewSquareClipper(0, 0, 10)
	// Line from outside to inside to outside
	pts := [][2]float64{{-20, 0}, {0, 0}, {20, 0}}
	segs := c.ClipLinestring(pts)
	if len(segs) == 0 {
		t.Fatal("expected at least one segment")
	}
}

func TestRectangleClipperContains(t *testing.T) {
	c := core.NewRectangleClipper(0, 0, 20, 10)
	if !c.IsInside(15, 5) {
		t.Fatal("should be inside")
	}
	if c.IsInside(25, 0) {
		t.Fatal("should be outside x")
	}
	if c.IsInside(5, 15) {
		t.Fatal("should be outside z")
	}
}
