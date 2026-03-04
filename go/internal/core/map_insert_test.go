package core_test

import (
	"strings"
	"testing"

	"github.com/huslage/topo-shadow-box/internal/core"
	"github.com/huslage/topo-shadow-box/internal/session"
)

func TestGenerateMapInsertSVG(t *testing.T) {
	b := session.Bounds{North: 36, South: 35, East: -82, West: -83, IsSet: true}
	svg := core.GenerateMapInsertSVG(b, nil, nil, session.DefaultColors())
	if !strings.Contains(svg, "<svg") {
		t.Fatal("expected svg output")
	}
}

func TestGenerateMapInsertPlate(t *testing.T) {
	b := session.Bounds{North: 36, South: 35, East: -82, West: -83, IsSet: true}
	tr := core.NewGeoToModelTransform(b, 200)
	m := core.GenerateMapInsertPlate(b, tr, 1.0)
	if m == nil {
		t.Fatal("expected plate mesh")
	}
	if len(m.Vertices) == 0 || len(m.Faces) == 0 {
		t.Fatal("plate mesh should have vertices/faces")
	}
}
