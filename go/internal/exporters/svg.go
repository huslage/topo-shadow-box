package exporters

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/huslage/topo-shadow-box/internal/session"
)

func ExportSVG(s *session.Session, outputPath string) error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}
	b := s.Config.Bounds
	if !b.IsSet {
		return fmt.Errorf("area bounds are not set")
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	width := 1000.0
	height := 1000.0
	if latSpan := b.North - b.South; latSpan > 0 {
		if lonSpan := b.East - b.West; lonSpan > 0 {
			height = width * (latSpan / lonSpan)
		}
	}
	if height < 200 {
		height = 200
	}
	if height > 2000 {
		height = 2000
	}

	project := func(lat, lon float64) (float64, float64) {
		x := (lon - b.West) / (b.East - b.West) * width
		y := (b.North - lat) / (b.North - b.South) * height
		return x, y
	}

	var out strings.Builder
	fmt.Fprintf(&out, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">\n", width, height, width, height)
	fmt.Fprintf(&out, "  <rect x=\"0\" y=\"0\" width=\"%.0f\" height=\"%.0f\" fill=\"%s\"/>\n", width, height, s.Config.Colors.Terrain)

	if s.FetchedData.Features != nil {
		for _, water := range s.FetchedData.Features.Water {
			if len(water.Coordinates) < 3 {
				continue
			}
			pts := make([]string, 0, len(water.Coordinates))
			for _, c := range water.Coordinates {
				x, y := project(c.Lat, c.Lon)
				pts = append(pts, fmt.Sprintf("%.2f,%.2f", x, y))
			}
			fmt.Fprintf(&out, "  <polygon points=\"%s\" fill=\"%s\" stroke=\"none\"/>\n", strings.Join(pts, " "), s.Config.Colors.Water)
		}

		for _, bld := range s.FetchedData.Features.Buildings {
			if len(bld.Coordinates) < 3 {
				continue
			}
			pts := make([]string, 0, len(bld.Coordinates))
			for _, c := range bld.Coordinates {
				x, y := project(c.Lat, c.Lon)
				pts = append(pts, fmt.Sprintf("%.2f,%.2f", x, y))
			}
			fmt.Fprintf(&out, "  <polygon points=\"%s\" fill=\"%s\" stroke=\"none\"/>\n", strings.Join(pts, " "), s.Config.Colors.Buildings)
		}

		for _, road := range s.FetchedData.Features.Roads {
			if len(road.Coordinates) < 2 {
				continue
			}
			pts := make([]string, 0, len(road.Coordinates))
			for _, c := range road.Coordinates {
				x, y := project(c.Lat, c.Lon)
				pts = append(pts, fmt.Sprintf("%.2f,%.2f", x, y))
			}
			fmt.Fprintf(&out, "  <polyline points=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n", strings.Join(pts, " "), s.Config.Colors.Roads)
		}
	}

	for _, tr := range s.Config.GpxTracks {
		if len(tr.Points) < 2 {
			continue
		}
		pts := make([]string, 0, len(tr.Points))
		for _, p := range tr.Points {
			x, y := project(p.Lat, p.Lon)
			pts = append(pts, fmt.Sprintf("%.2f,%.2f", x, y))
		}
		fmt.Fprintf(&out, "  <polyline points=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n", strings.Join(pts, " "), s.Config.Colors.GpxTrack)
	}

	out.WriteString("</svg>\n")
	if err := os.WriteFile(outputPath, []byte(out.String()), 0o644); err != nil {
		return fmt.Errorf("write svg: %w", err)
	}
	return nil
}
