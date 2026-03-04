package core

import (
	"fmt"
	"math"
	"strings"

	"github.com/huslage/topo-shadow-box/internal/session"
)

func GenerateMapInsertSVG(bounds session.Bounds, features *session.OsmFeatureSet, gpxTracks []session.GpxTrack, colors session.Colors) string {
	width := 800.0
	latRange := bounds.LatRange()
	lonRange := bounds.LonRange()
	lonScale := math.Cos(math.Pi / 180.0 * bounds.CenterLat())
	aspect := 1.0
	if latRange > 0 {
		aspect = (lonRange * lonScale) / latRange
		if aspect <= 0 {
			aspect = 1
		}
	}
	height := width / aspect
	if height < 100 {
		height = 100
	}
	if height > 2000 {
		height = 2000
	}

	geoToSVG := func(lat, lon float64) (float64, float64) {
		x := 0.0
		y := 0.0
		if lonRange > 0 {
			x = (lon - bounds.West) / lonRange * width
		}
		if latRange > 0 {
			y = (bounds.North - lat) / latRange * height
		}
		return x, y
	}

	parts := []string{
		fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f">`, width, height, width, height),
		fmt.Sprintf(`<rect width="%.0f" height="%.0f" fill="%s"/>`, width, height, colors.MapInsert),
	}

	if features != nil {
		for _, w := range features.Water {
			if len(w.Coordinates) < 3 {
				continue
			}
			pts := make([]string, 0, len(w.Coordinates))
			for _, c := range w.Coordinates {
				x, y := geoToSVG(c.Lat, c.Lon)
				pts = append(pts, fmt.Sprintf("%.1f,%.1f", x, y))
			}
			parts = append(parts, fmt.Sprintf(`<polygon points="%s" fill="%s" opacity="0.6"/>`, strings.Join(pts, " "), colors.Water))
		}

		for _, r := range features.Roads {
			if len(r.Coordinates) < 2 {
				continue
			}
			pts := make([]string, 0, len(r.Coordinates))
			for _, c := range r.Coordinates {
				x, y := geoToSVG(c.Lat, c.Lon)
				pts = append(pts, fmt.Sprintf("%.1f,%.1f", x, y))
			}
			parts = append(parts, fmt.Sprintf(`<polyline points="%s" fill="none" stroke="%s" stroke-width="1" opacity="0.5"/>`, strings.Join(pts, " "), colors.Roads))
		}

		for _, b := range features.Buildings {
			if len(b.Coordinates) < 3 {
				continue
			}
			pts := make([]string, 0, len(b.Coordinates))
			for _, c := range b.Coordinates {
				x, y := geoToSVG(c.Lat, c.Lon)
				pts = append(pts, fmt.Sprintf("%.1f,%.1f", x, y))
			}
			parts = append(parts, fmt.Sprintf(`<polygon points="%s" fill="%s" opacity="0.4"/>`, strings.Join(pts, " "), colors.Buildings))
		}
	}

	for _, tr := range gpxTracks {
		if len(tr.Points) < 2 {
			continue
		}
		pts := make([]string, 0, len(tr.Points))
		for _, p := range tr.Points {
			x, y := geoToSVG(p.Lat, p.Lon)
			pts = append(pts, fmt.Sprintf("%.1f,%.1f", x, y))
		}
		parts = append(parts, fmt.Sprintf(`<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>`, strings.Join(pts, " "), colors.GpxTrack))
	}

	parts = append(parts, `</svg>`)
	return strings.Join(parts, "\n")
}

func GenerateMapInsertPlate(bounds session.Bounds, transform GeoToModelTransform, plateThicknessMM float64) *session.Mesh {
	_ = bounds
	if plateThicknessMM <= 0 {
		plateThicknessMM = 1.0
	}
	w := transform.ModelWidthX
	h := transform.ModelWidthZ
	t := plateThicknessMM

	yTop := 0.0
	yBot := -t
	vertices := [][3]float64{
		{0, yTop, 0}, {w, yTop, 0}, {w, yTop, h}, {0, yTop, h},
		{0, yBot, 0}, {w, yBot, 0}, {w, yBot, h}, {0, yBot, h},
	}
	faces := [][3]int{
		{0, 1, 2}, {0, 2, 3},
		{4, 6, 5}, {4, 7, 6},
		{3, 2, 6}, {3, 6, 7},
		{0, 5, 1}, {0, 4, 5},
		{0, 3, 7}, {0, 7, 4},
		{1, 6, 2}, {1, 5, 6},
	}
	return &session.Mesh{Vertices: vertices, Faces: faces, Name: "Map Insert", FeatureType: "map_insert"}
}
