package main

import (
	"flag"
	"fmt"
	"math"
	"regexp"
)

type cliFlags struct {
	// Input — coordinates
	lat    float64
	lon    float64
	radius float64
	// Input — bounding box
	north float64
	south float64
	east  float64
	west  float64
	// Input — GPX
	gpx string
	// Output
	output string
	// Model params
	width         float64
	verticalScale float64
	baseHeight    float64
	shape         string
	resolution    int
	// Features
	features string
	// Colors
	colorTerrain   string
	colorRoads     string
	colorWater     string
	colorBuildings string
	colorGpxTrack  string
}

func parseFlags(args []string) (cliFlags, error) {
	fs := flag.NewFlagSet("topo-shadow-box", flag.ContinueOnError)
	var f cliFlags

	fs.Float64Var(&f.lat, "lat", math.NaN(), "Center latitude")
	fs.Float64Var(&f.lon, "lon", math.NaN(), "Center longitude")
	fs.Float64Var(&f.radius, "radius", 0, "Radius in meters")
	fs.Float64Var(&f.north, "north", math.NaN(), "North boundary")
	fs.Float64Var(&f.south, "south", math.NaN(), "South boundary")
	fs.Float64Var(&f.east, "east", math.NaN(), "East boundary")
	fs.Float64Var(&f.west, "west", math.NaN(), "West boundary")
	fs.StringVar(&f.gpx, "gpx", "", "Path to GPX file")

	const outputUsage = "Output file path (.3mf, .scad, or .svg)"
	fs.StringVar(&f.output, "output", "", outputUsage)
	fs.StringVar(&f.output, "o", "", outputUsage)

	fs.Float64Var(&f.width, "width", 200, "Model width in mm")
	fs.Float64Var(&f.verticalScale, "vertical-scale", 1.5, "Elevation exaggeration")
	fs.Float64Var(&f.baseHeight, "base-height", 10, "Base height in mm")
	fs.StringVar(&f.shape, "shape", "square", "Model shape (square/circle/hexagon/rectangle)")
	fs.IntVar(&f.resolution, "resolution", 200, "Grid resolution")

	fs.StringVar(&f.features, "features", "roads,water,buildings", "OSM features (roads,water,buildings)")

	fs.StringVar(&f.colorTerrain, "color-terrain", "#C8A882", "Terrain color (#RRGGBB)")
	fs.StringVar(&f.colorRoads, "color-roads", "#D4C5A9", "Roads color (#RRGGBB)")
	fs.StringVar(&f.colorWater, "color-water", "#4682B4", "Water color (#RRGGBB)")
	fs.StringVar(&f.colorBuildings, "color-buildings", "#E8D5B7", "Buildings color (#RRGGBB)")
	fs.StringVar(&f.colorGpxTrack, "color-gpx-track", "#FF0000", "GPX track color (#RRGGBB)")

	if err := fs.Parse(args); err != nil {
		return cliFlags{}, err
	}
	return f, nil
}

var validShapes = map[string]bool{"square": true, "circle": true, "hexagon": true, "rectangle": true}
var hexColorRe = regexp.MustCompile(`^#[0-9A-Fa-f]{6}$`)

func validateFlags(f cliFlags) error {
	if f.output == "" {
		return fmt.Errorf("--output is required")
	}

	hasCoords := !math.IsNaN(f.lat) || !math.IsNaN(f.lon) || f.radius != 0
	hasBbox := !math.IsNaN(f.north) || !math.IsNaN(f.south) || !math.IsNaN(f.east) || !math.IsNaN(f.west)
	hasGPX := f.gpx != ""

	inputCount := 0
	if hasCoords {
		inputCount++
	}
	if hasBbox {
		inputCount++
	}
	if hasGPX {
		inputCount++
	}

	if inputCount == 0 {
		return fmt.Errorf("one of --lat/--lon/--radius, --north/--south/--east/--west, or --gpx is required")
	}
	if inputCount > 1 {
		return fmt.Errorf("--lat/--lon/--radius, --north/--south/--east/--west, and --gpx are mutually exclusive")
	}

	if hasCoords {
		if math.IsNaN(f.lat) || math.IsNaN(f.lon) || f.radius <= 0 {
			return fmt.Errorf("--lat, --lon, and --radius must all be provided and radius must be positive")
		}
	}
	if hasBbox {
		if math.IsNaN(f.north) || math.IsNaN(f.south) || math.IsNaN(f.east) || math.IsNaN(f.west) {
			return fmt.Errorf("--north, --south, --east, and --west must all be provided together")
		}
		if f.north <= f.south {
			return fmt.Errorf("--north must be greater than --south")
		}
		if f.east <= f.west {
			return fmt.Errorf("--east must be greater than --west")
		}
	}

	if !validShapes[f.shape] {
		return fmt.Errorf("invalid --shape %q: must be square, circle, hexagon, or rectangle", f.shape)
	}

	for name, color := range map[string]string{
		"--color-terrain":   f.colorTerrain,
		"--color-roads":     f.colorRoads,
		"--color-water":     f.colorWater,
		"--color-buildings": f.colorBuildings,
		"--color-gpx-track": f.colorGpxTrack,
	} {
		if !hexColorRe.MatchString(color) {
			return fmt.Errorf("%s must be in #RRGGBB format, got %q", name, color)
		}
	}

	return nil
}

func runCLI(args []string) error {
	f, err := parseFlags(args)
	if err != nil {
		return err
	}
	_ = f
	return nil // pipeline wired in later tasks
}
