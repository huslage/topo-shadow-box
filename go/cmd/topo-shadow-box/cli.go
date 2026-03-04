package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/huslage/topo-shadow-box/internal/core"
	"github.com/huslage/topo-shadow-box/internal/exporters"
	"github.com/huslage/topo-shadow-box/internal/session"
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

func inferFormat(path string) (string, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".3mf":
		return "3mf", nil
	case ".scad":
		return "openscad", nil
	case ".svg":
		return "svg", nil
	default:
		return "", fmt.Errorf("cannot infer output format from extension %q; use .3mf, .scad, or .svg", ext)
	}
}

var validFeatures = map[string]bool{"roads": true, "water": true, "buildings": true}

func parseFeatures(s string) ([]string, error) {
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	for _, p := range parts {
		if !validFeatures[p] {
			return nil, fmt.Errorf("invalid feature %q: must be roads, water, or buildings", p)
		}
	}
	return parts, nil
}

func runCLI(args []string) error {
	f, err := parseFlags(args)
	if err != nil {
		return err
	}
	if err := validateFlags(f); err != nil {
		return err
	}
	return executeCLI(f)
}

func executeCLI(f cliFlags) error {
	format, err := inferFormat(f.output)
	if err != nil {
		return err
	}
	features, err := parseFeatures(f.features)
	if err != nil {
		return err
	}

	sess := session.New()
	ctx := context.Background()

	sess.Config.ModelParams.WidthMM = f.width
	sess.Config.ModelParams.VerticalScale = f.verticalScale
	sess.Config.ModelParams.BaseHeightMM = f.baseHeight
	sess.Config.ModelParams.Shape = f.shape

	sess.Config.Colors.Terrain = f.colorTerrain
	sess.Config.Colors.Roads = f.colorRoads
	sess.Config.Colors.Water = f.colorWater
	sess.Config.Colors.Buildings = f.colorBuildings
	sess.Config.Colors.GpxTrack = f.colorGpxTrack

	fmt.Fprintln(os.Stderr, "Setting area...")
	switch {
	case f.gpx != "":
		if err := core.SetAreaFromGPX(ctx, sess, f.gpx, 500); err != nil {
			return fmt.Errorf("set area from GPX: %w", err)
		}
	case f.radius > 0:
		if err := core.SetAreaFromCoordinates(ctx, sess, f.lat, f.lon, f.radius); err != nil {
			return fmt.Errorf("set area from coordinates: %w", err)
		}
	default:
		if err := core.SetAreaFromBbox(ctx, sess, f.north, f.south, f.east, f.west); err != nil {
			return fmt.Errorf("set area from bounding box: %w", err)
		}
	}

	fmt.Fprintln(os.Stderr, "Fetching elevation...")
	if err := core.FetchElevation(ctx, sess, f.resolution); err != nil {
		return fmt.Errorf("fetch elevation: %w", err)
	}

	if len(features) > 0 {
		fmt.Fprintln(os.Stderr, "Fetching map features...")
		if err := core.FetchFeatures(ctx, sess, features); err != nil {
			return fmt.Errorf("fetch features: %w", err)
		}
	}

	fmt.Fprintln(os.Stderr, "Generating model...")
	if err := core.GenerateModel(ctx, sess); err != nil {
		return fmt.Errorf("generate model: %w", err)
	}

	fmt.Fprintf(os.Stderr, "Exporting to %s...\n", f.output)
	switch format {
	case "3mf":
		err = exporters.Export3MF(sess, f.output)
	case "openscad":
		err = exporters.ExportOpenSCAD(sess, f.output)
	case "svg":
		err = exporters.ExportSVG(sess, f.output)
	}
	if err != nil {
		return fmt.Errorf("export: %w", err)
	}
	fmt.Fprintf(os.Stderr, "Done: %s\n", f.output)
	return nil
}
