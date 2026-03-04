package main

import (
	"flag"
	"math"
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

func runCLI(args []string) error {
	f, err := parseFlags(args)
	if err != nil {
		return err
	}
	_ = f
	return nil // pipeline wired in later tasks
}
