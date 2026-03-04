package main

import (
	"math"
	"testing"
)

func TestParseFlagsCoordinates(t *testing.T) {
	f, err := parseFlags([]string{"--lat", "35.99", "--lon", "-78.90", "--radius", "5000", "--output", "out.3mf"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f.lat != 35.99 {
		t.Errorf("lat: got %v, want 35.99", f.lat)
	}
	if f.output != "out.3mf" {
		t.Errorf("output: got %v, want out.3mf", f.output)
	}
}

func TestParseFlagsGPX(t *testing.T) {
	f, err := parseFlags([]string{"--gpx", "ride.gpx", "--output", "out.3mf"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f.gpx != "ride.gpx" {
		t.Errorf("gpx: got %v, want ride.gpx", f.gpx)
	}
}

func TestParseFlagsDefaults(t *testing.T) {
	// Use --gpx to provide a valid input without setting coordinates
	f, err := parseFlags([]string{"--gpx", "ride.gpx", "--output", "out.3mf"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f.width != 200 {
		t.Errorf("width default: got %v, want 200", f.width)
	}
	if f.verticalScale != 1.5 {
		t.Errorf("verticalScale default: got %v, want 1.5", f.verticalScale)
	}
	if f.shape != "square" {
		t.Errorf("shape default: got %v, want square", f.shape)
	}
	if f.features != "roads,water,buildings" {
		t.Errorf("features default: got %v, want roads,water,buildings", f.features)
	}
	if f.colorTerrain != "#C8A882" {
		t.Errorf("colorTerrain default: got %v, want #C8A882", f.colorTerrain)
	}
	if !math.IsNaN(f.lat) {
		t.Errorf("lat should be NaN when not set, got %v", f.lat)
	}
}

func TestParseFlagsShorthandOutput(t *testing.T) {
	f, err := parseFlags([]string{"--lat", "35.99", "--lon", "-78.90", "--radius", "5000", "-o", "out.3mf"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f.output != "out.3mf" {
		t.Errorf("output via -o: got %v, want out.3mf", f.output)
	}
}

func TestParseFlagsBbox(t *testing.T) {
	f, err := parseFlags([]string{"--north", "36.1", "--south", "35.9", "--east", "-78.8", "--west", "-79.0", "--output", "out.3mf"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f.north != 36.1 {
		t.Errorf("north: got %v, want 36.1", f.north)
	}
	if f.south != 35.9 {
		t.Errorf("south: got %v, want 35.9", f.south)
	}
}

func TestParseFlagsModelParamOverride(t *testing.T) {
	f, err := parseFlags([]string{"--lat", "35.99", "--lon", "-78.90", "--radius", "5000", "--output", "out.3mf",
		"--width", "150", "--shape", "circle", "--vertical-scale", "2.0"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f.width != 150 {
		t.Errorf("width: got %v, want 150", f.width)
	}
	if f.shape != "circle" {
		t.Errorf("shape: got %v, want circle", f.shape)
	}
	if f.verticalScale != 2.0 {
		t.Errorf("verticalScale: got %v, want 2.0", f.verticalScale)
	}
}

func TestParseFlagsColorOverride(t *testing.T) {
	f, err := parseFlags([]string{"--lat", "35.99", "--lon", "-78.90", "--radius", "5000", "--output", "out.3mf",
		"--color-terrain", "#8B7355"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f.colorTerrain != "#8B7355" {
		t.Errorf("colorTerrain: got %v, want #8B7355", f.colorTerrain)
	}
}

func TestParseFlagsUnknownFlag(t *testing.T) {
	_, err := parseFlags([]string{"--bogus", "value"})
	if err == nil {
		t.Error("expected error for unknown flag, got nil")
	}
}

func TestParseFlagsNotSetCoordsSentinel(t *testing.T) {
	// When no coords are passed, lat/lon should be NaN (not set)
	f, err := parseFlags([]string{"--gpx", "ride.gpx", "--output", "out.3mf"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !math.IsNaN(f.lat) {
		t.Errorf("lat should be NaN when not set, got %v", f.lat)
	}
	if !math.IsNaN(f.lon) {
		t.Errorf("lon should be NaN when not set, got %v", f.lon)
	}
}

func TestValidateFlagsMissingOutput(t *testing.T) {
	f := cliFlags{lat: 35.99, lon: -78.90, radius: 5000, north: math.NaN(), south: math.NaN(), east: math.NaN(), west: math.NaN()}
	err := validateFlags(f)
	if err == nil || err.Error() != "--output is required" {
		t.Errorf("got %v, want '--output is required'", err)
	}
}

func TestValidateFlagsNoInput(t *testing.T) {
	f := cliFlags{output: "out.3mf", lat: math.NaN(), lon: math.NaN(), north: math.NaN(), south: math.NaN(), east: math.NaN(), west: math.NaN()}
	err := validateFlags(f)
	if err == nil {
		t.Error("expected error for no input method, got nil")
	}
}

func TestValidateFlagsMutualExclusion(t *testing.T) {
	f := cliFlags{lat: 35.99, lon: -78.90, radius: 5000, gpx: "ride.gpx", output: "out.3mf",
		north: math.NaN(), south: math.NaN(), east: math.NaN(), west: math.NaN()}
	err := validateFlags(f)
	if err == nil {
		t.Error("expected error for multiple input methods, got nil")
	}
}

func TestValidateFlagsCoordsMissingRadius(t *testing.T) {
	f := cliFlags{lat: 35.99, lon: -78.90, radius: 0, output: "out.3mf",
		north: math.NaN(), south: math.NaN(), east: math.NaN(), west: math.NaN()}
	err := validateFlags(f)
	if err == nil {
		t.Error("expected error for lat/lon without radius, got nil")
	}
}

func TestValidateFlagsInvalidShape(t *testing.T) {
	f := cliFlags{lat: 35.99, lon: -78.90, radius: 5000, output: "out.3mf", shape: "triangle",
		north: math.NaN(), south: math.NaN(), east: math.NaN(), west: math.NaN(),
		colorTerrain: "#C8A882", colorRoads: "#D4C5A9", colorWater: "#4682B4",
		colorBuildings: "#E8D5B7", colorGpxTrack: "#FF0000"}
	err := validateFlags(f)
	if err == nil {
		t.Error("expected error for invalid shape, got nil")
	}
}

func TestValidateFlagsInvalidColor(t *testing.T) {
	f := cliFlags{lat: 35.99, lon: -78.90, radius: 5000, output: "out.3mf", shape: "square",
		north: math.NaN(), south: math.NaN(), east: math.NaN(), west: math.NaN(),
		colorTerrain: "not-a-color", colorRoads: "#D4C5A9", colorWater: "#4682B4",
		colorBuildings: "#E8D5B7", colorGpxTrack: "#FF0000"}
	err := validateFlags(f)
	if err == nil {
		t.Error("expected error for invalid color, got nil")
	}
}

func TestValidateFlagsValid(t *testing.T) {
	f := cliFlags{lat: 35.99, lon: -78.90, radius: 5000, output: "out.3mf",
		north: math.NaN(), south: math.NaN(), east: math.NaN(), west: math.NaN(),
		shape: "square", features: "roads,water,buildings",
		colorTerrain: "#C8A882", colorRoads: "#D4C5A9",
		colorWater: "#4682B4", colorBuildings: "#E8D5B7", colorGpxTrack: "#FF0000"}
	if err := validateFlags(f); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidateFlagsValidGPX(t *testing.T) {
	f := cliFlags{gpx: "ride.gpx", output: "out.3mf",
		lat: math.NaN(), lon: math.NaN(), north: math.NaN(), south: math.NaN(), east: math.NaN(), west: math.NaN(),
		shape: "square", features: "roads,water,buildings",
		colorTerrain: "#C8A882", colorRoads: "#D4C5A9",
		colorWater: "#4682B4", colorBuildings: "#E8D5B7", colorGpxTrack: "#FF0000"}
	if err := validateFlags(f); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidateFlagsValidBbox(t *testing.T) {
	f := cliFlags{north: 36.1, south: 35.9, east: -78.8, west: -79.0, output: "out.3mf",
		lat: math.NaN(), lon: math.NaN(),
		shape: "square", features: "roads,water,buildings",
		colorTerrain: "#C8A882", colorRoads: "#D4C5A9",
		colorWater: "#4682B4", colorBuildings: "#E8D5B7", colorGpxTrack: "#FF0000"}
	if err := validateFlags(f); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestInferFormat(t *testing.T) {
	tests := []struct {
		path    string
		want    string
		wantErr bool
	}{
		{"out.3mf", "3mf", false},
		{"model.scad", "openscad", false},
		{"map.svg", "svg", false},
		{"OUT.3MF", "3mf", false},
		{"out.txt", "", true},
		{"out", "", true},
	}
	for _, tt := range tests {
		got, err := inferFormat(tt.path)
		if tt.wantErr {
			if err == nil {
				t.Errorf("inferFormat(%q): expected error, got nil", tt.path)
			}
			continue
		}
		if err != nil {
			t.Errorf("inferFormat(%q): unexpected error: %v", tt.path, err)
			continue
		}
		if got != tt.want {
			t.Errorf("inferFormat(%q): got %q, want %q", tt.path, got, tt.want)
		}
	}
}

func TestParseFeatures(t *testing.T) {
	tests := []struct {
		input   string
		want    []string
		wantErr bool
	}{
		{"roads,water,buildings", []string{"roads", "water", "buildings"}, false},
		{"roads", []string{"roads"}, false},
		{"water,buildings", []string{"water", "buildings"}, false},
		{"", nil, false},
		{"roads,invalid", nil, true},
	}
	for _, tt := range tests {
		got, err := parseFeatures(tt.input)
		if tt.wantErr {
			if err == nil {
				t.Errorf("parseFeatures(%q): expected error, got nil", tt.input)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseFeatures(%q): unexpected error: %v", tt.input, err)
			continue
		}
		if len(got) != len(tt.want) {
			t.Errorf("parseFeatures(%q): got %v, want %v", tt.input, got, tt.want)
			continue
		}
		for i := range tt.want {
			if got[i] != tt.want[i] {
				t.Errorf("parseFeatures(%q)[%d]: got %q, want %q", tt.input, i, got[i], tt.want[i])
			}
		}
	}
}
