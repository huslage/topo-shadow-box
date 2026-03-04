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
