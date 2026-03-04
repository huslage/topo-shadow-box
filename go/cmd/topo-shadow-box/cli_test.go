package main

import (
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
	f, err := parseFlags([]string{"--lat", "35.99", "--lon", "-78.90", "--radius", "5000", "--output", "out.3mf"})
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
}
