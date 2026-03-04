package core_test

import (
	"testing"

	"github.com/huslage/topo-shadow-box/internal/core"
	"github.com/huslage/topo-shadow-box/internal/session"
)

func testElev() *session.ElevationData {
	return &session.ElevationData{
		Grid: [][]float64{
			{100, 105, 110},
			{95, 100, 105},
			{90, 95, 100},
		},
		Lats:         []float64{36.0, 35.95, 35.9},
		Lons:         []float64{-79.0, -78.95, -78.9},
		Resolution:   3,
		MinElevation: 90,
		MaxElevation: 110,
		IsSet:        true,
	}
}

func testBounds() session.Bounds {
	return session.Bounds{North: 36.0, South: 35.9, East: -78.9, West: -79.0, IsSet: true}
}

func TestGenerateTerrainMesh(t *testing.T) {
	mesh := core.GenerateTerrainMesh(testElev(), testBounds(), session.DefaultModelParams())
	if mesh == nil {
		t.Fatal("expected terrain mesh")
	}
	if len(mesh.Vertices) == 0 || len(mesh.Faces) == 0 {
		t.Fatal("terrain mesh should have vertices and faces")
	}
}

func TestGenerateSingleFeatureMeshRoad(t *testing.T) {
	road := session.RoadFeature{Coordinates: []session.Coordinate{{Lat: 35.99, Lon: -78.99}, {Lat: 35.95, Lon: -78.95}}}
	mesh := core.GenerateSingleFeatureMesh(road, "road", testElev(), testBounds(), session.DefaultModelParams())
	if mesh == nil {
		t.Fatal("expected road mesh")
	}
	if len(mesh.Faces) == 0 {
		t.Fatal("road mesh should have faces")
	}
}

func TestGenerateSingleFeatureMeshBuilding(t *testing.T) {
	b := session.BuildingFeature{Coordinates: []session.Coordinate{{Lat: 35.96, Lon: -78.98}, {Lat: 35.96, Lon: -78.97}, {Lat: 35.95, Lon: -78.97}, {Lat: 35.95, Lon: -78.98}}, Height: 12}
	mesh := core.GenerateSingleFeatureMesh(b, "building", testElev(), testBounds(), session.DefaultModelParams())
	if mesh == nil {
		t.Fatal("expected building mesh")
	}
	if len(mesh.Faces) == 0 {
		t.Fatal("building mesh should have faces")
	}
}

func TestGenerateGpxTrackMesh(t *testing.T) {
	tracks := []session.GpxTrack{{
		Name: "test",
		Points: []session.GpxPoint{
			{Lat: 35.99, Lon: -78.99},
			{Lat: 35.97, Lon: -78.97},
			{Lat: 35.95, Lon: -78.95},
		},
	}}
	mesh := core.GenerateGpxTrackMesh(tracks, testElev(), testBounds(), session.DefaultModelParams())
	if mesh == nil {
		t.Fatal("expected gpx mesh")
	}
	if len(mesh.Faces) == 0 {
		t.Fatal("gpx mesh should have faces")
	}
}
