package core_test

import (
	"context"
	"testing"

	"github.com/huslage/topo-shadow-box/internal/core"
	"github.com/huslage/topo-shadow-box/internal/session"
)

func TestSetAreaFromCoordinates(t *testing.T) {
	s := session.New()
	if err := core.SetAreaFromCoordinates(context.Background(), s, 35.99, -78.9, 1000); err != nil {
		t.Fatalf("set area failed: %v", err)
	}
	if !s.Config.Bounds.IsSet {
		t.Fatal("bounds should be set")
	}
}

func TestGenerateModelRequiresElevation(t *testing.T) {
	s := session.New()
	s.Config.Bounds = session.Bounds{North: 1, South: 0, East: 1, West: 0, IsSet: true}
	err := core.GenerateModel(context.Background(), s)
	if err == nil {
		t.Fatal("expected error when elevation is missing")
	}
}

func TestGenerateModelCreatesTerrain(t *testing.T) {
	s := session.New()
	s.Config.Bounds = session.Bounds{North: 1, South: 0, East: 1, West: 0, IsSet: true}
	s.FetchedData.Elevation = &session.ElevationData{
		Grid:         [][]float64{{100, 110}, {90, 95}},
		Lats:         []float64{1, 0},
		Lons:         []float64{0, 1},
		Resolution:   2,
		MinElevation: 90,
		MaxElevation: 110,
		IsSet:        true,
	}
	if err := core.GenerateModel(context.Background(), s); err != nil {
		t.Fatalf("generate model failed: %v", err)
	}
	if s.Results.TerrainMesh == nil {
		t.Fatal("expected terrain mesh")
	}
	if len(s.Results.TerrainMesh.Vertices) == 0 || len(s.Results.TerrainMesh.Faces) == 0 {
		t.Fatal("terrain mesh should have vertices and faces")
	}
}

func TestGenerateModelCreatesFeatureAndGpxMeshes(t *testing.T) {
	s := session.New()
	s.Config.Bounds = session.Bounds{North: 1, South: 0, East: 1, West: 0, IsSet: true}
	s.Config.GpxTracks = []session.GpxTrack{{
		Name: "test",
		Points: []session.GpxPoint{
			{Lat: 0.9, Lon: 0.1},
			{Lat: 0.7, Lon: 0.3},
		},
	}}
	s.FetchedData.Elevation = &session.ElevationData{
		Grid:         [][]float64{{100, 110, 120}, {90, 95, 100}, {80, 85, 90}},
		Lats:         []float64{1, 0.5, 0},
		Lons:         []float64{0, 0.5, 1},
		Resolution:   3,
		MinElevation: 80,
		MaxElevation: 120,
		IsSet:        true,
	}
	s.FetchedData.Features = &session.OsmFeatureSet{
		Roads: []session.RoadFeature{{
			ID: 1, Coordinates: []session.Coordinate{{Lat: 0.9, Lon: 0.2}, {Lat: 0.8, Lon: 0.5}},
		}},
		Water: []session.WaterFeature{{
			ID: 2, Coordinates: []session.Coordinate{{Lat: 0.7, Lon: 0.2}, {Lat: 0.7, Lon: 0.4}, {Lat: 0.6, Lon: 0.4}},
		}},
		Buildings: []session.BuildingFeature{{
			ID: 3, Height: 10,
			Coordinates: []session.Coordinate{{Lat: 0.4, Lon: 0.2}, {Lat: 0.4, Lon: 0.3}, {Lat: 0.3, Lon: 0.3}},
		}},
	}

	if err := core.GenerateModel(context.Background(), s); err != nil {
		t.Fatalf("generate model failed: %v", err)
	}
	if len(s.Results.FeatureMeshes) == 0 {
		t.Fatal("expected feature meshes")
	}
	if s.Results.GpxMesh == nil {
		t.Fatal("expected gpx mesh")
	}
}
