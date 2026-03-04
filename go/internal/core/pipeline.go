package core

import (
	"context"
	"fmt"
	"math"

	"github.com/huslage/topo-shadow-box/internal/session"
)

func SetAreaFromCoordinates(ctx context.Context, s *session.Session, lat, lon, radiusM float64) error {
	_ = ctx
	if s == nil {
		return fmt.Errorf("session is nil")
	}
	if math.IsNaN(lat) || math.IsNaN(lon) || radiusM <= 0 {
		return fmt.Errorf("lat/lon must be set and radius must be positive")
	}

	s.Lock()
	defer s.Unlock()

	b := AddPaddingToBounds(session.Bounds{
		North: lat,
		South: lat,
		East:  lon,
		West:  lon,
	}, radiusM, true)
	if err := b.Validate(); err != nil {
		return err
	}
	s.Config.Bounds = b
	s.ClearDownstream()
	return nil
}

func SetAreaFromBbox(ctx context.Context, s *session.Session, north, south, east, west float64) error {
	_ = ctx
	if s == nil {
		return fmt.Errorf("session is nil")
	}
	b := session.Bounds{North: north, South: south, East: east, West: west, IsSet: true}
	if err := b.Validate(); err != nil {
		return err
	}

	s.Lock()
	defer s.Unlock()
	s.Config.Bounds = b
	s.ClearDownstream()
	return nil
}

func SetAreaFromGPX(ctx context.Context, s *session.Session, filePath string, paddingM float64) error {
	_ = ctx
	if s == nil {
		return fmt.Errorf("session is nil")
	}
	if paddingM < 0 {
		return fmt.Errorf("padding must be non-negative")
	}

	tracks, waypoints, bounds, err := ParseGPXFile(filePath)
	if err != nil {
		return err
	}
	if !bounds.IsSet {
		return fmt.Errorf("gpx file has no bounds")
	}

	padded := AddPaddingToBounds(bounds, paddingM, true)
	if err := padded.Validate(); err != nil {
		return err
	}

	s.Lock()
	defer s.Unlock()
	s.Config.Bounds = padded
	s.Config.GpxTracks = tracks
	s.Config.GpxWaypoints = waypoints
	s.ClearDownstream()
	return nil
}

func FetchElevation(ctx context.Context, s *session.Session, resolution int) error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}
	if resolution <= 1 {
		return fmt.Errorf("resolution must be > 1")
	}

	s.Lock()
	bounds := s.Config.Bounds
	s.Unlock()

	if !bounds.IsSet {
		return fmt.Errorf("area bounds are not set")
	}

	elev, err := FetchTerrainElevation(ctx, nil, bounds.North, bounds.South, bounds.East, bounds.West, resolution)
	if err != nil {
		return err
	}

	s.Lock()
	defer s.Unlock()
	s.FetchedData.Elevation = elev
	s.Results.TerrainMesh = nil
	s.Results.FeatureMeshes = nil
	s.Results.GpxMesh = nil
	return nil
}

func FetchFeatures(ctx context.Context, s *session.Session, featureTypes []string) error {
	if s == nil {
		return fmt.Errorf("session is nil")
	}

	s.Lock()
	bounds := s.Config.Bounds
	s.Unlock()

	if !bounds.IsSet {
		return fmt.Errorf("area bounds are not set")
	}

	features, err := FetchOSMFeatures(ctx, nil, bounds.North, bounds.South, bounds.East, bounds.West, featureTypes)
	if err != nil {
		return err
	}

	s.Lock()
	defer s.Unlock()
	s.FetchedData.Features = features
	s.Results.FeatureMeshes = nil
	return nil
}

func GenerateModel(ctx context.Context, s *session.Session) error {
	_ = ctx
	if s == nil {
		return fmt.Errorf("session is nil")
	}

	s.Lock()
	defer s.Unlock()

	if !s.Config.Bounds.IsSet {
		return fmt.Errorf("area bounds are not set")
	}
	if s.FetchedData.Elevation == nil || !s.FetchedData.Elevation.IsSet {
		return fmt.Errorf("elevation is not fetched")
	}

	terrain := GenerateTerrainMesh(s.FetchedData.Elevation, s.Config.Bounds, s.Config.ModelParams)
	s.Results.TerrainMesh = terrain

	s.Results.FeatureMeshes = nil
	if s.FetchedData.Features != nil {
		for _, road := range s.FetchedData.Features.Roads {
			if m := GenerateSingleFeatureMesh(road, "road", s.FetchedData.Elevation, s.Config.Bounds, s.Config.ModelParams); m != nil {
				s.Results.FeatureMeshes = append(s.Results.FeatureMeshes, *m)
			}
		}
		for _, water := range s.FetchedData.Features.Water {
			if m := GenerateSingleFeatureMesh(water, "water", s.FetchedData.Elevation, s.Config.Bounds, s.Config.ModelParams); m != nil {
				s.Results.FeatureMeshes = append(s.Results.FeatureMeshes, *m)
			}
		}
		for _, bld := range s.FetchedData.Features.Buildings {
			if m := GenerateSingleFeatureMesh(bld, "building", s.FetchedData.Elevation, s.Config.Bounds, s.Config.ModelParams); m != nil {
				s.Results.FeatureMeshes = append(s.Results.FeatureMeshes, *m)
			}
		}
	}

	s.Results.GpxMesh = GenerateGpxTrackMesh(s.Config.GpxTracks, s.FetchedData.Elevation, s.Config.Bounds, s.Config.ModelParams)
	return nil
}
