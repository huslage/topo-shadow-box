package session

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type sessionFile struct {
	Bounds      *boundsFile     `json:"bounds,omitempty"`
	ModelParams modelParamsFile `json:"model_params"`
	Colors      colorsFile      `json:"colors"`
	GpxTracks   []gpxTrackFile  `json:"gpx_tracks,omitempty"`
}

type boundsFile struct {
	North float64 `json:"north"`
	South float64 `json:"south"`
	East  float64 `json:"east"`
	West  float64 `json:"west"`
}

type modelParamsFile struct {
	WidthMM       float64 `json:"width_mm"`
	VerticalScale float64 `json:"vertical_scale"`
	BaseHeightMM  float64 `json:"base_height_mm"`
	Shape         string  `json:"shape"`
}

type colorsFile struct {
	Terrain   string `json:"terrain"`
	Water     string `json:"water"`
	Roads     string `json:"roads"`
	Buildings string `json:"buildings"`
	GpxTrack  string `json:"gpx_track"`
	MapInsert string `json:"map_insert"`
}

type gpxPointFile struct {
	Lat       float64 `json:"lat"`
	Lon       float64 `json:"lon"`
	Elevation float64 `json:"elevation"`
}

type gpxTrackFile struct {
	Name   string         `json:"name"`
	Points []gpxPointFile `json:"points"`
}

func DefaultPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "topo-shadow-box", "session.json")
}

func SaveSession(s *Session, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create dir: %w", err)
	}

	f := sessionFile{
		ModelParams: modelParamsFile{
			WidthMM:       s.Config.ModelParams.WidthMM,
			VerticalScale: s.Config.ModelParams.VerticalScale,
			BaseHeightMM:  s.Config.ModelParams.BaseHeightMM,
			Shape:         s.Config.ModelParams.Shape,
		},
		Colors: colorsFile{
			Terrain: s.Config.Colors.Terrain, Water: s.Config.Colors.Water,
			Roads: s.Config.Colors.Roads, Buildings: s.Config.Colors.Buildings,
			GpxTrack: s.Config.Colors.GpxTrack, MapInsert: s.Config.Colors.MapInsert,
		},
	}
	if s.Config.Bounds.IsSet {
		f.Bounds = &boundsFile{
			North: s.Config.Bounds.North, South: s.Config.Bounds.South,
			East: s.Config.Bounds.East, West: s.Config.Bounds.West,
		}
	}
	for _, t := range s.Config.GpxTracks {
		tf := gpxTrackFile{Name: t.Name}
		for _, p := range t.Points {
			tf.Points = append(tf.Points, gpxPointFile{Lat: p.Lat, Lon: p.Lon, Elevation: p.Elevation})
		}
		f.GpxTracks = append(f.GpxTracks, tf)
	}

	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

func LoadSession(s *Session, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}
	var f sessionFile
	if err := json.Unmarshal(data, &f); err != nil {
		return fmt.Errorf("parse json: %w", err)
	}

	if f.Bounds != nil {
		s.Config.Bounds = Bounds{
			North: f.Bounds.North, South: f.Bounds.South,
			East: f.Bounds.East, West: f.Bounds.West, IsSet: true,
		}
	}
	s.Config.ModelParams = ModelParams{
		WidthMM:       f.ModelParams.WidthMM,
		VerticalScale: f.ModelParams.VerticalScale,
		BaseHeightMM:  f.ModelParams.BaseHeightMM,
		Shape:         f.ModelParams.Shape,
	}
	s.Config.Colors = Colors{
		Terrain: f.Colors.Terrain, Water: f.Colors.Water, Roads: f.Colors.Roads,
		Buildings: f.Colors.Buildings, GpxTrack: f.Colors.GpxTrack, MapInsert: f.Colors.MapInsert,
	}
	s.Config.GpxTracks = nil
	for _, t := range f.GpxTracks {
		track := GpxTrack{Name: t.Name}
		for _, p := range t.Points {
			track.Points = append(track.Points, GpxPoint{Lat: p.Lat, Lon: p.Lon, Elevation: p.Elevation})
		}
		s.Config.GpxTracks = append(s.Config.GpxTracks, track)
	}
	s.ClearDownstream()
	return nil
}
