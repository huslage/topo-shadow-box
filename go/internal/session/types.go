package session

import (
	"fmt"
	"regexp"
)

type Bounds struct {
	North float64
	South float64
	East  float64
	West  float64
	IsSet bool
}

func (b Bounds) Validate() error {
	if !b.IsSet {
		return nil
	}
	if b.North <= b.South {
		return fmt.Errorf("north (%.6f) must be greater than south (%.6f)", b.North, b.South)
	}
	if b.East <= b.West {
		return fmt.Errorf("east (%.6f) must be greater than west (%.6f)", b.East, b.West)
	}
	return nil
}

func (b Bounds) LatRange() float64  { return b.North - b.South }
func (b Bounds) LonRange() float64  { return b.East - b.West }
func (b Bounds) CenterLat() float64 { return (b.North + b.South) / 2 }
func (b Bounds) CenterLon() float64 { return (b.East + b.West) / 2 }

type ModelParams struct {
	WidthMM       float64
	VerticalScale float64
	BaseHeightMM  float64
	Shape         string // "square" | "circle" | "hexagon" | "rectangle"
}

func DefaultModelParams() ModelParams {
	return ModelParams{WidthMM: 200, VerticalScale: 1.5, BaseHeightMM: 10, Shape: "square"}
}

var hexColorRe = regexp.MustCompile(`^#[0-9A-Fa-f]{6}$`)

type Colors struct {
	Terrain   string
	Water     string
	Roads     string
	Buildings string
	GpxTrack  string
	MapInsert string
}

func DefaultColors() Colors {
	return Colors{
		Terrain: "#C8A882", Water: "#4682B4", Roads: "#D4C5A9",
		Buildings: "#E8D5B7", GpxTrack: "#FF0000", MapInsert: "#FFFFFF",
	}
}

func (c Colors) Validate() error {
	for name, val := range map[string]string{
		"terrain": c.Terrain, "water": c.Water, "roads": c.Roads,
		"buildings": c.Buildings, "gpx_track": c.GpxTrack, "map_insert": c.MapInsert,
	} {
		if !hexColorRe.MatchString(val) {
			return fmt.Errorf("invalid hex color %q for %s", val, name)
		}
	}
	return nil
}

func HexToRGB(hex string) (r, g, b uint8) {
	fmt.Sscanf(hex[1:], "%02x%02x%02x", &r, &g, &b)
	return
}

type Coordinate struct {
	Lat float64
	Lon float64
}

type GpxPoint struct {
	Lat       float64
	Lon       float64
	Elevation float64
}

type GpxTrack struct {
	Name   string
	Points []GpxPoint
}

type GpxWaypoint struct {
	Name        string
	Lat         float64
	Lon         float64
	Elevation   float64
	Description string
}

type RoadFeature struct {
	ID          int
	Coordinates []Coordinate
	Tags        map[string]string
	Name        string
	RoadType    string
}

type WaterFeature struct {
	ID          int
	Coordinates []Coordinate
	Tags        map[string]string
	Name        string
}

type BuildingFeature struct {
	ID          int
	Coordinates []Coordinate
	Tags        map[string]string
	Name        string
	Height      float64
}

type GeocodeCandidate struct {
	DisplayName string
	Lat         float64
	Lon         float64
	PlaceType   string
	BboxNorth   float64
	BboxSouth   float64
	BboxEast    float64
	BboxWest    float64
}

type OsmFeatureSet struct {
	Roads     []RoadFeature
	Water     []WaterFeature
	Buildings []BuildingFeature
}

type ElevationData struct {
	Grid         [][]float64
	Lats         []float64
	Lons         []float64
	Resolution   int
	MinElevation float64
	MaxElevation float64
	IsSet        bool
}

type Mesh struct {
	Vertices    [][3]float64
	Faces       [][3]int
	Name        string
	FeatureType string
}
