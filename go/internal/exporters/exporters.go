package exporters

import (
	"fmt"

	"github.com/huslage/topo-shadow-box/internal/session"
)

type MeshExport struct {
	Name        string
	FeatureType string
	Color       string
	Vertices    [][3]float64
	Faces       [][3]int
}

func collectMeshes(s *session.Session) ([]MeshExport, error) {
	if s == nil {
		return nil, fmt.Errorf("session is nil")
	}
	var out []MeshExport
	if s.Results.TerrainMesh != nil {
		out = append(out, MeshExport{
			Name:        s.Results.TerrainMesh.Name,
			FeatureType: s.Results.TerrainMesh.FeatureType,
			Color:       s.Config.Colors.Terrain,
			Vertices:    s.Results.TerrainMesh.Vertices,
			Faces:       s.Results.TerrainMesh.Faces,
		})
	}
	for _, m := range s.Results.FeatureMeshes {
		out = append(out, MeshExport{
			Name:        m.Name,
			FeatureType: m.FeatureType,
			Color:       colorForFeatureType(s.Config.Colors, m.FeatureType),
			Vertices:    m.Vertices,
			Faces:       m.Faces,
		})
	}
	if s.Results.GpxMesh != nil {
		out = append(out, MeshExport{
			Name:        s.Results.GpxMesh.Name,
			FeatureType: s.Results.GpxMesh.FeatureType,
			Color:       s.Config.Colors.GpxTrack,
			Vertices:    s.Results.GpxMesh.Vertices,
			Faces:       s.Results.GpxMesh.Faces,
		})
	}
	if s.Results.MapInsertMesh != nil {
		out = append(out, MeshExport{
			Name:        s.Results.MapInsertMesh.Name,
			FeatureType: s.Results.MapInsertMesh.FeatureType,
			Color:       s.Config.Colors.MapInsert,
			Vertices:    s.Results.MapInsertMesh.Vertices,
			Faces:       s.Results.MapInsertMesh.Faces,
		})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no mesh data to export")
	}
	return out, nil
}

func colorForFeatureType(c session.Colors, ft string) string {
	switch ft {
	case "road", "roads":
		return c.Roads
	case "water":
		return c.Water
	case "building", "buildings":
		return c.Buildings
	case "gpx", "gpx_track":
		return c.GpxTrack
	default:
		return "#808080"
	}
}
