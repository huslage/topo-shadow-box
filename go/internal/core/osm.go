package core

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/huslage/topo-shadow-box/internal/session"
)

// OverpassClient sends Overpass API requests (POST-based).
type OverpassClient interface {
	Post(url, contentType string, body io.Reader) (*http.Response, error)
}

var overpassServers = []string{
	"https://overpass-api.de/api/interpreter",
	"https://overpass.kumi.systems/api/interpreter",
}

// FetchOSMFeatures fetches OSM roads, water, and/or buildings for the given
// bounding box via the Overpass API. Pass nil for client to use http.DefaultClient.
// featureTypes is a subset of ["roads", "water", "buildings"].
func FetchOSMFeatures(ctx context.Context, client OverpassClient, north, south, east, west float64, featureTypes []string) (*session.OsmFeatureSet, error) {
	if client == nil {
		client = http.DefaultClient
	}
	result := &session.OsmFeatureSet{}
	wantRoads := contains(featureTypes, "roads")
	wantWater := contains(featureTypes, "water")
	wantBuildings := contains(featureTypes, "buildings")

	bbox := fmt.Sprintf("%.6f,%.6f,%.6f,%.6f", south, west, north, east)

	if wantRoads {
		query := fmt.Sprintf(`[out:json][timeout:25];(way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|service|unclassified)$"](%s););out geom;`, bbox)
		elements, err := runOverpassQuery(ctx, client, query)
		if err == nil {
			for _, el := range elements {
				if el.Type != "way" {
					continue
				}
				road := session.RoadFeature{
					ID:       int(el.ID),
					Name:     el.Tags["name"],
					RoadType: el.Tags["highway"],
					Tags:     el.Tags,
				}
				for _, g := range el.Geometry {
					road.Coordinates = append(road.Coordinates, session.Coordinate{Lat: g.Lat, Lon: g.Lon})
				}
				if len(road.Coordinates) >= 2 {
					result.Roads = append(result.Roads, road)
				}
			}
			if len(result.Roads) > 200 {
				result.Roads = result.Roads[:200]
			}
		}
	}

	if wantWater {
		query := fmt.Sprintf(`[out:json][timeout:25];(way["natural"~"^(water|coastline)$"](%s);way["waterway"](%s);relation["natural"="water"](%s););out geom;`, bbox, bbox, bbox)
		elements, err := runOverpassQuery(ctx, client, query)
		if err == nil {
			for _, el := range elements {
				if el.Type != "way" {
					continue
				}
				water := session.WaterFeature{
					ID:   int(el.ID),
					Name: el.Tags["name"],
					Tags: el.Tags,
				}
				for _, g := range el.Geometry {
					water.Coordinates = append(water.Coordinates, session.Coordinate{Lat: g.Lat, Lon: g.Lon})
				}
				if len(water.Coordinates) >= 2 {
					result.Water = append(result.Water, water)
				}
			}
			if len(result.Water) > 50 {
				result.Water = result.Water[:50]
			}
		}
	}

	if wantBuildings {
		query := fmt.Sprintf(`[out:json][timeout:25];(way["building"](%s););out geom;`, bbox)
		elements, err := runOverpassQuery(ctx, client, query)
		if err == nil {
			for _, el := range elements {
				if el.Type != "way" {
					continue
				}
				b := session.BuildingFeature{
					ID:   int(el.ID),
					Name: el.Tags["name"],
					Tags: el.Tags,
				}
				// Height parsing: prefer explicit "height" tag, fall back to building:levels
				if h, ok := el.Tags["height"]; ok {
					h = strings.TrimSuffix(strings.TrimSpace(h), "m")
					if v, err2 := strconv.ParseFloat(strings.TrimSpace(h), 64); err2 == nil {
						b.Height = v
					}
				} else if lvl, ok := el.Tags["building:levels"]; ok {
					if v, err2 := strconv.ParseFloat(strings.TrimSpace(lvl), 64); err2 == nil {
						b.Height = v * 3.0
					}
				}
				if b.Height <= 0 {
					b.Height = 6.0 // default: 2 floors at 3m each
				}
				for _, g := range el.Geometry {
					b.Coordinates = append(b.Coordinates, session.Coordinate{Lat: g.Lat, Lon: g.Lon})
				}
				if len(b.Coordinates) >= 3 {
					result.Buildings = append(result.Buildings, b)
				}
			}
			if len(result.Buildings) > 150 {
				result.Buildings = result.Buildings[:150]
			}
		}
	}

	return result, nil
}

type overpassElement struct {
	Type     string            `json:"type"`
	ID       int64             `json:"id"`
	Tags     map[string]string `json:"tags"`
	Geometry []struct {
		Lat float64 `json:"lat"`
		Lon float64 `json:"lon"`
	} `json:"geometry"`
}

type overpassResponse struct {
	Elements []overpassElement `json:"elements"`
}

func runOverpassQuery(ctx context.Context, client OverpassClient, query string) ([]overpassElement, error) {
	var lastErr error
	for _, server := range overpassServers {
		resp, err := client.Post(server, "application/x-www-form-urlencoded", strings.NewReader("data="+query))
		if err != nil {
			lastErr = err
			continue
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			lastErr = fmt.Errorf("overpass returned %d", resp.StatusCode)
			continue
		}
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			lastErr = err
			continue
		}
		var result overpassResponse
		if err := json.Unmarshal(data, &result); err != nil {
			lastErr = err
			continue
		}
		return result.Elements, nil
	}
	return nil, fmt.Errorf("all overpass servers failed: %w", lastErr)
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
