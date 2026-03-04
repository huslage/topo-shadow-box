package core

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"

	"github.com/huslage/topo-shadow-box/internal/session"
)

type nominatimResult struct {
	DisplayName string   `json:"display_name"`
	Lat         string   `json:"lat"`
	Lon         string   `json:"lon"`
	Type        string   `json:"type"`
	BoundingBox []string `json:"boundingbox"`
}

func GeocodePlace(ctx context.Context, query string, limit int) ([]session.GeocodeCandidate, error) {
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if limit < 1 {
		limit = 1
	}
	if limit > 10 {
		limit = 10
	}

	endpoint := "https://nominatim.openstreetmap.org/search"
	params := url.Values{}
	params.Set("q", query)
	params.Set("format", "json")
	params.Set("limit", strconv.Itoa(limit))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"?"+params.Encode(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "topo-shadow-box/1.0")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("nominatim returned %d", resp.StatusCode)
	}

	var raw []nominatimResult
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("decode nominatim: %w", err)
	}

	out := make([]session.GeocodeCandidate, 0, len(raw))
	for _, r := range raw {
		lat, err1 := strconv.ParseFloat(r.Lat, 64)
		lon, err2 := strconv.ParseFloat(r.Lon, 64)
		if err1 != nil || err2 != nil {
			continue
		}
		cand := session.GeocodeCandidate{
			DisplayName: r.DisplayName,
			Lat:         lat,
			Lon:         lon,
			PlaceType:   r.Type,
			BboxNorth:   lat,
			BboxSouth:   lat,
			BboxEast:    lon,
			BboxWest:    lon,
		}
		if len(r.BoundingBox) >= 4 {
			south, e1 := strconv.ParseFloat(r.BoundingBox[0], 64)
			north, e2 := strconv.ParseFloat(r.BoundingBox[1], 64)
			west, e3 := strconv.ParseFloat(r.BoundingBox[2], 64)
			east, e4 := strconv.ParseFloat(r.BoundingBox[3], 64)
			if e1 == nil && e2 == nil && e3 == nil && e4 == nil {
				cand.BboxSouth = south
				cand.BboxNorth = north
				cand.BboxWest = west
				cand.BboxEast = east
			}
		}
		out = append(out, cand)
	}

	return out, nil
}

func SetAreaFromGeocodeCandidate(ctx context.Context, s *session.Session, c session.GeocodeCandidate) error {
	_ = ctx
	if s == nil {
		return fmt.Errorf("session is nil")
	}
	b := session.Bounds{North: c.BboxNorth, South: c.BboxSouth, East: c.BboxEast, West: c.BboxWest, IsSet: true}
	if err := b.Validate(); err != nil {
		return err
	}
	s.Lock()
	defer s.Unlock()
	s.Config.Bounds = b
	s.Config.PendingGeocodeCandidates = nil
	s.ClearDownstream()
	return nil
}
