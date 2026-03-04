package core

import (
	"encoding/xml"
	"fmt"
	"io"
	"os"

	"github.com/huslage/topo-shadow-box/internal/session"
)

type gpxDoc struct {
	Tracks    []gpxTrack    `xml:"trk"`
	Waypoints []gpxWaypoint `xml:"wpt"`
}

type gpxTrack struct {
	Name     string       `xml:"name"`
	Segments []gpxSegment `xml:"trkseg"`
}

type gpxSegment struct {
	Points []gpxTrackPoint `xml:"trkpt"`
}

type gpxTrackPoint struct {
	Lat float64  `xml:"lat,attr"`
	Lon float64  `xml:"lon,attr"`
	Ele *float64 `xml:"ele"`
}

type gpxWaypoint struct {
	Name string   `xml:"name"`
	Desc string   `xml:"desc"`
	Lat  float64  `xml:"lat,attr"`
	Lon  float64  `xml:"lon,attr"`
	Ele  *float64 `xml:"ele"`
}

func ParseGPXFile(path string) ([]session.GpxTrack, []session.GpxWaypoint, session.Bounds, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, session.Bounds{}, fmt.Errorf("open gpx: %w", err)
	}
	defer f.Close()
	return ParseGPX(f)
}

func ParseGPX(r io.Reader) ([]session.GpxTrack, []session.GpxWaypoint, session.Bounds, error) {
	var doc gpxDoc
	if err := xml.NewDecoder(r).Decode(&doc); err != nil {
		return nil, nil, session.Bounds{}, fmt.Errorf("parse gpx xml: %w", err)
	}

	var tracks []session.GpxTrack
	var waypoints []session.GpxWaypoint

	bounds := session.Bounds{IsSet: false}
	updateBounds := func(lat, lon float64) {
		if !bounds.IsSet {
			bounds = session.Bounds{North: lat, South: lat, East: lon, West: lon, IsSet: true}
			return
		}
		if lat > bounds.North {
			bounds.North = lat
		}
		if lat < bounds.South {
			bounds.South = lat
		}
		if lon > bounds.East {
			bounds.East = lon
		}
		if lon < bounds.West {
			bounds.West = lon
		}
	}

	for _, tr := range doc.Tracks {
		track := session.GpxTrack{Name: tr.Name}
		if track.Name == "" {
			track.Name = "Unnamed Track"
		}
		for _, seg := range tr.Segments {
			for _, pt := range seg.Points {
				ele := 0.0
				if pt.Ele != nil {
					ele = *pt.Ele
				}
				track.Points = append(track.Points, session.GpxPoint{Lat: pt.Lat, Lon: pt.Lon, Elevation: ele})
				updateBounds(pt.Lat, pt.Lon)
			}
		}
		if len(track.Points) >= 2 {
			tracks = append(tracks, track)
		}
	}

	for _, wp := range doc.Waypoints {
		ele := 0.0
		if wp.Ele != nil {
			ele = *wp.Ele
		}
		waypoints = append(waypoints, session.GpxWaypoint{
			Name: wp.Name, Description: wp.Desc,
			Lat: wp.Lat, Lon: wp.Lon, Elevation: ele,
		})
		updateBounds(wp.Lat, wp.Lon)
	}

	return tracks, waypoints, bounds, nil
}
