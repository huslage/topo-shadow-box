package core

import (
	"math"

	"github.com/huslage/topo-shadow-box/internal/session"
)

type GeoToModelTransform struct {
	bounds      session.Bounds
	lonScale    float64
	scaleFactor float64
	ModelWidthX float64
	ModelWidthZ float64
}

func NewGeoToModelTransform(bounds session.Bounds, modelWidthMM float64) GeoToModelTransform {
	avgLat := bounds.CenterLat()
	lonScale := math.Cos(math.Pi / 180 * avgLat)
	latRange := bounds.LatRange()
	lonRange := bounds.LonRange() * lonScale
	maxSpan := math.Max(latRange, lonRange)
	scaleFactor := 0.0
	if maxSpan > 0 {
		scaleFactor = modelWidthMM / maxSpan
	}
	return GeoToModelTransform{
		bounds: bounds, lonScale: lonScale, scaleFactor: scaleFactor,
		ModelWidthX: lonRange * scaleFactor,
		ModelWidthZ: latRange * scaleFactor,
	}
}

func (t GeoToModelTransform) GeoToModel(lat, lon float64) (x, z float64) {
	x = (lon - t.bounds.West) * t.lonScale * t.scaleFactor
	z = (t.bounds.North - lat) * t.scaleFactor
	return
}

func (t GeoToModelTransform) ElevationToY(elevation, minElev, elevRange, verticalScale, sizeScale float64) float64 {
	if elevRange <= 0 {
		return 0
	}
	clamped := math.Max(minElev, math.Min(minElev+elevRange, elevation))
	return ((clamped - minElev) / elevRange) * 20.0 * sizeScale * verticalScale
}

func AddPaddingToBounds(b session.Bounds, paddingM float64, isSet bool) session.Bounds {
	latPadding := paddingM / 111_000.0
	avgLat := (b.North + b.South) / 2
	lonPadding := paddingM / (111_000.0 * math.Cos(math.Pi/180*avgLat))
	return session.Bounds{
		North: b.North + latPadding, South: b.South - latPadding,
		East: b.East + lonPadding, West: b.West - lonPadding,
		IsSet: isSet,
	}
}
