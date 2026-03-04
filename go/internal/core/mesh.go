package core

import (
	"math"

	"github.com/huslage/topo-shadow-box/internal/session"
)

func GenerateTerrainMesh(e *session.ElevationData, bounds session.Bounds, mp session.ModelParams) *session.Mesh {
	if e == nil || len(e.Grid) == 0 || len(e.Grid[0]) == 0 || len(e.Lats) == 0 || len(e.Lons) == 0 {
		return nil
	}

	rows := len(e.Grid)
	cols := len(e.Grid[0])
	tr := NewGeoToModelTransform(bounds, mp.WidthMM)
	modelWidth := math.Max(tr.ModelWidthX, tr.ModelWidthZ)
	sizeScale := modelWidth / 200.0
	if sizeScale <= 0 {
		sizeScale = 1
	}
	minElev := e.MinElevation
	elevRange := e.MaxElevation - e.MinElevation
	if elevRange <= 0 {
		elevRange = 1
	}

	vertices := make([][3]float64, 0, rows*cols*2)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			x, z := tr.GeoToModel(e.Lats[r], e.Lons[c])
			y := tr.ElevationToY(e.Grid[r][c], minElev, elevRange, mp.VerticalScale, sizeScale)
			vertices = append(vertices, [3]float64{x, y, z})
		}
	}

	faces := make([][3]int, 0, (rows-1)*(cols-1)*4)
	for r := 0; r < rows-1; r++ {
		for c := 0; c < cols-1; c++ {
			i := r*cols + c
			ir := i + 1
			id := i + cols
			idr := i + cols + 1
			faces = append(faces, [3]int{i, id, ir})
			faces = append(faces, [3]int{ir, id, idr})
		}
	}

	// Add bottom shell to make a printable solid.
	n := len(vertices)
	for i := 0; i < n; i++ {
		v := vertices[i]
		vertices = append(vertices, [3]float64{v[0], -mp.BaseHeightMM, v[2]})
	}

	for r := 0; r < rows-1; r++ {
		t0 := r * cols
		t1 := (r + 1) * cols
		b0 := n + t0
		b1 := n + t1
		faces = append(faces, [3]int{t1, t0, b0}, [3]int{t1, b0, b1})

		t0 = r*cols + (cols - 1)
		t1 = (r+1)*cols + (cols - 1)
		b0 = n + t0
		b1 = n + t1
		faces = append(faces, [3]int{t0, t1, b1}, [3]int{t0, b1, b0})
	}
	for c := 0; c < cols-1; c++ {
		t0 := c
		t1 := c + 1
		b0 := n + t0
		b1 := n + t1
		faces = append(faces, [3]int{t0, t1, b1}, [3]int{t0, b1, b0})

		t0 = (rows-1)*cols + c
		t1 = (rows-1)*cols + c + 1
		b0 = n + t0
		b1 = n + t1
		faces = append(faces, [3]int{t1, t0, b0}, [3]int{t1, b0, b1})
	}

	for r := 0; r < rows-1; r++ {
		for c := 0; c < cols-1; c++ {
			i := n + r*cols + c
			ir := i + 1
			id := i + cols
			idr := i + cols + 1
			faces = append(faces, [3]int{i, ir, id}, [3]int{ir, idr, id})
		}
	}

	return &session.Mesh{Vertices: vertices, Faces: faces, Name: "Terrain", FeatureType: "terrain"}
}

func GenerateSingleFeatureMesh(feature interface{}, _ string, elev *session.ElevationData, bounds session.Bounds, mp session.ModelParams) *session.Mesh {
	if elev == nil || !elev.IsSet {
		return nil
	}
	tr := NewGeoToModelTransform(bounds, mp.WidthMM)
	clipper := buildShapeClipper(tr, mp.Shape)
	minElev := elev.MinElevation
	elevRange := elev.MaxElevation - elev.MinElevation
	if elevRange <= 0 {
		elevRange = 1
	}
	modelWidth := math.Max(tr.ModelWidthX, tr.ModelWidthZ)
	sizeScale := modelWidth / 200.0
	if sizeScale <= 0 {
		sizeScale = 1
	}

	switch f := feature.(type) {
	case session.RoadFeature:
		return generateLineFeatureMesh(f.Coordinates, "Road", "road", 0.8, 0.5, clipper, elev, tr, minElev, elevRange, mp.VerticalScale, sizeScale)
	case session.WaterFeature:
		return generatePolygonFeatureMesh(f.Coordinates, "Water", "water", 0.15, clipper, elev, tr, minElev, elevRange, mp.VerticalScale, sizeScale)
	case session.BuildingFeature:
		h := 0.8 + (f.Height/10.0)*0.4
		if h > 3.0 {
			h = 3.0
		}
		return generatePolygonFeatureMesh(f.Coordinates, "Building", "building", h, clipper, elev, tr, minElev, elevRange, mp.VerticalScale, sizeScale)
	default:
		return nil
	}
}

func GenerateGpxTrackMesh(tracks []session.GpxTrack, elev *session.ElevationData, bounds session.Bounds, mp session.ModelParams) *session.Mesh {
	if elev == nil || !elev.IsSet || len(tracks) == 0 {
		return nil
	}
	tr := NewGeoToModelTransform(bounds, mp.WidthMM)
	clipper := buildShapeClipper(tr, mp.Shape)
	minElev := elev.MinElevation
	elevRange := elev.MaxElevation - elev.MinElevation
	if elevRange <= 0 {
		elevRange = 1
	}
	modelWidth := math.Max(tr.ModelWidthX, tr.ModelWidthZ)
	sizeScale := modelWidth / 200.0
	if sizeScale <= 0 {
		sizeScale = 1
	}

	var merged *session.Mesh
	for _, t := range tracks {
		coords := make([]session.Coordinate, 0, len(t.Points))
		for _, p := range t.Points {
			coords = append(coords, session.Coordinate{Lat: p.Lat, Lon: p.Lon})
		}
		m := generateLineFeatureMesh(coords, "GPX Track", "gpx_track", 1.2, 0.8, clipper, elev, tr, minElev, elevRange, mp.VerticalScale, sizeScale)
		if m == nil {
			continue
		}
		if merged == nil {
			merged = &session.Mesh{Name: "GPX Track", FeatureType: "gpx_track"}
		}
		base := len(merged.Vertices)
		merged.Vertices = append(merged.Vertices, m.Vertices...)
		for _, f := range m.Faces {
			merged.Faces = append(merged.Faces, [3]int{f[0] + base, f[1] + base, f[2] + base})
		}
	}
	return merged
}

func generateLineFeatureMesh(coords []session.Coordinate, name, featureType string, widthMM, heightMM float64, clipper ShapeClipper, elev *session.ElevationData, tr GeoToModelTransform, minElev, elevRange, verticalScale, sizeScale float64) *session.Mesh {
	if len(coords) < 2 {
		return nil
	}

	vertices := make([][3]float64, 0, len(coords)*4)
	faces := make([][3]int, 0, len(coords)*2)
	half := widthMM / 2

	for i := 0; i < len(coords)-1; i++ {
		p0 := coords[i]
		p1 := coords[i+1]
		x0, z0 := tr.GeoToModel(p0.Lat, p0.Lon)
		x1, z1 := tr.GeoToModel(p1.Lat, p1.Lon)
		if clipper != nil {
			if !clipper.IsInside(x0, z0) || !clipper.IsInside(x1, z1) {
				continue
			}
		}

		dx := x1 - x0
		dz := z1 - z0
		segLen := math.Hypot(dx, dz)
		if segLen < 1e-6 {
			continue
		}
		px := -dz / segLen * half
		pz := dx / segLen * half

		e0 := sampleElevationAtPoint(elev, p0.Lat, p0.Lon)
		e1 := sampleElevationAtPoint(elev, p1.Lat, p1.Lon)
		y0 := tr.ElevationToY(e0, minElev, elevRange, verticalScale, sizeScale) + heightMM
		y1 := tr.ElevationToY(e1, minElev, elevRange, verticalScale, sizeScale) + heightMM

		base := len(vertices)
		vertices = append(vertices,
			[3]float64{x0 + px, y0, z0 + pz},
			[3]float64{x0 - px, y0, z0 - pz},
			[3]float64{x1 + px, y1, z1 + pz},
			[3]float64{x1 - px, y1, z1 - pz},
		)
		faces = append(faces, [3]int{base, base + 2, base + 1}, [3]int{base + 1, base + 2, base + 3})
	}

	if len(vertices) == 0 || len(faces) == 0 {
		return nil
	}
	return &session.Mesh{Vertices: vertices, Faces: faces, Name: name, FeatureType: featureType}
}

func generatePolygonFeatureMesh(coords []session.Coordinate, name, featureType string, yOffset float64, clipper ShapeClipper, elev *session.ElevationData, tr GeoToModelTransform, minElev, elevRange, verticalScale, sizeScale float64) *session.Mesh {
	if len(coords) < 3 {
		return nil
	}

	verts2d := make([][2]float64, 0, len(coords))
	for _, c := range coords {
		x, z := tr.GeoToModel(c.Lat, c.Lon)
		verts2d = append(verts2d, [2]float64{x, z})
	}
	if clipper != nil {
		clipped := clipper.ClipPolygon(verts2d)
		if len(clipped) < 3 {
			return nil
		}
		verts2d = clipped
	}

	vertices := make([][3]float64, 0, len(verts2d))
	for i, p := range verts2d {
		lat := coords[i].Lat
		lon := coords[i].Lon
		e := sampleElevationAtPoint(elev, lat, lon)
		y := tr.ElevationToY(e, minElev, elevRange, verticalScale, sizeScale) + yOffset
		vertices = append(vertices, [3]float64{p[0], y, p[1]})
	}

	if len(vertices) < 3 {
		return nil
	}
	faces := make([][3]int, 0, len(vertices)-2)
	for i := 1; i < len(vertices)-1; i++ {
		faces = append(faces, [3]int{0, i, i + 1})
	}
	return &session.Mesh{Vertices: vertices, Faces: faces, Name: name, FeatureType: featureType}
}

func sampleElevationAtPoint(e *session.ElevationData, lat, lon float64) float64 {
	if e == nil || len(e.Grid) < 2 || len(e.Grid[0]) < 2 || len(e.Lats) < 2 || len(e.Lons) < 2 {
		return 0
	}
	return BilinearInterp(e.Grid, e.Lats, e.Lons, lat, lon)
}

func buildShapeClipper(tr GeoToModelTransform, shape string) ShapeClipper {
	cx := tr.ModelWidthX / 2
	cz := tr.ModelWidthZ / 2
	switch shape {
	case "circle":
		r := math.Min(tr.ModelWidthX, tr.ModelWidthZ) / 2
		return NewCircleClipper(cx, cz, r)
	case "hexagon":
		r := math.Min(tr.ModelWidthX, tr.ModelWidthZ) / 2
		return NewHexagonClipper(cx, cz, r)
	case "rectangle":
		return NewRectangleClipper(cx, cz, tr.ModelWidthX/2, tr.ModelWidthZ/2)
	default:
		return nil
	}
}
