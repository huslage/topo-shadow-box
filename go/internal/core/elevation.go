package core

import (
	"context"
	"fmt"
	"image"
	_ "image/png"
	"io"
	"math"
	"net/http"
	"sync"

	"github.com/huslage/topo-shadow-box/internal/session"
)

const awsTerrainURL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/%d/%d/%d.png"

// HTTPClient interface for testability
type HTTPClient interface {
	Get(url string) (*http.Response, error)
}

func LatLonToTile(lat, lon float64, zoom int) (x, y int) {
	n := math.Pow(2, float64(zoom))
	x = int((lon + 180.0) / 360.0 * n)
	latRad := lat * math.Pi / 180
	y = int((1.0 - math.Log(math.Tan(latRad)+1/math.Cos(latRad))/math.Pi) / 2.0 * n)
	return
}

func TileToLatLon(x, y, zoom int) (lat, lon float64) {
	n := math.Pow(2, float64(zoom))
	lon = float64(x)/n*360.0 - 180.0
	latRad := math.Atan(math.Sinh(math.Pi * (1 - 2*float64(y)/n)))
	lat = latRad * 180 / math.Pi
	return
}

func DecodeTerrariumPixel(r, g, b uint8) float64 {
	return float64(r)*256.0 + float64(g) + float64(b)/256.0 - 32768.0
}

func PickZoom(north, south, east, west float64) int {
	maxSpan := math.Max(north-south, east-west)
	switch {
	case maxSpan > 1.0:
		return 10
	case maxSpan > 0.5:
		return 11
	case maxSpan > 0.1:
		return 12
	case maxSpan > 0.05:
		return 13
	default:
		return 14
	}
}

func BilinearInterp(grid [][]float64, lats, lons []float64, lat, lon float64) float64 {
	rows := len(lats)
	cols := len(lons)
	if rows < 2 || cols < 2 {
		return 0
	}

	latStep := (lats[rows-1] - lats[0]) / float64(rows-1)
	lonStep := (lons[cols-1] - lons[0]) / float64(cols-1)

	r0 := int((lat - lats[0]) / latStep)
	c0 := int((lon - lons[0]) / lonStep)

	r0 = clampInt(r0, 0, rows-2)
	c0 = clampInt(c0, 0, cols-2)
	r1, c1 := r0+1, c0+1

	tLat := (lat - lats[r0]) / (lats[r1] - lats[r0])
	tLon := (lon - lons[c0]) / (lons[c1] - lons[c0])

	return (1-tLat)*(1-tLon)*grid[r0][c0] +
		(1-tLat)*tLon*grid[r0][c1] +
		tLat*(1-tLon)*grid[r1][c0] +
		tLat*tLon*grid[r1][c1]
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// gaussianFilter3x3 applies a mild 3x3 Gaussian blur (sigma~0.5)
func gaussianFilter3x3(grid [][]float64) [][]float64 {
	k := [3][3]float64{
		{0.0751, 0.1238, 0.0751},
		{0.1238, 0.2042, 0.1238},
		{0.0751, 0.1238, 0.0751},
	}
	rows := len(grid)
	if rows == 0 {
		return grid
	}
	cols := len(grid[0])
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, cols)
	}
	for r := 1; r < rows-1; r++ {
		for c := 1; c < cols-1; c++ {
			var v float64
			for dr := -1; dr <= 1; dr++ {
				for dc := -1; dc <= 1; dc++ {
					v += k[dr+1][dc+1] * grid[r+dr][c+dc]
				}
			}
			out[r][c] = v
		}
	}
	// Copy edges unchanged
	for r := 0; r < rows; r++ {
		out[r][0] = grid[r][0]
		out[r][cols-1] = grid[r][cols-1]
	}
	for c := 0; c < cols; c++ {
		out[0][c] = grid[0][c]
		out[rows-1][c] = grid[rows-1][c]
	}
	return out
}

func linspace(start, end float64, n int) []float64 {
	if n <= 1 {
		return []float64{start}
	}
	result := make([]float64, n)
	for i := range result {
		result[i] = start + float64(i)*(end-start)/float64(n-1)
	}
	return result
}

func flipUD(grid [][]float64) {
	n := len(grid)
	for i := 0; i < n/2; i++ {
		grid[i], grid[n-1-i] = grid[n-1-i], grid[i]
	}
}

func reverseFloat64(s []float64) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}

// findRange returns the start and end indices in sorted slice s that cover [lo, hi].
func findRange(s []float64, lo, hi float64) (start, end int) {
	start = -1
	end = -1
	for i, v := range s {
		if v >= lo && start < 0 {
			start = i
		}
		if v <= hi {
			end = i
		}
	}
	return
}

func cropGrid(grid [][]float64, rStart, rEnd, cStart, cEnd int) [][]float64 {
	rows := rEnd - rStart + 1
	cols := cEnd - cStart + 1
	if rows <= 0 || cols <= 0 {
		return nil
	}
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, cols)
		copy(out[i], grid[rStart+i][cStart:cEnd+1])
	}
	return out
}

func gridMinMax(grid [][]float64) (min, max float64) {
	min = math.MaxFloat64
	max = -math.MaxFloat64
	for _, row := range grid {
		for _, v := range row {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
	}
	return
}

func decodeTileInto(body io.Reader, stitched [][]float64, tileX, tileY, tileSize int) {
	img, _, err := image.Decode(body)
	if err != nil {
		return
	}
	for py := 0; py < tileSize && py < img.Bounds().Dy(); py++ {
		for px := 0; px < tileSize && px < img.Bounds().Dx(); px++ {
			c := img.At(px, py)
			r, g, b, _ := c.RGBA()
			elev := DecodeTerrariumPixel(uint8(r>>8), uint8(g>>8), uint8(b>>8))
			rowIdx := tileY*tileSize + py
			colIdx := tileX*tileSize + px
			if rowIdx < len(stitched) && colIdx < len(stitched[rowIdx]) {
				stitched[rowIdx][colIdx] = elev
			}
		}
	}
}

func FetchTerrainElevation(ctx context.Context, client HTTPClient, north, south, east, west float64, resolution int) (*session.ElevationData, error) {
	if client == nil {
		client = http.DefaultClient
	}
	zoom := PickZoom(north, south, east, west)

	xMin, yMax := LatLonToTile(south, west, zoom)
	xMax, yMin := LatLonToTile(north, east, zoom)
	if xMin > xMax {
		xMin, xMax = xMax, xMin
	}
	if yMin > yMax {
		yMin, yMax = yMax, yMin
	}

	numX := xMax - xMin + 1
	numY := yMax - yMin + 1
	if numX*numY > 25 {
		if zoom > 8 {
			zoom--
		}
		xMin, yMax = LatLonToTile(south, west, zoom)
		xMax, yMin = LatLonToTile(north, east, zoom)
		if xMin > xMax {
			xMin, xMax = xMax, xMin
		}
		if yMin > yMax {
			yMin, yMax = yMax, yMin
		}
		numX = xMax - xMin + 1
		numY = yMax - yMin + 1
	}

	const tileSize = 256
	stitchedH := numY * tileSize
	stitchedW := numX * tileSize
	stitched := make([][]float64, stitchedH)
	for i := range stitched {
		stitched[i] = make([]float64, stitchedW)
	}

	var wg sync.WaitGroup
	for ty := yMin; ty <= yMax; ty++ {
		for tx := xMin; tx <= xMax; tx++ {
			wg.Add(1)
			go func(tx, ty int) {
				defer wg.Done()
				url := fmt.Sprintf(awsTerrainURL, zoom, tx, ty)
				resp, err := client.Get(url)
				if err != nil || resp.StatusCode != 200 {
					return
				}
				defer resp.Body.Close()
				decodeTileInto(resp.Body, stitched, tx-xMin, ty-yMin, tileSize)
			}(tx, ty)
		}
	}
	wg.Wait()

	tileNorth, tileWest := TileToLatLon(xMin, yMin, zoom)
	tileSouth, tileEast := TileToLatLon(xMax+1, yMax+1, zoom)

	stitchedLats := linspace(tileSouth, tileNorth, stitchedH)
	stitchedLons := linspace(tileWest, tileEast, stitchedW)

	flipUD(stitched)
	reverseFloat64(stitchedLats)

	rStart, rEnd := findRange(stitchedLats, south, north)
	cStart, cEnd := findRange(stitchedLons, west, east)
	if rStart < 0 || cStart < 0 {
		return nil, fmt.Errorf("bounding box doesn't overlap with fetched tiles")
	}
	cropped := cropGrid(stitched, rStart, rEnd, cStart, cEnd)
	croppedLats := stitchedLats[rStart : rEnd+1]
	croppedLons := stitchedLons[cStart : cEnd+1]

	if len(croppedLats) < 4 || len(croppedLons) < 4 {
		return nil, fmt.Errorf("cropped region too small for interpolation")
	}

	targetLats := linspace(south, north, resolution)
	targetLons := linspace(west, east, resolution)
	resampled := make([][]float64, resolution)
	for r := range resampled {
		resampled[r] = make([]float64, resolution)
		for c := range resampled[r] {
			resampled[r][c] = BilinearInterp(cropped, croppedLats, croppedLons, targetLats[r], targetLons[c])
		}
	}

	resampled = gaussianFilter3x3(resampled)

	minElev, maxElev := gridMinMax(resampled)
	return &session.ElevationData{
		Grid:         resampled,
		Lats:         targetLats,
		Lons:         targetLons,
		Resolution:   resolution,
		MinElevation: minElev,
		MaxElevation: maxElev,
		IsSet:        true,
	}, nil
}
