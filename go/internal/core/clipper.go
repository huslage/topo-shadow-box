package core

import "math"

// ShapeClipper defines the boundary clipping interface.
type ShapeClipper interface {
	IsInside(x, z float64) bool
	ClipLinestring(pts [][2]float64) [][][2]float64
	ClipPolygon(pts [][2]float64) [][2]float64 // nil if any vertex outside
	ProjectToBoundary(x, z float64) (float64, float64)
}

// --- CircleClipper ---

type CircleClipper struct {
	cx, cz, radius float64
}

func NewCircleClipper(cx, cz, radius float64) *CircleClipper {
	return &CircleClipper{cx: cx, cz: cz, radius: radius}
}

func (c *CircleClipper) IsInside(x, z float64) bool {
	dx := x - c.cx
	dz := z - c.cz
	return dx*dx+dz*dz <= c.radius*c.radius
}

func (c *CircleClipper) lineCircleIntersection(x1, z1, x2, z2 float64) [][2]float64 {
	dx := x2 - x1
	dz := z2 - z1
	fx := x1 - c.cx
	fz := z1 - c.cz
	a := dx*dx + dz*dz
	b := 2 * (fx*dx + fz*dz)
	cc := fx*fx + fz*fz - c.radius*c.radius
	if a < 1e-10 {
		return nil
	}
	disc := b*b - 4*a*cc
	if disc < 0 {
		return nil
	}
	sqrtDisc := math.Sqrt(disc)
	var result [][2]float64
	for _, t := range []float64{(-b - sqrtDisc) / (2 * a), (-b + sqrtDisc) / (2 * a)} {
		if t >= 0 && t <= 1 {
			result = append(result, [2]float64{x1 + t*dx, z1 + t*dz})
		}
	}
	return result
}

func (c *CircleClipper) ClipLinestring(pts [][2]float64) [][][2]float64 {
	if len(pts) < 2 {
		return nil
	}
	var segments [][][2]float64
	var cur [][2]float64
	for i, p := range pts {
		inside := c.IsInside(p[0], p[1])
		if inside {
			cur = append(cur, p)
		}
		if i < len(pts)-1 {
			next := pts[i+1]
			insideNext := c.IsInside(next[0], next[1])
			if inside != insideNext {
				ixs := c.lineCircleIntersection(p[0], p[1], next[0], next[1])
				if len(ixs) > 0 {
					ix := ixs[0]
					if inside {
						cur = append(cur, ix)
						if len(cur) >= 2 {
							segments = append(segments, cur)
						}
						cur = nil
					} else {
						cur = [][2]float64{ix}
					}
				}
			}
		}
	}
	if len(cur) >= 2 {
		segments = append(segments, cur)
	}
	return segments
}

func (c *CircleClipper) ClipPolygon(pts [][2]float64) [][2]float64 {
	for _, p := range pts {
		if !c.IsInside(p[0], p[1]) {
			return nil
		}
	}
	return pts
}

func (c *CircleClipper) ProjectToBoundary(x, z float64) (float64, float64) {
	dx := x - c.cx
	dz := z - c.cz
	dist := math.Sqrt(dx*dx + dz*dz)
	if dist == 0 {
		return c.cx + c.radius, c.cz
	}
	scale := c.radius / dist
	return c.cx + dx*scale, c.cz + dz*scale
}

// --- SquareClipper ---

type SquareClipper struct {
	cx, cz, halfWidth float64
}

func NewSquareClipper(cx, cz, halfWidth float64) *SquareClipper {
	return &SquareClipper{cx: cx, cz: cz, halfWidth: halfWidth}
}

func (c *SquareClipper) IsInside(x, z float64) bool {
	return math.Abs(x-c.cx) <= c.halfWidth && math.Abs(z-c.cz) <= c.halfWidth
}

func (c *SquareClipper) lineBoxIntersection(x1, z1, x2, z2 float64) [][2]float64 {
	minX := c.cx - c.halfWidth
	maxX := c.cx + c.halfWidth
	minZ := c.cz - c.halfWidth
	maxZ := c.cz + c.halfWidth
	var result [][2]float64
	addIfValid := func(t, px, pz, lo, hi float64, checkX bool) {
		if t >= 0 && t <= 1 {
			if checkX {
				if pz >= lo && pz <= hi {
					result = append(result, [2]float64{px, pz})
				}
			} else {
				if px >= lo && px <= hi {
					result = append(result, [2]float64{px, pz})
				}
			}
		}
	}
	if x1 != x2 {
		t := (minX - x1) / (x2 - x1)
		addIfValid(t, minX, z1+t*(z2-z1), minZ, maxZ, true)
		t = (maxX - x1) / (x2 - x1)
		addIfValid(t, maxX, z1+t*(z2-z1), minZ, maxZ, true)
	}
	if z1 != z2 {
		t := (minZ - z1) / (z2 - z1)
		addIfValid(t, x1+t*(x2-x1), minZ, minX, maxX, false)
		t = (maxZ - z1) / (z2 - z1)
		addIfValid(t, x1+t*(x2-x1), maxZ, minX, maxX, false)
	}
	// Deduplicate
	var unique [][2]float64
	for _, p := range result {
		dup := false
		for _, u := range unique {
			if math.Abs(p[0]-u[0]) < 1e-6 && math.Abs(p[1]-u[1]) < 1e-6 {
				dup = true
				break
			}
		}
		if !dup {
			unique = append(unique, p)
		}
	}
	return unique
}

func clipLinestring(isInsideFn func(float64, float64) bool, intersectFn func(float64, float64, float64, float64) [][2]float64, pts [][2]float64) [][][2]float64 {
	if len(pts) < 2 {
		return nil
	}
	var segments [][][2]float64
	var cur [][2]float64
	for i, p := range pts {
		inside := isInsideFn(p[0], p[1])
		if inside {
			cur = append(cur, p)
		}
		if i < len(pts)-1 {
			next := pts[i+1]
			insideNext := isInsideFn(next[0], next[1])
			if inside != insideNext {
				ixs := intersectFn(p[0], p[1], next[0], next[1])
				if len(ixs) > 0 {
					ix := ixs[0]
					if inside {
						cur = append(cur, ix)
						if len(cur) >= 2 {
							segments = append(segments, cur)
						}
						cur = nil
					} else {
						cur = [][2]float64{ix}
					}
				}
			}
		}
	}
	if len(cur) >= 2 {
		segments = append(segments, cur)
	}
	return segments
}

func clipPolygon(isInsideFn func(float64, float64) bool, pts [][2]float64) [][2]float64 {
	for _, p := range pts {
		if !isInsideFn(p[0], p[1]) {
			return nil
		}
	}
	return pts
}

func (c *SquareClipper) ClipLinestring(pts [][2]float64) [][][2]float64 {
	return clipLinestring(c.IsInside, c.lineBoxIntersection, pts)
}

func (c *SquareClipper) ClipPolygon(pts [][2]float64) [][2]float64 {
	return clipPolygon(c.IsInside, pts)
}

func (c *SquareClipper) ProjectToBoundary(x, z float64) (float64, float64) {
	dx := x - c.cx
	dz := z - c.cz
	if math.Abs(dx) >= math.Abs(dz) {
		if dx >= 0 {
			return c.cx + c.halfWidth, z
		}
		return c.cx - c.halfWidth, z
	}
	if dz >= 0 {
		return x, c.cz + c.halfWidth
	}
	return x, c.cz - c.halfWidth
}

// --- RectangleClipper ---

type RectangleClipper struct {
	cx, cz, halfWidth, halfHeight float64
}

func NewRectangleClipper(cx, cz, halfWidth, halfHeight float64) *RectangleClipper {
	return &RectangleClipper{cx: cx, cz: cz, halfWidth: halfWidth, halfHeight: halfHeight}
}

func (c *RectangleClipper) IsInside(x, z float64) bool {
	return math.Abs(x-c.cx) <= c.halfWidth && math.Abs(z-c.cz) <= c.halfHeight
}

func (c *RectangleClipper) lineBoxIntersection(x1, z1, x2, z2 float64) [][2]float64 {
	minX := c.cx - c.halfWidth
	maxX := c.cx + c.halfWidth
	minZ := c.cz - c.halfHeight
	maxZ := c.cz + c.halfHeight
	var result [][2]float64
	if x1 != x2 {
		t := (minX - x1) / (x2 - x1)
		if t >= 0 && t <= 1 {
			zz := z1 + t*(z2-z1)
			if zz >= minZ && zz <= maxZ {
				result = append(result, [2]float64{minX, zz})
			}
		}
		t = (maxX - x1) / (x2 - x1)
		if t >= 0 && t <= 1 {
			zz := z1 + t*(z2-z1)
			if zz >= minZ && zz <= maxZ {
				result = append(result, [2]float64{maxX, zz})
			}
		}
	}
	if z1 != z2 {
		t := (minZ - z1) / (z2 - z1)
		if t >= 0 && t <= 1 {
			xx := x1 + t*(x2-x1)
			if xx >= minX && xx <= maxX {
				result = append(result, [2]float64{xx, minZ})
			}
		}
		t = (maxZ - z1) / (z2 - z1)
		if t >= 0 && t <= 1 {
			xx := x1 + t*(x2-x1)
			if xx >= minX && xx <= maxX {
				result = append(result, [2]float64{xx, maxZ})
			}
		}
	}
	var unique [][2]float64
	for _, p := range result {
		dup := false
		for _, u := range unique {
			if math.Abs(p[0]-u[0]) < 1e-6 && math.Abs(p[1]-u[1]) < 1e-6 {
				dup = true
				break
			}
		}
		if !dup {
			unique = append(unique, p)
		}
	}
	return unique
}

func (c *RectangleClipper) ClipLinestring(pts [][2]float64) [][][2]float64 {
	return clipLinestring(c.IsInside, c.lineBoxIntersection, pts)
}

func (c *RectangleClipper) ClipPolygon(pts [][2]float64) [][2]float64 {
	return clipPolygon(c.IsInside, pts)
}

func (c *RectangleClipper) ProjectToBoundary(x, z float64) (float64, float64) {
	dx := x - c.cx
	dz := z - c.cz
	// Guard against zero dimensions — treat as square with side 1 if degenerate.
	hw := c.halfWidth
	hh := c.halfHeight
	if hw <= 0 {
		hw = 1
	}
	if hh <= 0 {
		hh = 1
	}
	rx := math.Abs(dx) / hw
	rz := math.Abs(dz) / hh
	if rx >= rz {
		if dx >= 0 {
			return c.cx + c.halfWidth, z
		}
		return c.cx - c.halfWidth, z
	}
	if dz >= 0 {
		return x, c.cz + c.halfHeight
	}
	return x, c.cz - c.halfHeight
}

// --- HexagonClipper ---

type HexagonClipper struct {
	cx, cz   float64
	radius   float64
	vertices [6][2]float64
}

func NewHexagonClipper(cx, cz, radius float64) *HexagonClipper {
	h := &HexagonClipper{cx: cx, cz: cz, radius: radius}
	for i, deg := range []float64{0, 60, 120, 180, 240, 300} {
		a := deg * math.Pi / 180
		h.vertices[i] = [2]float64{cx + radius*math.Cos(a), cz + radius*math.Sin(a)}
	}
	return h
}

func (c *HexagonClipper) IsInside(x, z float64) bool {
	inside := false
	for j := 0; j < 6; j++ {
		k := (j + 1) % 6
		vx1, vz1 := c.vertices[j][0], c.vertices[j][1]
		vx2, vz2 := c.vertices[k][0], c.vertices[k][1]
		if ((vz1 > z) != (vz2 > z)) && (x < (vx2-vx1)*(z-vz1)/(vz2-vz1)+vx1) {
			inside = !inside
		}
	}
	return inside
}

func (c *HexagonClipper) ClipLinestring(pts [][2]float64) [][][2]float64 {
	// Hexagon uses simple include/exclude (no edge intersection, matching Python reference)
	if len(pts) < 2 {
		return nil
	}
	var segments [][][2]float64
	var cur [][2]float64
	for _, p := range pts {
		if c.IsInside(p[0], p[1]) {
			cur = append(cur, p)
		} else if len(cur) > 0 {
			if len(cur) >= 2 {
				segments = append(segments, cur)
			}
			cur = nil
		}
	}
	if len(cur) >= 2 {
		segments = append(segments, cur)
	}
	return segments
}

func (c *HexagonClipper) ClipPolygon(pts [][2]float64) [][2]float64 {
	return clipPolygon(c.IsInside, pts)
}

func (c *HexagonClipper) ProjectToBoundary(x, z float64) (float64, float64) {
	minDistSq := math.MaxFloat64
	var nearest [2]float64
	for j := 0; j < 6; j++ {
		k := (j + 1) % 6
		v1 := c.vertices[j]
		v2 := c.vertices[k]
		ex := v2[0] - v1[0]
		ez := v2[1] - v1[1]
		px := x - v1[0]
		pz := z - v1[1]
		edgeLenSq := ex*ex + ez*ez
		var proj [2]float64
		if edgeLenSq < 1e-12 {
			proj = v1
		} else {
			t := math.Max(0, math.Min(1, (px*ex+pz*ez)/edgeLenSq))
			proj = [2]float64{v1[0] + t*ex, v1[1] + t*ez}
		}
		d := (proj[0]-x)*(proj[0]-x) + (proj[1]-z)*(proj[1]-z)
		if d < minDistSq {
			minDistSq = d
			nearest = proj
		}
	}
	return nearest[0], nearest[1]
}

// NewShapeClipper returns the appropriate clipper for the given shape.
// cx, cz are the model center; halfSize is half the model width.
func NewShapeClipper(shape string, cx, cz, halfSize float64) ShapeClipper {
	switch shape {
	case "circle":
		return NewCircleClipper(cx, cz, halfSize)
	case "hexagon":
		return NewHexagonClipper(cx, cz, halfSize)
	case "rectangle":
		return NewRectangleClipper(cx, cz, halfSize, halfSize)
	default: // "square"
		return NewSquareClipper(cx, cz, halfSize)
	}
}
