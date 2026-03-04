package tools

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/huslage/topo-shadow-box/internal/core"
	"github.com/huslage/topo-shadow-box/internal/exporters"
	"github.com/huslage/topo-shadow-box/internal/preview"
	"github.com/huslage/topo-shadow-box/internal/session"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func RegisterAll(srv *server.MCPServer, sess *session.Session) {
	registerStatusTools(srv, sess)
	registerAreaTools(srv, sess)
	registerDataTools(srv, sess)
	registerModelTools(srv, sess)
	registerGenerateTools(srv, sess)
	registerPreviewTools(srv, sess)
	registerExportTools(srv, sess)
	registerSessionTools(srv, sess)
}

func registerStatusTools(srv *server.MCPServer, sess *session.Session) {
	srv.AddTool(
		mcp.NewTool("get_status", mcp.WithDescription("Return current in-memory session status.")),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			_ = req
			msg := fmt.Sprintf(
				"status: bounds_set=%t elevation_set=%t terrain_mesh=%t feature_meshes=%d gpx_mesh=%t",
				sess.Config.Bounds.IsSet,
				sess.FetchedData.Elevation != nil && sess.FetchedData.Elevation.IsSet,
				sess.Results.TerrainMesh != nil,
				len(sess.Results.FeatureMeshes),
				sess.Results.GpxMesh != nil,
			)
			return mcp.NewToolResultText(msg), nil
		},
	)
}

func registerAreaTools(srv *server.MCPServer, sess *session.Session) {
	srv.AddTool(
		mcp.NewTool(
			"set_area_from_coordinates",
			mcp.WithDescription("Set area by center+radius or explicit bounding box."),
			mcp.WithNumber("lat", mcp.Description("Center latitude")),
			mcp.WithNumber("lon", mcp.Description("Center longitude")),
			mcp.WithNumber("radius_m", mcp.Description("Radius in meters")),
			mcp.WithNumber("north", mcp.Description("North bound")),
			mcp.WithNumber("south", mcp.Description("South bound")),
			mcp.WithNumber("east", mcp.Description("East bound")),
			mcp.WithNumber("west", mcp.Description("West bound")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			args := req.GetArguments()
			_, hasLat := args["lat"]
			_, hasLon := args["lon"]
			_, hasRadius := args["radius_m"]
			_, hasNorth := args["north"]
			_, hasSouth := args["south"]
			_, hasEast := args["east"]
			_, hasWest := args["west"]

			var err error
			switch {
			case hasLat && hasLon && hasRadius:
				lat := mcp.ParseFloat64(req, "lat", 0)
				lon := mcp.ParseFloat64(req, "lon", 0)
				radius := mcp.ParseFloat64(req, "radius_m", 0)
				err = core.SetAreaFromCoordinates(ctx, sess, lat, lon, radius)
			case hasNorth && hasSouth && hasEast && hasWest:
				north := mcp.ParseFloat64(req, "north", 0)
				south := mcp.ParseFloat64(req, "south", 0)
				east := mcp.ParseFloat64(req, "east", 0)
				west := mcp.ParseFloat64(req, "west", 0)
				err = core.SetAreaFromBbox(ctx, sess, north, south, east, west)
			default:
				return mcp.NewToolResultError("Provide either lat/lon/radius_m or north/south/east/west."), nil
			}
			if err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			b := sess.Config.Bounds
			return mcp.NewToolResultText(
				fmt.Sprintf("Area set: N=%.6f S=%.6f E=%.6f W=%.6f", b.North, b.South, b.East, b.West),
			), nil
		},
	)

	srv.AddTool(
		mcp.NewTool(
			"set_area_from_gpx",
			mcp.WithDescription("Set area from GPX file bounds (+ optional padding)."),
			mcp.WithString("file_path", mcp.Required(), mcp.Description("Path to .gpx file")),
			mcp.WithNumber("padding_m", mcp.Description("Optional padding in meters (default 500)")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			path := mcp.ParseString(req, "file_path", "")
			if path == "" {
				return mcp.NewToolResultError("file_path is required"), nil
			}
			padding := mcp.ParseFloat64(req, "padding_m", 500)
			if err := core.SetAreaFromGPX(ctx, sess, path, padding); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			b := sess.Config.Bounds
			return mcp.NewToolResultText(
				fmt.Sprintf("GPX area set: N=%.6f S=%.6f E=%.6f W=%.6f", b.North, b.South, b.East, b.West),
			), nil
		},
	)

	srv.AddTool(
		mcp.NewTool("validate_area", mcp.WithDescription("Validate currently selected area and report warnings.")),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			_ = req
			b := sess.Config.Bounds
			if !b.IsSet {
				return mcp.NewToolResultError("set area first"), nil
			}
			latM := b.LatRange() * 111_000.0
			lonM := b.LonRange() * 111_000.0 * math.Abs(math.Cos(math.Pi/180.0*b.CenterLat()))
			minSpan := math.Min(latM, lonM)
			maxSpan := math.Max(latM, lonM)
			if minSpan < 100 {
				return mcp.NewToolResultError(fmt.Sprintf("area too small (min span %.0fm)", minSpan)), nil
			}
			warnings := []string{}
			if maxSpan > 500_000 {
				warnings = append(warnings, fmt.Sprintf("very large area (%.0fkm span)", maxSpan/1000.0))
			}
			if sess.FetchedData.Elevation != nil && sess.FetchedData.Elevation.IsSet {
				relief := sess.FetchedData.Elevation.MaxElevation - sess.FetchedData.Elevation.MinElevation
				if relief < 20 {
					warnings = append(warnings, fmt.Sprintf("low elevation relief (%.0fm)", relief))
				}
			}
			if len(warnings) == 0 {
				return mcp.NewToolResultText("Area looks good."), nil
			}
			return mcp.NewToolResultText("Warnings: " + strings.Join(warnings, " | ")), nil
		},
	)

	srv.AddTool(
		mcp.NewTool(
			"geocode_place",
			mcp.WithDescription("Search place name and set area (auto-select on single result)."),
			mcp.WithString("query", mcp.Required(), mcp.Description("Place query")),
			mcp.WithNumber("limit", mcp.Description("Maximum candidates (1-10, default 5)")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			query := strings.TrimSpace(mcp.ParseString(req, "query", ""))
			if query == "" {
				return mcp.NewToolResultError("query is required"), nil
			}
			limit := mcp.ParseInt(req, "limit", 5)
			candidates, err := core.GeocodePlace(ctx, query, limit)
			if err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			if len(candidates) == 0 {
				return mcp.NewToolResultText(fmt.Sprintf("No locations found for %q", query)), nil
			}
			if len(candidates) == 1 {
				if err := core.SetAreaFromGeocodeCandidate(ctx, sess, candidates[0]); err != nil {
					return mcp.NewToolResultError(err.Error()), nil
				}
				b := sess.Config.Bounds
				return mcp.NewToolResultText(
					fmt.Sprintf("Found 1 result and set area: N=%.6f S=%.6f E=%.6f W=%.6f", b.North, b.South, b.East, b.West),
				), nil
			}

			sess.Config.PendingGeocodeCandidates = candidates
			lines := []string{fmt.Sprintf("Found %d locations for %q:", len(candidates), query)}
			for i, c := range candidates {
				lines = append(lines, fmt.Sprintf("%d. %s (%.5f, %.5f) type=%s", i+1, c.DisplayName, c.Lat, c.Lon, c.PlaceType))
			}
			lines = append(lines, fmt.Sprintf("Use select_geocode_result with number 1-%d.", len(candidates)))
			return mcp.NewToolResultError(strings.Join(lines, "\n")), nil
		},
	)

	srv.AddTool(
		mcp.NewTool(
			"select_geocode_result",
			mcp.WithDescription("Select one pending geocode candidate and set area."),
			mcp.WithNumber("number", mcp.Required(), mcp.Description("1-based candidate index")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			n := mcp.ParseInt(req, "number", 0)
			if len(sess.Config.PendingGeocodeCandidates) == 0 {
				return mcp.NewToolResultError("no geocode candidates pending; call geocode_place first"), nil
			}
			if n < 1 || n > len(sess.Config.PendingGeocodeCandidates) {
				return mcp.NewToolResultError(fmt.Sprintf("number must be between 1 and %d", len(sess.Config.PendingGeocodeCandidates))), nil
			}
			c := sess.Config.PendingGeocodeCandidates[n-1]
			if err := core.SetAreaFromGeocodeCandidate(ctx, sess, c); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			b := sess.Config.Bounds
			return mcp.NewToolResultText(
				fmt.Sprintf("Selected %q. Area set: N=%.6f S=%.6f E=%.6f W=%.6f", c.DisplayName, b.North, b.South, b.East, b.West),
			), nil
		},
	)
}

func registerPreviewTools(srv *server.MCPServer, sess *session.Session) {
	srv.AddTool(
		mcp.NewTool("preview", mcp.WithDescription("Open or refresh local 3D preview at http://localhost:3333.")),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			_ = req
			if sess.Results.TerrainMesh == nil {
				return mcp.NewToolResultError("generate_model first"), nil
			}
			url, err := preview.StartOrUpdate(sess, 3333)
			if err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText("Preview available at " + url), nil
		},
	)
}

func registerDataTools(srv *server.MCPServer, sess *session.Session) {
	srv.AddTool(
		mcp.NewTool(
			"fetch_elevation",
			mcp.WithDescription("Fetch elevation grid for the currently configured bounds."),
			mcp.WithNumber("resolution", mcp.Description("Grid resolution (default 200)")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			resolution := mcp.ParseInt(req, "resolution", 200)
			if err := core.FetchElevation(ctx, sess, resolution); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			e := sess.FetchedData.Elevation
			return mcp.NewToolResultText(
				fmt.Sprintf("Elevation fetched: %dx%d (%.0fm to %.0fm)", e.Resolution, e.Resolution, e.MinElevation, e.MaxElevation),
			), nil
		},
	)

	srv.AddTool(
		mcp.NewTool(
			"fetch_features",
			mcp.WithDescription("Fetch OSM feature overlays: roads, water, buildings."),
			mcp.WithString("include", mcp.Description("CSV list: roads,water,buildings (default all)")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			include := mcp.ParseString(req, "include", "roads,water,buildings")
			features := parseFeatureCSV(include)
			if len(features) == 0 {
				features = []string{"roads", "water", "buildings"}
			}
			if err := core.FetchFeatures(ctx, sess, features); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			if sess.FetchedData.Features == nil {
				return mcp.NewToolResultText("Features fetched: none"), nil
			}
			f := sess.FetchedData.Features
			return mcp.NewToolResultText(
				fmt.Sprintf("Features fetched: roads=%d water=%d buildings=%d", len(f.Roads), len(f.Water), len(f.Buildings)),
			), nil
		},
	)
}

func registerGenerateTools(srv *server.MCPServer, sess *session.Session) {
	srv.AddTool(
		mcp.NewTool("generate_model", mcp.WithDescription("Generate terrain + overlay meshes from fetched data.")),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = req
			if err := core.GenerateModel(ctx, sess); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			terrainVerts := 0
			if sess.Results.TerrainMesh != nil {
				terrainVerts = len(sess.Results.TerrainMesh.Vertices)
			}
			return mcp.NewToolResultText(
				fmt.Sprintf("Model generated: terrain_vertices=%d feature_meshes=%d gpx_mesh=%t", terrainVerts, len(sess.Results.FeatureMeshes), sess.Results.GpxMesh != nil),
			), nil
		},
	)

	srv.AddTool(
		mcp.NewTool(
			"generate_map_insert",
			mcp.WithDescription("Generate map insert artifacts (SVG metadata and/or printable plate mesh)."),
			mcp.WithString("format", mcp.Description("svg|plate|both (default both)")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			if !sess.Config.Bounds.IsSet {
				return mcp.NewToolResultError("set area first"), nil
			}
			format := strings.TrimSpace(strings.ToLower(mcp.ParseString(req, "format", "both")))
			if format == "" {
				format = "both"
			}
			if format != "svg" && format != "plate" && format != "both" {
				return mcp.NewToolResultError("format must be svg, plate, or both"), nil
			}

			results := []string{}
			if format == "svg" || format == "both" {
				_ = core.GenerateMapInsertSVG(sess.Config.Bounds, sess.FetchedData.Features, sess.Config.GpxTracks, sess.Config.Colors)
				results = append(results, "svg generated")
			}
			if format == "plate" || format == "both" {
				tr := core.NewGeoToModelTransform(sess.Config.Bounds, sess.Config.ModelParams.WidthMM)
				sess.Results.MapInsertMesh = core.GenerateMapInsertPlate(sess.Config.Bounds, tr, 1.0)
				results = append(results, "plate generated")
			}
			return mcp.NewToolResultText("Map insert: " + strings.Join(results, ", ")), nil
		},
	)
}

func registerModelTools(srv *server.MCPServer, sess *session.Session) {
	srv.AddTool(
		mcp.NewTool(
			"set_model_params",
			mcp.WithDescription("Update model parameters used by generate_model."),
			mcp.WithNumber("width_mm", mcp.Description("Model width in mm")),
			mcp.WithNumber("vertical_scale", mcp.Description("Elevation exaggeration")),
			mcp.WithNumber("base_height_mm", mcp.Description("Base thickness in mm")),
			mcp.WithString("shape", mcp.Description("square|circle|hexagon|rectangle")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			args := req.GetArguments()
			if _, ok := args["width_mm"]; ok {
				sess.Config.ModelParams.WidthMM = mcp.ParseFloat64(req, "width_mm", sess.Config.ModelParams.WidthMM)
			}
			if _, ok := args["vertical_scale"]; ok {
				sess.Config.ModelParams.VerticalScale = mcp.ParseFloat64(req, "vertical_scale", sess.Config.ModelParams.VerticalScale)
			}
			if _, ok := args["base_height_mm"]; ok {
				sess.Config.ModelParams.BaseHeightMM = mcp.ParseFloat64(req, "base_height_mm", sess.Config.ModelParams.BaseHeightMM)
			}
			if _, ok := args["shape"]; ok {
				shape := strings.TrimSpace(strings.ToLower(mcp.ParseString(req, "shape", sess.Config.ModelParams.Shape)))
				switch shape {
				case "square", "circle", "hexagon", "rectangle":
					sess.Config.ModelParams.Shape = shape
				default:
					return mcp.NewToolResultError("invalid shape: must be square, circle, hexagon, or rectangle"), nil
				}
			}
			sess.Results.TerrainMesh = nil
			sess.Results.FeatureMeshes = nil
			sess.Results.GpxMesh = nil
			return mcp.NewToolResultText(
				fmt.Sprintf(
					"Model params updated: width_mm=%.2f vertical_scale=%.2f base_height_mm=%.2f shape=%s",
					sess.Config.ModelParams.WidthMM,
					sess.Config.ModelParams.VerticalScale,
					sess.Config.ModelParams.BaseHeightMM,
					sess.Config.ModelParams.Shape,
				),
			), nil
		},
	)

	srv.AddTool(
		mcp.NewTool(
			"set_colors",
			mcp.WithDescription("Update color configuration used by exporters."),
			mcp.WithString("terrain"),
			mcp.WithString("roads"),
			mcp.WithString("water"),
			mcp.WithString("buildings"),
			mcp.WithString("gpx_track"),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			args := req.GetArguments()
			if _, ok := args["terrain"]; ok {
				sess.Config.Colors.Terrain = mcp.ParseString(req, "terrain", sess.Config.Colors.Terrain)
			}
			if _, ok := args["roads"]; ok {
				sess.Config.Colors.Roads = mcp.ParseString(req, "roads", sess.Config.Colors.Roads)
			}
			if _, ok := args["water"]; ok {
				sess.Config.Colors.Water = mcp.ParseString(req, "water", sess.Config.Colors.Water)
			}
			if _, ok := args["buildings"]; ok {
				sess.Config.Colors.Buildings = mcp.ParseString(req, "buildings", sess.Config.Colors.Buildings)
			}
			if _, ok := args["gpx_track"]; ok {
				sess.Config.Colors.GpxTrack = mcp.ParseString(req, "gpx_track", sess.Config.Colors.GpxTrack)
			}
			if err := sess.Config.Colors.Validate(); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText("Colors updated"), nil
		},
	)
}

func registerExportTools(srv *server.MCPServer, sess *session.Session) {
	srv.AddTool(
		mcp.NewTool("export_3mf", mcp.WithDescription("Export current model to .3mf."), mcp.WithString("output_path", mcp.Required())),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			path := mcp.ParseString(req, "output_path", "")
			if path == "" {
				return mcp.NewToolResultError("output_path is required"), nil
			}
			if err := exporters.Export3MF(sess, path); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText("3MF exported: " + path), nil
		},
	)

	srv.AddTool(
		mcp.NewTool("export_openscad", mcp.WithDescription("Export current model to .scad."), mcp.WithString("output_path", mcp.Required())),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			path := mcp.ParseString(req, "output_path", "")
			if path == "" {
				return mcp.NewToolResultError("output_path is required"), nil
			}
			if err := exporters.ExportOpenSCAD(sess, path); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText("OpenSCAD exported: " + path), nil
		},
	)

	srv.AddTool(
		mcp.NewTool("export_svg", mcp.WithDescription("Export map insert style SVG."), mcp.WithString("output_path", mcp.Required())),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			path := mcp.ParseString(req, "output_path", "")
			if path == "" {
				return mcp.NewToolResultError("output_path is required"), nil
			}
			if err := exporters.ExportSVG(sess, path); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText("SVG exported: " + path), nil
		},
	)
}

func parseFeatureCSV(v string) []string {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	out := make([]string, 0, len(parts))
	seen := map[string]bool{}
	for _, p := range parts {
		p = strings.TrimSpace(strings.ToLower(p))
		switch p {
		case "roads", "water", "buildings":
			if !seen[p] {
				seen[p] = true
				out = append(out, p)
			}
		}
	}
	return out
}

func registerSessionTools(srv *server.MCPServer, sess *session.Session) {
	srv.AddTool(
		mcp.NewTool(
			"save_session",
			mcp.WithDescription("Save current session config to JSON."),
			mcp.WithString("path", mcp.Description("Optional path; default ~/.cache/topo-shadow-box/session.json")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			path := mcp.ParseString(req, "path", session.DefaultPath())
			if err := session.SaveSession(sess, path); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText("Session saved: " + path), nil
		},
	)

	srv.AddTool(
		mcp.NewTool(
			"load_session",
			mcp.WithDescription("Load session config from JSON and clear downstream data."),
			mcp.WithString("path", mcp.Description("Optional path; default ~/.cache/topo-shadow-box/session.json")),
		),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			_ = ctx
			path := mcp.ParseString(req, "path", session.DefaultPath())
			if err := session.LoadSession(sess, path); err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText("Session loaded: " + path + ". Re-run fetch_elevation and generate_model."), nil
		},
	)
}
