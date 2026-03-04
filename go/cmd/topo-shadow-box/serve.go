package main

import (
	"github.com/huslage/topo-shadow-box/internal/session"
	"github.com/huslage/topo-shadow-box/internal/tools"
	"github.com/mark3labs/mcp-go/server"
)

func runServer() error {
	sess := session.New()
	srv := server.NewMCPServer(
		"topo-shadow-box",
		"0.1.0",
		server.WithInstructions("Generate 3D-printable topographic shadow boxes from area bounds, OSM features, and GPX tracks."),
	)

	tools.RegisterAll(srv, sess)

	return server.ServeStdio(srv)
}
