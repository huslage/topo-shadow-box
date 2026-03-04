package main

import (
	"fmt"
	"os"
)

func runServer() error {
	fmt.Fprintln(os.Stderr, "error: MCP server not yet implemented — use the Python server")
	os.Exit(1)
	return nil // unreachable, satisfies compiler
}
