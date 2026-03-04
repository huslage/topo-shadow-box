# Development

```bash
# Run all Go tests
cd go
go test ./...

# Run package tests
cd go
go test ./internal/core/...

# Build local binary
cd go
go build -o /tmp/topo-shadow-box ./cmd/topo-shadow-box

# Start MCP server
/tmp/topo-shadow-box serve
```
