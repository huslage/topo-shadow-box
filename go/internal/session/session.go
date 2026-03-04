package session

import "sync"

type Config struct {
	Bounds                   Bounds
	ModelParams              ModelParams
	Colors                   Colors
	GpxTracks                []GpxTrack
	GpxWaypoints             []GpxWaypoint
	PendingGeocodeCandidates []GeocodeCandidate
}

type FetchedData struct {
	Elevation *ElevationData
	Features  *OsmFeatureSet
}

type Results struct {
	TerrainMesh   *Mesh
	FeatureMeshes []Mesh
	GpxMesh       *Mesh
	MapInsertMesh *Mesh
}

type Session struct {
	Config      Config
	FetchedData FetchedData
	Results     Results
	mu          sync.Mutex
}

func New() *Session {
	return &Session{
		Config: Config{
			ModelParams: DefaultModelParams(),
			Colors:      DefaultColors(),
		},
	}
}

func (s *Session) Lock()   { s.mu.Lock() }
func (s *Session) Unlock() { s.mu.Unlock() }

// ClearDownstream resets all fetched data and results.
// Call when the area changes.
func (s *Session) ClearDownstream() {
	s.FetchedData.Elevation = nil
	s.FetchedData.Features = nil
	s.Results.TerrainMesh = nil
	s.Results.FeatureMeshes = nil
	s.Results.GpxMesh = nil
	s.Results.MapInsertMesh = nil
}
