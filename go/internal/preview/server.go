package preview

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/huslage/topo-shadow-box/internal/session"
)

var (
	mu       sync.RWMutex
	running  bool
	stateRef *session.Session
)

type meshPayload struct {
	Name     string       `json:"name"`
	Type     string       `json:"type"`
	Color    string       `json:"color"`
	Vertices [][3]float64 `json:"vertices"`
	Faces    [][3]int     `json:"faces"`
}

type scenePayload struct {
	Meshes []meshPayload `json:"meshes"`
}

func StartOrUpdate(s *session.Session, port int) (string, error) {
	if s == nil {
		return "", fmt.Errorf("session is nil")
	}
	mu.Lock()
	stateRef = s
	if running {
		mu.Unlock()
		return fmt.Sprintf("http://localhost:%d", port), nil
	}
	running = true
	mu.Unlock()

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/data", handleData)

	go func() {
		addr := fmt.Sprintf("127.0.0.1:%d", port)
		_ = http.ListenAndServe(addr, mux)
	}()

	return fmt.Sprintf("http://localhost:%d", port), nil
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(viewerHTML))
}

func handleData(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	s := stateRef
	mu.RUnlock()
	if s == nil {
		http.Error(w, `{"error":"no session"}`, http.StatusServiceUnavailable)
		return
	}

	payload := scenePayload{}
	if s.Results.TerrainMesh != nil {
		payload.Meshes = append(payload.Meshes, meshPayload{
			Name: s.Results.TerrainMesh.Name, Type: s.Results.TerrainMesh.FeatureType,
			Color:    s.Config.Colors.Terrain,
			Vertices: s.Results.TerrainMesh.Vertices, Faces: s.Results.TerrainMesh.Faces,
		})
	}
	for _, m := range s.Results.FeatureMeshes {
		payload.Meshes = append(payload.Meshes, meshPayload{
			Name: m.Name, Type: m.FeatureType,
			Color:    colorForFeatureType(s.Config.Colors, m.FeatureType),
			Vertices: m.Vertices, Faces: m.Faces,
		})
	}
	if s.Results.GpxMesh != nil {
		payload.Meshes = append(payload.Meshes, meshPayload{
			Name: s.Results.GpxMesh.Name, Type: s.Results.GpxMesh.FeatureType,
			Color:    s.Config.Colors.GpxTrack,
			Vertices: s.Results.GpxMesh.Vertices, Faces: s.Results.GpxMesh.Faces,
		})
	}
	if s.Results.MapInsertMesh != nil {
		payload.Meshes = append(payload.Meshes, meshPayload{
			Name: s.Results.MapInsertMesh.Name, Type: s.Results.MapInsertMesh.FeatureType,
			Color:    s.Config.Colors.MapInsert,
			Vertices: s.Results.MapInsertMesh.Vertices, Faces: s.Results.MapInsertMesh.Faces,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(payload)
}

func colorForFeatureType(c session.Colors, ft string) string {
	switch ft {
	case "road", "roads":
		return c.Roads
	case "water":
		return c.Water
	case "building", "buildings":
		return c.Buildings
	case "gpx", "gpx_track":
		return c.GpxTrack
	case "map_insert":
		return c.MapInsert
	default:
		return "#808080"
	}
}

const viewerHTML = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Topo Shadow Box Preview</title>
<style>
body { margin:0; background:#111827; color:#e5e7eb; font-family: ui-sans-serif, system-ui, sans-serif; }
#status { position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,.65); padding: 8px 10px; border-radius: 6px; font-size: 12px; }
</style>
</head>
<body>
<div id="status">connecting...</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111827);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 10000);
camera.position.set(220, 220, 220);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const dl = new THREE.DirectionalLight(0xffffff, 0.95); dl.position.set(200,300,100); scene.add(dl);
scene.add(new THREE.GridHelper(400, 40, 0x374151, 0x1f2937));

let groups = [];
function clearMeshes(){ for (const g of groups) scene.remove(g); groups = []; }
function addMesh(m){
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(m.vertices.flat()), 3));
  g.setIndex(m.faces.flat());
  g.computeVertexNormals();
  const mat = new THREE.MeshPhongMaterial({ color: new THREE.Color(m.color || '#888888'), side: THREE.DoubleSide, flatShading: true });
  const mesh = new THREE.Mesh(g, mat);
  scene.add(mesh);
  groups.push(mesh);
}
function fit(){
  const box = new THREE.Box3();
  for (const g of groups) box.expandByObject(g);
  const c = box.getCenter(new THREE.Vector3());
  const s = box.getSize(new THREE.Vector3());
  controls.target.copy(c);
  camera.position.set(c.x + s.x*1.2 + 20, c.y + s.y*1.3 + 20, c.z + s.z*1.2 + 20);
  controls.update();
}
async function tick(){
  try {
    const res = await fetch('/data', {cache:'no-store'});
    const data = await res.json();
    clearMeshes();
    for (const m of (data.meshes || [])) addMesh(m);
    if ((data.meshes || []).length) fit();
    document.getElementById('status').textContent = 'meshes: ' + (data.meshes || []).length + ' | updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('status').textContent = 'preview fetch failed';
  }
}
setInterval(tick, 1000);
tick();
function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();
window.addEventListener('resize', () => { camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); });
</script>
</body>
</html>`
