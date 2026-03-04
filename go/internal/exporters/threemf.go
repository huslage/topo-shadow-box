package exporters

import (
	"archive/zip"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/huslage/topo-shadow-box/internal/session"
)

const contentTypesXML = `<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>`

const relsXML = `<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>`

func Export3MF(s *session.Session, outputPath string) error {
	meshes, err := collectMeshes(s)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}
	f, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}
	defer f.Close()

	zw := zip.NewWriter(f)
	if err := writeZipEntry(zw, "[Content_Types].xml", contentTypesXML); err != nil {
		return err
	}
	if err := writeZipEntry(zw, "_rels/.rels", relsXML); err != nil {
		return err
	}
	if err := writeZipEntry(zw, "3D/3dmodel.model", buildModelXML(meshes)); err != nil {
		return err
	}
	if err := zw.Close(); err != nil {
		return fmt.Errorf("close zip writer: %w", err)
	}
	return nil
}

func writeZipEntry(w *zip.Writer, name, body string) error {
	f, err := w.Create(name)
	if err != nil {
		return fmt.Errorf("create zip entry %s: %w", name, err)
	}
	if _, err := f.Write([]byte(body)); err != nil {
		return fmt.Errorf("write zip entry %s: %w", name, err)
	}
	return nil
}

func buildModelXML(meshes []MeshExport) string {
	var b strings.Builder
	b.WriteString("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
	b.WriteString("<model unit=\"millimeter\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\" xml:lang=\"en-US\">\n")
	b.WriteString("  <resources>\n")

	for i, m := range meshes {
		if len(m.Vertices) == 0 || len(m.Faces) == 0 {
			continue
		}
		objID := i + 1
		fmt.Fprintf(&b, "    <object id=\"%d\" type=\"model\" name=\"%s\">\n", objID, xmlEscape(m.Name))
		b.WriteString("      <mesh>\n")
		b.WriteString("        <vertices>\n")
		for _, v := range m.Vertices {
			fmt.Fprintf(&b, "          <vertex x=\"%.6f\" y=\"%.6f\" z=\"%.6f\"/>\n", v[0], v[1], v[2])
		}
		b.WriteString("        </vertices>\n")
		b.WriteString("        <triangles>\n")
		for _, f := range m.Faces {
			fmt.Fprintf(&b, "          <triangle v1=\"%d\" v2=\"%d\" v3=\"%d\"/>\n", f[0], f[1], f[2])
		}
		b.WriteString("        </triangles>\n")
		b.WriteString("      </mesh>\n")
		b.WriteString("    </object>\n")
	}

	b.WriteString("  </resources>\n")
	b.WriteString("  <build>\n")
	for i := range meshes {
		fmt.Fprintf(&b, "    <item objectid=\"%d\"/>\n", i+1)
	}
	b.WriteString("  </build>\n")
	b.WriteString("</model>\n")
	return b.String()
}

func xmlEscape(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}
