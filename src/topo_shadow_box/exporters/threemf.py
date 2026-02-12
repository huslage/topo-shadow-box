"""3MF multi-material export using custom XML + ZIP."""

import zipfile


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def export_3mf(meshes: list[dict], output_path: str) -> dict:
    """Export meshes as a multi-material 3MF file.

    Each mesh dict has: name, type, vertices, faces, color (hex string).

    The 3MF file is a ZIP archive containing:
    - [Content_Types].xml
    - _rels/.rels
    - 3D/3dmodel.model (the actual model XML)
    """
    objects = []
    for m in meshes:
        if not m.get("vertices") or not m.get("faces"):
            continue
        rgb = _hex_to_rgb(m["color"])
        objects.append((m["name"], m["vertices"], m["faces"], rgb))

    if not objects:
        raise ValueError("No mesh data to export")

    model_xml = _build_model_xml(objects)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("3D/3dmodel.model", model_xml)

    return {"success": True, "filepath": output_path, "objects": len(objects)}


def _build_model_xml(objects: list[tuple]) -> str:
    """Build the 3MF model XML with multiple colored objects."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US"',
        '  xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"',
        '  xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">',
        '  <metadata name="Application">topo-shadow-box</metadata>',
        "  <resources>",
    ]

    # Base materials
    parts.append('    <m:basematerials id="1">')
    for name, _, _, (r, g, b) in objects:
        safe_name = name.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
        parts.append(f'      <m:base name="{safe_name}" displaycolor="#{r:02X}{g:02X}{b:02X}"/>')
    parts.append("    </m:basematerials>")

    # Objects
    for obj_idx, (name, vertices, faces, _) in enumerate(objects):
        obj_id = obj_idx + 2
        safe_name = name.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
        parts.append(
            f'    <object id="{obj_id}" name="{safe_name}" '
            f'pid="1" pindex="{obj_idx}" type="model">'
        )
        parts.append("      <mesh>")

        # Vertices
        parts.append("        <vertices>")
        for v in vertices:
            parts.append(f'          <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>')
        parts.append("        </vertices>")

        # Triangles
        parts.append("        <triangles>")
        for f in faces:
            parts.append(f'          <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>')
        parts.append("        </triangles>")

        parts.append("      </mesh>")
        parts.append("    </object>")

    parts.append("  </resources>")

    # Build items
    parts.append("  <build>")
    for obj_idx in range(len(objects)):
        parts.append(f'    <item objectid="{obj_idx + 2}"/>')
    parts.append("  </build>")
    parts.append("</model>")

    return "\n".join(parts)


_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""
