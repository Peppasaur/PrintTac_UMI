#!/usr/bin/env python3
"""Generate thin, raised-pattern 3D models for the UMI ArUco marker pair."""

import argparse
import math
from pathlib import Path


MARKERS = {
    0: (
        (1, 1, 1, 1, 1, 1),
        (1, 0, 1, 0, 0, 1),
        (1, 1, 0, 1, 0, 1),
        (1, 1, 1, 0, 0, 1),
        (1, 1, 1, 0, 1, 1),
        (1, 1, 1, 1, 1, 1),
    ),
    1: (
        (1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
        (1, 0, 0, 0, 0, 1),
        (1, 0, 1, 1, 0, 1),
        (1, 0, 1, 0, 1, 1),
        (1, 1, 1, 1, 1, 1),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--marker-edge-mm", type=float, default=16.0)
    parser.add_argument("--base-thickness-mm", type=float, default=1.20)
    parser.add_argument("--pattern-height-mm", type=float, default=0.15)
    return parser.parse_args()


def _validate(marker_edge_mm, base_thickness_mm, pattern_height_mm):
    values = {
        "marker_edge_mm": marker_edge_mm,
        "base_thickness_mm": base_thickness_mm,
        "pattern_height_mm": pattern_height_mm,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("All dimensions must be finite")
    if marker_edge_mm <= 0 or base_thickness_mm <= 0 or pattern_height_mm <= 0:
        raise ValueError("marker edge, base thickness, and pattern height must be positive")


def _box_vertices(x0, y0, z0, x1, y1, z1):
    return [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]


def _append_box(vertices, faces, materials, material, x0, y0, z0, x1, y1, z1):
    start = len(vertices)
    vertices.extend(_box_vertices(x0, y0, z0, x1, y1, z1))
    faces.extend(
        (
            (start + 0, start + 3, start + 2, start + 1),
            (start + 4, start + 5, start + 6, start + 7),
            (start + 0, start + 1, start + 5, start + 4),
            (start + 1, start + 2, start + 6, start + 5),
            (start + 2, start + 3, start + 7, start + 6),
            (start + 3, start + 0, start + 4, start + 7),
        )
    )
    materials.extend([material] * 6)


def build_mesh(marker, edge, base, height):
    cell = edge / 6.0
    vertices = []
    faces = []
    materials = []
    _append_box(vertices, faces, materials, "white", 0.0, 0.0, 0.0, edge, edge, base)
    for row, values in enumerate(marker):
        for col, is_black in enumerate(values):
            if is_black:
                _append_box(
                    vertices,
                    faces,
                    materials,
                    "black",
                    col * cell,
                    (5 - row) * cell,
                    base,
                    (col + 1) * cell,
                    (6 - row) * cell,
                    base + height,
                )
    return vertices, faces, materials


def write_ascii_stl(path, vertices, faces, name):
    with path.open("w") as output:
        output.write(f"solid {name}\n")
        for face in faces:
            a, b, c, d = (vertices[index] for index in face)
            for triangle in ((a, b, c), (a, c, d)):
                ux = triangle[1][0] - triangle[0][0]
                uy = triangle[1][1] - triangle[0][1]
                uz = triangle[1][2] - triangle[0][2]
                vx = triangle[2][0] - triangle[0][0]
                vy = triangle[2][1] - triangle[0][1]
                vz = triangle[2][2] - triangle[0][2]
                nx = uy * vz - uz * vy
                ny = uz * vx - ux * vz
                nz = ux * vy - uy * vx
                norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                output.write(f"  facet normal {nx / norm:.9g} {ny / norm:.9g} {nz / norm:.9g}\n")
                output.write("    outer loop\n")
                for point in triangle:
                    output.write(f"      vertex {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
                output.write("    endloop\n  endfacet\n")
        output.write(f"endsolid {name}\n")


def write_obj(path, vertices, faces, materials, name):
    material_path = path.with_suffix(".mtl")
    material_path.write_text(
        "newmtl white\nKd 1 1 1\nKa 0.1 0.1 0.1\n\n"
        "newmtl black\nKd 0.005 0.005 0.005\nKa 0 0 0\n"
    )
    with path.open("w") as output:
        output.write(f"# {name}; dimensions in millimeters\n")
        output.write(f"mtllib {material_path.name}\n")
        for vertex in vertices:
            output.write("v %.9g %.9g %.9g\n" % vertex)
        previous_material = None
        for face, material in zip(faces, materials):
            if material != previous_material:
                output.write(f"usemtl {material}\n")
                previous_material = material
            output.write("f %s\n" % " ".join(str(index + 1) for index in face))


def _mesh_for_material(vertices, faces, materials, selected_material):
    selected_faces = [
        face for face, material in zip(faces, materials) if material == selected_material
    ]
    used_indices = sorted({index for face in selected_faces for index in face})
    index_map = {old: new for new, old in enumerate(used_indices)}
    return (
        [vertices[index] for index in used_indices],
        [tuple(index_map[index] for index in face) for face in selected_faces],
    )


def build_union_mesh(marker, edge, base, height):
    """Build a watertight height-field mesh for the single-material STL."""
    cell = edge / 6.0
    vertices = []
    faces = []
    materials = []

    def add_quad(points):
        start = len(vertices)
        vertices.extend(points)
        faces.append((start, start + 1, start + 2, start + 3))
        materials.append("single")

    for row, values in enumerate(marker):
        for col, is_black in enumerate(values):
            x0, x1 = col * cell, (col + 1) * cell
            y0, y1 = (5 - row) * cell, (6 - row) * cell
            top = base + (height if is_black else 0.0)
            add_quad(((x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top)))
            add_quad(((x0, y0, 0.0), (x0, y1, 0.0), (x1, y1, 0.0), (x1, y0, 0.0)))

            neighbors = (
                (-1, 0, ((x0, y1), (x1, y1))),
                (1, 0, ((x1, y0), (x0, y0))),
                (0, -1, ((x0, y0), (x0, y1))),
                (0, 1, ((x1, y1), (x1, y0))),
            )
            for d_row, d_col, ((ax, ay), (bx, by)) in neighbors:
                other_row = row + d_row
                other_col = col + d_col
                if 0 <= other_row < 6 and 0 <= other_col < 6:
                    other_top = base + (height if marker[other_row][other_col] else 0.0)
                else:
                    other_top = 0.0
                if top > other_top:
                    add_quad(
                        (
                            (ax, ay, other_top),
                            (bx, by, other_top),
                            (bx, by, top),
                            (ax, ay, top),
                        )
                    )
    return vertices, faces, materials


def generate_marker(marker_id, output_dir, edge, base, height):
    vertices, faces, materials = build_mesh(MARKERS[marker_id], edge, base, height)
    union_vertices, union_faces, _ = build_union_mesh(
        MARKERS[marker_id], edge, base, height
    )
    stem = output_dir / f"aruco_marker_id{marker_id}_16mm"
    write_ascii_stl(stem.with_suffix(".stl"), union_vertices, union_faces, stem.name)
    write_obj(stem.with_suffix(".obj"), vertices, faces, materials, stem.name)
    for material in ("white", "black"):
        part_vertices, part_faces = _mesh_for_material(
            vertices,
            faces,
            materials,
            material,
        )
        write_ascii_stl(
            stem.with_name(f"{stem.name}_{material}.stl"),
            part_vertices,
            part_faces,
            f"{stem.name}_{material}",
        )
    return stem.with_suffix(".stl"), stem.with_suffix(".obj"), len(vertices), len(faces)


def main():
    args = parse_args()
    _validate(
        args.marker_edge_mm,
        args.base_thickness_mm,
        args.pattern_height_mm,
    )
    if abs(args.marker_edge_mm - 16.0) > 1e-9:
        raise ValueError("This generator currently writes the 16 mm output filename; use marker_edge_mm=16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for marker_id in MARKERS:
        stl, obj, vertex_count, face_count = generate_marker(
            marker_id,
            args.output_dir,
            args.marker_edge_mm,
            args.base_thickness_mm,
            args.pattern_height_mm,
        )
        print(f"Wrote {stl} and {obj} ({vertex_count} vertices, {face_count} quad faces)")


if __name__ == "__main__":
    main()
