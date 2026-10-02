#!/usr/bin/env python3
"""Blender-side exporter for one materialized ArUco OBJ model."""

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def main():
    args = parse_args()
    for path in (args.glb, args.blend, args.preview):
        path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=str(args.input.resolve()))
    marker = bpy.context.selected_objects[0]
    marker.name = args.input.stem
    marker.scale = (0.001, 0.001, 0.001)  # OBJ dimensions are millimeters.
    bpy.context.view_layer.objects.active = marker
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    for material in marker.data.materials:
        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled is not None:
            value = 0.005 if material.name.lower().startswith("black") else 1.0
            principled.inputs["Base Color"].default_value = (value, value, value, 1.0)
            principled.inputs["Roughness"].default_value = 0.72
            principled.inputs["Metallic"].default_value = 0.0

    bpy.ops.object.select_all(action="DESELECT")
    marker.select_set(True)
    bpy.context.view_layer.objects.active = marker
    bpy.ops.export_scene.gltf(
        filepath=str(args.glb.resolve()),
        export_format="GLB",
        use_selection=True,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))

    camera_data = bpy.data.cameras.new("PreviewCamera")
    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.024, -0.025, 0.020)
    point_at(camera, (0.008, 0.008, 0.0005))
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 0.026
    camera_data.clip_start = 0.001
    bpy.context.scene.camera = camera

    for name, location, energy, size in (
        ("Key", (-0.025, -0.020, 0.040), 500.0, 0.025),
        ("Fill", (0.035, 0.010, 0.025), 300.0, 0.020),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = location
        point_at(light, (0.008, 0.008, 0.0))

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.render.filepath = str(args.preview.resolve())
    scene.view_settings.look = "AgX - Medium High Contrast"
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
