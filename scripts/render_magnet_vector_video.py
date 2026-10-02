#!/usr/bin/env python3
"""Render each magnetic sensor's XYZ reading as an animated 3D vector video."""

import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np


AXIS_COLORS = {
    "X": (50, 80, 220),
    "Y": (70, 165, 65),
    "Z": (220, 120, 45),
}
SENSOR_COLORS = (
    (214, 39, 40),
    (31, 119, 180),
    (44, 160, 44),
    (255, 127, 14),
    (148, 103, 189),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-csv", required=True, help="CSV containing sensor XYZ readings.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument(
        "--value-prefix",
        default="auto",
        choices=("auto", "magnet", "raw", "normalized", "s"),
        help=(
            "CSV field prefix. auto prefers normalized, raw, magnet, then "
            "ARPoseStreamer s0..sN fields."
        ),
    )
    parser.add_argument(
        "--sensor-count",
        type=int,
        default=5,
        help="Number of sensors to render (default: 5).",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Output frame rate.")
    parser.add_argument("--width", type=int, default=1280, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Output height in pixels.")
    parser.add_argument(
        "--vector-limit",
        type=float,
        default=None,
        help="Fixed symmetric XYZ limit. Defaults to the robust 95th percentile norm.",
    )
    parser.add_argument(
        "--baseline",
        choices=("none", "first"),
        default="first",
        help="Subtract the first finite reading for each sensor/axis (default: first).",
    )
    parser.add_argument("--codec", default="mp4v", help="FourCC codec (default: mp4v).")
    return parser.parse_args()


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def available_prefixes(fieldnames, sensor_count):
    prefixes = []
    for prefix in ("normalized", "raw", "magnet"):
        if all(
            f"{prefix}_s{sensor}_x" in fieldnames
            and f"{prefix}_s{sensor}_y" in fieldnames
            and f"{prefix}_s{sensor}_z" in fieldnames
            for sensor in range(1, sensor_count + 1)
        ):
            prefixes.append(prefix)
    if all(
        f"s{sensor - 1}_x" in fieldnames
        and f"s{sensor - 1}_y" in fieldnames
        and f"s{sensor - 1}_z" in fieldnames
        for sensor in range(1, sensor_count + 1)
    ):
        prefixes.append("s")
    return prefixes


def load_magnet_csv(path, sensor_count=5, value_prefix="auto"):
    path = Path(path).expanduser()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Magnet CSV is empty: {path}")
    available = available_prefixes(fieldnames, sensor_count)
    if value_prefix == "auto":
        if not available:
            raise KeyError(
                f"No complete S1-S{sensor_count} XYZ field group in {path}; "
                "expected normalized_sN_x/y/z, raw_sN_x/y/z, magnet_sN_x/y/z, "
                "or ARPoseStreamer s0_x/y/z fields"
            )
        value_prefix = available[0]
    elif value_prefix not in available:
        raise KeyError(
            f"Prefix {value_prefix!r} does not contain complete S1-S{sensor_count} XYZ fields; "
            f"available complete groups: {available or 'none'}"
        )

    elapsed_key = "elapsed" if "elapsed" in fieldnames else "relative_time"
    elapsed = np.asarray([_float_or_nan(row.get(elapsed_key)) for row in rows], dtype=np.float32)
    if not np.isfinite(elapsed).any():
        elapsed = np.arange(len(rows), dtype=np.float32)
    values = np.full((len(rows), sensor_count, 3), np.nan, dtype=np.float32)
    for row_index, row in enumerate(rows):
        for sensor_index in range(sensor_count):
            sensor = sensor_index + 1
            for axis_index, axis in enumerate(("x", "y", "z")):
                key = (
                    f"s{sensor_index}_{axis}"
                    if value_prefix == "s"
                    else f"{value_prefix}_s{sensor}_{axis}"
                )
                values[row_index, sensor_index, axis_index] = _float_or_nan(
                    row.get(key)
                )
    order = np.argsort(np.nan_to_num(elapsed, nan=np.inf))
    return elapsed[order], values[order], value_prefix


def first_finite_baseline(values):
    baseline = np.full(values.shape[1:], np.nan, dtype=np.float32)
    for sensor_index in range(values.shape[1]):
        for axis_index in range(3):
            series = values[:, sensor_index, axis_index]
            valid = np.flatnonzero(np.isfinite(series))
            if valid.size:
                baseline[sensor_index, axis_index] = series[valid[0]]
    return baseline


def vector_limit(values, explicit_limit=None):
    if explicit_limit is not None:
        limit = float(explicit_limit)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("--vector-limit must be positive and finite")
        return limit
    norms = np.linalg.norm(values, axis=2)
    finite = norms[np.isfinite(norms)]
    if finite.size == 0:
        return 1.0
    return max(1.0, float(np.percentile(finite, 95.0)))


def _draw_arrow(image, start, end, color, thickness=3):
    cv2.arrowedLine(image, start, end, color, thickness, cv2.LINE_AA, tipLength=0.16)


def _draw_sensor_panel(image, rect, sensor_index, value, limit):
    x0, y0, x1, y1 = rect
    panel = image[y0:y1, x0:x1]
    height, width = panel.shape[:2]
    cv2.rectangle(image, (x0, y0), (x1, y1), (235, 235, 235), 1)
    cv2.putText(
        image,
        f"S{sensor_index + 1}",
        (x0 + 12, y0 + 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        SENSOR_COLORS[sensor_index % len(SENSOR_COLORS)],
        2,
        cv2.LINE_AA,
    )
    origin = (int(width * 0.45), int(height * 0.61))
    scale = min(width, height) * 0.30 / max(limit, 1e-6)
    axis_length = int(min(width, height) * 0.30)
    # Fixed reference coordinate system: X right, Y up-right, Z up.
    axis_directions = ((1.0, 0.0), (0.48, -0.48), (0.0, -1.0))
    for axis_name, (dx, dy) in zip(("X", "Y", "Z"), axis_directions):
        end = (int(origin[0] + dx * axis_length), int(origin[1] + dy * axis_length))
        _draw_arrow(panel, origin, end, AXIS_COLORS[axis_name], thickness=1)
        cv2.putText(
            panel,
            axis_name,
            (end[0] + 4, end[1] + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            AXIS_COLORS[axis_name],
            1,
            cv2.LINE_AA,
        )
    if np.isfinite(value).all():
        # Keep outliers on the panel boundary rather than allowing them to
        # obscure neighboring sensor panels.
        clipped = np.clip(value, -limit, limit)
        dx = clipped[0] * scale + clipped[1] * scale * 0.48
        dy = -clipped[2] * scale - clipped[1] * scale * 0.48
        end = (int(origin[0] + dx), int(origin[1] + dy))
        _draw_arrow(panel, origin, end, SENSOR_COLORS[sensor_index % len(SENSOR_COLORS)], thickness=3)
        text = f"X={value[0]:.1f}  Y={value[1]:.1f}  Z={value[2]:.1f}"
    else:
        text = "XYZ unavailable"
    cv2.putText(
        panel,
        text,
        (12, height - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )


def render_vector_frame(values, elapsed, frame_index, limit, width=1280, height=720):
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(
        image,
        "Magnetic XYZ vectors",
        (26, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"t={float(elapsed[frame_index]):.2f} s    vector range: +/-{limit:.1f}",
        (26, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    sensor_count = values.shape[1]
    cols = min(3, sensor_count)
    rows = int(math.ceil(sensor_count / cols))
    margin = 18
    top = 84
    cell_width = (width - margin * (cols + 1)) // cols
    cell_height = (height - top - margin * (rows + 1)) // rows
    for sensor_index in range(sensor_count):
        row, col = divmod(sensor_index, cols)
        x0 = margin + col * (cell_width + margin)
        y0 = top + margin + row * (cell_height + margin)
        _draw_sensor_panel(
            image,
            (x0, y0, x0 + cell_width, y0 + cell_height),
            sensor_index,
            values[frame_index, sensor_index],
            limit,
        )
    return image


def render_vector_video(
    elapsed,
    values,
    output,
    fps=12.0,
    width=1280,
    height=720,
    vector_limit_value=None,
    codec="mp4v",
):
    elapsed = np.asarray(elapsed, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(elapsed) or values.shape[2] != 3:
        raise ValueError("values must have shape (frames, sensors, 3) matching elapsed")
    if fps <= 0.0 or width <= 0 or height <= 0:
        raise ValueError("fps, width, and height must be positive")
    limit = vector_limit(values, vector_limit_value)
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*str(codec)[:4]),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")
    try:
        for frame_index in range(len(values)):
            writer.write(render_vector_frame(values, elapsed, frame_index, limit, width, height))
    finally:
        writer.release()
    return output, limit


def main():
    args = parse_args()
    if args.sensor_count <= 0:
        raise ValueError("--sensor-count must be positive")
    elapsed, values, prefix = load_magnet_csv(
        args.trace_csv,
        sensor_count=args.sensor_count,
        value_prefix=args.value_prefix,
    )
    if args.baseline == "first":
        values = values - first_finite_baseline(values)[None, :, :]
    output, limit = render_vector_video(
        elapsed,
        values,
        args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        vector_limit_value=args.vector_limit,
        codec=args.codec,
    )
    print(
        f"Wrote {output}: frames={len(values)}, sensors={args.sensor_count}, "
        f"prefix={prefix}, baseline={args.baseline}, vector_limit={limit:.3f}"
    )


if __name__ == "__main__":
    main()
