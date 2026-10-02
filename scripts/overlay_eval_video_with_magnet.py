#!/usr/bin/env python3
"""Overlay policy-eval magnet traces next to an existing video.

The trace CSV is produced by FrankaPolymetisEnv policy recording
(`*_magnet_trace.csv`). Frames are aligned by video time and the CSV
`elapsed` column.
"""

import argparse
import csv
import os
import re
from pathlib import Path

import cv2
import numpy as np


AXIS_COLORS = {
    # OpenCV BGR colors used by the live video panel.
    "X": (220, 60, 60),
    "Y": (60, 170, 60),
    "Z": (60, 100, 220),
}

PLOT_AXIS_COLORS = {
    # Matplotlib RGB colors matching the force-plot style supplied by the user.
    "X": "#f00000",
    "Y": "#00d900",
    "Z": "#0000e8",
}
PLOT_SENSOR_LINESTYLES = ("-", "--", "-.", ":")
PLOT_SENSOR_COLORS = ("#e41a1c", "#377eb8", "#4daf4a", "#ff7f00")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate an MP4 with a time-aligned magnet visualization panel."
    )
    parser.add_argument("--trace-csv", required=True, help="Policy recording *_magnet_trace.csv")
    parser.add_argument("--video", required=True, help="Input MP4/video path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--panel-width", type=int, default=620)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--sensor-count", type=int, default=4)
    parser.add_argument(
        "--raw-limit",
        type=float,
        default=None,
        help="Fixed raw-delta y-axis limit. Defaults to global 99th percentile over the whole trace.",
    )
    parser.add_argument(
        "--trim-video-start-sec",
        type=float,
        default=0.0,
        help=(
            "Drop the first N seconds of the input video before overlaying. "
            "The trimmed output starts at video time 0 and is aligned to CSV elapsed 0."
        ),
    )
    parser.add_argument(
        "--trim-trace-start-sec",
        type=float,
        default=0.0,
        help=(
            "Drop trace samples before N seconds and shift remaining CSV elapsed "
            "times so the trimmed trace starts at 0."
        ),
    )
    parser.add_argument(
        "--crop-video-left-ratio",
        "--crop-left-ratio",
        dest="crop_video_left_ratio",
        type=float,
        default=0.0,
        help="Fraction of the input video width to crop from the left (for example, 0.1 for 10%%).",
    )
    parser.add_argument(
        "--crop-video-right-ratio",
        "--crop-right-ratio",
        dest="crop_video_right_ratio",
        type=float,
        default=0.0,
        help="Fraction of the input video width to crop from the right (for example, 0.1 for 10%%).",
    )
    parser.add_argument(
        "--time-offset-sec",
        type=float,
        default=0.0,
        help="Additive offset: trace_time = video_time + offset.",
    )
    parser.add_argument(
        "--start-at-trace-zero",
        action="store_true",
        help="Map video frame 0 to the first CSV elapsed timestamp instead of 0.",
    )
    parser.add_argument(
        "--frame-output-dir",
        default=None,
        help=(
            "Optional directory for exporting cropped source-video frames. "
            "When set, one PNG is saved every --frame-save-interval frames."
        ),
    )
    parser.add_argument(
        "--frame-save-interval",
        type=int,
        default=30,
        help="Save one source-video image every N output frames.",
    )
    parser.add_argument(
        "--magnet-plot-output",
        default=None,
        help=(
            "Optional PNG/JPEG path for one full-duration magnetic-change plot. "
            "All sensors and XYZ axes are combined in the same image."
        ),
    )
    parser.add_argument(
        "--magnet-x-range",
        type=float,
        nargs=2,
        metavar=("X_MIN", "X_MAX"),
        default=None,
        help=(
            "Exact time-axis range in seconds for --magnet-plot-output, for example "
            "--magnet-x-range 2 12. Defaults to the full trace duration."
        ),
    )
    parser.add_argument(
        "--magnet-y-range",
        type=float,
        nargs=2,
        metavar=("Y_MIN", "Y_MAX"),
        default=None,
        help=(
            "Exact y-axis range for --magnet-plot-output, for example "
            "--magnet-y-range -2000 1000. Defaults to the symmetric --raw-limit range."
        ),
    )
    parser.add_argument(
        "--magnet-y-tick",
        type=float,
        default=1000.0,
        help=(
            "Major y-axis tick interval for --magnet-plot-output, for example "
            "--magnet-y-tick 100. Must be positive."
        ),
    )
    parser.add_argument(
        "--magnet-offset-file",
        "--normalized-offset-file",
        dest="magnet_offset_file",
        default=None,
        help=(
            "Optional normalized magnet offset file. The S1-S4 XYZ values are "
            "added to every channel after the trace first-sample baseline is removed. "
            "Accepts *_last_normalized.txt or a CSV with normalized_sN_x/y/z columns."
        ),
    )
    parser.add_argument("--codec", default="mp4v")
    return parser.parse_args()


def _float_or_nan(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_trace_csv(path, sensor_count):
    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Trace CSV is empty: {path}")

    elapsed = np.asarray([_float_or_nan(row.get("elapsed", "")) for row in rows], dtype=np.float32)
    if not np.isfinite(elapsed).any():
        raise ValueError(f"Trace CSV has no finite elapsed values: {path}")

    magnet = np.full((len(rows), sensor_count, 3), np.nan, dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for sensor_idx in range(sensor_count):
            for axis_idx, axis_name in enumerate(("x", "y", "z")):
                magnet_key = f"magnet_s{sensor_idx + 1}_{axis_name}"
                magnet[row_idx, sensor_idx, axis_idx] = _float_or_nan(row.get(magnet_key, ""))

    order = np.argsort(elapsed)
    return elapsed[order], magnet[order]


def trim_trace_start(elapsed, magnet, trim_sec):
    trim_sec = max(0.0, float(trim_sec))
    if trim_sec <= 0.0:
        return elapsed, magnet
    keep = elapsed >= trim_sec
    if not np.any(keep):
        raise ValueError(
            f"--trim-trace-start-sec={trim_sec:g} removed all trace samples "
            f"(trace ends at {float(np.nanmax(elapsed)):.3f}s)"
    )
    elapsed = elapsed[keep].astype(np.float32) - np.float32(trim_sec)
    return elapsed, magnet[keep]


def first_finite_baseline(values):
    baseline = np.full(values.shape[1:], np.nan, dtype=np.float32)
    for sensor_idx in range(values.shape[1]):
        for axis_idx in range(values.shape[2]):
            series = values[:, sensor_idx, axis_idx]
            valid = np.flatnonzero(np.isfinite(series))
            if valid.size > 0:
                baseline[sensor_idx, axis_idx] = series[valid[0]]
    return baseline


_NORMALIZED_OFFSET_LINE = re.compile(
    r"^\s*S(?P<sensor>\d+)\s*:\s*"
    r"x\s*=\s*(?P<x>[-+0-9.eE]+)\s+"
    r"y\s*=\s*(?P<y>[-+0-9.eE]+)\s+"
    r"z\s*=\s*(?P<z>[-+0-9.eE]+)\s*$",
    re.IGNORECASE,
)


def load_magnet_offset_file(path, sensor_count):
    """Load one normalized XYZ offset for each sensor from text or CSV."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Magnet offset file does not exist: {path}")
    sensor_count = int(sensor_count)
    offset = np.full((sensor_count, 3), np.nan, dtype=np.float32)

    # The gripper script's human-readable output has one ``S#: x=...`` line
    # per sensor.  Parse those lines without depending on metadata ordering.
    for line in path.read_text().splitlines():
        match = _NORMALIZED_OFFSET_LINE.match(line)
        if match is None:
            continue
        sensor_idx = int(match.group("sensor")) - 1
        if 0 <= sensor_idx < sensor_count:
            offset[sensor_idx] = [
                float(match.group("x")),
                float(match.group("y")),
                float(match.group("z")),
            ]

    # Also accept a trace-like CSV containing normalized_sN_x/y/z columns.
    if not np.isfinite(offset).all():
        try:
            with path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
        except (UnicodeDecodeError, csv.Error):
            rows = []
        if rows:
            row = rows[-1]
            for sensor_idx in range(sensor_count):
                values = [
                    _float_or_nan(row.get(f"normalized_s{sensor_idx + 1}_{axis}", ""))
                    for axis in ("x", "y", "z")
                ]
                if np.isfinite(values).all():
                    offset[sensor_idx] = values

    if not np.isfinite(offset).all():
        missing = [
            f"S{idx + 1}"
            for idx in range(sensor_count)
            if not np.isfinite(offset[idx]).all()
        ]
        raise ValueError(
            f"Magnet offset file {path} is missing finite XYZ values for {', '.join(missing)}"
        )
    return offset


def robust_limit(values, default=1.0):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(default)
    return max(float(default), float(np.percentile(np.abs(finite), 99.0)))


def video_crop_bounds(width, left_ratio, right_ratio):
    width = int(width)
    left_ratio = float(left_ratio)
    right_ratio = float(right_ratio)
    if width <= 0:
        raise ValueError(f"Video width must be positive, got {width}")
    if not np.isfinite(left_ratio) or not 0.0 <= left_ratio < 1.0:
        raise ValueError(
            f"--crop-video-left-ratio must be in [0, 1), got {left_ratio}"
        )
    if not np.isfinite(right_ratio) or not 0.0 <= right_ratio < 1.0:
        raise ValueError(
            f"--crop-video-right-ratio must be in [0, 1), got {right_ratio}"
        )
    if left_ratio + right_ratio >= 1.0:
        raise ValueError(
            "The left and right video crop ratios must sum to less than 1, "
            f"got {left_ratio + right_ratio:.6f}"
        )

    left = int(round(width * left_ratio))
    right = width - int(round(width * right_ratio))
    if right <= left:
        raise ValueError(
            "The requested video crop removes every pixel: "
            f"width={width}, left={left}, right={right}"
        )
    return left, right


def export_video_frame(frame, frame_idx, output_dir):
    output_path = output_dir / f"frame_{frame_idx:06d}.png"
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not write video frame image: {output_path}")
    return output_path


def draw_series(panel, times, values, sensor_idx, graph_left, graph_right, center_y, amplitude, value_limit):
    x_span = max(1e-6, float(times[-1] - times[0])) if len(times) > 1 else 1.0
    latest = []
    for axis_idx, axis_label in enumerate(("X", "Y", "Z")):
        series = values[:, sensor_idx, axis_idx]
        valid = np.flatnonzero(np.isfinite(series))
        latest.append(float(series[valid[-1]]) if valid.size else float("nan"))
        points = []
        for idx in valid:
            x = int(graph_left + (float(times[idx] - times[0]) / x_span) * (graph_right - graph_left))
            y = int(center_y - np.clip(float(series[idx]) / value_limit, -1.0, 1.0) * amplitude)
            points.append((x, y))
        if len(points) >= 2:
            cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, AXIS_COLORS[axis_label], 1, cv2.LINE_AA)
        elif len(points) == 1:
            cv2.circle(panel, points[0], 2, AXIS_COLORS[axis_label], -1, cv2.LINE_AA)
    return latest


def render_panel(
    elapsed,
    magnet,
    now,
    height,
    panel_width,
    window_sec,
    raw_limit,
    show_annotations=True,
):
    panel = np.full((height, panel_width, 3), 250, dtype=np.uint8)
    sensor_count = magnet.shape[1]
    if show_annotations:
        cv2.putText(panel, "Magnet", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    raw_left = 58 if show_annotations else 48
    raw_right = panel_width - 12
    if show_annotations:
        cv2.putText(panel, "Raw delta", (raw_left + 40, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        legend_x = panel_width - 112
        for axis_label, color in AXIS_COLORS.items():
            cv2.putText(panel, axis_label, (legend_x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            legend_x += 34

    end_idx = int(np.searchsorted(elapsed, now, side="right"))
    if end_idx <= 0:
        if show_annotations:
            cv2.putText(panel, "No trace samples yet", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
        return panel

    start_time = max(float(elapsed[0]), float(now) - float(window_sec))
    start_idx = int(np.searchsorted(elapsed, start_time, side="left"))
    times = elapsed[start_idx:end_idx]
    recent_magnet = magnet[start_idx:end_idx]
    if len(times) == 0:
        return panel

    baseline = first_finite_baseline(magnet[:end_idx])
    raw_delta = recent_magnet - baseline[None, :, :]
    header_height = 24
    footer_height = 18 if show_annotations else 4
    plot_height = max(1, height - header_height - footer_height)
    row_height = max(1, plot_height // sensor_count)
    for sensor_idx in range(sensor_count):
        y0 = header_height + sensor_idx * row_height
        y1 = header_height + (sensor_idx + 1) * row_height
        center_y = (y0 + y1) // 2
        amplitude_margin = 16 if show_annotations else 8
        amplitude = max(4, row_height // 2 - amplitude_margin)
        if show_annotations:
            cv2.line(panel, (raw_left, center_y), (raw_right, center_y), (210, 210, 210), 1)
        cv2.putText(panel, f"S{sensor_idx + 1}", (10, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

        raw_latest = draw_series(panel, times, raw_delta, sensor_idx, raw_left, raw_right, center_y, amplitude, raw_limit)
        if show_annotations:
            raw_text = " ".join(
                f"d{axis}={value:.0f}" if np.isfinite(value) else f"d{axis}=nan"
                for axis, value in zip(("X", "Y", "Z"), raw_latest)
            )
            cv2.putText(panel, raw_text, (raw_left, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (45, 45, 45), 1, cv2.LINE_AA)

    if show_annotations:
        cv2.putText(panel, f"raw +/-{raw_limit:.0f}", (12, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(panel, f"t={now:5.2f}s", (panel_width - 88, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1, cv2.LINE_AA)
    return panel


def render_full_trace_plot(
    elapsed,
    magnet,
    raw_limit,
    min_width=1450,
    row_height=150,
    magnet_offset=None,
    magnet_y_range=None,
    magnet_x_range=None,
    magnet_y_tick=1000.0,
):
    """Render one line per sensor, summing its three magnetic axes."""
    elapsed = np.asarray(elapsed, dtype=np.float32).reshape(-1)
    magnet = np.asarray(magnet, dtype=np.float32)
    if magnet.ndim != 3 or magnet.shape[0] != len(elapsed) or magnet.shape[2] != 3:
        raise ValueError(
            "Expected magnet shape (samples, sensors, 3) matching elapsed; "
            f"got elapsed={elapsed.shape}, magnet={magnet.shape}"
        )
    finite_time = np.isfinite(elapsed)
    if not np.any(finite_time):
        raise ValueError("Cannot render full trace without a finite elapsed timestamp")
    elapsed = elapsed[finite_time]
    magnet = magnet[finite_time]
    order = np.argsort(elapsed)
    elapsed = elapsed[order]
    magnet = magnet[order]

    sensor_count = int(magnet.shape[1])
    if sensor_count < 1:
        raise ValueError("Full magnetic trace plot requires at least one sensor")
    magnet_y_tick = float(magnet_y_tick)
    if not np.isfinite(magnet_y_tick) or magnet_y_tick <= 0.0:
        raise ValueError(
            f"--magnet-y-tick must be positive and finite, got {magnet_y_tick}"
        )

    # Import lazily so video overlay remains usable in minimal OpenCV-only
    # environments when --magnet-plot-output is not requested.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator

    width = max(int(min_width), 320)
    # Keep the compact wide layout used by the original overview.
    height = max(320, int(round(width * 0.288)))
    start_time = float(elapsed[0])
    end_time = float(elapsed[-1])
    if magnet_x_range is None:
        x_min = start_time
        x_max = end_time if end_time > start_time else start_time + 1.0
    else:
        x_min, x_max = (float(value) for value in magnet_x_range)
        if not np.isfinite([x_min, x_max]).all():
            raise ValueError("--magnet-x-range values must be finite")
        if x_min >= x_max:
            raise ValueError(
                "--magnet-x-range requires X_MIN < X_MAX, got "
                f"{x_min:g} >= {x_max:g}"
            )
    baseline = first_finite_baseline(magnet)
    delta = magnet - baseline[None, :, :]
    if magnet_offset is not None:
        magnet_offset = np.asarray(magnet_offset, dtype=np.float32)
        if magnet_offset.shape != (sensor_count, 3):
            raise ValueError(
                "magnet_offset must have shape (sensor_count, 3), got "
                f"{magnet_offset.shape} for sensor_count={sensor_count}"
            )
        if not np.isfinite(magnet_offset).all():
            raise ValueError("magnet_offset must contain only finite values")
        # Add the externally measured normalized state after baseline removal;
        # adding it to raw readings would cancel when the trace baseline is fit.
        delta = delta + magnet_offset[None, :, :]
    value_limit = max(1e-6, float(raw_limit))
    if magnet_y_range is None:
        y_min, y_max = -value_limit * 1.05, value_limit * 1.05
    else:
        y_min, y_max = (float(value) for value in magnet_y_range)
        if not np.isfinite([y_min, y_max]).all():
            raise ValueError("--magnet-y-range values must be finite")
        if y_min >= y_max:
            raise ValueError(
                "--magnet-y-range requires Y_MIN < Y_MAX, got "
                f"{y_min:g} >= {y_max:g}"
            )

    figure = Figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.078, 0.18, 0.90, 0.74), facecolor="white")
    times = elapsed.astype(np.float64)
    summed = np.sum(delta, axis=2)
    for sensor_idx in range(sensor_count):
        series = summed[:, sensor_idx]
        valid = np.isfinite(series) & np.isfinite(times)
        if not np.any(valid):
            continue
        axes.plot(
            times[valid],
            series[valid],
            color=PLOT_SENSOR_COLORS[sensor_idx % len(PLOT_SENSOR_COLORS)],
            linewidth=2.6,
            alpha=0.92,
            solid_capstyle="round",
            label=f"S{sensor_idx + 1}",
        )

    axes.axhline(0.0, color="#9a9a9a", linewidth=0.8, zorder=0)
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.yaxis.set_major_locator(MultipleLocator(magnet_y_tick))
    axes.set_xlabel("Time [s]", fontsize=18, labelpad=8)
    axes.set_ylabel("Magnetic change (X + Y + Z)", fontsize=18, labelpad=8)
    axes.tick_params(axis="both", labelsize=13, colors="#222222")
    axes.grid(False)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color("#555555")
    axes.spines["bottom"].set_color("#555555")
    axes.legend(
        loc="upper right",
        frameon=False,
        ncol=min(sensor_count, 4),
        fontsize=14,
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.25,
    )

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


def save_full_trace_plot(
    path,
    elapsed,
    magnet,
    raw_limit,
    magnet_offset=None,
    magnet_y_range=None,
    magnet_x_range=None,
    magnet_y_tick=1000.0,
):
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_full_trace_plot(
        elapsed,
        magnet,
        raw_limit,
        magnet_offset=magnet_offset,
        magnet_y_range=magnet_y_range,
        magnet_x_range=magnet_x_range,
        magnet_y_tick=magnet_y_tick,
    )
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write magnetic trace image: {output_path}")
    return output_path


def main():
    args = parse_args()
    if args.sensor_count <= 0:
        raise ValueError(f"--sensor-count must be positive, got {args.sensor_count}")
    if args.frame_save_interval <= 0:
        raise ValueError(
            f"--frame-save-interval must be positive, got {args.frame_save_interval}"
        )
    elapsed, magnet = load_trace_csv(args.trace_csv, args.sensor_count)
    elapsed, magnet = trim_trace_start(
        elapsed,
        magnet,
        args.trim_trace_start_sec,
    )
    baseline = first_finite_baseline(magnet)
    global_raw_delta = magnet - baseline[None, :, :]
    magnet_offset = None
    if args.magnet_offset_file is not None:
        magnet_offset = load_magnet_offset_file(
            args.magnet_offset_file,
            args.sensor_count,
        )
        print(
            f"Loaded normalized magnet offset from {args.magnet_offset_file}: "
            f"shape={magnet_offset.shape}"
        )
        # Keep automatic y-limits large enough to show the shifted overview.
        global_raw_delta = global_raw_delta + magnet_offset[None, :, :]
    # The static overview plots the same per-sensor X+Y+Z sums as the renderer.
    global_raw_delta = np.sum(global_raw_delta, axis=2)
    raw_limit = (
        robust_limit(global_raw_delta, default=1.0)
        if args.raw_limit is None
        else max(1e-6, float(args.raw_limit))
    )
    if args.magnet_plot_output is not None:
        magnet_plot_path = save_full_trace_plot(
            args.magnet_plot_output,
            elapsed,
            magnet,
            raw_limit,
            magnet_offset=magnet_offset,
            magnet_y_range=args.magnet_y_range,
            magnet_x_range=args.magnet_x_range,
            magnet_y_tick=args.magnet_y_tick,
        )
        print(f"Wrote full magnetic trace plot to {magnet_plot_path}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid input video size: {width}x{height}")
    crop_left, crop_right = video_crop_bounds(
        width,
        args.crop_video_left_ratio,
        args.crop_video_right_ratio,
    )
    cropped_width = crop_right - crop_left
    trim_video_start_sec = max(0.0, float(args.trim_video_start_sec))
    if trim_video_start_sec > 0.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, trim_video_start_sec * 1000.0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_output_dir = None
    if args.frame_output_dir is not None:
        frame_output_dir = Path(args.frame_output_dir).expanduser()
        frame_output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*args.codec[:4]),
        fps,
        (cropped_width + int(args.panel_width), height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output}")

    frame_idx = 0
    exported_frame_count = 0
    trace_origin = float(elapsed[0]) if args.start_at_trace_zero else 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = frame[:, crop_left:crop_right]
            if frame_output_dir is not None and frame_idx % args.frame_save_interval == 0:
                export_video_frame(frame, frame_idx, frame_output_dir)
                exported_frame_count += 1
            video_time = frame_idx / fps
            trace_time = trace_origin + video_time + float(args.time_offset_sec)
            panel = render_panel(
                elapsed,
                magnet,
                trace_time,
                height,
                int(args.panel_width),
                float(args.window_sec),
                raw_limit,
            )
            writer.write(np.concatenate([frame, panel], axis=1))
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(
        f"Wrote {frame_idx} frames to {output} "
        f"(video crop: x=[{crop_left}, {crop_right}), "
        f"{width}px -> {cropped_width}px)"
    )
    if frame_output_dir is not None:
        print(
            f"Exported {exported_frame_count} source-video frames to "
            f"{frame_output_dir} (every {args.frame_save_interval} frames)"
        )


if __name__ == "__main__":
    main()
