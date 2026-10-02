import cv2
import numpy as np

from scripts.extract_video_frames_without_magnet import extract_camera_frames


def test_extracts_every_interval_from_left_camera_region(tmp_path):
    video_path = tmp_path / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (10, 4),
    )
    assert writer.isOpened()
    for frame_idx in range(5):
        frame = np.zeros((4, 10, 3), dtype=np.uint8)
        frame[:, :6] = frame_idx * 20
        frame[:, 6:] = 255
        writer.write(frame)
    writer.release()

    output_dir = tmp_path / "frames"
    written, frame_count = extract_camera_frames(
        video_path,
        output_dir=output_dir,
        frame_interval=2,
        magnet_panel_width=4,
    )

    assert frame_count == 5
    assert [path.name for path in written] == [
        "frame_000000.png",
        "frame_000002.png",
        "frame_000004.png",
    ]
    image = cv2.imread(str(output_dir / "frame_000002.png"))
    assert image.shape == (4, 6, 3)
    assert int(image[:, -1].mean()) < 100


def test_rejects_panel_that_removes_full_frame(tmp_path):
    video_path = tmp_path / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (4, 2),
    )
    assert writer.isOpened()
    writer.write(np.zeros((2, 4, 3), dtype=np.uint8))
    writer.release()

    try:
        extract_camera_frames(video_path, tmp_path / "frames", magnet_panel_width=4)
    except ValueError as exc:
        assert "complete video frame" in str(exc)
    else:
        raise AssertionError("Expected invalid panel width to fail")
