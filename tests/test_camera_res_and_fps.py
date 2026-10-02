import cv2


def test_camera_capabilities():
    # Create a list of common resolutions to test
    test_resolutions = [
        (352, 288),
        (640, 480),  # VGA
        (800, 600),  # SVGA
        (1280, 720),  # HD
        (1920, 1080),  # Full HD
        (2560, 1440),  # 2K
        (3840, 2160)  # 4K
    ]

    # Common frame rates to test
    test_fps = [15, 25, 30, 60]

    # Initialize the camera (0 is usually the default USB camera)
    cap = cv2.VideoCapture(12)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera detected. Testing capabilities...")
    print("\nSupported combinations of resolution and frame rate:")
    print("-" * 50)

    # Test each resolution
    for resolution in test_resolutions:
        width, height = resolution

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get actual resolution (might be different from requested)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Test each frame rate
        for fps in test_fps:
            # Set frame rate
            cap.set(cv2.CAP_PROP_FPS, fps)

            # Get actual frame rate
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            # Capture a few frames to test if this combination works
            success = True
            for _ in range(5):
                ret, frame = cap.read()
                if not ret:
                    success = False
                    break

            if success:
                print(f"Resolution: {actual_width}x{actual_height}")
                print(f"Frame Rate: {actual_fps:.1f} FPS")
                print("-" * 50)

    # Release the camera
    cap.release()

    print("\nTesting completed!")


if __name__ == "__main__":
    test_camera_capabilities()
