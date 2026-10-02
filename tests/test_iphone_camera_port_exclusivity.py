from pathlib import Path


def test_iphone_video_port_is_not_shared_between_eval_receivers():
    source = (
        Path(__file__).parents[1]
        / "reactive_diffusion_policy/env/franka_polymetis/franka_polymetis_env.py"
    ).read_text()

    start = source.index("self.video_socket = socket.socket", source.index("class _IPhoneUDPCamera"))
    end = source.index("self.video_socket.setblocking(False)", start)
    setup = source[start:end]
    assert "SO_REUSEADDR" not in setup
    assert "Another iPhone receiver" in setup
