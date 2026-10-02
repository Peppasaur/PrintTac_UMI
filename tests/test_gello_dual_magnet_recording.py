import unittest

import numpy as np

from scripts.gello_polymetis_tcp_teleop import (
    collect_magnet_frame_fields,
    normalize_record_magnet_ports,
)


class _Reader:
    def __init__(self, value):
        self.value = float(value)

    def get_recent_samples(self):
        return {
            "magnet_xyz": np.full((2, 4, 3), self.value, dtype=np.float32),
            "magnet_timestamp_ns": np.full(2, int(self.value), dtype=np.int64),
            "magnet_sample_count": np.asarray([2], dtype=np.int32),
        }


class DualMagnetRecordingTest(unittest.TestCase):
    def test_collects_two_readers_without_overwriting_primary_keys(self):
        fields = collect_magnet_frame_fields([_Reader(1), _Reader(2)])

        self.assertEqual(
            set(fields),
            {
                "magnet_xyz",
                "magnet_timestamp_ns",
                "magnet_sample_count",
                "magnet2_xyz",
                "magnet2_timestamp_ns",
                "magnet2_sample_count",
            },
        )
        np.testing.assert_array_equal(fields["magnet_xyz"], 1.0)
        np.testing.assert_array_equal(fields["magnet2_xyz"], 2.0)

    def test_normalizes_one_or_two_unique_ports(self):
        self.assertEqual(normalize_record_magnet_ports("/dev/ttyACM0"), ("/dev/ttyACM0",))
        self.assertEqual(
            normalize_record_magnet_ports(["/dev/ttyACM0", "/dev/ttyACM1"]),
            ("/dev/ttyACM0", "/dev/ttyACM1"),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_record_magnet_ports(["/dev/ttyACM0", "/dev/ttyACM0"])
        with self.assertRaisesRegex(ValueError, "At most 2"):
            normalize_record_magnet_ports(["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
