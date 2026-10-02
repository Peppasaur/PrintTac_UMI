import unittest
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf


class DualMagnetConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OmegaConf.register_new_resolver("eval", eval, replace=True)

    def _compose(self, config_name, task):
        with initialize(
            version_base=None,
            config_path="../reactive_diffusion_policy/config",
        ):
            cfg = compose(config_name=config_name, overrides=[f"task={task}"])
        OmegaConf.resolve(cfg)
        return cfg

    def _assert_dual_keys(self, cfg):
        expected = {
            "left_gripper1_marker_offset_emb",
            "left_gripper2_marker_offset_emb",
        }
        self.assertTrue(expected.issubset(cfg.task.shape_meta.obs.keys()))
        self.assertTrue(expected.issubset(cfg.task.dataset.shape_meta.obs.keys()))
        self.assertEqual(set(cfg.task.dataset.gaussian_normalizer_keys), expected)

    def test_dp_training_and_eval_configs_include_both_inputs(self):
        train_cfg = self._compose(
            "train_diffusion_unet_real_image_workspace",
            "real_wipe_image_dual_magnet_emb_dp_absolute_12fps",
        )
        eval_cfg = self._compose(
            "train_diffusion_unet_real_image_workspace",
            "franka_polymetis_image_dual_magnet_emb_dp_absolute_12fps",
        )

        self._assert_dual_keys(train_cfg)
        self._assert_dual_keys(eval_cfg)
        self.assertEqual(
            eval_cfg.task.env_runner.env_params.magnet2_port,
            "/dev/ttyACM1",
        )
        self.assertTrue(eval_cfg.task.env_runner.env_params.magnet2_required)

    def test_at_and_ldp_configs_include_both_regular_and_extended_inputs(self):
        cases = (
            (
                "train_at_workspace",
                "real_wipe_image_dual_magnet_emb_at_absolute_12fps",
            ),
            (
                "train_latent_diffusion_unet_real_image_workspace",
                "real_wipe_image_dual_magnet_emb_ldp_absolute_12fps",
            ),
            (
                "train_latent_diffusion_unet_real_image_workspace",
                "franka_polymetis_image_dual_magnet_emb_ldp_absolute_12fps",
            ),
        )
        for config_name, task in cases:
            with self.subTest(task=task):
                cfg = self._compose(config_name, task)
                self._assert_dual_keys(cfg)
                self.assertIn(
                    "left_gripper2_marker_offset_emb",
                    cfg.task.shape_meta.extended_obs,
                )

    def test_eval_script_exposes_second_magnetometer_overrides(self):
        script = (Path(__file__).parents[1] / "eval.sh").read_text()

        self.assertIn('MAGNET2_PORT="${MAGNET2_PORT:-}"', script)
        self.assertIn('MAGNET2_REQUIRED="${MAGNET2_REQUIRED:-}"', script)
        self.assertIn("env_params.magnet2_port=${MAGNET2_PORT}", script)
        self.assertIn("env_params.magnet2_sensor_order", script)
        self.assertIn("env_params.magnet2_zero_channels", script)


if __name__ == "__main__":
    unittest.main()
