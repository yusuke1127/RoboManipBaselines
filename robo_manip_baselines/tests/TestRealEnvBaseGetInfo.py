import unittest

import cv2
import numpy as np

from robo_manip_baselines.envs.real.RealEnvBase import RealEnvBase


class DummyRealEnv(RealEnvBase):
    def __init__(self, gelsight_ids):
        camera_ids = {}
        super().__init__(
            robot_ip=None, camera_ids=camera_ids, gelsight_ids=gelsight_ids
        )
        self.setup_realsense(camera_ids)

        # gsrobotics/examples/show3d.py
        #     the device ID can change after unplugging and changing the usb ports.
        #     on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
        self.setup_gelsight(gelsight_ids)

    def _reset_robot(self, *args, **kwargs):
        pass

    def _set_action(self, *args, **kwargs):
        pass

    def _get_obs(self, *args, **kwargs):
        pass

    @property
    def tactile_names(self):
        return self.tactiles.keys()


class TestRealEnvBaseGetInfo(unittest.TestCase):
    def assert_env_info_valid(self, dummy_real_env):
        info = dummy_real_env._get_info()
        for tactile_name in dummy_real_env.tactile_names:
            rgb_image = info["rgb_images"][tactile_name]
            depth_image = info["depth_images"][tactile_name]

            self.assertIsInstance(rgb_image, np.ndarray)
            self.assertEqual(rgb_image.dtype, np.uint8)
            self.assertEqual(rgb_image.shape[-1], 3)
            self.assertIsNone(depth_image)

    def show_image_loop(self, dummy_real_env):
        try:
            print("press q on image to exit")
            while True:
                # get rgb image
                info = dummy_real_env._get_info()
                for tactile_name in dummy_real_env.tactile_names:
                    rgb_image = info["rgb_images"][tactile_name]
                    cv2.imshow(tactile_name, rgb_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Interrupted!")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @unittest.skip("Skipping.")
    def test_dummy_real_env_get_info_case1(self):
        dummy_real_env = DummyRealEnv(
            gelsight_ids={"tactile": "GelSight Mini R0B 2D16-V7R5: Ge"}
        )
        self.assert_env_info_valid(dummy_real_env)
        self.show_image_loop(dummy_real_env)

    def test_dummy_real_env_get_info_case2(self):
        dummy_real_env = DummyRealEnv(
            gelsight_ids={
                "tactile_left": "GelSight Mini R0B 2BNK-CE0U: Ge",
                "tactile_right": "GelSight Mini R0B 2BG8-0H3X: Ge",
            }
        )
        self.assert_env_info_valid(dummy_real_env)
        self.show_image_loop(dummy_real_env)


if __name__ == "__main__":
    unittest.main()
