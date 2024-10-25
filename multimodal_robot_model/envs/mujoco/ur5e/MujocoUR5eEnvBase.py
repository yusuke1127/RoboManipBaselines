from os import path
import numpy as np
import mujoco

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from gymnasium.spaces import Box

class MujocoUR5eEnvBase(MujocoEnv, utils.EzPickle):
    frame_skip = 8
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(1 / (0.004 * frame_skip)),
    }
    default_camera_config = {
        "azimuth": -135.0,
        "elevation": -45.0,
        "distance": 1.8,
        "lookat": [-0.2, -0.2, 0.8]
    }

    def __init__(
        self,
        xml_file,
        init_qpos,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            init_qpos,
            **kwargs,
        )

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=self.frame_skip,
            observation_space=observation_space,
            width=640,
            height=480,
            default_camera_config=self.default_camera_config,
            **kwargs,
        )
        self.mujoco_renderer.width = None
        self.mujoco_renderer.height = None

        # Setup robot
        mujoco.mj_kinematics(self.model, self.data)
        self.arm_urdf_path = path.join(path.dirname(__file__), "../../assets/common/robots/ur5e/ur5e.urdf")
        self.arm_root_pose = self.get_body_pose("ur5e_root_frame")
        self.init_qpos[:len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        # Setup camera
        self.cameras = {}
        for camera_id in range(self.model.ncam):
            camera = {}
            camera_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
            camera["name"] = camera_name
            camera["id"] = camera_id
            camera["viewer"] = OffScreenViewer(self.model, self.data, width=640, height=480)
            self.cameras[camera_name] = camera

        # This is required to automatically switch context to free camera in render()
        # https://github.com/Farama-Foundation/Gymnasium/blob/81b87efb9f011e975f3b646bab6b7871c522e15e/gymnasium/envs/mujoco/mujoco_rendering.py#L695-L697
        self.mujoco_renderer._viewers["dummy"] = None

        self._first_render = True

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward = 0.0
        terminated = False
        info = self._get_info()

        if self.render_mode == "human":
            if self._first_render:
                self._first_render = False
                self.mujoco_renderer.viewer._hide_menu = True
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_obs(self):
        arm_joint_name_list = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        gripper_joint_name_list = [
            "right_driver_joint",
            "right_spring_link_joint",
            "left_driver_joint",
            "left_spring_link_joint",
        ]

        arm_qpos = np.array([self.data.joint(joint_name).qpos[0] for joint_name in arm_joint_name_list])
        arm_qvel = np.array([self.data.joint(joint_name).qvel[0] for joint_name in arm_joint_name_list])
        gripper_qpos = np.array([self.data.joint(joint_name).qpos[0] for joint_name in gripper_joint_name_list])
        gripper_pos = np.rad2deg(gripper_qpos.mean(keepdims=True)) / 45.0 * 255.0
        force = self.data.sensor("force_sensor").data.flat.copy()
        torque = self.data.sensor("torque_sensor").data.flat.copy()

        return np.concatenate((arm_qpos, arm_qvel, gripper_pos, force, torque))

    def _get_info(self):
        info = {}

        if len(self.camera_names) == 0:
            return info

        info["rgb_images"] = {}
        info["depth_images"] = {}
        for camera in self.cameras.values():
            camera["viewer"].make_context_current()
            rgb_image = camera["viewer"].render(render_mode="rgb_array", camera_id=camera["id"])
            info["rgb_images"][camera["name"]] = rgb_image
            depth_image = camera["viewer"].render(render_mode="depth_array", camera_id=camera["id"])
            # See https://github.com/google-deepmind/mujoco/blob/631b16e7ad192df936195658fe79f2ada85f755c/python/mujoco/renderer.py#L170-L178
            extent = self.model.stat.extent
            near = self.model.vis.map.znear * extent
            far = self.model.vis.map.zfar * extent
            depth_image = near / (1 - depth_image * (1 - near / far))
            info["depth_images"][camera["name"]] = depth_image

        return info

    def _get_reset_info(self):
        return self._get_info()

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def close(self):
        for camera in self.cameras.values():
            camera["viewer"].close()
        MujocoEnv.close(self)

    def get_arm_qpos_from_obs(self, obs):
        """Grm arm joint position (6D array) from observation."""
        return obs[0:6]

    def get_arm_qvel_from_obs(self, obs):
        """Grm arm joint velocity (6D array) from observation."""
        return obs[6:12]

    def get_gripper_pos_from_obs(self, obs):
        """Grm gripper joint position (1D array) from observation."""
        return obs[12:13]

    def get_eef_wrench_from_obs(self, obs):
        """Grm end-effector wrench (6D array) from observation."""
        return obs[13:19]

    def get_sim_time(self):
        """Get simulation time. [s]"""
        return self.data.time

    def get_body_pose(self, body_name):
        """Get body pose in the format [tx, ty, tz, qw, qx, qy, qz]."""
        body = self.data.body(body_name)
        return np.concatenate((body.xpos, body.xquat))

    def get_geom_pose(self, geom_name):
        """Get geom pose in the format [tx, ty, tz, qw, qx, qy, qz]."""
        geom = self.data.geom(geom_name)
        xquat = np.zeros(4)
        mujoco.mju_mat2Quat(xquat, geom.xmat.flatten())
        return np.concatenate((geom.xpos, xquat))

    @property
    def camera_names(self):
        """Camera names being measured."""
        return self.cameras.keys()

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        return self.model.cam(camera_name).fovy[0]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        raise NotImplementedError("[MujocoUR5eEnvBase] modify_world is not implemented.")

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        self.mujoco_renderer.viewer.add_marker(
            pos=pos,
            mat=mat,
            label="",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            rgba=rgba)
