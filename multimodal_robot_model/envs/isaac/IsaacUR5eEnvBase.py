from os import path
import numpy as np

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import gymnasium as gym
from gymnasium import utils
from gymnasium.spaces import Box

class IsaacUR5eEnvBase(gym.Env, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
        ],
    }

    def __init__(
        self,
        init_qpos,
        camera_configs=None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            init_qpos,
            camera_configs,
            **kwargs,
        )

        self.init_qpos = init_qpos
        self.render_mode = kwargs.get("render_mode")
        self.setupSim()

        # Setup robot
        self.arm_urdf_path = path.join(path.dirname(__file__), "../assets/common/robots/ur5e/ur5e.urdf")
        self.arm_root_pose = self.get_link_pose("ur5e", "base_link")

        # Setup environment parameters
        self.skip_sim = 2
        self.dt = self.skip_sim * self.gym.get_sim_params(self.sim).dt
        robot_dof_props = self.gym.get_actor_dof_properties(self.env, self.robot_handle)
        self.action_space = Box(
            low=np.concatenate((robot_dof_props["lower"][0:6], np.array([0.0], dtype=np.float32))),
            high=np.concatenate((robot_dof_props["upper"][0:6], np.array([255.0], dtype=np.float32))),
            dtype=np.float32
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64
        )

        # Setup camera
        self.camera_configs = camera_configs

        # Setup internal variables
        self._quit_flag = False
        self._pause_flag = False

    def setupSim(self):
        # Setup sim
        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.80665) # [m/s^2]
        sim_params.dt = 1.0 / 60.0 # [s]
        sim_params.substeps = 4
        sim_params.use_gpu_pipeline = False
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.0005
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True # False
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        # Setup robot asset
        robot_asset_root = path.join(path.dirname(__file__), "../assets/isaac/robots/ur5e")
        robot_asset_file = "ur5e_integrated.urdf"
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.armature = 0.01
        robot_asset_options.fix_base_link = True
        robot_asset_options.flip_visual_attachments = True
        self.robot_asset = self.gym.load_asset(self.sim, robot_asset_root, robot_asset_file, robot_asset_options)

        # Setup force sensor
        force_sensor_body_name = "wrist_3_link"
        force_sensor_body_idx = self.gym.find_asset_rigid_body_index(self.robot_asset, force_sensor_body_name)
        force_sensor_pose_local = gymapi.Transform(r=gymapi.Quat.from_euler_zyx(0, np.pi, 0))
        force_sensor_idx = self.gym.create_asset_force_sensor(self.robot_asset, force_sensor_body_idx, force_sensor_pose_local)

        # Setup ground
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Setup task-specific assets
        self.setup_task_specific_assets()

        # Setup env
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        # Setup viewer
        if self.render_mode == "human":
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            camera_origin_pos = gymapi.Vec3(1.0, 0.5, 1.0)
            camera_lookat_pos = gymapi.Vec3(0.3, 0.0, 0.3)
            self.gym.viewer_camera_look_at(self.viewer, self.env, camera_origin_pos, camera_lookat_pos)
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "quit")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "quit")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause_resume")

        # Setup robot actor
        robot_pose = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
        self.robot_handle = self.gym.create_actor(self.env, self.robot_asset, robot_pose, "ur5e", 1, 0)
        self.force_sensor = self.gym.get_actor_force_sensor(self.env, self.robot_handle, force_sensor_idx)

        # Setup gripper mimic joints
        gripper_mimic_multiplier_map = {
            "robotiq_85_left_knuckle_joint": 1.0,
            "robotiq_85_right_knuckle_joint": -1.0,
            "robotiq_85_left_inner_knuckle_joint": 1.0,
            "robotiq_85_right_inner_knuckle_joint": -1.0,
            "robotiq_85_left_finger_tip_joint": -1.0,
            "robotiq_85_right_finger_tip_joint": 1.0,
        }
        self.gripper_mimic_multiplier_list = np.zeros(len(gripper_mimic_multiplier_map), dtype=np.float32)
        for joint_name, mimic_multiplier in gripper_mimic_multiplier_map.items():
            dof_idx = self.gym.find_actor_dof_index(self.env, self.robot_handle, joint_name, gymapi.DOMAIN_ACTOR)
            self.gripper_mimic_multiplier_list[dof_idx - 6] = mimic_multiplier

        # Setup joint control mode
        robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        robot_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
        robot_dof_props["armature"] = np.array([0.1] * 6 + [0.001] * 6, dtype=np.float32)
        robot_dof_props["stiffness"] = np.array([2000, 2000, 2000, 500, 500, 500] + [400] * 6, dtype=np.int32)
        robot_dof_props["damping"] = np.array([400, 400, 400, 100, 100, 100] + [80] * 6, dtype=np.int32)
        self.gym.set_actor_dof_properties(self.env, self.robot_handle, robot_dof_props)

        # Setup gripper command scale
        gripper_dof_idx = self.gym.find_actor_dof_index(
            self.env, self.robot_handle, "robotiq_85_left_knuckle_joint", gymapi.DOMAIN_ACTOR)
        original_gripper_range = robot_dof_props["upper"][gripper_dof_idx] - robot_dof_props["lower"][gripper_dof_idx]
        new_gripper_range = 255.0
        self.gripper_command_scale = original_gripper_range / new_gripper_range

        # Setup joint control command
        robot_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        self.init_robot_dof_state = np.zeros(robot_num_dofs, gymapi.DofState.dtype)
        self.init_robot_dof_state["pos"] = self.get_robot_dof_pos_from_qpos(self.init_qpos)
        self.gym.set_actor_dof_states(self.env, self.robot_handle, self.init_robot_dof_state, gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, self.init_robot_dof_state["pos"])

        # Setup task-specific actors
        self.setup_task_specific_actors()

        # Setup fixed cameras
        self.camera_handles = []
        self.setup_task_specific_cameras()

        # Setup robot cameras
        self.camera_properties = gymapi.CameraProperties()
        # TODO: Set from camera_config
        self.camera_properties.width = 640
        self.camera_properties.height = 480
        camera_handle = self.gym.create_camera_sensor(self.env, self.camera_properties)
        body_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_handle, "camera_link")
        camera_pose_local = gymapi.Transform(p=gymapi.Vec3(0, -0.02, 0.005))
        self.gym.attach_camera_to_body(
            camera_handle, self.env, body_handle, camera_pose_local, gymapi.FOLLOW_TRANSFORM)
        self.camera_handles.append(camera_handle)

        # Setup marker
        self.box_geom_target = gymutil.WireframeBoxGeometry(0.08, 0.08, 0.12, color=(0, 1, 0))
        self.box_geom_current = gymutil.WireframeBoxGeometry(0.08, 0.08, 0.12, color=(1, 0, 0))

        # Store state
        self.init_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

    def setup_task_specific_assets(self):
        raise NotImplementedError("[IsaacUR5eEnvBase] setup_task_specific_assets is not implemented.")

    def setup_task_specific_actors(self):
        raise NotImplementedError("[IsaacUR5eEnvBase] setup_task_specific_actors is not implemented.")

    def setup_task_specific_cameras(self):
        raise NotImplementedError("[IsaacUR5eEnvBase] setup_task_specific_cameras is not implemented.")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.gym.set_sim_rigid_body_states(self.sim, self.init_state, gymapi.STATE_ALL)
        self.gym.set_actor_dof_states(self.env, self.robot_handle, self.init_robot_dof_state, gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, self.init_robot_dof_state["pos"])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Check key input
        if self.render_mode == "human":
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "quit" and evt.value > 0:
                    self._quit_flag = True
                elif evt.action == "pause_resume" and evt.value > 0:
                    self._pause_flag = not self._pause_flag
        if self._quit_flag:
            self.close()
            return

        # Set joint command
        robot_dof_pos = self.get_robot_dof_pos_from_qpos(action)
        self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, robot_dof_pos)

        # Update simulation
        if not self._pause_flag:
            for _ in range(self.skip_sim):
                self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Update viewer
        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        reward = 0.0
        terminated = False
        info = self._get_info()

        # self.gym.sync_frame_time(self.sim)

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_obs(self):
        robot_dof_state = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)

        arm_qpos = robot_dof_state["pos"][0:6]
        arm_qvel = robot_dof_state["vel"][0:6]
        gripper_pos = self.get_gripper_pos_from_gripper_dof_pos(robot_dof_state["pos"][6:12])
        wrench = self.force_sensor.get_forces()
        force = np.array([wrench.force.x, wrench.force.y, wrench.force.z])
        torque = np.array([wrench.torque.x, wrench.torque.y, wrench.torque.z])

        return np.concatenate((arm_qpos, arm_qvel, gripper_pos, force, torque), dtype=np.float64)

    def _get_info(self):
        info = {}

        if self.num_cameras == 0:
            return info

        # Update camera image
        self.gym.clear_lines(self.viewer)
        self.gym.render_all_camera_sensors(self.sim)

        # Get camera images
        info["rgb_images"] = {}
        info["depth_images"] = {}
        for camera_config in self.camera_configs:
            # TODO: Make it a generic implementation, not a special treatment of the camera name
            if camera_config["name"] == "front":
                camera_handle = self.camera_handles[0]
            elif camera_config["name"] == "side":
                camera_handle = self.camera_handles[1]
            elif camera_config["name"] == "hand":
                camera_handle = self.camera_handles[2]
            rgb_image = self.gym.get_camera_image(self.sim, self.env, camera_handle, gymapi.IMAGE_COLOR)
            rgb_image = rgb_image.reshape(rgb_image.shape[0], -1, 4)[:, :, :3]
            depth_image = -1 * self.gym.get_camera_image(self.sim, self.env, camera_handle, gymapi.IMAGE_DEPTH)
            info["rgb_images"][camera_config["name"]] = rgb_image
            info["depth_images"][camera_config["name"]] = depth_image

        return info

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def render(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)

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
        return self.gym.get_sim_time(self.sim)

    def get_link_pose(self, actor_name, link_name=None, link_idx=0):
        """Get link pose in the format [tx, ty, tz, qw, qx, qy, qz]."""
        actor_idx = self.gym.find_actor_index(self.env, actor_name, gymapi.DOMAIN_ENV)
        if link_name is not None:
            link_idx = self.gym.find_actor_rigid_body_index(self.env, actor_idx, link_name, gymapi.DOMAIN_ACTOR)
        link_pose = gymapi.Transform.from_buffer(
            self.gym.get_actor_rigid_body_states(self.env, actor_idx, gymapi.STATE_POS)["pose"][link_idx])
        return np.array([link_pose.p.x, link_pose.p.y, link_pose.p.z,
                         link_pose.r.w, link_pose.r.x, link_pose.r.y, link_pose.r.z])

    @property
    def num_cameras(self):
        """Number of cameras."""
        return len(self.camera_configs)

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        # TODO: Get the fovy of the corresponding camera
        camera_fovy = self.camera_properties.height / self.camera_properties.width * self.camera_properties.horizontal_fov
        return camera_fovy

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        raise NotImplementedError("[IsaacUR5eEnvBase] modify_world is not implemented.")

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        # TODO: Implement a method
        pass

    def get_robot_dof_pos_from_qpos(self, qpos):
        robot_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        robot_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        robot_dof_pos[0:6] = qpos[:6]
        robot_dof_pos[6:12] = self.get_gripper_dof_pos_from_gripper_pos(qpos[6])
        return robot_dof_pos

    def get_gripper_dof_pos_from_gripper_pos(self, gripper_pos):
        return gripper_pos * self.gripper_command_scale * self.gripper_mimic_multiplier_list

    def get_gripper_pos_from_gripper_dof_pos(self, gripper_dof_pos):
        return (gripper_dof_pos / (self.gripper_command_scale * self.gripper_mimic_multiplier_list)).mean(keepdims=True)
