# Dataset list

## Demonstrations in MuJoCo environments
| Task  | Link |
| --- | --- |
| [TeleopMujocoUR5eCable](./environment_catalog.md#MujocoUR5eCableEnv) | [Download (9GB)]() |
| [TeleopMujocoUR5eRing](./environment_catalog.md#MujocoUR5eRingEnv) | [Download (8GB)](https://www.dropbox.com/scl/fi/cg3qd7k5scmpxnj0t4qa5/TeleopMujocoUR5eRing_Dataset30_20241031.zip?rlkey=jgbwglrqi7svvrggpawrazg5r&dl=1) |
| [TeleopMujocoUR5eParticle](./environment_catalog.md#MujocoUR5eParticleEnv) | [Download (13GB)]() |
| [TeleopMujocoUR5eCloth](./environment_catalog.md#MujocoUR5eClothEnv) | [Download (8GB)](https://www.dropbox.com/scl/fi/ums7qz2rom9focuf91j87/TeleopMujocoUR5eCloth_Dataset30_20241031.zip?rlkey=qq10s4y5gi8stbondnsoso31l&dl=1) |

## Demonstrations in real-world environments
Coming soon.

## Dataset format
| Entry  | Description | Shape | Dtype |
| --- | --- | --- | --- |
| `time` | time | `(T,)` | `float64` |
| `measured_joint_pos` | Measured joint position [rad] | `(T, JointDim)` | `float64` |
| `command_joint_pos` | Command joint position [rad] | `(T, JointDim)` | `float64` |
| `measured_joint_vel` | Measured joint velocity [rad/s] | `(T, JointDim)` | `float64` |
| `command_joint_vel` | Not set |  |  |
| `measured_joint_torque` | Not set |  |  |
| `command_joint_torque` | Not set |  |  |
| `measured_eef_pose` | Measured pose of end-effector [m], [rad] | `(T, PoseDim)` | `float64` |
| `command_eef_pose` | Command pose of end-effector [m], [rad] | `(T, PoseDim)` | `float64` |
| `measured_eef_vel` | Not set |  |  |
| `command_eef_vel` | Not set |  |  |
| `measured_eef_wrench` | Measured force and torque of end-effector [N], [Nm] | `(T, WrenchDim)` | `float64` |
| `command_eef_wrench` | Not set |  |  |
| `<camera_name>_rgb_image` | RGB image | `(T,)` | `object` |
| `<camera_name>_depth_image` | Depth image | `(T, Height, Width)` | `float32` |
| `<camera_name>_depth_image_fovy` | Field of view of depth image height | `()` | `float64` |
| `format` | Data format (fixed with `"RoboManipBaselines-TeleopData"`) | `()` | `str` |
| `version` | Version of RoboManipBaselines | `()` | `str` |
| `demo` | Demonstration name | `()` | `str` |
| `world_idx` | World index | `()` | `int64` |
`T` is the length of the time sequence, `JointDim` is the number of joints, `PoseDim = 7` is pose (tx, ty, tz, qw, qx, qy, qz), `WrenchDim = 6` is wrench (fx, fy, fz, nx, ny, nz), respectively.
