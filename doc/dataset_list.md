# Dataset list

## Demonstrations in MuJoCo environments
| Dataset | Environment | Operation method | # Demo | Date | Link |
| --- | --- | --- | --- | --- | --- |
| **TeleopMujocoUR5eCable_Dataset30** | [MujocoUR5eCable](./environment_catalog.md#MujocoUR5eCableEnv) | 3D mouse | 30 | 10/31/2024 | [Download (9GB)](https://www.dropbox.com/scl/fi/2e2on6pl26x0m2l4c0qr6/TeleopMujocoUR5eCable_Dataset30_20241028.zip?rlkey=ua659cleqn2ncqd5ik9zri4h7&st=bbnduu2w&dl=1) |
| **TeleopMujocoUR5eRing_Dataset30** | [MujocoUR5eRing](./environment_catalog.md#MujocoUR5eRingEnv) | 3D mouse | 30 | 10/31/2024 | [Download (8GB)](https://www.dropbox.com/scl/fi/cg3qd7k5scmpxnj0t4qa5/TeleopMujocoUR5eRing_Dataset30_20241031.zip?rlkey=jgbwglrqi7svvrggpawrazg5r&dl=1) |
| **TeleopMujocoUR5eParticle_Dataset30** | [MujocoUR5eParticle](./environment_catalog.md#MujocoUR5eParticleEnv) | 3D mouse | 30 | 10/31/2024 | [Download (13GB)](https://www.dropbox.com/scl/fi/y5aocgzpc85fx2tjtdcqs/TeleopMujocoUR5eParticle_Dataset30_20241031.zip?rlkey=3lwul2am7tlxjoayluf9j3yjy&dl=1) |
| **TeleopMujocoUR5eCloth_Dataset30** | [MujocoUR5eCloth](./environment_catalog.md#MujocoUR5eClothEnv) | 3D mouse | 30 | 10/31/2024 | [Download (8GB)](https://www.dropbox.com/scl/fi/ums7qz2rom9focuf91j87/TeleopMujocoUR5eCloth_Dataset30_20241031.zip?rlkey=qq10s4y5gi8stbondnsoso31l&dl=1) |

## Demonstrations in real-world environments
| Dataset | Environment | Operation method | # Demo | Date | Link |
| --- | --- | --- | --- | --- | --- |
| **TeleopRealUR5eFoldBlueHandkerchief_Dataset30** | [RealUR5eDemo](./environment_catalog.md#RealUR5eDemoEnv) | 3D mouse | 30 | 11/15/2024 | [Download (3.1GB)](https://www.dropbox.com/scl/fi/878r1olspwtgclfyi3hwy/RealUR5eFoldBlueHandkerchief_20241115.zip?rlkey=rkv6iwv0t3xzqtn8ufkc0g4gk&dl=1) |

## Dataset format
Each demonstration data is saved as an npz file. The npz file contains the following entries.

| Entry  | Description | Shape | Dtype |
| --- | --- | --- | --- |
| `time` | Time | `(T,)` | `float64` |
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
| `<camera_name>_rgb_image` | RGB image (*1) | `(T,)` | `object` |
| `<camera_name>_depth_image` | Depth image | `(T, Height, Width)` | `float32` |
| `<camera_name>_depth_image_fovy` | Field of view of depth image height | `()` | `float64` |
| `format` | Data format (fixed with `"RoboManipBaselines-TeleopData"`) | `()` | `str` |
| `version` | Version of RoboManipBaselines | `()` | `str` |
| `demo` | Demonstration name | `()` | `str` |
| `world_idx` | World index | `()` | `int64` |

- `T` is the length of the time sequence, `JointDim` is the number of joints, `PoseDim = 7` is pose (tx, ty, tz, qw, qx, qy, qz), `WrenchDim = 6` is wrench (fx, fy, fz, nx, ny, nz), respectively.
- *1: RGB images are compressed in JPEG using `cv2.imencode`.
