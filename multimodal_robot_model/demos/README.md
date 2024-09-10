# Demonstration data collection

Sample data can be downloaded [here](https://www.dropbox.com/scl/fi/15r33msj4vd1potaosirh/teleop_data_20240414.zip?rlkey=2vt1h5gde7l42vrwz4axj10da&dl=0).
**This data is in an old format and will be replaced with a new format soon.**

## Record teleoperation data
Connect SpaceMouse to your PC before launching the script.

### Task to pass the cable between two poles
```console
$ python DemoTeleopUR5eCable.py
```
https://github.com/user-attachments/assets/59736023-a7f7-4aca-a860-176db84579f7

### Task to pick a ring and put it around the pole
```console
$ python DemoTeleopUR5eRing.py
```
https://github.com/user-attachments/assets/0eb76bbc-6b9d-43f6-95b1-8600e12a47cf

### Task to scoop up particles
```console
$ python DemoTeleopUR5eParticle.py
```
https://github.com/user-attachments/assets/305300bd-6685-46ab-9704-5a15d901ed7a

### Task to roll up the cloth
```console
$ python DemoTeleopUR5eCloth.py
```
https://github.com/user-attachments/assets/88bb9d84-7ca9-4d45-b457-cb9931cfb9a1

### Other options
If you want to collect data only in a simulation environment with limited world indices (for example, only 0 and 5), add the following option:
```console
$ python DemoTeleopUR5eCable.py --world_idx_list 0 5
```

To add a 3D plot of the point cloud, add the following option:
```console
$ python DemoTeleopUR5eCable.py --enable_3d_plot
```
If you cannot zoom the point cloud view by right-clicking, try changing the matplotlib version: `pip install matplotlib=="3.6.1"`.

To replay the teleoperation motion of the log, add the following option:
```console
$ python DemoTeleopUR5eCable.py --replay_log ./teleop_data/UR5eCable/env0/UR5eCable_env0_000.npz
```
