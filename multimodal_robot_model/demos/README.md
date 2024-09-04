# Demonstration data collection

Sample data can be downloaded [here](https://www.dropbox.com/scl/fi/15r33msj4vd1potaosirh/teleop_data_20240414.zip?rlkey=2vt1h5gde7l42vrwz4axj10da&dl=0).
**This data is in an old format and will be replaced with a new format soon.**

## Record teleoperation data
Connect SpaceMouse to your PC before launching the script.

Task to pass the cable between two poles:
```console
$ python DemoTeleopUR5eCable.py
```
https://github.com/user-attachments/assets/59736023-a7f7-4aca-a860-176db84579f7

Task to pick a ring and put it around the pole:
```console
$ python DemoTeleopUR5eRing.py
```
https://github.com/user-attachments/assets/0eb76bbc-6b9d-43f6-95b1-8600e12a47cf

To add a 3D plot of the point cloud, add the following option:
```console
$ python DemoTeleopUR5eCable.py --enable-3d-plot
```
If you cannot zoom the point cloud view by right-clicking, try changing the matplotlib version: `pip install matplotlib=="3.6.1"`.

To replay the teleoperation motion of the log, add the following option:
```console
$ python DemoTeleopUR5eCable.py --replay-log ./teleop_data/UR5eCable/env0/UR5eCable_env0_000.npz
```
