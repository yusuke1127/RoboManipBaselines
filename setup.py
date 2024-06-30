from setuptools import setup

setup(
    name="multimodal_robot_model",
    version="0.0.1",
    install_requires=[
        "matplotlib>=3.3.4",
        "gymnasium==0.29.1",
        "mujoco==2.3.7",
        "imageio >=2.14.1",
        "pyspacemouse",
        "opencv-python",
        "rosbag",
        "pycryptodomex",
        "gnupg",
    ],
)
