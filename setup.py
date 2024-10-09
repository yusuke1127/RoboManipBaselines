from setuptools import setup
from distutils.util import convert_path

# https://stackoverflow.com/a/24517154
main_ns = {}
version_path = convert_path("multimodal_robot_model/version.py")
with open(version_path) as version_file:
    exec(version_file.read(), main_ns)

setup(
    name="multimodal_robot_model",
    version=main_ns["__version__"],
    install_requires=[
        "matplotlib>=3.3.4",
        "gymnasium==0.29.1",
        "mujoco==2.3.7",
        "imageio >=2.14.1",
        "pyspacemouse",
        "opencv-python",
        "pycryptodomex",
        "gnupg",
    ],
)
