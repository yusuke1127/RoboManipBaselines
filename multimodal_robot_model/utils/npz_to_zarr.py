import numpy as np
import glob
import zarr
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, default="./data/")
args = parser.parse_args()

file_names = glob.glob(os.path.join(args.in_dir, "**/*.npz"), recursive=True)
file_names.sort()

npz_files = []
for filename in file_names:
    npz_file = dict(np.load(filename))
    npz_file["joint"] = npz_file["joint"][::10]
    npz_file["joint"] = np.delete(npz_file["joint"], -1, 1)
    npz_file["wrench"] = npz_file["wrench"][::10]
    npz_file["front_image"] = npz_file["front_image"][::10]
    npz_file["side_image"] = npz_file["side_image"][::10]
    npz_file["time"] = npz_file["time"][::10]
    npz_files.append(npz_file)

ep_ends = np.zeros(len(npz_files), dtype=np.int64)
joints = npz_files[0]["joint"]
images = npz_files[0]["front_image"]
ep_end = 0
for i in range(len(npz_files)):
    ep_end += npz_files[i]["front_image"].shape[0]
    ep_ends[i] = ep_end
    if i != 0:
        joints = np.concatenate([joints, npz_files[i]["joint"]])
        images = np.concatenate([images, npz_files[i]["front_image"]])

zarr_root = zarr.open(os.path.join(args.in_dir, "learning_data.zarr"), mode="w")
zarr_root.create_group("meta")
zarr_root["meta"].create_dataset("episode_ends", data=ep_ends)
zarr_root.create_group("data")
zarr_root["data"].create_dataset("action", data=joints)
zarr_root["data"].create_dataset("img", data=images)
