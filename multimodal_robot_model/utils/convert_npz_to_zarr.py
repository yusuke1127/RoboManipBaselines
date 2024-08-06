import numpy as np
import glob
import zarr
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", type=str)
parser.add_argument("--out_path", type=str)
parser.add_argument("--skip", type=int, default=4)
args = parser.parse_args()

in_file_names = glob.glob(os.path.join(args.in_dir, "**/*.npz"), recursive=True)
in_file_names.sort()

joints = None
images = None
ep_ends = np.zeros(len(in_file_names), dtype=np.int64)
print("[convert_npz_to_zarr] Load npz files:")
for idx, in_file_name in enumerate(in_file_names):
    print(" " * 4 + f"{in_file_name}")
    npz_data = dict(np.load(in_file_name))
    if idx == 0:
        joints = npz_data["joint"][::args.skip]
        images = npz_data["front_image"][::args.skip]
        ep_ends[idx] = len(npz_data["joint"][::args.skip])
    else:
        joints = np.concatenate((joints, npz_data["joint"][::args.skip]))
        images = np.concatenate((images, npz_data["front_image"][::args.skip]))
        ep_ends[idx] = ep_ends[idx - 1] + len(npz_data["joint"][::args.skip])

if args.out_path is None:
    out_path = os.path.join(args.in_dir, "learning_data.zarr")
else:
    out_path = args.out_path
print(f"[convert_npz_to_zarr] Save a zarr file: {out_path}")
zarr_root = zarr.open(out_path, mode="w")
zarr_root.create_group("meta")
zarr_root["meta"].create_dataset("episode_ends", data=ep_ends)
zarr_root.create_group("data")
zarr_root["data"].create_dataset("action", data=joints)
zarr_root["data"].create_dataset("joint", data=joints)
zarr_root["data"].create_dataset("img", data=images)

for key in ("meta/episode_ends", "data/action", "data/joint", "data/img"):
    shape = zarr_root[f"{key}"].shape
    print(" " * 4 + f"{key}:\t{shape}")
print(" " * 4 + f"episode_ends: {list(zarr_root.meta.episode_ends)}")
