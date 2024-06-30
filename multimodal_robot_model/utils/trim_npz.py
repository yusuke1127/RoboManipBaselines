import numpy as np
import glob
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", type=str)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--play_freq", type=int, default=100)
args = parser.parse_args()

in_npz_path_list = glob.glob(os.path.join(args.in_dir, "**/*.npz"), recursive=True)
in_npz_path_list.sort()
for in_npz_path in in_npz_path_list:
    print(f"[trim_npz] Load a npz file: {in_npz_path}")
    npz_data = np.load(in_npz_path)
    images = npz_data["front_image"]

    print("[trim_npz] Press 's' to decide the start, press 'e' to decide the end.")
    start, end = None, None
    for i in range(len(images)):
        cv2.imshow("image", cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(args.play_freq)
        if key == 115: # s key
            start = i
            print(f"[trim_npz] Start index: {i}")

        if key == 101: # e key
            end = i
            print(f"[trim_npz] End index: {i}")

    if (start is not None) and (end is not None) and (start >= end):
        print("[trim_npz] Start index is larger than end index")
        start, end = None, None

    if start is None:
        start = 0
    if end is None:
        end = len(images)

    npz_data = dict(npz_data)
    for key in npz_data:
        npz_data[key] = npz_data[key][start:end]

    if args.out_dir is None:
        out_npz_path = in_npz_path
    else:
        out_npz_path = os.path.join(args.out_dir, os.path.basename(in_npz_path))
    print(f"[trim_npz] Save a npz file: {out_npz_path}")
    np.savez(
        out_npz_path,
        time=npz_data["time"],
        joint=npz_data["joint"],
        wrench=npz_data["wrench"],
        front_image=npz_data["front_image"],
        side_image=npz_data["side_image"],
    )

    print("[trim_npz] Press any key to proceed to the next image.")
    cv2.imshow("image", cv2.cvtColor(np.zeros_like(images[0]), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
