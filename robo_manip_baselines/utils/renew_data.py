import argparse
import os

from robo_manip_baselines.common import DataManager

parser = argparse.ArgumentParser()
parser.add_argument("in_filename", type=str)
parser.add_argument("--out_filename", type=str, default=None)
parser.add_argument("--demo_name", type=str, default=None)
args = parser.parse_args()

if args.out_filename is None:
    args.out_filename = os.path.splitext(args.in_filename)[0] + "-renewed.npz"

data_manager = DataManager(env=None)
data_manager.load_data(args.in_filename)
if args.demo_name is not None:
    data_manager.general_info["demo"] = args.demo_name
data_manager.save_data(args.out_filename)

print("[renew_data] Renew data:")
print(f"  in: {args.in_filename}")
print(f"  out: {args.out_filename}")
