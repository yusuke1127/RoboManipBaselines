import argparse
import os
import shutil

import h5py
import numpy as np
import videoio

from robo_manip_baselines.common import DataKey


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("src_dirname", type=str)
    parser.add_argument("dst_filename", type=str)
    parser.add_argument("--force_overwrite", action="store_true")

    return parser.parse_args()


class ConvertRmbToHdf5:
    def __init__(self, src_dirname, dst_filename, force_overwrite=False):
        self.src_dirname = src_dirname
        self.dst_filename = dst_filename
        self.force_overwrite = force_overwrite

        _, src_ext = os.path.splitext(self.src_dirname.rstrip("/"))
        if src_ext.lower() != ".rmb":
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid source file extension '{src_ext}'. Expected '.rmb': {self.src_dirname}"
            )
        _, dst_ext = os.path.splitext(self.dst_filename)
        if dst_ext.lower() != ".hdf5":
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid destination file extension '{dst_ext}'. Expected '.hdf5': {self.dst_filename}"
            )

    def run(self):
        if os.path.exists(self.dst_filename):
            if not self.force_overwrite:
                print(
                    f"[{self.__class__.__name__}] A file already exists: {self.dst_filename}"
                )
            if self.get_answer() == "y":
                if os.path.isdir(self.dst_filename):
                    shutil.rmtree(self.dst_filename)
                else:
                    os.remove(self.dst_filename)
            else:
                return

        src_filename = os.path.join(self.src_dirname, "main.rmb.hdf5")
        with h5py.File(self.dst_filename, "w") as dst_h5file:
            with h5py.File(src_filename, "r") as src_h5file:
                for key in src_h5file.keys():
                    src_h5file.copy(key, dst_h5file)
                for key in src_h5file.attrs.keys():
                    dst_h5file.attrs[key] = src_h5file.attrs[key]

            for src_filename in os.listdir(self.src_dirname):
                if not src_filename.endswith(".rmb.mp4"):
                    continue
                key = src_filename[: -len(".rmb.mp4")]
                if DataKey.is_rgb_image_key(key):
                    dst_h5file.create_dataset(
                        key, data=self.load_rgb_image(src_filename)
                    )
                elif DataKey.is_depth_image_key(key):
                    dst_h5file.create_dataset(
                        key, data=self.load_depth_image(src_filename)
                    )

    def load_rgb_image(self, filename):
        filename = os.path.join(self.src_dirname, filename)
        return videoio.videoread(filename)

    def load_depth_image(self, filename):
        filename = os.path.join(self.src_dirname, filename)
        return (1e-3 * videoio.uint16read(filename)).astype(np.float32)

    def get_answer(self):
        if self.force_overwrite:
            return "y"
        else:
            return (
                input(f"[{self.__class__.__name__}] Do you want to delete it? (y/n): ")
                .strip()
                .lower()
            )


if __name__ == "__main__":
    conv = ConvertRmbToHdf5(**vars(parse_argument()))
    conv.run()
