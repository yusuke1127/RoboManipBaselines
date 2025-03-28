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

    parser.add_argument("src_filename", type=str)
    parser.add_argument("dst_dirname", type=str)
    parser.add_argument("--force_overwrite", action="store_true")

    return parser.parse_args()


class ConvertHdf5ToRmb:
    def __init__(self, src_filename, dst_dirname, force_overwrite=False):
        self.src_filename = src_filename
        self.dst_dirname = dst_dirname
        self.force_overwrite = force_overwrite

        _, src_ext = os.path.splitext(self.src_filename)
        if src_ext.lower() != ".hdf5":
            raise ValueError(
                f"Invalid source file extension '{src_ext}'. Expected '.hdf5': {self.src_filename}"
            )
        _, dst_ext = os.path.splitext(self.dst_dirname.rstrip("/"))
        if dst_ext.lower() != ".rmb":
            raise ValueError(
                f"Invalid destination file extension '{dst_ext}'. Expected '.rmb': {self.dst_filename}"
            )

    def run(self):
        if os.path.isfile(self.dst_dirname):
            if not self.force_overwrite:
                print(
                    f"[{self.__class__.__name__}] A file already exists: {self.dst_dirname}"
                )
            if self.get_answer() == "y":
                os.remove(self.dst_dirname)
            else:
                return
        elif os.path.isdir(self.dst_dirname) and os.listdir(self.dst_dirname):
            if not self.force_overwrite:
                print(
                    f"[{self.__class__.__name__}] A non-empty directory already exists: {self.dst_dirname}"
                )
            if self.get_answer() == "y":
                shutil.rmtree(self.dst_dirname)
            else:
                return
        os.makedirs(self.dst_dirname, exist_ok="True")

        dst_filename = os.path.join(self.dst_dirname, "main.rmb.hdf5")
        with (
            h5py.File(self.src_filename, "r") as src_h5file,
            h5py.File(dst_filename, "w") as dst_h5file,
        ):
            for key in src_h5file.keys():
                if DataKey.is_rgb_image_key(key):
                    self.dump_rgb_image(key, src_h5file[key][()])
                elif DataKey.is_depth_image_key(key):
                    self.dump_depth_image(key, src_h5file[key][()])
                else:
                    src_h5file.copy(key, dst_h5file)
            for key in src_h5file.attrs.keys():
                dst_h5file.attrs[key] = src_h5file.attrs[key]

    def dump_rgb_image(self, key, data):
        filename = os.path.join(self.dst_dirname, f"{key}.rmb.mp4")
        videoio.videosave(filename, data)

    def dump_depth_image(self, key, data):
        filename = os.path.join(self.dst_dirname, f"{key}.rmb.mp4")
        videoio.uint16save(filename, (1e3 * data).astype(np.uint16))

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
    conv = ConvertHdf5ToRmb(**vars(parse_argument()))
    conv.run()
