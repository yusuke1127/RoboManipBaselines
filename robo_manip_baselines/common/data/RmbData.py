import os

import h5py
import numpy as np
import videoio

from .DataKey import DataKey


class RmbData:
    """Data in RoboManipBaselines format."""

    class RmbVideo:
        def __len__(self):
            return len(self[()])

        @property
        def shape(self):
            return self[()].shape

    class RmbRgbVideo(RmbVideo):
        def __init__(self, path):
            self.path = path

        def __getitem__(self, key):
            return videoio.videoread(self.path)[key]

        @property
        def dtype(self):
            return np.uint8

    class RmbDepthVideo(RmbVideo):
        def __init__(self, path):
            self.path = path

        def __getitem__(self, key):
            return (1e-3 * videoio.uint16read(self.path)[key]).astype(np.float32)

        @property
        def dtype(self):
            return np.float32

    def __init__(self):
        self.path = None
        self.is_single_hdf5 = None
        self.h5file = None

    @classmethod
    def from_file(cls, path):
        inst = cls()

        inst.path = path

        _, ext = os.path.splitext(path.rstrip("/"))
        if ext.lower() == ".hdf5":
            inst.is_single_hdf5 = True
        elif ext.lower() == ".rmb":
            inst.is_single_hdf5 = False
        else:
            raise ValueError(
                f"[{inst.__class__.__name__}] Invalid file extension '{ext}'. Expected '.hdf5' or '.rmb': {path}"
            )

        return inst

    def open(self):
        if self.path is None:
            raise ValueError(f"[{self.__class__.__name__}] The file path is not set.")

        if self.h5file is not None:
            self.close()

        if self.is_single_hdf5:
            path = self.path
        else:
            path = os.path.join(self.path, "main.rmb.hdf5")
        self.h5file = h5py.File(path, "r")

    def close(self):
        if self.h5file is None:
            return

        self.h5file.close()
        self.h5file = None

    @property
    def closed(self):
        return self.h5file is None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, key):
        if self.is_single_hdf5:
            return self.h5file[key]
        elif DataKey.is_rgb_image_key(key):
            return self.RmbRgbVideo(os.path.join(self.path, f"{key}.rmb.mp4"))
        elif DataKey.is_depth_image_key(key):
            return self.RmbDepthVideo(os.path.join(self.path, f"{key}.rmb.mp4"))
        else:
            return self.h5file[key]

    def keys(self):
        if self.is_single_hdf5:
            return self.h5file.keys()
        else:
            ret = list(self.h5file.keys())

            for filename in os.listdir(self.path):
                if not filename.endswith(".rmb.mp4"):
                    continue
                key = filename[: -len(".rmb.mp4")]
                if DataKey.is_rgb_image_key(key) or DataKey.is_depth_image_key(key):
                    ret.append(key)

            return ret

    @property
    def attrs(self):
        return self.h5file.attrs

    def dump_hdf5(self):
        pass  # TODO

    def dump_rmb(self):
        pass  # TODO
