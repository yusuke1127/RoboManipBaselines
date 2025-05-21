import argparse
import os

import h5py


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("path", type=str, help="path to data (*.hdf5 or *.rmb)")
    parser.add_argument(
        "--task_desc", type=str, required=True, help="task description to set"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )

    return parser.parse_args()


class RefineRmbData:
    def __init__(self, path, task_desc, overwrite=False):
        self.path = path.rstrip("/")
        self.task_desc_new = task_desc
        self.overwrite = overwrite
        self.hdf5_path = self.resolve_hdf5_path(self.path)

    def resolve_hdf5_path(self, path):
        if path.endswith(".rmb"):
            hdf5_path = os.path.join(path, "main.rmb.hdf5")
        elif path.endswith(".hdf5"):
            hdf5_path = path
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported file extension: {path}"
            )

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] HDF5 file not found: {hdf5_path}"
            )
        return hdf5_path

    def run(self):
        print(f"[{self.__class__.__name__}] Open {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r+") as f:
            task_desc_orig = f.attrs.get("task_desc", "")
            if isinstance(task_desc_orig, bytes):
                task_desc_orig = task_desc_orig.decode("utf-8")

            if task_desc_orig and not self.overwrite:
                print(
                    f"[{self.__class__.__name__}] task_desc already exists and is non-empty: {task_desc_orig} (use --overwrite to replace)"
                )
                return

            print(f'Set task_desc from "{task_desc_orig}" to "{self.task_desc_new}"')
            f.attrs["task_desc"] = self.task_desc_new


if __name__ == "__main__":
    refine = RefineRmbData(**vars(parse_argument()))
    refine.run()
