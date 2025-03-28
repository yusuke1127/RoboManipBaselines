import argparse

import h5py
import numpy as np


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("filename1", type=str)
    parser.add_argument("filename2", type=str)

    return parser.parse_args()


class CompareHdf5:
    def __init__(self, filename1, filename2):
        self.filename1 = filename1
        self.filename2 = filename2

    def run(self):
        with (
            h5py.File(self.filename1, "r") as h5file1,
            h5py.File(self.filename2, "r") as h5file2,
        ):
            # Check data
            keys1 = set(h5file1.keys())
            keys2 = set(h5file2.keys())

            only_in_file1 = keys1 - keys2
            only_in_file2 = keys2 - keys1

            if only_in_file1:
                print(
                    f"[{self.__class__.__name__}] Keys only in {self.filename1}: {only_in_file1}"
                )
            if only_in_file2:
                print(
                    f"[{self.__class__.__name__}] Keys only in {self.filename2}: {only_in_file2}"
                )

            common_keys = sorted(keys1 & keys2)
            for key in common_keys:
                data1 = h5file1[key]
                data2 = h5file2[key]

                if data1.shape != data2.shape:
                    print(
                        f"[{self.__class__.__name__}] {key:25}: Shape mismatch. {data1.shape} vs {data2.shape}"
                    )
                    continue

                if data1.dtype != data2.dtype:
                    print(
                        f"[{self.__class__.__name__}] {key:25}: Type mismatch. {data1.dtype} vs {data2.dtype}"
                    )
                    if not (
                        np.issubdtype(data1.dtype, np.floating)
                        and np.issubdtype(data2.dtype, np.floating)
                    ):
                        continue

                if np.issubdtype(data1.dtype, np.unsignedinteger):
                    diff = np.abs(
                        data1[...].astype(np.int64) - data2[...].astype(np.int64)
                    )
                else:
                    diff = np.abs(data1[...] - data2[...])
                if np.max(diff) > 0.0:
                    print(
                        f"[{self.__class__.__name__}] {key:25}: Mean err = {np.mean(diff):.5g}, Max err = {np.max(diff):.5g}"
                    )

            # Check attrs
            keys1 = set(h5file1.attrs.keys())
            keys2 = set(h5file2.attrs.keys())

            only_in_file1 = keys1 - keys2
            only_in_file2 = keys2 - keys1

            if only_in_file1:
                print(
                    f"[{self.__class__.__name__}] Attributes only in {self.filename1}: {only_in_file1}"
                )
            if only_in_file2:
                print(
                    f"[{self.__class__.__name__}] Attributes only in {self.filename2}: {only_in_file2}"
                )

            common_keys = sorted(keys1 & keys2)
            for key in common_keys:
                data1 = h5file1.attrs[key]
                data2 = h5file2.attrs[key]

                if type(data1) is not type(data2):
                    print(
                        f"[{self.__class__.__name__}] Attribute {key:25}: Type mismatch. {type(data1)} vs {type(data2)}"
                    )
                    continue


if __name__ == "__main__":
    conv = CompareHdf5(**vars(parse_argument()))
    conv.run()
