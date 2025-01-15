import argparse

import h5py
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import DataKey, get_skipped_data_seq


def add_arrays_to_smaller_length(arr1, arr2):
    min_length = min(arr1.shape[0], arr2.shape[0])
    return arr1[:min_length] + arr2[:min_length]


def test_get_skipped_data_seq_joint_pos(filename, skip=3):
    abs_key = DataKey.MEASURED_JOINT_POS
    rel_key = DataKey.get_rel_key(abs_key)

    with h5py.File(filename, "r") as h5file:
        joint_abs_seq = h5file[abs_key][()]
        joint_rel_seq = h5file[rel_key][()]

    skipped_joint_abs_seq = joint_abs_seq[::skip][1:]

    skipped_joint_rel_seq = get_skipped_data_seq(joint_rel_seq[1:], rel_key, skip)

    skipped_joint_abs_seq2 = add_arrays_to_smaller_length(
        joint_abs_seq[::skip], skipped_joint_rel_seq
    )

    error = np.sum(
        np.abs(
            add_arrays_to_smaller_length(
                skipped_joint_abs_seq, -1 * skipped_joint_abs_seq2
            )
        )
    )

    print(f"[test_get_skipped_data_seq_joint_pos] error: {error}")


def test_get_skipped_data_seq_eef_pose(filename, skip=3):
    abs_key = DataKey.MEASURED_EEF_POSE
    rel_key = DataKey.get_rel_key(abs_key)

    with h5py.File(filename, "r") as h5file:
        eef_abs_seq = h5file[abs_key][()]
        eef_rel_seq = h5file[rel_key][()]

    skipped_eef_abs_seq = eef_abs_seq[::skip][1:]

    skipped_eef_rel_seq = get_skipped_data_seq(eef_rel_seq[1:], rel_key, skip)

    skipped_eef_abs_seq2 = []
    for eef_abs, eef_rel in zip(eef_abs_seq[::skip][:-1], skipped_eef_rel_seq):
        new_eef_abs = np.zeros_like(eef_abs)
        new_eef_abs[0:3] = eef_abs[0:3] + eef_rel[0:3]
        new_eef_abs[3:7] = pin.Quaternion(
            pin.Quaternion(*eef_abs[3:7]).toRotationMatrix()
            @ pin.rpy.rpyToMatrix(eef_rel[3:6])
        ).coeffs()[[3, 0, 1, 2]]
        skipped_eef_abs_seq2.append(new_eef_abs)
    skipped_eef_abs_seq2 = np.array(skipped_eef_abs_seq2)

    error = np.sum(
        np.abs(
            add_arrays_to_smaller_length(skipped_eef_abs_seq, -1 * skipped_eef_abs_seq2)
        )
    )

    print(f"[test_get_skipped_data_seq_eef_pose] error: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test data utils",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        type=str,
        help="filename of teleoperation data",
    )
    args = parser.parse_args()

    test_get_skipped_data_seq_joint_pos(args.filename)
    test_get_skipped_data_seq_eef_pose(args.filename)
