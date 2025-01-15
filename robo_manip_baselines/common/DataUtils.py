from functools import reduce

import numpy as np
import pinocchio as pin

from .DataKey import DataKey


def aggregate_data_seq_with_skip(data_seq, skip, agg_func):
    """
    Aggregate elements along the first axis (axis=0) in chunks of skip.

    Parameters:
        data_seq (np.ndarray): Input data sequence.
        skip (int): Group size.
        agg_func (callable): A function that takes a group (shape: (skip, ...)) as input and returns the aggregated result.

    Returns:
        np.ndarray: New data sequence where the specified function has been applied to grouped elements along the first axis.
    """
    # Add zero padding to the first axis
    size = data_seq.shape[0]
    padding = (skip - size % skip) % skip
    if padding > 0:
        pad_shape = [(0, padding)] + [(0, 0)] * (data_seq.ndim - 1)
        data_seq = np.pad(
            data_seq, pad_width=pad_shape, mode="constant", constant_values=0
        )

    # Reshape into skip groups along the first axis
    grouped_data_seq = data_seq.reshape(-1, skip, *data_seq.shape[1:])

    # Aggregate array by applying the user-defined function to each group
    aggregated_data_seq = np.array([agg_func(arr) for arr in grouped_data_seq])

    return aggregated_data_seq


def get_skipped_data_seq(data_seq, key, skip):
    """Get skipped data sequence."""
    if key in (
        DataKey.MEASURED_JOINT_POS_REL,
        DataKey.COMMAND_JOINT_POS_REL,
    ):

        def sum_regular(arr):
            return np.sum(arr, axis=0)

        skipped_data_seq = aggregate_data_seq_with_skip(data_seq, skip, sum_regular)
    elif key in (
        DataKey.MEASURED_EEF_POSE_REL,
        DataKey.COMMAND_EEF_POSE_REL,
    ):

        def sum_pose_rel(arr):
            def add_pose_rel(pose_rel1, pose_rel2):
                new_pose_rel = np.empty_like(pose_rel1)
                new_pose_rel[0:3] = pose_rel1[0:3] + pose_rel2[0:3]
                new_pose_rel[3:6] = pin.rpy.matrixToRpy(
                    pin.rpy.rpyToMatrix(pose_rel1[3:6])
                    @ pin.rpy.rpyToMatrix(pose_rel2[3:6])
                )
                return new_pose_rel

            return reduce(add_pose_rel, arr)

        skipped_data_seq = aggregate_data_seq_with_skip(data_seq, skip, sum_pose_rel)
    else:
        skipped_data_seq = data_seq[::skip]

    return skipped_data_seq
