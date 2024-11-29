import numpy as np


def calc_minmax(array):
    axis = tuple(np.arange(array.ndim - 1))
    return np.array([np.min(array, axis=axis), np.max(array, axis=axis)])


def stack_arrays_with_padding(arr_list, seq_len=None):
    if seq_len is None:
        seq_len = np.max([len(arr) for arr in arr_list])

    arr_concat = np.empty(
        (len(arr_list), seq_len) + arr_list[0].shape[1:], dtype=arr_list[0].dtype
    )

    for arr_idx, arr in enumerate(arr_list):
        arr_concat[arr_idx][: len(arr)] = arr
        arr_concat[arr_idx][len(arr) :] = arr[-1]

    return arr_concat
