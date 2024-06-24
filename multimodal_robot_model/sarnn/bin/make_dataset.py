#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import sys
sys.path.append("../../third_party/eipl/")
import os
import glob
import argparse
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from eipl.utils import resize_img, calc_minmax, list_to_numpy
from sklearn.model_selection import train_test_split
from sklearn.utils._param_validation import InvalidParameterError


parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, default="./bag/")
parser.add_argument("--skip", type=int, default=10)
#parser.add_argument("--train_size", type=float, required=False)
#parser.add_argument("--test_size", type=float, required=False)
#parser.add_argument("--random_state", type=int, required=False)
#parser.add_argument("--not_shuffle", action="store_true")
parser.add_argument("--cropped_img_size", type=int, required=False)
parser.add_argument("-j", "--nproc", type=int, default=1)
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()
print(args)


def load_skip_data(file_info):
    skip, filename = file_info
    print(filename)
    npz_data = np.load(filename)
    try:
        _front_images = npz_data["front_image"][::skip]
        _side_images = npz_data["side_image"][::skip]
        if args.cropped_img_size is not None:
            [fro_lef, fro_top, sid_lef, sid_top] = [
                (
                    images_shape[ax] - args.cropped_img_size
                ) // 2 for images_shape in [
                    _front_images.shape, 
                    _side_images.shape
                ] for ax in [1, 2]
            ]
            [fro_rig, fro_bot, sid_rig, sid_bot] = [
                (p + args.cropped_img_size) for p in [
                    fro_lef, fro_top, sid_lef, sid_top
                ]
            ]
            _front_images = _front_images[:, fro_lef:fro_rig, fro_top:fro_bot, :]
            _side_images = _side_images[:, sid_lef:sid_rig, sid_top:sid_bot, :]
        _front_images = resize_img(_front_images, (64, 64))
        _side_images = resize_img(_side_images, (64, 64))
        _wrenches = npz_data["wrench"][::skip]
        _joints = npz_data["joint"][::skip]
    except KeyError as e:
        print(f"{e.__class__.__name__}: filename={filename}")
        raise e
    assert len(_front_images) == len(_side_images)
    assert len(_front_images) == len(_wrenches)
    assert len(_front_images) == len(_joints)
    return (_front_images, _side_images, _wrenches, _joints)


def save_arr(file_name, arr_data, verbose):
    """ almost an alias for np.save() """
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    np.save(file_name, arr_data)
    if verbose:
        print(f"(save file, shape):\t( {file_name},\t{arr_data.shape} )")


def load_data(in_dir, skip, nproc):
    front_images = []
    side_images = []
    wrenches = []
    joints = []
    masks = []

    file_names = glob.glob(os.path.join(in_dir, "**/*.npz"))
    file_names.sort()
    try:
        assert len(file_names) >= 1, f"Not asserted len(file_names):\t{len(file_names)}"
    except AssertionError as e:
        print("in_dir:\t", in_dir)
        raise e
    pool = Pool(nproc)
    loaded_data = pool.map(
        load_skip_data, [(skip, file_name) for file_name in file_names]
    )
    seq_length = []
    for (_front_images, _side_images, _wrenches, _joints) in loaded_data:
        seq_length.append(len(_joints))
    max_seq = max(seq_length)

    for (_front_images, _side_images, _wrenches, _joints) in loaded_data:
        front_images.append(_front_images)
        side_images.append(_side_images)
        wrenches.append(_wrenches)
        joints.append(_joints)
        masks.append(np.concatenate((np.ones(len(_joints)), np.zeros(max_seq - len(_joints)))))

    front_images = list_to_numpy(front_images, max_seq)
    side_images = list_to_numpy(side_images, max_seq)
    wrenches = list_to_numpy(wrenches, max_seq)
    joints = list_to_numpy(joints, max_seq)
    masks = np.stack(masks)

    return front_images, side_images, wrenches, joints, masks, file_names


def split_dataset_index(in_list, train_size, test_size, random_state, shuffle):

    # cast to int if necessary 
    if train_size is not None and train_size >= 1.0:
        train_size = int(train_size)
    if test_size is not None and test_size >= 1.0:
        test_size = int(test_size)

    # train test split
    try:
        train_list, test_list = train_test_split(
            in_list,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle
        )
    except InvalidParameterError as e:
        print(
            f"{e.__class__.__name__}: "
            f"(test_size, train_size)={(test_size, train_size)}"
        )
        raise

    return train_list, test_list


if __name__ == "__main__":
    # load data
    front_images, side_images, wrenches, joints, masks, file_names = load_data(
        args.in_dir, args.skip, args.nproc
    )

    # dataset index
    train_list, test_list = list(), list()
    for i, file_name in enumerate(file_names):
        if "env3" in file_name:
            test_list.append(i)
        else:
            train_list.append(i)
    """
    train_list, test_list = split_dataset_index(
        range(len(joints)), 
        train_size=args.train_size, 
        test_size=args.test_size, 
        random_state=args.random_state, 
        shuffle=(not args.not_shuffle)
    )
    """
    if args.verbose:
        print("train_list:\t", train_list)
        print("test_list:\t", test_list)

    # save
    save_arr("./data/train/front_images.npy", front_images[train_list].astype(np.uint8), args.verbose)
    save_arr("./data/train/side_images.npy", side_images[train_list].astype(np.uint8), args.verbose)
    save_arr("./data/train/wrenches.npy", wrenches[train_list].astype(np.float32), args.verbose)
    save_arr("./data/train/joints.npy", joints[train_list].astype(np.float32), args.verbose)
    save_arr("./data/train/masks.npy", masks[train_list].astype(np.float32), args.verbose)
    save_arr("./data/test/front_images.npy", front_images[test_list].astype(np.uint8), args.verbose)
    save_arr("./data/test/side_images.npy", side_images[test_list].astype(np.uint8), args.verbose)
    save_arr("./data/test/wrenches.npy", wrenches[test_list].astype(np.float32), args.verbose)
    save_arr("./data/test/joints.npy", joints[test_list].astype(np.float32), args.verbose)
    save_arr("./data/test/masks.npy", masks[test_list].astype(np.float32), args.verbose)

    # save joint bounds
    joint_bounds = calc_minmax(joints)
    save_arr("./data/joint_bounds.npy", joint_bounds, args.verbose)

    # save wrench bounds
    wrench_bounds = calc_minmax(wrenches)
    save_arr("./data/wrench_bounds.npy", wrench_bounds, args.verbose)
