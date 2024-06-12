import numpy as np
import torch
import os
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.__getitem__(0) # initialize self

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        import pdb;pdb.set_trace()
        joints_raw = np.load(sorted(dataset_dir.glob("**/joints.npy"))[0])
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            joint = root['/observations/joint'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

        is_pad = np.zeros(episode_len)

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        joint_data = torch.from_numpy(joint).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        joint_data = (joint_data - self.norm_stats["joint_mean"]) / self.norm_stats["joint_std"]

        return image_data, joint_data, is_pad


def get_norm_stats(train_dataset_dir, val_dataset_dir):
    all_joint_data = []
    for dataset_dir in (train_dataset_dir, val_dataset_dir):
        joints_raw = np.load(sorted(dataset_dir.glob("**/joints.npy"))[0])
        all_joint_data.append(torch.from_numpy(joints_raw))
    all_joint_data = torch.stack(all_joint_data)

    # normalize joint data
    joint_mean = all_joint_data.mean(dim=[0, 1], keepdim=True)
    joint_std = all_joint_data.std(dim=[0, 1], keepdim=True)
    joint_std = torch.clip(joint_std, 1e-2, np.inf) # clipping

    stats = {"joint_mean": joint_mean.numpy().squeeze(), "joint_std": joint_std.numpy().squeeze(),
             "example_joint": joints_raw}

    return stats


def load_data(dataset_dir, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    dataset_dir = Path(dataset_dir)
    # obtain train test dataset dir
    train_dataset_dir = dataset_dir / "train"
    val_dataset_dir = dataset_dir / "test"

    # obtain normalization stats for joint
    norm_stats = get_norm_stats(train_dataset_dir, val_dataset_dir)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

