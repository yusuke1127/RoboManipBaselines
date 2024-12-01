import numpy as np
import torch
from functools import lru_cache
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from robo_manip_baselines.mt_act import CAMERA_NAMES, TEXT_EMBEDDINGS, TASKS


@lru_cache(maxsize=128)
def load_array(dir_path, glob_pattern):
    globbed_list = list(dir_path.glob(glob_pattern))
    assert (
        len(globbed_list) == 1
    ), f"{(dir_path, glob_pattern, globbed_list, len(globbed_list))=}"
    return np.load(globbed_list[0])


@dataclass
class Trials:
    original_joints: np.ndarray = None
    original_actions: np.ndarray = None
    original_images: np.ndarray = None
    original_masks: np.ndarray = None


class EpisodicDatasetRobopen(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, norm_stats):
        super(EpisodicDatasetRobopen).__init__()
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.is_sim = None
        self.trials = Trials()
        self.task_emb_per_trial = []
        self.verbose = True

        tasks = load_array(self.dataset_dir, "**/tasks.npy")
        self.task_emb_per_trial += list(
            map(
                lambda x: TEXT_EMBEDDINGS[
                    {t: i for i, t in enumerate(TASKS)}.get(
                        x,
                        0,  # SINGLE TASK embedding wont be used
                    )
                ],
                tasks,
            )
        )

        self.trials.original_actions = load_array(self.dataset_dir, "**/actions.npy")
        self.trials.original_joints = load_array(self.dataset_dir, "**/joints.npy")
        for cam_name in CAMERA_NAMES:
            try:
                self.trials.original_images = load_array(
                    self.dataset_dir, f"**/{cam_name}_images.npy"
                )
            except IndexError:
                print(f"self.dataset_dir:\t{self.dataset_dir}")
                print(f"cam_name:\t{cam_name}")
                raise
        self.trials.original_masks = load_array(self.dataset_dir, "**/masks.npy")

        print("TOTAL TRIALS", len(self.trials.original_joints))
        self.__getitem__(0)

    def __len__(self):
        return len(load_array(self.dataset_dir, "**/joints.npy"))

    def __getitem__(self, episode_id):
        sample_full_episode = False  # hardcode
        task_emb = self.task_emb_per_trial[episode_id]

        original_action = self.trials.original_actions[episode_id]
        original_joint = self.trials.original_joints[episode_id]
        original_action_shape = original_action.shape
        cutoff = 2  # 10#5
        episode_len = original_action_shape[0] - cutoff  ## cutoff last few

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        joint = original_joint[start_ts]
        image_dict = dict()
        for cam_name in CAMERA_NAMES:
            image_dict[cam_name] = self.trials.original_images[episode_id][start_ts]
        # get mask
        original_mask = self.trials.original_masks[episode_id].astype(bool)
        # get all actions after and including start_ts
        action = original_action[max(0, start_ts - 1) :].astype(
            np.float32
        )  # hack, to make timesteps more aligned
        action_len = episode_len - max(
            0, start_ts - 1
        )  # hack, to make timesteps more aligned
        mask = original_mask[max(0, start_ts - 1) :]

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action[:-cutoff]
        padded_mask = np.zeros(episode_len, dtype=bool)
        padded_mask[:action_len] = mask[:-cutoff]
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in CAMERA_NAMES:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        joint_data = torch.from_numpy(joint).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        joint_data = (joint_data - self.norm_stats["joint_mean"]) / self.norm_stats[
            "joint_std"
        ]

        task_emb = torch.from_numpy(np.asarray(task_emb)).float()

        return image_data, joint_data, action_data, is_pad, task_emb


def get_norm_stats_robopen(train_dataset_dir, val_dataset_dir):
    all_joint_data = []
    all_action_data = []
    for dataset_dir in (train_dataset_dir, val_dataset_dir):
        try:
            joint = load_array(Path(dataset_dir), "**/joints.npy")
            action = load_array(Path(dataset_dir), "**/actions.npy")
        except IndexError:
            print(f"dataset_dir:\t{dataset_dir}")
            raise
        all_joint_data.append(torch.from_numpy(joint))
        all_action_data.append(torch.from_numpy(action))
    all_joint_data = torch.cat(all_joint_data)
    all_action_data = torch.cat(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10)  # clipping

    # normalize joint data
    joint_mean = all_joint_data.mean(dim=[0, 1], keepdim=True)
    joint_std = all_joint_data.std(dim=[0, 1], keepdim=True)
    joint_std = torch.clip(joint_std, 1e-2, 10)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "joint_mean": joint_mean.numpy().squeeze(),
        "joint_std": joint_std.numpy().squeeze(),
        "example_joint": joint,
    }

    return stats


def load_data(dataset_dir, batch_size_train, batch_size_val):
    print(f"\nData from: {dataset_dir}\n")
    dataset_dir = Path(dataset_dir)
    # obtain train test dataset dir
    train_dataset_dir = dataset_dir / "train"
    val_dataset_dir = dataset_dir / "test"

    # obtain normalization stats for joint and action
    norm_stats = get_norm_stats_robopen(train_dataset_dir, val_dataset_dir)

    # construct dataset and dataloader
    train_dataset = EpisodicDatasetRobopen(train_dataset_dir, norm_stats)
    val_dataset = EpisodicDatasetRobopen(val_dataset_dir, norm_stats)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim
