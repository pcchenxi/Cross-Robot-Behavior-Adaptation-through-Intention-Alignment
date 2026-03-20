"""Shared training dataset utilities for local picking tasks."""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import crop


class BaseTrainingDataset(Dataset):
    def __init__(self, path, transform=None, task_name='ur5f', init_stats=False):
        self._path = path
        self.transform = transform
        self.task_name = task_name

        action_path = os.path.join(self._path, 'action')
        self.file_list = os.listdir(action_path)
        self.n_episodes = len(self.file_list)
        print('loading', task_name, action_path, self.n_episodes)

        self.actions = self._preload_actions()
        self.action_mean, self.action_std = self.get_action_stats(
            task_name,
            init_stats=init_stats,
        )
        self.normalized_actions = (
            (self.actions - self.action_mean) / (self.action_std + 1e-8)
        ).astype(np.float32)

    def _get_action_stats_paths(self, task_name):
        stats_folder = Path(__file__).resolve().parents[2] / 'results' / 'action_stats'
        file_name_mean = stats_folder / f'action_{task_name}_mean.pkl'
        file_name_std = stats_folder / f'action_{task_name}_std.pkl'
        return stats_folder, file_name_mean, file_name_std

    def _to_numpy_stats(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _load_field(self, field, file_idx):
        path = os.path.join(self._path, field, file_idx)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_action_from_episode(self, episode_id):
        file_idx = self.file_list[episode_id]
        ee_state = self._load_field('ee_state', file_idx)
        if len(ee_state) < 2:
            return None
        return np.asarray(ee_state[1], dtype=np.float32)

    def _get_lang_goal(self, episode_id):
        file_idx = self.file_list[episode_id]
        info = self._load_field('info', file_idx)
        if not info:
            raise ValueError(f'Episode {file_idx} does not contain info entries.')
        return info[0]['lang_goal']

    def _preload_actions(self):
        all_actions = []
        for idx in range(self.n_episodes):
            action = self._load_action_from_episode(idx)
            if action is None:
                raise IndexError(
                    f'Episode {self.file_list[idx]} does not contain the second timestep required for training.'
                )
            all_actions.append(action)

        if not all_actions:
            raise ValueError('No valid actions found to preload.')

        actions = np.stack(all_actions, axis=0).astype(np.float32)
        print('preloaded actions', actions.shape)
        return actions

    def compute_and_save_action_stats(self, task_name=None):
        task_name = self.task_name if task_name is None else task_name
        action_mean = self.actions.mean(axis=0).astype(np.float32)
        action_std = self.actions.std(axis=0).astype(np.float32)

        stats_folder_name, file_name_mean, file_name_std = self._get_action_stats_paths(
            task_name,
        )
        os.makedirs(stats_folder_name, exist_ok=True)

        with open(file_name_mean, 'wb') as f:
            pickle.dump(torch.from_numpy(action_mean), f)
        with open(file_name_std, 'wb') as f:
            pickle.dump(torch.from_numpy(action_std), f)

        print('saved action stats to', stats_folder_name)
        return action_mean, action_std

    def get_action_stats(self, task_name, init_stats=False):
        _, file_name_mean, file_name_std = self._get_action_stats_paths(task_name)
        print('action stats folder', file_name_mean, file_name_std)

        if init_stats:
            return self.compute_and_save_action_stats(task_name)

        try:
            with open(file_name_mean, 'rb') as f:
                action_mean = self._to_numpy_stats(pickle.load(f))
            with open(file_name_std, 'rb') as f:
                action_std = self._to_numpy_stats(pickle.load(f))
            print('--- action stats loaded')
            return action_mean, action_std
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return self.compute_and_save_action_stats(task_name)

    def _get_camera_image(self, color):
        if self.task_name == 'ur5f':
            return color[0]
        if self.task_name == 'ur5l':
            return color[1]
        if self.task_name == 'ur5r':
            return color[2]
        raise ValueError(f'Unsupported task_name: {self.task_name}')

    def load(self, episode_id):
        file_idx = self.file_list[episode_id]
        color = self._load_field('color', file_idx)
        obs = {
            'color': color[0],
        }
        return obs, self.normalized_actions[episode_id]

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        return self.get_item(idx)


def crop_img(image):
    return crop(image, 100, 100, 400, 450)


def create_training_dataloader(
    dataset,
    batch_size=16,
    num_workers=4,
    epoch_step_num=100,
    persistent_workers=False,
):
    return DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(
            dataset,
            num_samples=batch_size * epoch_step_num,
            replacement=True,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers and num_workers > 0,
    )
