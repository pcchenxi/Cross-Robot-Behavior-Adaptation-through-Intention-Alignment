"""Shared runtime associator utilities."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from iail_sim_picking.intention_extractor.intention_extractor_network import (
    IntentionExtractorNetwork,
)
from iail_sim_picking.motion_generator.motion_generator_trainer import MotionGenerator

SUPPORTED_TASK_NAMES = ("ur5f", "ur5l", "ur5r")
TASK_CAMERA_INDEX = {"ur5f": 0, "ur5l": 1, "ur5r": 2}
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets"


def _trace(message):
    print(f"[RuntimeAssociator] {message}", flush=True)


def load_torch_checkpoint(checkpoint_path, map_location):
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def ensure_supported_task_name(task_name, field_name):
    if task_name not in SUPPORTED_TASK_NAMES:
        raise ValueError(
            f"{field_name} must be one of {SUPPORTED_TASK_NAMES}, got {task_name!r}"
        )


def load_pickle_array(path):
    with open(path, "rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def to_device_tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    raise TypeError("Expected torch.Tensor or numpy.ndarray input.")


def add_batch_dim(tensor):
    if tensor.ndim in (1, 3):
        return tensor.unsqueeze(0)
    return tensor


def load_action_stats(task_name, results_dir=DEFAULT_RESULTS_DIR):
    ensure_supported_task_name(task_name, "task_name")
    stats_dir = Path(results_dir) / "action_stats"
    mean_path = stats_dir / f"action_{task_name}_mean.pkl"
    std_path = stats_dir / f"action_{task_name}_std.pkl"
    if not mean_path.exists():
        raise FileNotFoundError(f"Action mean stats do not exist: {mean_path}")
    if not std_path.exists():
        raise FileNotFoundError(f"Action std stats do not exist: {std_path}")

    action_mean = load_pickle_array(mean_path)
    action_std = load_pickle_array(std_path)
    if action_mean.shape != action_std.shape:
        raise ValueError(
            f"Action stats shape mismatch for {task_name}: "
            f"mean={action_mean.shape}, std={action_std.shape}"
        )
    return action_mean, action_std


def load_intention_extractor(
    checkpoint_path,
    *,
    action_dim,
    device,
    text_encoder_model="distilbert-base-uncased",
    projection_dim=256,
    dropout=0.1,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Intention extractor checkpoint does not exist: {checkpoint_path}"
        )
    _trace(f"Loading intention extractor checkpoint from {checkpoint_path}")

    network = IntentionExtractorNetwork(
        action_dim=action_dim,
        text_encoder_model=text_encoder_model,
        pretrained=False,
        trainable=False,
        projection_dim=projection_dim,
        dropout=dropout,
        device=device,
    ).to(device)
    state_dict = load_torch_checkpoint(checkpoint_path, map_location=device)
    try:
        network.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load intention extractor checkpoint {checkpoint_path}. "
            f"Check action_dim={action_dim}, projection_dim={projection_dim}, and dropout={dropout}."
        ) from exc
    network.eval()
    return network


def load_motion_generator(
    checkpoint_path,
    *,
    action_dim,
    latent_dim,
    device,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Motion generator checkpoint does not exist: {checkpoint_path}"
        )
    _trace(f"Loading motion generator checkpoint from {checkpoint_path}")

    motion_generator = MotionGenerator(
        action_dim=action_dim,
        latent_dim=latent_dim,
        device=device,
    ).to(device)
    state_dict = load_torch_checkpoint(checkpoint_path, map_location=device)
    try:
        motion_generator.actor_vae.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load motion generator checkpoint {checkpoint_path}. "
            f"Check action_dim={action_dim} and latent_dim={latent_dim}."
        ) from exc
    motion_generator.eval()
    motion_generator.actor_vae.eval()
    return motion_generator


def save_summary_json(summary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
