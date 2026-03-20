"""Recompute and compare motion-generator action stats for a task."""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data_paper"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
SUPPORTED_TASK_NAMES = ("ur5f", "ur5l", "ur5r")


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-name",
        choices=SUPPORTED_TASK_NAMES,
        required=True,
        help="Task name used to resolve dataset and stored stats.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing <task-name>-train folders.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split suffix used to resolve <task-name>-<split>.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory containing results/action_stats.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for np.allclose comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for np.allclose comparison.",
    )
    return parser


def resolve_dataset_path(data_root, task_name, split):
    dataset_path = Path(data_root) / f"{task_name}-{split}"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    return dataset_path


def resolve_stats_paths(results_dir, task_name):
    stats_dir = Path(results_dir) / "action_stats"
    mean_path = stats_dir / f"action_{task_name}_mean.pkl"
    std_path = stats_dir / f"action_{task_name}_std.pkl"
    if not mean_path.exists():
        raise FileNotFoundError(f"Stored action mean does not exist: {mean_path}")
    if not std_path.exists():
        raise FileNotFoundError(f"Stored action std does not exist: {std_path}")
    return mean_path, std_path


def load_pickle_array(path):
    with open(path, "rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_actions(dataset_path):
    action_dir = Path(dataset_path) / "action"
    ee_state_dir = Path(dataset_path) / "ee_state"
    if not action_dir.exists():
        raise FileNotFoundError(f"Action directory does not exist: {action_dir}")
    if not ee_state_dir.exists():
        raise FileNotFoundError(f"ee_state directory does not exist: {ee_state_dir}")

    file_list = sorted(os.listdir(action_dir))
    if not file_list:
        raise ValueError(f"No episode files found in: {action_dir}")

    actions = []
    for file_name in file_list:
        ee_state_path = ee_state_dir / file_name
        if not ee_state_path.exists():
            raise FileNotFoundError(f"Missing ee_state file for episode {file_name}: {ee_state_path}")
        with open(ee_state_path, "rb") as handle:
            ee_state = pickle.load(handle)
        if len(ee_state) < 2:
            raise ValueError(
                f"Episode {file_name} does not contain the second timestep required for stats."
            )
        actions.append(np.asarray(ee_state[1], dtype=np.float32))

    return np.stack(actions, axis=0).astype(np.float32)


def summarize_difference(name, recomputed, stored, atol, rtol):
    diff = recomputed - stored
    abs_diff = np.abs(diff)
    is_close = np.allclose(recomputed, stored, atol=atol, rtol=rtol)
    print(f"{name}_recomputed: {recomputed}")
    print(f"{name}_stored:     {stored}")
    print(f"{name}_diff:       {diff}")
    print(f"{name}_max_abs_diff: {abs_diff.max():.10f}")
    print(f"{name}_allclose(atol={atol}, rtol={rtol}): {bool(is_close)}")
    return bool(is_close)


def main():
    args = build_argparser().parse_args()
    dataset_path = resolve_dataset_path(args.data_root, args.task_name, args.split)
    mean_path, std_path = resolve_stats_paths(args.results_dir, args.task_name)

    actions = load_actions(dataset_path)
    recomputed_mean = actions.mean(axis=0).astype(np.float32)
    recomputed_std = actions.std(axis=0).astype(np.float32)

    stored_mean = load_pickle_array(mean_path)
    stored_std = load_pickle_array(std_path)

    if recomputed_mean.shape != stored_mean.shape:
        raise ValueError(
            f"Mean shape mismatch: recomputed={recomputed_mean.shape}, stored={stored_mean.shape}"
        )
    if recomputed_std.shape != stored_std.shape:
        raise ValueError(
            f"Std shape mismatch: recomputed={recomputed_std.shape}, stored={stored_std.shape}"
        )

    print(f"task_name: {args.task_name}")
    print(f"dataset_path: {dataset_path}")
    print(f"num_episodes: {actions.shape[0]}")
    print(f"action_dim: {actions.shape[1]}")
    print(f"stored_mean_path: {mean_path}")
    print(f"stored_std_path: {std_path}")

    mean_ok = summarize_difference(
        "mean",
        recomputed_mean,
        stored_mean,
        atol=args.atol,
        rtol=args.rtol,
    )
    std_ok = summarize_difference(
        "std",
        recomputed_std,
        stored_std,
        atol=args.atol,
        rtol=args.rtol,
    )

    if mean_ok and std_ok:
        print("status: MATCH")
    else:
        print("status: MISMATCH")


if __name__ == "__main__":
    main()
