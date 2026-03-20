"""Run online demonstrator -> learner imitation evaluation."""

import argparse
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from iail_sim_picking.runtime_associator import (
    Demonstrator,
    Learner,
    Runner,
    save_summary_json,
)


def resolve_demo_data_dir(data_root: Path, task_name: str) -> Path:
    demo_data_dir = data_root / f"{task_name}-test"
    if not demo_data_dir.exists():
        raise FileNotFoundError(
            f"Resolved demo data directory does not exist: {demo_data_dir}"
        )
    return demo_data_dir


def resolve_learner_motion_checkpoint(
    task_name: str,
    motion_root: Path,
) -> Path:
    checkpoint = motion_root / f"cvae_{task_name}_latest.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Resolved learner motion checkpoint does not exist: {checkpoint}"
        )
    return checkpoint


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)

    demonstrator_group = parser.add_argument_group("demonstrator")
    demonstrator_group.add_argument(
        "--demo-data-root",
        type=Path,
        default=REPO_ROOT / "data_paper",
        help="Root directory containing demo task folders such as <task_name>-test.",
    )
    demonstrator_group.add_argument(
        "--demo-task-name",
        choices=("ur5f", "ur5l", "ur5r"),
        required=True,
        help="Demo task name used to resolve the directory as <demo_data_root>/<task_name>-test.",
    )
    demonstrator_group.add_argument(
        "--demo-intention-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "intention_extractor"
        / "clip_intention_extractor_latest.pth",
    )

    learner_group = parser.add_argument_group("learner")
    learner_group.add_argument(
        "--learner-task-name",
        choices=("ur5f", "ur5l", "ur5r"),
        required=True,
    )
    learner_group.add_argument(
        "--learner-intention-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "intention_extractor"
        / "clip_intention_extractor_latest.pth",
    )
    learner_group.add_argument(
        "--learner-motion-root",
        type=Path,
        default=REPO_ROOT / "results" / "motion_generator",
        help=(
            "Directory containing learner motion generator checkpoints. "
            "The script resolves cvae_<learner_task_name>_latest.pth from this root."
        ),
    )
    learner_group.add_argument("--latent-dim", type=int, default=6)
    learner_group.add_argument("--valid-threshold", type=float, default=0.5)
    learner_group.add_argument("--aligned-threshold", type=float, default=0.3)
    learner_group.add_argument("--learner-mode", default="test")

    runner_group = parser.add_argument_group("runner")
    runner_group.add_argument(
        "--assets-root",
        type=Path,
        default=REPO_ROOT / "iail_sim_picking" / "assets",
    )
    runner_group.add_argument(
        "--env-display",
        action="store_true",
        help="Open the live OpenCV environment display while running evaluation.",
    )
    runner_group.add_argument(
        "--save-learner-image-dir",
        type=Path,
        default=None,
        help="Optional directory to save the reset observation sent to the learner.",
    )

    evaluation_group = parser.add_argument_group("evaluation")
    evaluation_group.add_argument("--num-candidates", type=int, default=50)
    evaluation_group.add_argument("--candidate-batch-size", type=int, default=250)
    evaluation_group.add_argument("--num-episodes", type=int, default=500)
    evaluation_group.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    evaluation_group.add_argument("--seed", type=int, default=12345)
    evaluation_group.add_argument("--save-log", type=Path, default=None)
    return parser


def resolve_evaluation_csv_path(
    learner_task_name: str,
    demo_task_name: str,
) -> Path:
    return (
        REPO_ROOT
        / "results"
        / "evaluation"
        / f"{learner_task_name}-{demo_task_name}.csv"
    )


def _resolve_similarity(episode, accepted, key):
    if not accepted:
        return 0.0
    value = episode.get(key, 0.0)
    if value is None:
        return 0.0
    return float(value)


def _evaluation_csv_fieldnames():
    return [
        "learner_task",
        "demo_task",
        "episode_type",
        "episode_result",
        "episode_score",
        "optimal_score",
        "demo_similarity",
        "failed_similarity",
    ]


def _build_evaluation_row(episode):
    accepted = bool(episode.get("accepted", False))
    return {
        "learner_task": episode.get("learner_task_name", ""),
        "demo_task": episode.get("demo_task_name", ""),
        "episode_type": episode.get("episode_type", ""),
        "episode_result": episode.get("episode_result", ""),
        "episode_score": episode.get("episode_score", 0.0),
        "optimal_score": episode.get("optimal_score", 0.0),
        "demo_similarity": _resolve_similarity(
            episode,
            accepted,
            "demo_similarity",
        ),
        "failed_similarity": _resolve_similarity(
            episode,
            accepted,
            "failed_similarity",
        ),
    }


def initialize_evaluation_csv(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_evaluation_csv_fieldnames())
        writer.writeheader()


def append_evaluation_csv_row(path, episode):
    path = Path(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_evaluation_csv_fieldnames())
        writer.writerow(_build_evaluation_row(episode))


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    demo_data_dir = resolve_demo_data_dir(args.demo_data_root, args.demo_task_name)
    learner_motion_checkpoint = resolve_learner_motion_checkpoint(
        args.learner_task_name,
        args.learner_motion_root,
    )

    demonstrator = Demonstrator(
        data_dir=demo_data_dir,
        task_name=args.demo_task_name,
        intention_checkpoint=args.demo_intention_checkpoint,
        device=device,
    )
    learner = Learner(
        task_name=args.learner_task_name,
        intention_checkpoint=args.learner_intention_checkpoint,
        motion_checkpoint=learner_motion_checkpoint,
        latent_dim=args.latent_dim,
        valid_threshold=args.valid_threshold,
        aligned_threshold=args.aligned_threshold,
        device=device,
    )
    runner = Runner(
        demonstrator=demonstrator,
        learner=learner,
        assets_root=args.assets_root,
        learner_mode=args.learner_mode,
        env_display=args.env_display,
        save_learner_image_dir=args.save_learner_image_dir,
    )
    evaluation_csv_path = resolve_evaluation_csv_path(
        args.learner_task_name,
        args.demo_task_name,
    )
    initialize_evaluation_csv(evaluation_csv_path)
    print(f"writing evaluation csv: {evaluation_csv_path}")

    summary = runner.run(
        args.num_episodes,
        num_candidates=args.num_candidates,
        candidate_batch_size=args.candidate_batch_size,
        episode_callback=lambda episode: append_evaluation_csv_row(
            evaluation_csv_path,
            episode,
        ),
    )
    print(f"average episode score: {summary['average_episode_score']:.4f}")
    print(f"average optimal score: {summary['average_optimal_score']:.4f}")
    print(f"saved evaluation csv: {evaluation_csv_path}")

    if args.save_log is not None:
        save_summary_json(summary, args.save_log)


if __name__ == "__main__":
    main()
