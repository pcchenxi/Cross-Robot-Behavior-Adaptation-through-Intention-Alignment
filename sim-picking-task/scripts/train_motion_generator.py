"""Train the motion-generator CVAE on the local picking dataset."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm

from iail_sim_picking.motion_generator.data_loader import create_dataloader
from iail_sim_picking.motion_generator.motion_generator_trainer import MotionGenerator


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data_paper",
        help="Fast dataset root containing converted <task-name>-train folders.",
    )
    parser.add_argument(
        "--task-name",
        choices=("ur5f", "ur5l", "ur5r"),
        required=True,
        help="Task name used for dataset selection and camera routing.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        choices=(3,),
        default=3,
        help="Action dimension for ee_state. Only 3 is supported.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs when num_workers > 0.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "motion_generator",
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Checkpoint frequency in epochs.",
    )
    parser.add_argument(
        "--init-stats",
        action="store_true",
        help="Recompute action normalization statistics before training.",
    )
    return parser


def resolve_dataset_path(args):
    return args.data_root / f"{args.task_name}-train"


def train_epoch(model, train_loader, device, epoch_idx, num_epochs):
    model.train()
    running_recons = 0.0
    running_kl = 0.0
    num_batches = 0

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch_idx}/{num_epochs}",
    )

    for batch in progress:
        image = batch["image"].float().to(device, non_blocking=True)
        action = batch["action"].float().to(device, non_blocking=True)

        recons_loss, kl_loss = model.train_once(image, action)
        running_recons += recons_loss
        running_kl += kl_loss
        num_batches += 1

        progress.set_postfix(
            recons_loss=f"{running_recons / num_batches:.6f}",
            kl_loss=f"{running_kl / num_batches:.6f}",
        )

    return {
        "recons_loss": running_recons / max(num_batches, 1),
        "kl_loss": running_kl / max(num_batches, 1),
    }


def save_checkpoint(model, output_dir, checkpoint_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_name, str(output_dir))


def main():
    args = build_argparser().parse_args()
    dataset_path = resolve_dataset_path(args)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    action_dim = args.action_dim

    print(f"dataset_path: {dataset_path}")
    print(f"device: {device}")
    print(f"task_name: {args.task_name}")
    print(f"action_dim: {action_dim}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"persistent_workers: {args.persistent_workers}")

    train_loader = create_dataloader(
        data_dir=str(dataset_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        task_name=args.task_name,
        epoch_step_num=args.steps_per_epoch,
        init_stats=args.init_stats,
    )

    model = MotionGenerator(
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        device=device,
        vae_lr=args.lr,
    ).to(device)

    checkpoint_prefix = (
        f"cvae_{args.task_name}"
    )

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, train_loader, device, epoch, args.epochs)
        print(
            f"epoch={epoch} "
            f"recons_loss={metrics['recons_loss']:.6f} "
            f"kl_loss={metrics['kl_loss']:.6f}"
        )

        save_checkpoint(model, args.output_dir, f"{checkpoint_prefix}_latest")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(model, args.output_dir, f"{checkpoint_prefix}_epoch{epoch:04d}")


if __name__ == "__main__":
    main()
