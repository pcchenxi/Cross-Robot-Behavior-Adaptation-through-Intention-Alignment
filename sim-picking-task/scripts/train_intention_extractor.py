"""Train the CLIP-style intention extractor on the local picking dataset."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm

from iail_sim_picking.intention_extractor.data_loader import create_dataloader
from iail_sim_picking.intention_extractor.intention_extractor_trainer import (
    IntentionExtractor,
)

TASK_NAMES = ("ur5f", "ur5l", "ur5r")


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data_paper",
        help="Dataset root containing <task-name>-train folders.",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs when num_workers > 0.",
    )
    parser.add_argument("--epochs", type=int, default=301)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, for example cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "intention_extractor",
        help="Directory for checkpoints.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--init-stats",
        action="store_true",
        help="Recompute action normalization statistics before training.",
    )
    parser.add_argument("--anotate-prob", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--text-encoder-model",
        default="distilbert-base-uncased",
        help="Transformers model name for text encoding.",
    )
    return parser


def resolve_dataset_paths(data_root):
    dataset_paths = {}
    for task_name in TASK_NAMES:
        dataset_path = data_root / f"{task_name}-train"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        dataset_paths[task_name] = dataset_path
    return dataset_paths


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "caption"
    }


def create_train_loaders(args, dataset_paths):
    return {
        task_name: create_dataloader(
            data_dir=str(dataset_paths[task_name]),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            task_name=task_name,
            epoch_step_num=args.steps_per_epoch,
            init_stats=args.init_stats,
            action_dim=3,
            anotate_prob=args.anotate_prob,
            latent_dim=args.latent_dim,
        )
        for task_name in TASK_NAMES
    }


def save_checkpoint(model, output_dir, checkpoint_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_name, str(output_dir))


def train_epoch(model, train_loaders, device, epoch_idx, num_epochs):
    model.train()
    running = {"loss_img": 0.0, "loss_text": 0.0, "loss": 0.0}
    num_batches = 0

    main_loader = train_loaders[TASK_NAMES[0]]
    aux_loaders = [train_loaders[task_name] for task_name in TASK_NAMES[1:]]
    progress = tqdm(
        zip(main_loader, *aux_loaders),
        total=len(main_loader),
        desc=f"Epoch {epoch_idx}/{num_epochs}",
    )

    for batches in progress:
        batch_by_task = {
            task_name: move_batch_to_device(batch, device)
            for task_name, batch in zip(TASK_NAMES, batches)
        }
        metrics = model.train_once(batch_by_task)
        num_batches += 1

        for key in running:
            running[key] += metrics[key]

        progress.set_postfix(
            loss_img=f"{running['loss_img'] / num_batches:.6f}",
            loss_text=f"{running['loss_text'] / num_batches:.6f}",
            loss=f"{running['loss'] / num_batches:.6f}",
        )

    return {
        key: value / max(num_batches, 1)
        for key, value in running.items()
    }


def main():
    args = build_argparser().parse_args()
    dataset_paths = resolve_dataset_paths(args.data_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print(f"device: {device}")
    print(f"data_root: {args.data_root}")
    print(f"tasks: {TASK_NAMES}")
    print(f"num_workers: {args.num_workers}")
    print(f"batch_size: {args.batch_size}")
    print(f"persistent_workers: {args.persistent_workers}")

    train_loaders = create_train_loaders(args, dataset_paths)
    model = IntentionExtractor(
        action_dim=3,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        text_encoder_model=args.text_encoder_model,
        pretrained=False,
        trainable=True,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)

    checkpoint_prefix = "clip_intention_extractor"
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, train_loaders, device, epoch, args.epochs)
        print(
            f"epoch={epoch} "
            f"loss_img={metrics['loss_img']:.6f} "
            f"loss_text={metrics['loss_text']:.6f} "
            f"loss={metrics['loss']:.6f}"
        )

        save_checkpoint(model, args.output_dir, f"{checkpoint_prefix}_latest")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                model,
                args.output_dir,
                f"{checkpoint_prefix}_epoch{epoch:04d}",
            )


if __name__ == "__main__":
    main()
