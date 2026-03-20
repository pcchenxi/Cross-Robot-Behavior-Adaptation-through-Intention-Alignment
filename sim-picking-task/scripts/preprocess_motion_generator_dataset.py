#!/usr/bin/env python3
"""Precompute processed_color fields for motion-generator datasets."""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import crop


def crop_img(image):
    return crop(image, 100, 100, 400, 450)


def build_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(crop_img),
            transforms.Resize((size, size)),
        ]
    )


def process_color_episode(color_episode: np.ndarray, transform) -> np.ndarray:
    if color_episode.ndim != 5:
        raise ValueError(
            f"Expected color episode with shape (T, C, H, W, 3), got {color_episode.shape}."
        )

    processed_steps = []
    for step in color_episode:
        processed_cameras = []
        for image in step:
            processed_cameras.append(transform(image).numpy())
        processed_steps.append(np.stack(processed_cameras, axis=0))
    return np.stack(processed_steps, axis=0).astype(np.float32)


def process_split(
    split_dir: Path, transform, overwrite: bool, limit: Optional[int]
):
    color_dir = split_dir / "color"
    output_dir = split_dir / "processed_color"

    if not color_dir.is_dir():
        return 0, 0

    output_dir.mkdir(exist_ok=True)

    processed = 0
    skipped = 0
    for color_path in sorted(color_dir.glob("*.pkl")):
        output_path = output_dir / color_path.name
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        with color_path.open("rb") as f:
            color_episode = pickle.load(f)

        processed_color = process_color_episode(np.asarray(color_episode), transform)

        with output_path.open("wb") as f:
            pickle.dump(processed_color, f, protocol=pickle.HIGHEST_PROTOCOL)

        processed += 1
        if limit is not None and processed >= limit:
            break

    return processed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute processed_color pickles for datasets in data_paper."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_paper"),
        help="Root folder containing dataset splits such as ur5f-train.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output square image size after crop and resize.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate processed_color files even if they already exist.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        help="Optional split folder names to process, for example ur5f-train ur5f-test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional per-split cap for quick verification runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = build_transform(args.size)

    split_dirs = sorted(path for path in args.data_root.iterdir() if path.is_dir())
    if args.splits:
        requested = set(args.splits)
        split_dirs = [path for path in split_dirs if path.name in requested]
    if not split_dirs:
        raise FileNotFoundError(f"No dataset split directories found in {args.data_root}.")

    total_processed = 0
    total_skipped = 0
    for split_dir in split_dirs:
        processed, skipped = process_split(
            split_dir, transform, args.overwrite, args.limit
        )
        total_processed += processed
        total_skipped += skipped
        if processed or skipped:
            print(f"{split_dir}: processed={processed} skipped={skipped}")

    print(
        f"Finished preprocessing {args.data_root}: processed={total_processed} skipped={total_skipped}"
    )


if __name__ == "__main__":
    main()
