#!/usr/bin/env python3
"""Build a fast sharded motion-generator dataset from raw episode pickles."""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import crop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CAMERA_INDEX = {
    "ur5f": 0,
    "ur5l": 1,
    "ur5r": 2,
}


def crop_img(image):
    return crop(image, 100, 100, 400, 450)


def build_transform(size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(crop_img),
            transforms.Resize((size, size)),
        ]
    )


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(path, value):
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_sample(split_dir, file_name, task_name, transform):
    color = np.asarray(load_pickle(split_dir / "color" / file_name))
    ee_state = load_pickle(split_dir / "ee_state" / file_name)
    info = load_pickle(split_dir / "info" / file_name)

    if color.ndim != 5:
        raise ValueError(f"Unexpected color shape for {file_name}: {color.shape}")
    if len(ee_state) < 2:
        raise IndexError(
            f"Episode {file_name} does not contain the second timestep required for training."
        )

    image = transform(color[0][CAMERA_INDEX[task_name]]).numpy().astype(np.float32)
    action = np.asarray(ee_state[1], dtype=np.float32)

    return image, action, info


def iter_split_files(split_dir):
    action_dir = split_dir / "action"
    if not action_dir.is_dir():
        raise FileNotFoundError(f"Expected action directory in {split_dir}")
    return sorted(path.name for path in action_dir.glob("*.pkl"))


def write_manifest(output_dir, manifest):
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def convert_split(split_dir, output_root, shard_size, image_size, overwrite, limit):
    split_name = split_dir.name
    task_name = split_name.split("-")[0]
    if task_name not in CAMERA_INDEX:
        raise ValueError(f"Unsupported task name inferred from split {split_name}")

    output_dir = output_root / split_name
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists at {manifest_path}. Use --overwrite to rebuild."
        )

    transform = build_transform(image_size)
    file_names = iter_split_files(split_dir)
    if limit is not None:
        file_names = file_names[:limit]
    total_samples = len(file_names)
    num_shards = int(math.ceil(total_samples / float(shard_size))) if total_samples else 0

    shard_entries = []
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(total_samples, start + shard_size)
        shard_files = file_names[start:end]

        images = []
        actions = []
        infos = []
        sample_ids = []
        for file_name in shard_files:
            image, action, info_payload = build_sample(
                split_dir, file_name, task_name, transform
            )
            images.append(image)
            actions.append(action)
            infos.append(info_payload)
            sample_ids.append(file_name)

        shard_payload = {
            "images": np.stack(images, axis=0).astype(np.float32),
            "actions": np.stack(actions, axis=0).astype(np.float32),
            "info": infos,
            "sample_ids": sample_ids,
        }
        shard_file = shards_dir / f"shard_{shard_idx:05d}.pkl"
        dump_pickle(shard_file, shard_payload)

        shard_entries.append(
            {
                "file": str(shard_file.relative_to(output_dir)),
                "start_idx": start,
                "end_idx": end,
                "num_samples": end - start,
            }
        )
        print(f"{split_name}: wrote {shard_file.name} with {end - start} samples")

    manifest = {
        "task_name": task_name,
        "split_name": split_name,
        "num_samples": total_samples,
        "num_shards": num_shards,
        "camera_index": CAMERA_INDEX[task_name],
        "image_shape": [3, image_size, image_size],
        "action_shape": [3],
        "transform": {
            "to_tensor": True,
            "crop": [100, 100, 400, 450],
            "resize": [image_size, image_size],
        },
        "shards": shard_entries,
    }
    write_manifest(output_dir, manifest)
    return total_samples, num_shards


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data_paper",
        help="Root containing raw splits such as ur5f-train.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data_fast",
        help="Root where fast sharded splits will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        help="Optional split names to convert, for example ur5f-train ur5f-test.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1024,
        help="Number of samples per shard file.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Final square image size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing converted split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional per-split sample cap for smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split_dirs = sorted(path for path in args.raw_root.iterdir() if path.is_dir())
    if args.splits:
        requested = set(args.splits)
        split_dirs = [path for path in split_dirs if path.name in requested]

    if not split_dirs:
        raise FileNotFoundError(f"No split directories found in {args.raw_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    for split_dir in split_dirs:
        total_samples, num_shards = convert_split(
            split_dir=split_dir,
            output_root=args.output_root,
            shard_size=args.shard_size,
            image_size=args.image_size,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        print(
            f"{split_dir.name}: total_samples={total_samples} num_shards={num_shards}"
        )


if __name__ == "__main__":
    main()
