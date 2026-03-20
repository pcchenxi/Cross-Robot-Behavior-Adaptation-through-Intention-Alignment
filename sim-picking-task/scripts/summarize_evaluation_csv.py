"""Summarize average episode_score and optimal_score from evaluation CSV files."""

import argparse
import csv
from pathlib import Path
from typing import Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read evaluation CSV files and report average episode_score and "
            "average optimal_score."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Directory containing evaluation CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern used to find CSV files inside the input directory.",
    )
    return parser.parse_args()


def compute_file_averages(csv_path: Path) -> Tuple[int, float, float]:
    row_count = 0
    episode_total = 0.0
    optimal_total = 0.0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"episode_score", "optimal_score"}
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        for row in reader:
            episode_total += float(row["episode_score"])
            optimal_total += float(row["optimal_score"])
            row_count += 1

    if row_count == 0:
        raise ValueError(f"{csv_path} does not contain any data rows")

    return row_count, episode_total / row_count, optimal_total / row_count


def main() -> None:
    args = parse_args()
    csv_files = sorted(args.input_dir.glob(args.pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {args.input_dir} matching {args.pattern}"
        )

    total_rows = 0
    total_episode = 0.0
    total_optimal = 0.0

    print(f"Scanning {len(csv_files)} CSV file(s) in {args.input_dir}")
    print()

    for csv_file in csv_files:
        row_count, avg_episode, avg_optimal = compute_file_averages(csv_file)
        total_rows += row_count
        total_episode += avg_episode * row_count
        total_optimal += avg_optimal * row_count

        print(
            f"{csv_file.name}: rows={row_count}, "
            f"average episode_score={avg_episode:.6f}, "
            f"average optimal_score={avg_optimal:.6f}"
        )

    print()
    print(f"Overall rows={total_rows}")
    print(f"Overall average episode_score={total_episode / total_rows:.6f}")
    print(f"Overall average optimal_score={total_optimal / total_rows:.6f}")


if __name__ == "__main__":
    main()
