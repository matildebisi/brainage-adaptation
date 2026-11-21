#!/usr/bin/env python3
"""Merge per-run predictions into a subject-by-model CSV.

Both development and test predictions use the same naming convention:
    dl_tl_TL_<run>_ft1p0_seed1_pred_age
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PREDICTION_COLUMN = re.compile(r"^dl_tl_.+_pred_age$")
SEED_DIRECTORY = re.compile(r"seed(\d+)$", flags=re.IGNORECASE)
FT_DIRECTORY = re.compile(r"ft\d+p\d+$", flags=re.IGNORECASE)


def shorten_name(name: str) -> str:
    """Apply the naming conventions used in the original result merges."""
    name = str(name)
    for old, new in (
        ("alpha", "a"),
        ("dropout", "d"),
        ("_freezeBN", ""),
        ("freezeBN", ""),
        ("tmdense2_4_", "d24_"),
        ("tmdense4_", "d4_"),
        ("tmNone_", ""),
    ):
        name = name.replace(old, new)
    return name


def locate_run(path: Path) -> tuple[str, int, str]:
    """Find the run, fold seed and fine-tuning tag in a result file path."""
    names = [p.name for p in path.parents]
    run_tags = [p for p in names if p.startswith("TL_")]
    seeds = [int(m.group(1)) for p in names if (m := SEED_DIRECTORY.fullmatch(p))]
    ft_tags = [p for p in names if FT_DIRECTORY.fullmatch(p)]
    if len(run_tags) != 1 or len(seeds) != 1 or len(ft_tags) != 1:
        raise ValueError(
            f"Expected TL_<run>/seedN/ftXpY in the path: {path}"
        )
    return run_tags[0], seeds[0], ft_tags[0]


def output_column(input_column: str, ft_tag: str, seed: int, mode: str) -> str:
    """Generate consistent prediction column names for development and test."""
    stem = shorten_name(input_column.removesuffix("_pred_age"))
    ft_suffix = f"_{ft_tag}"

    # Ensure that the fine-tuning tag appears exactly once.
    while stem.endswith(ft_suffix):
        stem = stem[:-len(ft_suffix)]

    return f"{stem}{ft_suffix}_seed{seed}_pred_age"

def merge_predictions(
    runs_dir: Path,
    base_csv: Path,
    output: Path,
    mode: str,
    subject_col: str = "SUBJ_ID",
    ft_filter: str | None = None,
) -> pd.DataFrame:
    if mode not in ("test", "development"):
        raise ValueError("mode must be 'test' or 'development'")

    base = pd.read_csv(base_csv, dtype={subject_col: str}, low_memory=False)
    if subject_col not in base.columns:
        raise ValueError(f"{base_csv} has no '{subject_col}' column")
    if base[subject_col].isna().any() or base[subject_col].duplicated().any():
        raise ValueError(f"'{subject_col}' must uniquely identify each row in {base_csv}")

    glob_pattern = "test_TL*.csv" if mode == "test" else "development_predictions.csv"
    sources = sorted(runs_dir.rglob(glob_pattern))
    if not sources:
        raise FileNotFoundError(f"No {glob_pattern} files found under {runs_dir}")

    merged = base.set_index(subject_col, verify_integrity=True)
    base_ids = set(merged.index)
    added_columns: set[str] = set()
    skipped = 0

    for source in sources:
        run_tag, seed, ft_tag = locate_run(source)
        if ft_filter is not None and ft_tag != ft_filter:
            skipped += 1
            continue

        per_run = pd.read_csv(source, dtype={subject_col: str}, low_memory=False)
        if subject_col not in per_run.columns:
            raise ValueError(f"{source} has no '{subject_col}' column")
        if per_run[subject_col].isna().any() or per_run[subject_col].duplicated().any():
            raise ValueError(f"Duplicate or missing IDs in {source}; merge would be ambiguous")

        pred_cols = [c for c in per_run if PREDICTION_COLUMN.fullmatch(c)]
        if len(pred_cols) != 1:
            raise ValueError(f"Expected one dl_tl_*_pred_age column in {source}, got {pred_cols}")

        pred_col = pred_cols[0]
        expected_prefix = f"dl_tl_{run_tag}_"
        if not pred_col.startswith(expected_prefix):
            raise ValueError(
                f"Prediction column '{pred_col}' does not match run directory '{run_tag}'"
            )

        merged_col = output_column(pred_col, ft_tag, seed, mode)
        if merged_col in merged.columns or merged_col in added_columns:
            raise ValueError(f"Duplicate prediction column '{merged_col}' from {source}")

        unknown_ids = set(per_run[subject_col]) - base_ids
        if unknown_ids:
            raise ValueError(
                f"{source} contains {len(unknown_ids)} IDs missing from the base CSV"
            )

        values = pd.to_numeric(per_run.set_index(subject_col)[pred_col], errors="coerce")
        invalid = values.isna() & per_run.set_index(subject_col)[pred_col].notna()
        if invalid.any():
            raise ValueError(f"Non-numeric predictions in '{pred_col}' of {source}")
        merged[merged_col] = values.reindex(merged.index)
        added_columns.add(merged_col)

    if not added_columns:
        raise ValueError(f"No matching prediction files were merged ({skipped} skipped)")

    output.parent.mkdir(parents=True, exist_ok=True)
    merged = merged.reset_index()
    merged.to_csv(output, index=False)
    print(f"Merged {len(added_columns)} run/seed columns across {len(merged)} subjects")
    print(f"Saved to: {output}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("test", "development"), required=True)
    parser.add_argument("--subject-col", default="SUBJ_ID")
    parser.add_argument("--ft-tag", default="ft1p0", help="Only merge runs with this FT tag")
    args = parser.parse_args()
    merge_predictions(
        args.runs_dir, args.base_csv, args.output, args.mode,
        subject_col=args.subject_col, ft_filter=args.ft_tag,
    )


if __name__ == "__main__":
    main()
