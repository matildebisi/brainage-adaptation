import json
from pathlib import Path

import pandas as pd


# ----------------------------
# CONFIG
# ----------------------------

import argparse
_parser = argparse.ArgumentParser(description='Summarise completed fine-tuning runs.')
_parser.add_argument('--runs-dir', required=True, type=Path)
_parser.add_argument('--output', required=True, type=Path)
_parser.add_argument('--ft-tag', default='ft1p0')
_args = _parser.parse_args()
PARENT_DIR = _args.runs_dir
FT_TAG = _args.ft_tag
OUT_CSV = _args.output


# ----------------------------
# FUNCTIONS
# ----------------------------

def find_model_dirs(parent_dir: Path) -> list[Path]:
    """Find folders whose name starts with TL_TP."""
    return sorted(
        path for path in parent_dir.iterdir()
        if path.is_dir() and path.name.startswith("TL_TP")
    )

def extract_metrics(summary_file: Path) -> dict | None:
    """Extract metrics from one TL_summary.json."""
    with summary_file.open("r") as f:
        summary = json.load(f)

    stages = summary.get("stages", [])

    if not stages:
        print(f"Skipped empty stages: {summary_file}")
        return None

    stage = stages[-1]

    return {
        "best_val_loss": stage["best_val_loss"],
        "best_R2": stage["best_R2"],
        "best_corr": stage["best_corr"],
        "best_epoch": stage["best_epoch"],
        "trainable_pct": stage["trainable_pct"],
        "duration_min": stage["duration_min"],
        "duration_per_epoch_s": stage.get("mean_seconds_per_epoch", float("nan")),
    }


def summarize_model(model_dir: Path, ft_tag: str) -> dict | None:
    """Calculate mean and standard deviation across valid seeds."""
    summary_files = sorted(
        model_dir.glob(f"seed*/{ft_tag}/TL_summary.json")
    )

    if not summary_files:
        print(f"Skipped {model_dir.name}: no TL_summary.json found")
        return None

    metrics = []

    for summary_file in summary_files:
        result = extract_metrics(summary_file)

        if result is not None:
            metrics.append(result)

    if not metrics:
        print(f"Skipped {model_dir.name}: no valid seeds")
        return None

    seed_metrics = pd.DataFrame(metrics)

    result = {
        "model": model_dir.name,
        "n_seeds": len(seed_metrics),
        "n_seeds_found": len(summary_files),
    }

    for metric in seed_metrics.columns:
        result[f"{metric}_mean"] = seed_metrics[metric].mean()
        result[f"{metric}_std"] = seed_metrics[metric].std()

    return result

def create_summary_csv(
    parent_dir: Path,
    ft_tag: str,
    out_csv: Path,
) -> pd.DataFrame:
    """Create and save the summary table for all TL_TP folders."""
    rows = []

    for model_dir in find_model_dirs(parent_dir):
        result = summarize_model(model_dir, ft_tag)
        if result is not None:
            rows.append(result)

    if not rows:
        raise RuntimeError(
            f"No valid TL_summary.json files found in {parent_dir}"
        )

    summary_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    return summary_df


# ----------------------------
# MAIN
# ----------------------------

def main():
    summary_df = create_summary_csv(
        parent_dir=PARENT_DIR,
        ft_tag=FT_TAG,
        out_csv=OUT_CSV,
    )

    print(f"\nSaved: {OUT_CSV}")
    print(f"Models found: {len(summary_df)}")
    print(summary_df)


if __name__ == "__main__":
    main()