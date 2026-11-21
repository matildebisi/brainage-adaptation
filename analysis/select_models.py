#!/usr/bin/env python3
"""
Build validation metrics and select fine-tuned models using development folds.

Input:
  1) A wide CSV containing chronological age and prediction columns, e.g.
     lasidad_validation_results.csv
  2) A split directory containing files such as:
     LASI-DAD_fold_1_base_train_rows.csv
     LASI-DAD_fold_1_base_validation_rows.csv
     ...
  3) An output directory.

Outputs:
  - train_mae_validation_metrics_per_fold.csv
  - train_mae_validation_metrics_summary.csv
  - selected_models_validation_<TP>.csv, one for each TP/base model
  - selected_models_validation_all_TPs.csv

The selected-model files include the zero-shot row plus the best model of each
category, selected by lowest validation_MAE_mean.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


PRED_RE = re.compile(r"^dl_tl_(?P<model>.+)_seed(?P<seed>\d+)_pred_age$")

METHOD_RE = re.compile(
    r"^(?P<base>.+?)_"
    r"(?P<method>FilteredLoRA|LoRA|DoRA|PiSSA|dense2_4|dense4|head|all|none)"
    r"(?P<rest>_.*)?$"
)

TRADITIONAL_METHODS = {"head", "dense4", "dense2_4", "all", "none"}


def safe_filename(text: str) -> str:
    """Make a string safe for use in filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def strip_tl_prefix(model: str) -> str:
    """Remove leading TL_ if present."""
    return model[3:] if model.startswith("TL_") else model


def parse_model_name(raw_model: str) -> Dict[str, object]:
    """
    Parse a model string such as:
      TL_TP1_White_LoRA_btFalse_min_r16_d24_ft1p0
      TL_TP1_White_FilteredLoRA_btFalse_min_r8_d24_f5_sigmaS1.5_scale0.2_ft1p0
      TL_TP3_dense2_4_btFalse_min_ft1p0
    """
    no_tl = strip_tl_prefix(raw_model)
    m = METHOD_RE.match(no_tl)

    if not m:
        base = no_tl.split("_btFalse")[0]
        method = "unknown"
    else:
        base = m.group("base")
        method = m.group("method")

    if method == "FilteredLoRA":
        technique = "FiltLoRA"
        category = "FiltLoRA"
    elif method in {"LoRA", "DoRA", "PiSSA"}:
        technique = method
        category = method
    elif method in TRADITIONAL_METHODS:
        technique = "Traditional"
        category = "Traditional" if method != "none" else "Zero-shot"
    else:
        technique = "Unknown"
        category = "Unknown"

    rank = np.nan
    rank_m = re.search(r"_r(\d+)(?:_|$)", no_tl)
    if rank_m:
        rank = int(rank_m.group(1))

    if "_d24" in no_tl:
        target_code = "d24"
        target_module = "dense2_4"
    elif "_d4" in no_tl:
        target_code = "d4"
        target_module = "dense4"
    elif method in {"head", "dense4", "dense2_4", "all", "none"}:
        target_code = method
        target_module = method
    else:
        target_code = "all"
        target_module = "all"

    filt_kernel = np.nan
    filt_sigma = np.nan
    filt_scale = np.nan

    km = re.search(r"_f(\d+)(?:_|$)", no_tl)
    sm = re.search(r"_sigmaS([0-9.]+)(?:_|$)", no_tl)
    scm = re.search(r"_scale([0-9.]+)(?:_|$)", no_tl)

    if km:
        filt_kernel = int(km.group(1))
    if sm:
        filt_sigma = sm.group(1)
    if scm:
        filt_scale = scm.group(1)

    return {
        "raw_model": raw_model,
        "model_no_tl": no_tl,
        "base_model": base,
        "method": method,
        "technique": technique,
        "category": category,
        "rank": rank,
        "target_code": target_code,
        "target_module": target_module,
        "filter_kernel_size": filt_kernel,
        "filter_sigma_spatial": filt_sigma,
        "filter_scale": filt_scale,
    }


def _not_missing(x: object) -> bool:
    return not pd.isna(x)


def make_label(meta: Dict[str, object]) -> str:
    """Human-readable label used in selected_models_validation files."""
    category = str(meta["category"])
    method = str(meta["method"])
    target_code = str(meta["target_code"])
    rank = meta["rank"]

    if category == "Zero-shot":
        return "Traditional | zero-shot"

    if category == "Traditional":
        return f"Traditional | {method}"

    if category in {"LoRA", "DoRA", "PiSSA"}:
        bits = [category]
        if _not_missing(rank):
            bits.append(f"r{int(rank)}")
        bits.append(target_code)
        return " | ".join(bits)

    if category == "FiltLoRA":
        bits = ["FiltLoRA"]
        if _not_missing(rank):
            bits.append(f"r{int(rank)}")
        bits.append(target_code)
        if _not_missing(meta.get("filter_kernel_size")):
            bits.append(f"f{int(meta['filter_kernel_size'])}")
        if _not_missing(meta.get("filter_sigma_spatial")):
            bits.append(f"σ{meta['filter_sigma_spatial']}")
        if _not_missing(meta.get("filter_scale")):
            bits.append(f"scale{meta['filter_scale']}")
        return " | ".join(bits)

    return str(meta["model_no_tl"])


def make_validation_model_name(raw_model: str) -> str:
    """
    Convert a prediction-model name into the validation/training run name.

    Example:
      TL_TP1_White_LoRA_btFalse_min_r16_d24_ft1p0
      -> TL_TP1_White_LoRA_btFalse_min_r16_tmdense2_4
    """
    name = raw_model
    name = re.sub(r"_ft[0-9]+p[0-9]+$", "", name)
    name = name.replace("_d24", "_tmdense2_4")
    name = name.replace("_d4", "_tmdense4")
    return name


def load_split_rows(
    split_dir: Path,
    split_prefix: str,
    fold: int,
    subject_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation row CSVs for one fold."""
    train_path = split_dir / f"{split_prefix}_fold_{fold}_base_train_rows.csv"
    val_path = split_dir / f"{split_prefix}_fold_{fold}_base_validation_rows.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split for fold {fold}: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation split for fold {fold}: {val_path}")

    train_rows = pd.read_csv(train_path)
    val_rows = pd.read_csv(val_path)

    if subject_col not in train_rows.columns:
        raise ValueError(f"{train_path} does not contain subject column '{subject_col}'")
    if subject_col not in val_rows.columns:
        raise ValueError(f"{val_path} does not contain subject column '{subject_col}'")

    return train_rows, val_rows


def subset_by_subject(
    all_df: pd.DataFrame,
    rows_df: pd.DataFrame,
    subject_col: str,
) -> pd.DataFrame:
    """Subset the wide prediction table using subjects listed in a split rows file."""
    keep = set(rows_df[subject_col].astype(str))
    return all_df[all_df[subject_col].astype(str).isin(keep)]


def compute_metrics(df: pd.DataFrame, pred_col: str, age_col: str) -> Dict[str, float]:
    """Compute MAE, BAG, R2 and Pearson correlation for one subset."""
    valid = df[[age_col, pred_col]].dropna().copy()
    n = int(len(valid))

    if n == 0:
        return {"n": 0, "MAE": np.nan, "BAG": np.nan, "R2": np.nan, "corr": np.nan}

    y = valid[age_col].astype(float).to_numpy()
    yhat = valid[pred_col].astype(float).to_numpy()

    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    bag = float(np.mean(err))

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    if n > 1 and np.std(y) > 0 and np.std(yhat) > 0:
        corr = float(np.corrcoef(y, yhat)[0, 1])
    else:
        corr = np.nan

    return {"n": n, "MAE": mae, "BAG": bag, "R2": r2, "corr": corr}


def discover_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per prediction column with parsed model and seed."""
    rows = []
    for col in df.columns:
        m = PRED_RE.match(col)
        if not m:
            continue

        raw_model = m.group("model")
        seed = int(m.group("seed"))
        meta = parse_model_name(raw_model)

        rows.append({"prediction_column": col, "raw_model": raw_model, "seed": seed, **meta})

    pred_info = pd.DataFrame(rows)
    if pred_info.empty:
        raise ValueError("No prediction columns matching 'dl_tl_<model>_seed<k>_pred_age' were found.")

    return pred_info.sort_values(["base_model", "category", "raw_model", "seed"])


def build_per_fold_metrics(
    val_df: pd.DataFrame,
    pred_info: pd.DataFrame,
    split_dir: Path,
    split_prefix: str,
    subject_col: str,
    age_col: str,
) -> pd.DataFrame:
    """
    Build per-fold metrics. By default seed number is interpreted as fold number:
    seed1 -> fold 1, seed2 -> fold 2, etc.
    """
    split_cache: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    out_rows = []

    for row in pred_info.itertuples(index=False):
        seed = int(row.seed)
        fold = seed

        if fold not in split_cache:
            train_rows, val_rows = load_split_rows(split_dir, split_prefix, fold, subject_col)
            split_cache[fold] = (train_rows, val_rows)
        else:
            train_rows, val_rows = split_cache[fold]

        train_df = subset_by_subject(val_df, train_rows, subject_col)
        val_fold_df = subset_by_subject(val_df, val_rows, subject_col)

        train_met = compute_metrics(train_df, row.prediction_column, age_col)
        val_met = compute_metrics(val_fold_df, row.prediction_column, age_col)

        out_rows.append(
            {
                "base_model": row.base_model,
                "category": row.category,
                "technique": row.technique,
                "method": row.method,
                "target_module": row.target_module,
                "target_code": row.target_code,
                "rank": row.rank,
                "filter_kernel_size": row.filter_kernel_size,
                "filter_sigma_spatial": row.filter_sigma_spatial,
                "filter_scale": row.filter_scale,
                "model": row.raw_model,
                "model_no_tl": row.model_no_tl,
                "seed": seed,
                "fold": fold,
                "prediction_column": row.prediction_column,
                "n_train": train_met["n"],
                "n_validation": val_met["n"],
                "train_MAE": train_met["MAE"],
                "validation_MAE": val_met["MAE"],
                "validation_BAG": val_met["BAG"],
                "validation_R2": val_met["R2"],
                "validation_corr": val_met["corr"],
            }
        )

    return pd.DataFrame(out_rows)


def build_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics across folds."""
    index_cols = [
        "base_model",
        "category",
        "technique",
        "method",
        "target_module",
        "target_code",
        "rank",
        "filter_kernel_size",
        "filter_sigma_spatial",
        "filter_scale",
        "model",
        "model_no_tl",
    ]

    metric_cols = [
        "train_MAE",
        "validation_MAE",
        "validation_BAG",
        "validation_R2",
        "validation_corr",
    ]

    agg_dict = {
        "n_folds": ("fold", "nunique"),
        "n_train_mean": ("n_train", "mean"),
        "n_validation_mean": ("n_validation", "mean"),
    }

    for metric in metric_cols:
        agg_dict[f"{metric}_mean"] = (metric, "mean")
        agg_dict[f"{metric}_std"] = (metric, "std")

    summary = per_fold.groupby(index_cols, dropna=False, as_index=False).agg(**agg_dict)
    return summary.sort_values(["base_model", "category", "validation_MAE_mean", "model"])


def build_selected_for_base(
    summary: pd.DataFrame,
    base_model: str,
    score_scale: float = 1.0,
    include_zero_shot: bool = True,
) -> pd.DataFrame:
    """Select best model of each category for one base model using lowest validation_MAE_mean."""
    sub = summary[summary["base_model"] == base_model].copy()
    rows = []

    if include_zero_shot:
        rows.append(
            {
                "category": "Zero-shot",
                "technique": "Traditional",
                "validation_model": f"TL_{base_model}_none_btFalse_min",
                "model": f"{base_model}_none_btFalse_min_ft1p0",
                "label": "Traditional | zero-shot",
                "selection_metric": "validation_MAE_mean",
                "validation_score": np.nan,
            }
        )

    for category in ["Traditional", "LoRA", "DoRA", "PiSSA", "FiltLoRA"]:
        cat_df = sub[sub["category"] == category].copy()
        if cat_df.empty:
            continue

        if category == "Traditional":
            cat_df = cat_df[cat_df["method"] != "none"]
            if cat_df.empty:
                continue

        cat_df = cat_df.sort_values(
            ["validation_MAE_mean", "validation_R2_mean", "model"],
            ascending=[True, False, True],
            na_position="last",
        )
        best = cat_df.iloc[0].to_dict()
        raw_model = str(best["model"])

        rows.append(
            {
                "category": category,
                "technique": best["technique"],
                "validation_model": make_validation_model_name(raw_model),
                "model": strip_tl_prefix(raw_model),
                "label": make_label(best),
                "selection_metric": "validation_MAE_mean",
                "validation_score": best["validation_MAE_mean"] * score_scale,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute validation metrics and selected models from LASI-DAD prediction CSV."
    )
    parser.add_argument("--validation_results", required=True, type=Path)
    parser.add_argument("--split_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--split_prefix", default="dataset")
    parser.add_argument("--subject_col", default="SUBJ_ID")
    parser.add_argument("--age_col", default="AGE")
    parser.add_argument(
        "--selected_score_scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier for validation_score in selected_models files. "
            "Use 0.01 if you want old loss-like values around 0.04 instead of MAE in years."
        ),
    )
    parser.add_argument("--no_zero_shot", action="store_true")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading validation results:")
    print(f"  {args.validation_results}")
    val_df = pd.read_csv(args.validation_results)

    for col in [args.subject_col, args.age_col]:
        if col not in val_df.columns:
            raise ValueError(f"Validation-results CSV does not contain required column '{col}'")

    print("Discovering prediction columns...")
    pred_info = discover_prediction_columns(val_df)
    pred_info_path = args.out_dir / "prediction_columns_parsed.csv"
    pred_info.to_csv(pred_info_path, index=False)
    print(f"  Found {len(pred_info)} prediction columns.")
    print(f"  Saved parsed column metadata to: {pred_info_path}")

    print("Computing per-fold metrics...")
    per_fold = build_per_fold_metrics(
        val_df=val_df,
        pred_info=pred_info,
        split_dir=args.split_dir,
        split_prefix=args.split_prefix,
        subject_col=args.subject_col,
        age_col=args.age_col,
    )

    per_fold_path = args.out_dir / "train_mae_validation_metrics_per_fold.csv"
    per_fold.to_csv(per_fold_path, index=False)
    print(f"  Saved: {per_fold_path}")

    print("Computing summary metrics...")
    summary = build_summary(per_fold)
    summary_path = args.out_dir / "train_mae_validation_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    print("Selecting best models per TP/base model...")
    selected_all = []
    base_models = sorted(summary["base_model"].dropna().unique())

    for base in base_models:
        selected = build_selected_for_base(
            summary=summary,
            base_model=base,
            score_scale=args.selected_score_scale,
            include_zero_shot=not args.no_zero_shot,
        )

        selected_path = args.out_dir / f"selected_models_validation_{safe_filename(base)}.csv"
        selected.to_csv(selected_path, index=False)
        selected_all.append(selected.assign(base_model=base))
        print(f"  {base}: saved {selected_path}")

    if selected_all:
        selected_all_df = pd.concat(selected_all, ignore_index=True)
        cols = ["base_model"] + [c for c in selected_all_df.columns if c != "base_model"]
        selected_all_df = selected_all_df[cols]
        selected_all_path = args.out_dir / "selected_models_validation_all_TPs.csv"
        selected_all_df.to_csv(selected_all_path, index=False)
        print(f"  Saved combined selected file: {selected_all_path}")

    print("Done.")


if __name__ == "__main__":
    main()
