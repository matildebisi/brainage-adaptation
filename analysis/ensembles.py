#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# EXPERT SETS

EXPERT_SETS = {
    "TP1_only": [
        "TP1_White_pred_age",
        "TP1_Black_pred_age",
        "TP1_Asian_pred_age",
    ],
    "TP1_plus_TP3": [
        "TP1_White_pred_age",
        "TP1_Black_pred_age",
        "TP1_Asian_pred_age",
        "TP3_pred_age",
    ],
}

MODEL_TYPES = [
    "global_learned_weights",
    "subject_specific_gate",
    "ridge_stacking",
]

N_FOLDS = 4


# HELPERS

def find_column(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find {what}. Tried {candidates}.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def normalize_ids(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[ok]
    p = y_pred[ok]

    out = {
        "n": int(len(y)),
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Corr": np.nan,
        "BAG": np.nan,
        "AgeBiasSlope": np.nan,
    }

    if len(y) < 2:
        return out

    err = p - y
    corr = (
        np.corrcoef(y, p)[0, 1]
        if np.std(y) > 0 and np.std(p) > 0
        else np.nan
    )
    slope = np.polyfit(y, err, 1)[0] if np.unique(y).size > 1 else np.nan

    out.update(
        MAE=float(mean_absolute_error(y, p)),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        R2=float(r2_score(y, p)),
        Corr=float(corr),
        BAG=float(np.mean(err)),
        AgeBiasSlope=float(slope),
    )
    return out


def load_subject_ids(path, preferred_id_col="SUBJ_ID"):
    df = pd.read_csv(path)
    id_col = find_column(
        df,
        [preferred_id_col, "SUBJ_ID", "Subject", "subject"],
        f"subject ID in {path}",
    )
    return set(normalize_ids(df[id_col]))


def collect_development_ids(split_dir, split_prefix, id_col):
    """Collect subjects from mutually exclusive validation folds."""
    ids = set()
    fold_sets = []
    for fold in range(1, N_FOLDS + 1):
        path = split_dir / f"{split_prefix}_fold_{fold}_base_validation_subjects.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing validation split: {path}")
        s = load_subject_ids(path, id_col)
        ids.update(s)
        fold_sets.append(s)

    # In a proper 4-fold partition each development subject is validation exactly once.
    total_memberships = sum(len(s) for s in fold_sets)
    if total_memberships != len(ids):
        raise RuntimeError(
            "Validation folds overlap: at least one development subject appears "
            "in validation more than once."
        )
    return ids


def load_fold_data(dev, split_dir, split_prefix, fold, id_col, age_col, pred_cols):
    train_path = split_dir / f"{split_prefix}_fold_{fold}_base_train_subjects.csv"
    val_path = split_dir / f"{split_prefix}_fold_{fold}_base_validation_subjects.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing split file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing split file: {val_path}")

    train_ids = load_subject_ids(train_path, id_col)
    val_ids = load_subject_ids(val_path, id_col)

    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise RuntimeError(
            f"Fold {fold}: train/validation overlap: {sorted(overlap)[:10]}"
        )

    train_df = (
        dev[dev[id_col].isin(train_ids)]
        .dropna(subset=[age_col] + pred_cols)
        .copy()
        .reset_index(drop=True)
    )
    val_df = (
        dev[dev[id_col].isin(val_ids)]
        .dropna(subset=[age_col] + pred_cols)
        .copy()
        .reset_index(drop=True)
    )

    observed_train = set(train_df[id_col])
    observed_val = set(val_df[id_col])

    return train_df, val_df


def make_run_key(expert_set, model_type, fold, lr=None, weight_decay=None, ridge_alpha=None):
    def clean(x):
        if x is None or pd.isna(x):
            return None
        return float(x)

    return (
        str(expert_set),
        str(model_type),
        clean(lr),
        clean(weight_decay),
        clean(ridge_alpha),
        int(fold),
    )


# COMBINERS

class GlobalWeightEnsemble(nn.Module):
    """One global convex combination shared by all subjects."""

    def __init__(self, n_experts):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_experts))

    def forward(self, x_scaled, x_raw):
        del x_scaled
        weights = torch.softmax(self.logits, dim=0)
        pred = (x_raw * weights.unsqueeze(0)).sum(dim=1)
        subject_weights = weights.unsqueeze(0).expand(x_raw.shape[0], -1)
        return pred, subject_weights


class SubjectSpecificGate(nn.Module):
    """Linear prediction-conditioned softmax gate."""

    def __init__(self, n_experts):
        super().__init__()
        self.linear = nn.Linear(n_experts, n_experts)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x_scaled, x_raw):
        logits = self.linear(x_scaled)
        weights = torch.softmax(logits, dim=1)
        pred = (weights * x_raw).sum(dim=1)
        return pred, weights


def build_torch_model(model_type, n_experts):
    if model_type == "global_learned_weights":
        return GlobalWeightEnsemble(n_experts)
    if model_type == "subject_specific_gate":
        return SubjectSpecificGate(n_experts)
    raise ValueError(f"Not a torch combiner: {model_type}")


# TRAIN ONE FOLD: GLOBAL / GATE

def train_one_fold_torch(
    model_type,
    train_df,
    val_df,
    pred_cols,
    age_col,
    lr,
    weight_decay,
    fold_seed,
    max_epochs,
    patience,
    min_delta,
):
    X_train_raw = train_df[pred_cols].to_numpy(dtype=np.float32)
    X_val_raw = val_df[pred_cols].to_numpy(dtype=np.float32)
    y_train = train_df[age_col].to_numpy(dtype=np.float32)
    y_val = val_df[age_col].to_numpy(dtype=np.float32)

    mean_train = X_train_raw.mean(axis=0, keepdims=True)
    std_train = X_train_raw.std(axis=0, keepdims=True)
    std_train[std_train < 1e-6] = 1.0

    X_train_scaled = (X_train_raw - mean_train) / std_train
    X_val_scaled = (X_val_raw - mean_train) / std_train

    Xtr_raw_t = torch.tensor(X_train_raw, dtype=torch.float32)
    Xva_raw_t = torch.tensor(X_val_raw, dtype=torch.float32)
    Xtr_scaled_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    Xva_scaled_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.float32)
    yva_t = torch.tensor(y_val, dtype=torch.float32)

    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    model = build_torch_model(model_type, len(pred_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.L1Loss()

    best_val_mae = np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    total_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()

        train_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        train_pred, _ = model(Xtr_scaled_t, Xtr_raw_t)
        loss = criterion(train_pred, ytr_t)
        loss.backward()
        optimizer.step()
        train_time = time.perf_counter() - train_start

        eval_start = time.perf_counter()
        model.eval()
        with torch.no_grad():
            train_pred_eval, _ = model(Xtr_scaled_t, Xtr_raw_t)
            val_pred_eval, _ = model(Xva_scaled_t, Xva_raw_t)
            train_mae = torch.mean(torch.abs(train_pred_eval - ytr_t)).item()
            val_mae = torch.mean(torch.abs(val_pred_eval - yva_t)).item()
        validation_time = time.perf_counter() - eval_start
        epoch_time = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_MAE": train_mae,
                "val_MAE": val_mae,
                "train_time_s": train_time,
                "validation_time_s": validation_time,
                "epoch_time_s": epoch_time,
            }
        )

        if val_mae < best_val_mae - min_delta:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    total_training_time = time.perf_counter() - total_start

    if best_state is None:
        raise RuntimeError("No checkpoint was saved.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        best_train_pred, _ = model(Xtr_scaled_t, Xtr_raw_t)
        best_val_pred, _ = model(Xva_scaled_t, Xva_raw_t)

    history_df = pd.DataFrame(history)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "kind": "torch",
        "state_dict": copy.deepcopy(model.state_dict()),
        "mean": mean_train,
        "std": std_train,
        "train_metrics": compute_metrics(y_train, best_train_pred.numpy()),
        "val_metrics": compute_metrics(y_val, best_val_pred.numpy()),
        "history": history_df,
        "cost": {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "best_epoch": int(best_epoch),
            "epochs_run": int(len(history_df)),
            "mean_epoch_time_s": float(history_df["epoch_time_s"].mean()),
            "std_epoch_time_s": (
                float(history_df["epoch_time_s"].std(ddof=1))
                if len(history_df) > 1
                else 0.0
            ),
            "mean_train_time_s": float(history_df["train_time_s"].mean()),
            "mean_validation_time_s": float(history_df["validation_time_s"].mean()),
            "total_training_time_s": float(total_training_time),
        },
    }


# TRAIN ONE FOLD: RIDGE STACKING

def train_one_fold_ridge(train_df, val_df, pred_cols, age_col, ridge_alpha):
    """
    Affine stacking on standardized expert predictions:

        z_ik = (p_ik - mu_k) / sigma_k
        y_hat_i = beta_0 + sum_k beta_k z_ik

    sklearn Ridge minimizes squared error + alpha * ||beta||_2^2.
    Hyperparameter alpha is selected by mean validation MAE across the 4 folds.
    Intercept is not penalized.
    """
    X_train_raw = train_df[pred_cols].to_numpy(dtype=np.float64)
    X_val_raw = val_df[pred_cols].to_numpy(dtype=np.float64)
    y_train = train_df[age_col].to_numpy(dtype=np.float64)
    y_val = val_df[age_col].to_numpy(dtype=np.float64)

    mean_train = X_train_raw.mean(axis=0, keepdims=True)
    std_train = X_train_raw.std(axis=0, keepdims=True)
    std_train[std_train < 1e-12] = 1.0

    X_train_scaled = (X_train_raw - mean_train) / std_train
    X_val_scaled = (X_val_raw - mean_train) / std_train

    train_start = time.perf_counter()
    model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    model.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - train_start

    val_start = time.perf_counter()
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    validation_time = time.perf_counter() - val_start

    total_time = train_time + validation_time

    coef_scaled = np.asarray(model.coef_, dtype=float).reshape(-1)
    intercept_scaled = float(model.intercept_)

    # Equivalent coefficients in the original prediction-age space:
    # beta0_raw + sum beta_k_raw * p_ik
    scale = std_train.reshape(-1)
    mean = mean_train.reshape(-1)
    coef_raw = coef_scaled / scale
    intercept_raw = intercept_scaled - float(np.sum(coef_scaled * mean / scale))

    history_df = pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_MAE": mean_absolute_error(y_train, train_pred),
                "val_MAE": mean_absolute_error(y_val, val_pred),
                "train_time_s": train_time,
                "validation_time_s": validation_time,
                "epoch_time_s": total_time,
            }
        ]
    )

    # K coefficients + 1 intercept.
    n_params = len(pred_cols) + 1

    return {
        "kind": "ridge",
        "mean": mean_train,
        "std": std_train,
        "coef_scaled": coef_scaled,
        "intercept_scaled": intercept_scaled,
        "coef_raw": coef_raw,
        "intercept_raw": intercept_raw,
        "train_metrics": compute_metrics(y_train, train_pred),
        "val_metrics": compute_metrics(y_val, val_pred),
        "history": history_df,
        "cost": {
            "total_params": int(n_params),
            "trainable_params": int(n_params),
            "best_epoch": np.nan,
            "epochs_run": 1,
            "mean_epoch_time_s": float(total_time),
            "std_epoch_time_s": 0.0,
            "mean_train_time_s": float(train_time),
            "mean_validation_time_s": float(validation_time),
            "total_training_time_s": float(total_time),
        },
    }


# APPLY TRAINED FOLD MODEL

def apply_saved_fold_model(run_info, model_type, df, pred_cols):
    X = df[pred_cols].to_numpy(dtype=np.float64)
    complete = np.isfinite(X).all(axis=1)

    predictions = np.full(len(df), np.nan, dtype=float)
    weights = np.full((len(df), len(pred_cols)), np.nan, dtype=float)

    if not complete.any():
        return predictions, weights

    X_raw = X[complete]
    X_scaled = (X_raw - run_info["mean"]) / run_info["std"]

    if model_type == "ridge_stacking":
        pred = (
            float(run_info["intercept_scaled"])
            + X_scaled @ np.asarray(run_info["coef_scaled"], dtype=float)
        )
        predictions[complete] = pred
        # Ridge coefficients are not convex mixture weights, so weights stay NaN.
        return predictions, weights

    Xraw_t = torch.tensor(X_raw, dtype=torch.float32)
    Xscaled_t = torch.tensor(X_scaled, dtype=torch.float32)

    model = build_torch_model(model_type, len(pred_cols))
    model.load_state_dict(run_info["state_dict"])
    model.eval()

    with torch.no_grad():
        pred, w = model(Xscaled_t, Xraw_t)

    predictions[complete] = pred.numpy()
    weights[complete] = w.numpy()
    return predictions, weights


# GRID SUMMARY / SELECTION

def summarize_grid(fold_metrics):
    group_cols = [
        "expert_set",
        "model_type",
        "lr",
        "weight_decay",
        "ridge_alpha",
    ]

    rows = []
    for keys, g in fold_metrics.groupby(group_cols, dropna=False):
        expert_set, model_type, lr, wd, alpha = keys

        row = {
            "expert_set": expert_set,
            "model_type": model_type,
            "lr": float(lr) if pd.notna(lr) else np.nan,
            "weight_decay": float(wd) if pd.notna(wd) else np.nan,
            "ridge_alpha": float(alpha) if pd.notna(alpha) else np.nan,
            "n_folds": int(g["fold"].nunique()),
            "train_MAE_mean": float(g["train_MAE"].mean()),
            "train_MAE_std": float(g["train_MAE"].std(ddof=1)),
            "validation_MAE_mean": float(g["validation_MAE"].mean()),
            "validation_MAE_std": float(g["validation_MAE"].std(ddof=1)),
            "validation_RMSE_mean": float(g["validation_RMSE"].mean()),
            "validation_RMSE_std": float(g["validation_RMSE"].std(ddof=1)),
            "validation_R2_mean": float(g["validation_R2"].mean()),
            "validation_R2_std": float(g["validation_R2"].std(ddof=1)),
            "validation_Corr_mean": float(g["validation_Corr"].mean()),
            "validation_Corr_std": float(g["validation_Corr"].std(ddof=1)),
            "validation_BAG_mean": float(g["validation_BAG"].mean()),
            "validation_BAG_std": float(g["validation_BAG"].std(ddof=1)),
            "validation_AgeBiasSlope_mean": float(g["validation_AgeBiasSlope"].mean()),
            "validation_AgeBiasSlope_std": float(g["validation_AgeBiasSlope"].std(ddof=1)),
            "best_epoch_mean": float(g["best_epoch"].mean()) if g["best_epoch"].notna().any() else np.nan,
            "best_epoch_std": float(g["best_epoch"].std(ddof=1)) if g["best_epoch"].notna().sum() > 1 else np.nan,
            "epochs_run_mean": float(g["epochs_run"].mean()),
            "epochs_run_std": float(g["epochs_run"].std(ddof=1)),
            "epochs_run_total_4folds": int(g["epochs_run"].sum()),
            "total_training_time_s_4folds": float(g["total_training_time_s"].sum()),
            "mean_epoch_time_s": float(g["mean_epoch_time_s"].mean()),
            "trainable_params": int(g["trainable_params"].iloc[0]),
        }
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["expert_set", "model_type", "validation_MAE_mean", "validation_MAE_std"]
        )
        .reset_index(drop=True)
    )


def select_best_per_category(grid_summary):
    """Select one hyperparameter configuration per expert-set × model-type."""
    selected = []

    for (expert_set, model_type), g in grid_summary.groupby(
        ["expert_set", "model_type"]
    ):
        if model_type == "ridge_stacking":
            order = [
                "validation_MAE_mean",
                "validation_MAE_std",
                "ridge_alpha",
            ]
        else:
            order = [
                "validation_MAE_mean",
                "validation_MAE_std",
                "weight_decay",
                "lr",
            ]

        best = g.sort_values(order, ascending=True, na_position="last").iloc[0].copy()
        selected.append(best)

    return pd.DataFrame(selected).reset_index(drop=True)


def summarize_test_by_fold(test_by_fold):
    rows = []
    for (expert_set, model_type), g in test_by_fold.groupby(
        ["expert_set", "model_type"]
    ):
        row = {
            "expert_set": expert_set,
            "model_type": model_type,
            "n_folds": int(g["fold"].nunique()),
            "lr": float(g["lr"].iloc[0]) if pd.notna(g["lr"].iloc[0]) else np.nan,
            "weight_decay": (
                float(g["weight_decay"].iloc[0])
                if pd.notna(g["weight_decay"].iloc[0])
                else np.nan
            ),
            "ridge_alpha": (
                float(g["ridge_alpha"].iloc[0])
                if pd.notna(g["ridge_alpha"].iloc[0])
                else np.nan
            ),
        }

        for metric in ["MAE", "RMSE", "R2", "Corr", "BAG", "AgeBiasSlope"]:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description=(
            "4-fold grid search for frozen TP1 expert combinations: equal mean, "
            "global convex weights, subject-specific softmax gate, and Ridge stacking. "
            "All learned hyperparameters are selected only by mean validation MAE."
        )
    )

    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--split_prefix", type=str, default="LASI-DAD")
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=None,
        help=(
            "Optional authoritative test CSV (e.g. test_wave2_cdr0.csv). "
            "If supplied, test subjects are selected by SUBJ_ID membership in this file. "
            "Otherwise the input CSV's dataset_split/split column is used."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("tp1_ensemble_gridsearch_ridge_results"),
    )

    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        help="Learning rates for global weighting and subject-specific gate.",
    )
    parser.add_argument(
        "--weight_decays",
        type=float,
        nargs="+",
        default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        help="Weight-decay values for global weighting and subject-specific gate.",
    )
    parser.add_argument(
        "--ridge_alphas",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
        help="Ridge alpha values. alpha=0 corresponds to unregularized OLS stacking.",
    )
    parser.add_argument("--max_epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    id_col = find_column(df, ["SUBJ_ID", "Subject", "subject"], "subject ID")
    age_col = find_column(df, ["AGE", "real_age", "Age", "age"], "chronological age")
    df[id_col] = normalize_ids(df[id_col])

    all_pred_cols = sorted(
        set(col for cols in EXPERT_SETS.values() for col in cols)
    )
    missing_cols = [c for c in all_pred_cols if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing prediction columns:\n" + "\n".join(missing_cols))

    # Development subjects are defined from the original 4-fold split files.
    dev_ids = collect_development_ids(args.split_dir, args.split_prefix, id_col)
    dev = df[df[id_col].isin(dev_ids)].copy().reset_index(drop=True)

    missing_dev = dev_ids.difference(set(dev[id_col]))
    # if missing_dev:
    #     raise RuntimeError(
    #         f"{len(missing_dev)} development subjects from split files are missing "
    #         f"from the prediction CSV. Examples: {sorted(missing_dev)[:10]}"
    #     )

    # Test subjects: preferably from the authoritative separate test CSV.
    if args.test_csv is not None:
        test_ids = load_subject_ids(args.test_csv, id_col)
        test = df[df[id_col].isin(test_ids)].copy().reset_index(drop=True)
        missing_test = test_ids.difference(set(test[id_col]))
        if missing_test:
            raise RuntimeError(
                f"{len(missing_test)} subjects from --test_csv are missing from the "
                f"prediction CSV. Examples: {sorted(missing_test)[:10]}"
            )
        test_source = str(args.test_csv)
    else:
        split_col = find_column(
            df, ["dataset_split", "split"], "development/test split"
        )
        mask = (
            df[split_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("test")
        )
        test = df[mask].copy().reset_index(drop=True)
        test_ids = set(test[id_col])
        test_source = f"{args.csv}:{split_col} == test"

    if not dev_ids.isdisjoint(test_ids):
        overlap = sorted(dev_ids.intersection(test_ids))
        raise RuntimeError(
            f"Development/test overlap detected: {len(overlap)} subjects. "
            f"Examples: {overlap[:10]}"
        )

    if len(dev) == 0:
        raise RuntimeError("No development rows found.")
    if len(test) == 0:
        raise RuntimeError("No test rows found.")

    print("=" * 80)
    print("DATA")
    print("=" * 80)
    print(f"Development rows: {len(dev)}")
    print(f"Test rows:        {len(test)}")
    print(f"Test source:      {test_source}")
    print(f"LR grid:          {args.lrs}")
    print(f"Weight decay:     {args.weight_decays}")
    print(f"Ridge alpha:      {args.ridge_alphas}")

    # -------------------------------------------------------------------------
    # GRID SEARCH: DEVELOPMENT ONLY
    # -------------------------------------------------------------------------

    grid_fold_rows = []
    history_frames = []
    trained_runs = {}
    grid_start = time.perf_counter()

    for expert_set, pred_cols in EXPERT_SETS.items():
        for model_type in MODEL_TYPES:
            if model_type == "ridge_stacking":
                configs = [(None, None, float(a)) for a in args.ridge_alphas]
            else:
                configs = [
                    (float(lr), float(wd), None)
                    for lr, wd in itertools.product(args.lrs, args.weight_decays)
                ]

            for lr, wd, alpha in configs:
                if model_type == "ridge_stacking":
                    config_label = f"alpha={alpha:g}"
                else:
                    config_label = f"lr={lr:g} | wd={wd:g}"

                print(
                    "\n" + "=" * 80
                    + f"\n{expert_set} | {model_type} | {config_label}"
                    + "\n" + "=" * 80
                )

                for fold in range(1, N_FOLDS + 1):
                    train_df, val_df = load_fold_data(
                        dev=dev,
                        split_dir=args.split_dir,
                        split_prefix=args.split_prefix,
                        fold=fold,
                        id_col=id_col,
                        age_col=age_col,
                        pred_cols=pred_cols,
                    )

                    if model_type == "ridge_stacking":
                        run_info = train_one_fold_ridge(
                            train_df=train_df,
                            val_df=val_df,
                            pred_cols=pred_cols,
                            age_col=age_col,
                            ridge_alpha=alpha,
                        )
                    else:
                        run_info = train_one_fold_torch(
                            model_type=model_type,
                            train_df=train_df,
                            val_df=val_df,
                            pred_cols=pred_cols,
                            age_col=age_col,
                            lr=lr,
                            weight_decay=wd,
                            fold_seed=args.seed + fold - 1,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            min_delta=args.min_delta,
                        )

                    key = make_run_key(
                        expert_set,
                        model_type,
                        fold,
                        lr=lr,
                        weight_decay=wd,
                        ridge_alpha=alpha,
                    )
                    trained_runs[key] = run_info

                    train_met = run_info["train_metrics"]
                    val_met = run_info["val_metrics"]
                    cost = run_info["cost"]

                    grid_fold_rows.append(
                        {
                            "expert_set": expert_set,
                            "model_type": model_type,
                            "lr": lr,
                            "weight_decay": wd,
                            "ridge_alpha": alpha,
                            "fold": fold,
                            "n_train": len(train_df),
                            "n_validation": len(val_df),
                            "train_MAE": train_met["MAE"],
                            "train_RMSE": train_met["RMSE"],
                            "train_R2": train_met["R2"],
                            "train_Corr": train_met["Corr"],
                            "train_BAG": train_met["BAG"],
                            "train_AgeBiasSlope": train_met["AgeBiasSlope"],
                            "validation_MAE": val_met["MAE"],
                            "validation_RMSE": val_met["RMSE"],
                            "validation_R2": val_met["R2"],
                            "validation_Corr": val_met["Corr"],
                            "validation_BAG": val_met["BAG"],
                            "validation_AgeBiasSlope": val_met["AgeBiasSlope"],
                            **cost,
                        }
                    )

                    hist = run_info["history"].copy()
                    hist.insert(0, "fold", fold)
                    hist.insert(0, "ridge_alpha", alpha)
                    hist.insert(0, "weight_decay", wd)
                    hist.insert(0, "lr", lr)
                    hist.insert(0, "model_type", model_type)
                    hist.insert(0, "expert_set", expert_set)
                    history_frames.append(hist)

                    best_epoch_txt = (
                        "NA"
                        if pd.isna(cost["best_epoch"])
                        else str(int(cost["best_epoch"]))
                    )
                    print(
                        f"Fold {fold}: best epoch={best_epoch_txt} | "
                        f"train MAE={train_met['MAE']:.4f} | "
                        f"val MAE={val_met['MAE']:.4f}"
                    )

    grid_total_time_s = time.perf_counter() - grid_start

    grid_fold_df = pd.DataFrame(grid_fold_rows)
    grid_fold_df.to_csv(args.out_dir / "gridsearch_metrics_by_fold.csv", index=False)

    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(
            args.out_dir / "gridsearch_epoch_history.csv", index=False
        )

    grid_summary = summarize_grid(grid_fold_df)
    grid_summary.to_csv(
        args.out_dir / "gridsearch_validation_summary.csv", index=False
    )

    selected = select_best_per_category(grid_summary)
    selected.to_csv(args.out_dir / "selected_hyperparameters.csv", index=False)

    # -------------------------------------------------------------------------
    # TEST BASELINES: evaluated only after model selection
    # -------------------------------------------------------------------------

    baseline_rows = []
    for col in all_pred_cols:
        baseline_rows.append(
            {
                "expert_set": "individual",
                "approach": col.replace("_pred_age", ""),
                **compute_metrics(test[age_col], test[col]),
            }
        )

    for expert_set, pred_cols in EXPERT_SETS.items():
        complete = test[pred_cols].notna().all(axis=1)
        pred = np.full(len(test), np.nan, dtype=float)
        pred[complete] = test.loc[complete, pred_cols].mean(axis=1).to_numpy()
        baseline_rows.append(
            {
                "expert_set": expert_set,
                "approach": "simple_mean",
                **compute_metrics(test[age_col], pred),
            }
        )

    pd.DataFrame(baseline_rows).to_csv(
        args.out_dir / "deterministic_baseline_test_metrics.csv", index=False
    )

    # -------------------------------------------------------------------------
    # TEST: SELECTED CONFIGURATIONS ONLY
    # -------------------------------------------------------------------------

    test_fold_rows = []
    prediction_rows = []
    weight_rows = []
    ridge_coef_rows = []
    selected_cost_rows = []

    for sel in selected.itertuples(index=False):
        expert_set = sel.expert_set
        model_type = sel.model_type
        lr = float(sel.lr) if pd.notna(sel.lr) else None
        wd = float(sel.weight_decay) if pd.notna(sel.weight_decay) else None
        alpha = float(sel.ridge_alpha) if pd.notna(sel.ridge_alpha) else None
        pred_cols = EXPERT_SETS[expert_set]

        if model_type == "ridge_stacking":
            hp_text = f"alpha={alpha:g}"
        else:
            hp_text = f"lr={lr:g}, wd={wd:g}"

        print(
            "\nSELECTED "
            f"{expert_set} | {model_type}: {hp_text}, "
            f"validation MAE={sel.validation_MAE_mean:.4f} "
            f"± {sel.validation_MAE_std:.4f}"
        )

        for fold in range(1, N_FOLDS + 1):
            key = make_run_key(
                expert_set,
                model_type,
                fold,
                lr=lr,
                weight_decay=wd,
                ridge_alpha=alpha,
            )
            run_info = trained_runs[key]

            test_pred, test_weights = apply_saved_fold_model(
                run_info=run_info,
                model_type=model_type,
                df=test,
                pred_cols=pred_cols,
            )
            met = compute_metrics(test[age_col], test_pred)

            test_fold_rows.append(
                {
                    "expert_set": expert_set,
                    "model_type": model_type,
                    "lr": lr,
                    "weight_decay": wd,
                    "ridge_alpha": alpha,
                    "fold": fold,
                    **met,
                }
            )

            selected_cost_rows.append(
                {
                    "expert_set": expert_set,
                    "model_type": model_type,
                    "lr": lr,
                    "weight_decay": wd,
                    "ridge_alpha": alpha,
                    "fold": fold,
                    **run_info["cost"],
                }
            )

            for row_idx in range(len(test)):
                prediction_rows.append(
                    {
                        "expert_set": expert_set,
                        "model_type": model_type,
                        "lr": lr,
                        "weight_decay": wd,
                        "ridge_alpha": alpha,
                        "fold": fold,
                        id_col: test.iloc[row_idx][id_col],
                        age_col: test.iloc[row_idx][age_col],
                        "predicted_age": test_pred[row_idx],
                    }
                )

                if model_type != "ridge_stacking":
                    for expert_idx, pred_col in enumerate(pred_cols):
                        weight_rows.append(
                            {
                                "expert_set": expert_set,
                                "model_type": model_type,
                                "lr": lr,
                                "weight_decay": wd,
                                "ridge_alpha": alpha,
                                "fold": fold,
                                id_col: test.iloc[row_idx][id_col],
                                "expert": pred_col.replace("_pred_age", ""),
                                "weight": test_weights[row_idx, expert_idx],
                            }
                        )

            if model_type == "ridge_stacking":
                for j, pred_col in enumerate(pred_cols):
                    ridge_coef_rows.append(
                        {
                            "expert_set": expert_set,
                            "fold": fold,
                            "ridge_alpha": alpha,
                            "expert": pred_col.replace("_pred_age", ""),
                            "coefficient_scaled": run_info["coef_scaled"][j],
                            "coefficient_raw": run_info["coef_raw"][j],
                            "intercept_scaled": run_info["intercept_scaled"],
                            "intercept_raw": run_info["intercept_raw"],
                        }
                    )

                np.savez(
                    args.out_dir
                    / f"selected_{expert_set}_ridge_stacking_fold{fold}.npz",
                    coef_scaled=run_info["coef_scaled"],
                    intercept_scaled=np.array([run_info["intercept_scaled"]]),
                    coef_raw=run_info["coef_raw"],
                    intercept_raw=np.array([run_info["intercept_raw"]]),
                    mean=run_info["mean"],
                    std=run_info["std"],
                    prediction_columns=np.asarray(pred_cols, dtype=str),
                    ridge_alpha=np.array([alpha], dtype=float),
                )
            else:
                torch.save(
                    {
                        "state_dict": run_info["state_dict"],
                        "prediction_columns": pred_cols,
                        "mean": run_info["mean"],
                        "std": run_info["std"],
                        "fold": fold,
                        "expert_set": expert_set,
                        "model_type": model_type,
                        "lr": lr,
                        "weight_decay": wd,
                    },
                    args.out_dir
                    / f"selected_{expert_set}_{model_type}_fold{fold}.pth",
                )

            selected_history = run_info["history"].copy()
            selected_history.insert(0, "fold", fold)
            selected_history.insert(0, "ridge_alpha", alpha)
            selected_history.insert(0, "weight_decay", wd)
            selected_history.insert(0, "lr", lr)
            selected_history.insert(0, "model_type", model_type)
            selected_history.insert(0, "expert_set", expert_set)
            selected_history.to_csv(
                args.out_dir
                / f"selected_{expert_set}_{model_type}_fold{fold}_history.csv",
                index=False,
            )

    test_by_fold = pd.DataFrame(test_fold_rows)
    test_by_fold.to_csv(
        args.out_dir / "selected_test_metrics_by_fold.csv", index=False
    )

    test_summary = summarize_test_by_fold(test_by_fold)
    test_summary.to_csv(
        args.out_dir / "selected_test_metrics_mean_std.csv", index=False
    )

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(
        args.out_dir / "selected_test_predictions_by_fold.csv", index=False
    )

    pd.DataFrame(weight_rows).to_csv(
        args.out_dir / "selected_test_weights_by_fold.csv", index=False
    )
    pd.DataFrame(ridge_coef_rows).to_csv(
        args.out_dir / "selected_ridge_coefficients_by_fold.csv", index=False
    )

    # -------------------------------------------------------------------------
    # FOUR-FOLD PREDICTION ENSEMBLE OF EACH SELECTED COMBINER
    # -------------------------------------------------------------------------

    ensemble_rows = []
    for (expert_set, model_type), g in predictions_df.groupby(
        ["expert_set", "model_type"]
    ):
        wide = g.pivot(index=id_col, columns="fold", values="predicted_age")
        ensemble_pred = wide.mean(axis=1)

        age_df = test[[id_col, age_col]].copy()
        tmp = age_df.merge(
            ensemble_pred.rename("ensemble_pred"),
            left_on=id_col,
            right_index=True,
            how="left",
        )

        sel = selected[
            (selected["expert_set"] == expert_set)
            & (selected["model_type"] == model_type)
        ].iloc[0]

        ensemble_rows.append(
            {
                "expert_set": expert_set,
                "model_type": model_type,
                "lr": sel["lr"],
                "weight_decay": sel["weight_decay"],
                "ridge_alpha": sel["ridge_alpha"],
                **compute_metrics(tmp[age_col], tmp["ensemble_pred"]),
            }
        )

    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_df.to_csv(
        args.out_dir / "selected_test_four_fold_ensemble_metrics.csv", index=False
    )

    # -------------------------------------------------------------------------
    # COST SUMMARY
    # -------------------------------------------------------------------------

    selected_cost_df = pd.DataFrame(selected_cost_rows)
    selected_cost_df.to_csv(
        args.out_dir / "selected_cost_metrics_by_fold.csv", index=False
    )

    cost_summary_rows = []
    for (expert_set, model_type), g in selected_cost_df.groupby(
        ["expert_set", "model_type"]
    ):
        cost_summary_rows.append(
            {
                "expert_set": expert_set,
                "model_type": model_type,
                "lr": g["lr"].iloc[0],
                "weight_decay": g["weight_decay"].iloc[0],
                "ridge_alpha": g["ridge_alpha"].iloc[0],
                "trainable_params": int(g["trainable_params"].iloc[0]),
                "best_epoch_mean": (
                    float(g["best_epoch"].mean())
                    if g["best_epoch"].notna().any()
                    else np.nan
                ),
                "best_epoch_std": (
                    float(g["best_epoch"].std(ddof=1))
                    if g["best_epoch"].notna().sum() > 1
                    else np.nan
                ),
                "epochs_run_mean": float(g["epochs_run"].mean()),
                "epochs_run_std": float(g["epochs_run"].std(ddof=1)),
                "epochs_run_total_4folds": int(g["epochs_run"].sum()),
                "mean_epoch_time_s": float(g["mean_epoch_time_s"].mean()),
                "total_training_time_s_4folds": float(
                    g["total_training_time_s"].sum()
                ),
            }
        )

    selected_cost_summary = pd.DataFrame(cost_summary_rows)
    selected_cost_summary.to_csv(
        args.out_dir / "selected_cost_metrics_summary.csv", index=False
    )

    # -------------------------------------------------------------------------
    # COMPACT FINAL SUMMARY
    # -------------------------------------------------------------------------

    final_summary = selected.merge(
        test_summary,
        on=["expert_set", "model_type"],
        how="left",
        suffixes=("_validation_selection", "_test"),
    )
    final_summary.to_csv(args.out_dir / "final_results_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    n_neural_configs = len(args.lrs) * len(args.weight_decays)
    n_ridge_configs = len(args.ridge_alphas)
    total_training_runs = (
        len(EXPERT_SETS)
        * N_FOLDS
        * (2 * n_neural_configs + n_ridge_configs)
    )

    metadata = {
        "learning_rates": args.lrs,
        "weight_decays": args.weight_decays,
        "ridge_alphas": args.ridge_alphas,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "seed": args.seed,
        "n_folds": N_FOLDS,
        "n_expert_sets": len(EXPERT_SETS),
        "model_types": MODEL_TYPES,
        "total_number_of_training_runs": total_training_runs,
        "gridsearch_total_wall_time_s": grid_total_time_s,
        "test_source": test_source,
        "selection_rule": (
            "minimum mean validation MAE across 4 folds; tie-break lower validation "
            "MAE SD; then smaller alpha for Ridge, or smaller weight decay then "
            "learning rate for torch combiners"
        ),
        "ridge_training_objective": (
            "squared error + alpha * L2 penalty; alpha selected by validation MAE"
        ),
        "test_policy": (
            "test set evaluated only after validation-based hyperparameter selection"
        ),
    }

    with open(args.out_dir / "gridsearch_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("SELECTED HYPERPARAMETERS")
    print("=" * 80)
    print(
        selected[
            [
                "expert_set",
                "model_type",
                "lr",
                "weight_decay",
                "ridge_alpha",
                "validation_MAE_mean",
                "validation_MAE_std",
                "trainable_params",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("SELECTED TEST METRICS: MEAN ± SD ACROSS 4 FOLD MODELS")
    print("=" * 80)
    print(test_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("SELECTED 4-FOLD PREDICTION ENSEMBLE")
    print("=" * 80)
    print(ensemble_df.to_string(index=False))

    print("\nSaved all results to:")
    print(args.out_dir.resolve())


if __name__ == "__main__":
    main()
