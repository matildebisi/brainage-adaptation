#!/usr/bin/env python3
"""Build the subject-level error table used by the thesis R analyses.

Inputs are the wide held-out test prediction CSV and the model list selected
using development-set validation. Each condition uses its four fold-specific
predictions; the same complete set of subjects is kept for every condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRETRAININGS = ("TP1_White", "TP2_Asian", "TP2_Black", "TP3")
ADAPTATIONS = ("Zero-shot", "Traditional", "LoRA", "DoRA", "PiSSA", "FiltLoRA")
OUTPUT_NAME = "ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv"
PROVENANCE_NAME = "ART_pretraining_adaptation_model_provenance.csv"


def prediction_columns(model: str, folds: int) -> list[str]:
    """Column naming convention used by the original test-result merge."""
    return [f"dl_tl_TL_{model}_seed{fold}_pred_age" for fold in range(1, folds + 1)]


def prepare(
    predictions_path: Path,
    selection_path: Path,
    out_dir: Path,
    *,
    subject_col: str = "SUBJ_ID",
    age_col: str = "AGE",
    sex_col: str = "Sex",
    site_col: str = "Site",
    site_f: str = "F",
    qc_col: str = "QC_note_wave2",
    exclude_qc_notes: bool = False,
    folds: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if folds < 1:
        raise ValueError("--folds must be at least 1")

    test = pd.read_csv(predictions_path, encoding="utf-8-sig", low_memory=False,
                       dtype={subject_col: str})
    selection = pd.read_csv(selection_path, encoding="utf-8-sig")

    metadata_cols = [subject_col, age_col, sex_col, site_col]
    missing_meta = sorted(set(metadata_cols) - set(test.columns))
    if missing_meta:
        raise ValueError(f"Test predictions CSV is missing: {missing_meta}")

    required_selection = {"base_model", "category", "model"}
    missing_selection = sorted(required_selection - set(selection.columns))
    if missing_selection:
        raise ValueError(f"Selected models CSV is missing: {missing_selection}")

    if test[subject_col].isna().any() or test[subject_col].duplicated().any():
        raise ValueError("Each subject must occur exactly once in the test CSV. "
                         "Use a session identifier if subjects have multiple scans.")

    selected = selection[selection["base_model"].isin(PRETRAININGS)].copy()
    specs = []
    all_prediction_columns = []
    for pretraining in PRETRAININGS:
        for adaptation in ADAPTATIONS:
            match = selected[(selected["base_model"] == pretraining)
                             & (selected["category"] == adaptation)]
            if len(match) != 1:
                raise ValueError(
                    f"Expected exactly one validation-selected model for "
                    f"{pretraining} / {adaptation}; found {len(match)}."
                )
            model = match.iloc[0]["model"]
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"Missing model identifier for {pretraining} / {adaptation}")
            cols = prediction_columns(model, folds)
            specs.append((pretraining, adaptation, model, cols))
            all_prediction_columns.extend(cols)

    missing_predictions = sorted(set(all_prediction_columns) - set(test.columns))
    if missing_predictions:
        example = "\n  ".join(missing_predictions[:6])
        raise ValueError(
            f"Missing {len(missing_predictions)} prediction columns in the test CSV. "
            f"First missing columns:\n  {example}\n"
            "The input must be the merged WIDE test prediction CSV, with "
            "fold-specific columns named as in the original experiments."
        )

    # Keep only subjects with a complete set of predictions across all conditions.
    # A common cohort is essential for the repeated-measures statistical analysis.
    if exclude_qc_notes:
        if qc_col not in test.columns:
            raise ValueError(f"Cannot apply QC filtering: '{qc_col}' not found")
        valid_qc = test[qc_col].isna() | test[qc_col].astype(str).str.strip().eq("")
        test = test.loc[valid_qc].copy()

    age = pd.to_numeric(test[age_col], errors="coerce")
    valid_metadata = (
        age.notna() & np.isfinite(age.to_numpy(dtype=float))
        & test[sex_col].notna() & test[site_col].notna()
    )
    numeric_preds = test[all_prediction_columns].apply(pd.to_numeric, errors="coerce")
    valid_predictions = numeric_preds.notna().all(axis=1) & np.isfinite(
        numeric_preds.to_numpy(dtype=float)
    ).all(axis=1)

    n_before = len(test)
    valid = valid_metadata & valid_predictions
    test = test.loc[valid].copy()
    numeric_preds = numeric_preds.loc[valid]
    if test.empty:
        raise ValueError("No subjects have complete metadata and all selected predictions")

    # Important: abs_error is the MEAN of the four individual absolute errors,
    # not abs(predicted_age - Age), which would generally be smaller.
    ages = pd.to_numeric(test[age_col]).to_numpy(dtype=float)
    base = pd.DataFrame({
        "Subject": test[subject_col].to_numpy(),
        "Age": ages,
        "Sex": test[sex_col].to_numpy(),
        "Site": test[site_col].to_numpy(),
        "SiteGroup": np.where(test[site_col].astype(str).to_numpy() == site_f,
                              "Site F", "Other sites"),
    })

    results = []
    provenance = []
    for pretraining, adaptation, model, cols in specs:
        pred = numeric_preds[cols].to_numpy(dtype=float)
        errors = pred - ages[:, None]
        frame = base.copy()
        frame.insert(1, "Pretraining", pretraining)
        frame.insert(2, "Adaptation", adaptation)
        frame["predicted_age"] = pred.mean(axis=1)
        frame["signed_error"] = errors.mean(axis=1)
        frame["abs_error"] = np.abs(errors).mean(axis=1)
        results.append(frame)
        provenance.append({
            "Pretraining": pretraining,
            "Adaptation": adaptation,
            "model": model,
            "n_prediction_columns": len(cols),
            "prediction_columns": " | ".join(cols),
        })

    out = pd.concat(results, ignore_index=True)
    out = out[["Subject", "Pretraining", "Adaptation", "Age", "Sex", "Site",
               "predicted_age", "signed_error", "abs_error", "SiteGroup"]]
    out = out.sort_values(["Pretraining", "Adaptation", "Subject"], kind="stable")
    source = pd.DataFrame(provenance)

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / OUTPUT_NAME, index=False)
    source.to_csv(out_dir / PROVENANCE_NAME, index=False)
    print(f"Subjects with complete predictions: {len(test)} / {n_before}")
    print(f"Pretraining/adaptation conditions: {len(specs)}")
    print(f"Saved: {out_dir / OUTPUT_NAME} ({len(out)} rows)")
    print(f"Saved: {out_dir / PROVENANCE_NAME}")
    return out, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Merged wide CSV of held-out test predictions")
    parser.add_argument("--selected-models", type=Path, required=True,
                        help="selected_models_validation_all_TPs.csv from validation")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory read by the thesis R analysis scripts")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--subject-col", default="SUBJ_ID")
    parser.add_argument("--age-col", default="AGE")
    parser.add_argument("--sex-col", default="Sex")
    parser.add_argument("--site-col", default="Site")
    parser.add_argument("--site-f", default="F")
    parser.add_argument("--qc-col", default="QC_note_wave2")
    parser.add_argument("--exclude-qc-notes", action="store_true",
                        help="Exclude rows with non-empty QC notes (used for the thesis cohort)")
    args = parser.parse_args()

    prepare(
        args.predictions, args.selected_models, args.out_dir,
        subject_col=args.subject_col,
        age_col=args.age_col,
        sex_col=args.sex_col,
        site_col=args.site_col,
        site_f=args.site_f,
        qc_col=args.qc_col,
        exclude_qc_notes=args.exclude_qc_notes,
        folds=args.folds,
    )


if __name__ == "__main__":
    main()
