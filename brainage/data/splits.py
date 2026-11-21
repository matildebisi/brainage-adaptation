"""Create reproducible subject-level stratified K-fold train/validation CSVs.

The split is performed once at subject level. All rows/timepoints belonging to
one subject are kept in the same fold. Stratification uses a composite label
formed by age bin and sex.

For K=4, each fold contains approximately 75% of subjects in training and 25%
in validation. Validation subjects are mutually exclusive across folds and each
subject appears in validation exactly once.

Example
-------
python save_splits_kfold.py \
    --csv <PLACEHOLDER>/metadata.csv \
    --img-root <PLACEHOLDER>/preprocessed/images \
    --output-dir <PLACEHOLDER>/saved_splits \
    --prefix BRK \
    --n-splits 4 \
    --split-seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from brainage.data.utils import find_image_path


def safe_file_token(value: object) -> str:
    """Convert a value to a filesystem-friendly string."""
    value = str(value).strip()
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in value
    )


def make_subject_table(
    df: pd.DataFrame,
    subject_col: str = "SUBJ_ID",
    age_col: str = "AGE",
    sex_col: str = "Sex",
    site_col: str = "Site",
) -> pd.DataFrame:
    """Return one representative baseline row per subject.

    The representative row is used only to define age/sex strata. All original
    rows/timepoints are restored after subjects have been assigned to folds.
    """
    required = {subject_col, age_col, sex_col, site_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if df[subject_col].isna().any():
        raise ValueError(f"Column '{subject_col}' contains missing values.")
    if df[age_col].isna().any():
        raise ValueError(f"Column '{age_col}' contains missing values.")
    if df[sex_col].isna().any():
        raise ValueError(f"Column '{sex_col}' contains missing values.")

    sex_nunique = df.groupby(subject_col)[sex_col].nunique(dropna=False)
    inconsistent_sex = sex_nunique[sex_nunique > 1]
    if not inconsistent_sex.empty:
        raise ValueError(
            "Some subjects have inconsistent sex values across rows. "
            f"Examples: {inconsistent_sex.index[:10].tolist()}"
        )

    site_nunique = df.groupby(subject_col)[site_col].nunique(dropna=False)
    inconsistent_site = site_nunique[site_nunique > 1]

    if not inconsistent_site.empty:
        raise ValueError(
            "Some subjects have inconsistent site values across rows. "
            f"Examples: {inconsistent_site.index[:10].tolist()}"
        )

    sort_columns = [subject_col]
    if "timepoint" in df.columns:
        sort_columns.append("timepoint")

    subject_df = (
        df.sort_values(sort_columns)
        .drop_duplicates(subset=subject_col, keep="first")
        .reset_index(drop=True)
    )

    print(f"\nRows/scans available: {len(df)}")
    print(f"Unique subjects available: {len(subject_df)}")
    return subject_df


def add_stratification_labels(
    subject_df: pd.DataFrame,
    n_splits: int,
    num_age_bins: int = 8,
    use_quantiles: bool = True,
    age_bin_width: float = 5.0,
    dynamic_bins: bool = True,
    age_col: str = "AGE",
    sex_col: str = "Sex",
    site_col: str = "Site",
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Add age bins and composite age-bin × sex labels at subject level.

    StratifiedKFold requires every composite stratum to contain at least
    ``n_splits`` subjects. For quantile bins, the number of age bins is reduced
    automatically until this condition is satisfied. Fixed-width bins are not
    changed automatically; an informative error is raised instead.
    """
    subject_df = subject_df.copy()

    age_min = float(subject_df[age_col].min())
    age_max = float(subject_df[age_col].max())
    age_range = age_max - age_min

    requested_bins = int(num_age_bins)
    if dynamic_bins:
        dynamic_n = int(np.ceil(age_range / age_bin_width))
        requested_bins = max(2, min(dynamic_n, 20))
        print("\nDynamic age-bin mode enabled")
        print(f"Age range: {age_min:.2f}–{age_max:.2f}")
        print(f"Requested approximately {age_bin_width:g}-year resolution")
        print(f"Initial requested number of bins: {requested_bins}")

    bin_edges: np.ndarray | None = None

    if use_quantiles:
        accepted = False
        max_possible_bins = min(requested_bins, subject_df[age_col].nunique())

        for current_bins in range(max_possible_bins, 1, -1):
            age_bins, candidate_edges = pd.qcut(
                subject_df[age_col],
                q=current_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            candidate = subject_df.copy()
            candidate["age_bins"] = age_bins.astype(int)
            candidate["stratify_group"] = (
                candidate["age_bins"].astype(str)
                + "_"
                + candidate[sex_col].astype(str)
                + "_"
                + candidate[site_col].astype(str)
            )
            counts = candidate["stratify_group"].value_counts()

            if not counts.empty and int(counts.min()) >= n_splits:
                subject_df = candidate
                bin_edges = candidate_edges
                accepted = True
                print(
                    f"Using {subject_df['age_bins'].nunique()} quantile age bins; "
                    f"smallest age-bin × sex x site stratum has {int(counts.min())} subjects."
                )
                break

        if not accepted:
            raise ValueError(
                "Unable to create at least two age bins while keeping every "
                f"age-bin × sex stratum at or above {n_splits} subjects. "
                "Use fewer folds, inspect sex coding, or increase the sample size."
            )

    else:
        start = np.floor(age_min / age_bin_width) * age_bin_width
        stop = np.ceil(age_max / age_bin_width) * age_bin_width + age_bin_width
        bin_edges = np.arange(start, stop, age_bin_width)

        subject_df["age_bins"] = pd.cut(
            subject_df[age_col],
            bins=bin_edges,
            labels=False,
            include_lowest=True,
            right=False,
        )
        if subject_df["age_bins"].isna().any():
            raise ValueError("Some ages could not be assigned to fixed-width bins.")

        subject_df["age_bins"] = subject_df["age_bins"].astype(int)
        subject_df["stratify_group"] = (
            subject_df["age_bins"].astype(str)
            + "_"
            + subject_df[sex_col].astype(str)
            + "_"
            + subject_df[site_col].astype(str)
        )
        counts = subject_df["stratify_group"].value_counts()
        too_small = counts[counts < n_splits]
        if not too_small.empty:
            raise ValueError(
                "Fixed-width age bins created strata that are too small for "
                f"{n_splits}-fold stratification:\n{too_small}\n"
                "Increase --age-bin-width or use quantile bins."
            )

    print("\nAge-bin boundaries:")
    if bin_edges is not None:
        for index in range(len(bin_edges) - 1):
            print(f"  Bin {index}: {bin_edges[index]:.2f} -> {bin_edges[index + 1]:.2f}")

    print("\nSubject-level age-bin × sex x site counts:")
    print(subject_df["stratify_group"].value_counts().sort_index())

    return subject_df, bin_edges


def save_fold_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str | Path,
    fold: int,
    n_splits: int,
    split_seed: int,
    prefix: str = "dataset",
    subject_col: str = "SUBJ_ID",
) -> dict[str, str]:
    """Save the same five CSV types previously produced for each split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = safe_file_token(prefix)
    fold = int(fold)
    stem = f"{prefix}_fold_{fold}_base"

    train_subjects = (
        train_df[[subject_col]]
        .drop_duplicates()
        .sort_values(subject_col, key=lambda series: series.astype(str))
        .reset_index(drop=True)
    )
    val_subjects = (
        val_df[[subject_col]]
        .drop_duplicates()
        .sort_values(subject_col, key=lambda series: series.astype(str))
        .reset_index(drop=True)
    )

    train_ids = set(train_subjects[subject_col].astype(str))
    val_ids = set(val_subjects[subject_col].astype(str))
    overlap = sorted(train_ids.intersection(val_ids))
    if overlap:
        raise ValueError(
            f"Fold {fold}: found {len(overlap)} subjects in both train and validation. "
            f"Examples: {overlap[:10]}"
        )

    assignments = pd.concat(
        [
            train_subjects.assign(split="train"),
            val_subjects.assign(split="validation"),
        ],
        ignore_index=True,
    )
    assignments.insert(0, "split_seed", int(split_seed))
    assignments.insert(0, "n_splits", int(n_splits))
    assignments.insert(0, "fold", fold)
    assignments = assignments.sort_values(
        ["split", subject_col],
        key=lambda series: series.astype(str),
    ).reset_index(drop=True)

    paths = {
        "train_rows": output_dir / f"{stem}_train_rows.csv",
        "validation_rows": output_dir / f"{stem}_validation_rows.csv",
        "train_subjects": output_dir / f"{stem}_train_subjects.csv",
        "validation_subjects": output_dir / f"{stem}_validation_subjects.csv",
        "assignments": output_dir / f"{stem}_subject_assignments.csv",
    }

    train_df.reset_index(drop=True).to_csv(paths["train_rows"], index=False)
    val_df.reset_index(drop=True).to_csv(paths["validation_rows"], index=False)
    train_subjects.to_csv(paths["train_subjects"], index=False)
    val_subjects.to_csv(paths["validation_subjects"], index=False)
    assignments.to_csv(paths["assignments"], index=False)

    print("\nSaved fold CSVs")
    print(f"  Fold: {fold}/{n_splits}")
    print(f"  Split seed: {split_seed}")
    print(f"  Train rows: {len(train_df)}")
    print(f"  Validation rows: {len(val_df)}")
    print(f"  Unique train subjects: {len(train_subjects)}")
    print(f"  Unique validation subjects: {len(val_subjects)}")
    for label, path in paths.items():
        print(f"  {label}: {path}")

    return {label: str(path) for label, path in paths.items()}


def generate_kfold_splits(
    df: pd.DataFrame,
    img_root: str,
    output_dir: str | Path,
    prefix: str,
    n_splits: int = 4,
    split_seed: int = 42,
    ignore_subjects: list[str] | None = None,
    num_age_bins: int = 8,
    use_quantiles: bool = True,
    age_bin_width: float = 5.0,
    dynamic_bins: bool = True,
    subject_col: str = "SUBJ_ID",
    age_col: str = "AGE",
    sex_col: str = "Sex",
    site_col: str = "Site",
) -> None:
    """Generate one subject-level stratified K-fold partition and save all folds."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    work_df = df.copy()
    ignore_subjects = [] if ignore_subjects is None else [
        str(subject) for subject in ignore_subjects
    ]
    if ignore_subjects:
        work_df = work_df[
            ~work_df[subject_col].astype(str).isin(ignore_subjects)
        ].reset_index(drop=True)

    required_group_col = "transfer_learning_group"
    if required_group_col not in work_df.columns:
        raise KeyError(f"Missing required column: {required_group_col}")

    work_df = work_df[
        work_df[required_group_col]
        .str.lower()
        .str.contains("dev_set", na=False)
    ].copy()

    print(f"\nDevelopment set rows before image-path check: {len(work_df)}")

    work_df["img_path"] = work_df.apply(
        lambda row: find_image_path(row, img_root),
        axis=1,
    )
    work_df = work_df.dropna(subset=["img_path"]).reset_index(drop=True)

    print(f"Development set rows after image-path check: {len(work_df)}")

    subject_df = make_subject_table(
        work_df,
        subject_col=subject_col,
        age_col=age_col,
        sex_col=sex_col,
        site_col=site_col,
    )
    subject_df, bin_edges = add_stratification_labels(
        subject_df,
        n_splits=n_splits,
        num_age_bins=num_age_bins,
        use_quantiles=use_quantiles,
        age_bin_width=age_bin_width,
        dynamic_bins=dynamic_bins,
        age_col=age_col,
        sex_col=sex_col,
        site_col=site_col,
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(split_seed),
    )

    all_subject_ids = set(subject_df[subject_col].astype(str))
    validation_sets: list[set[str]] = []
    manifest_rows: list[dict[str, object]] = []

    for fold_index, (train_indices, val_indices) in enumerate(
        splitter.split(subject_df, subject_df["stratify_group"]),
        start=1,
    ):
        print("\n" + "=" * 80)
        print(f"GENERATING FOLD {fold_index}/{n_splits}")
        print("=" * 80)

        train_subject_ids = set(
            subject_df.iloc[train_indices][subject_col].astype(str)
        )
        val_subject_ids = set(
            subject_df.iloc[val_indices][subject_col].astype(str)
        )

        if not train_subject_ids.isdisjoint(val_subject_ids):
            raise RuntimeError(f"Fold {fold_index}: train/validation subject overlap detected.")

        train_df = work_df[
            work_df[subject_col].astype(str).isin(train_subject_ids)
        ].copy().reset_index(drop=True)
        val_df = work_df[
            work_df[subject_col].astype(str).isin(val_subject_ids)
        ].copy().reset_index(drop=True)

        save_fold_csvs(
            train_df=train_df,
            val_df=val_df,
            output_dir=output_dir,
            fold=fold_index,
            n_splits=n_splits,
            split_seed=int(split_seed),
            prefix=prefix,
            subject_col=subject_col,
        )

        validation_sets.append(val_subject_ids)
        manifest_rows.append(
            {
                "fold": fold_index,
                "n_splits": n_splits,
                "split_seed": int(split_seed),
                "n_train_subjects": len(train_subject_ids),
                "n_validation_subjects": len(val_subject_ids),
                "n_train_rows": len(train_df),
                "n_validation_rows": len(val_df),
            }
        )

        print("\nValidation age-bin × sex composition:")
        fold_val_subjects = subject_df.iloc[val_indices]
        print(pd.crosstab(
            fold_val_subjects["age_bins"],
            fold_val_subjects[sex_col],
            margins=True,
        ))

        print("\nValidation site × sex composition:")
        print(
            pd.crosstab(
                fold_val_subjects[site_col],
                fold_val_subjects[sex_col],
                margins=True,
            )
)


    # Global K-fold checks.
    for left in range(len(validation_sets)):
        for right in range(left + 1, len(validation_sets)):
            overlap = validation_sets[left].intersection(validation_sets[right])
            if overlap:
                raise RuntimeError(
                    f"Validation folds {left + 1} and {right + 1} overlap: "
                    f"{sorted(overlap)[:10]}"
                )

    validation_union = set().union(*validation_sets)
    if validation_union != all_subject_ids:
        missing = sorted(all_subject_ids.difference(validation_union))
        extra = sorted(validation_union.difference(all_subject_ids))
        raise RuntimeError(
            "Validation folds do not form an exact partition of all subjects. "
            f"Missing examples: {missing[:10]}; extra examples: {extra[:10]}"
        )

    output_dir = Path(output_dir)
    prefix_token = safe_file_token(prefix)
    manifest_path = output_dir / f"{prefix_token}_kfold_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    bins_path = output_dir / f"{prefix_token}_age_bin_boundaries.csv"
    if bin_edges is not None:
        pd.DataFrame(
            {
                "age_bin": np.arange(len(bin_edges) - 1),
                "lower": bin_edges[:-1],
                "upper": bin_edges[1:],
            }
        ).to_csv(bins_path, index=False)

    print("\n" + "=" * 80)
    print("K-FOLD SPLIT COMPLETE")
    print("=" * 80)
    print(f"Every subject appears in validation exactly once across {n_splits} folds.")
    print(f"Manifest: {manifest_path}")
    if bin_edges is not None:
        print(f"Age-bin boundaries: {bins_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible subject-level stratified K-fold train/validation "
            "CSVs using age-bin × sex strata."
        )
    )
    parser.add_argument("--csv", required=True, help="Metadata CSV used during training.")
    parser.add_argument("--img-root", required=True, help="Root directory containing MRI images.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for fold CSVs.")
    parser.add_argument("--prefix", default="dataset", help="Prefix for output filenames.")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used only to construct the shuffled K-fold partition.",
    )
    parser.add_argument("--ignore-subjects", nargs="*", default=[])
    parser.add_argument("--num-age-bins", type=int, default=8)
    parser.add_argument("--age-bin-width", type=float, default=5.0)
    parser.add_argument(
        "--fixed-width-bins",
        action="store_true",
        help="Use fixed-width bins instead of quantile bins.",
    )
    parser.add_argument(
        "--no-dynamic-bins",
        action="store_true",
        help="Disable dynamic initial selection of the number of age bins.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    generate_kfold_splits(
        df=df,
        img_root=args.img_root,
        output_dir=args.output_dir,
        prefix=args.prefix,
        n_splits=args.n_splits,
        split_seed=args.split_seed,
        ignore_subjects=args.ignore_subjects,
        num_age_bins=args.num_age_bins,
        use_quantiles=not args.fixed_width_bins,
        age_bin_width=args.age_bin_width,
        dynamic_bins=not args.no_dynamic_bins,
    )


if __name__ == "__main__":
    main()