# data_utils.py
"""
Data-related helpers: path resolution, path checking, transforms, loaders
"""

import os
import glob
import pandas as pd
import numpy as np
import monai as mn
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import nibabel as nib
from brainage.data.dataset import MyCustomDataset
from pathlib import Path

class Normalize01(torch.nn.Module):  
  def __init__(self):
    super().__init__()
  
  def forward(self, img):
      lo, hi = img.min(), img.max()
      return (img - lo) / max(float(hi - lo), 1e-8)

  def __repr__(self):
    return self.__class__.__name__ + '()'
  

def loader(path):
  final_volume = np.zeros((1,200,200,200), dtype="float32")
 
  data = nib.load(path)
  data = np.asarray(data.dataobj)
  data = np.float32(data)
  
  data = data[np.newaxis, :, :, :]
  final_volume[:,9:191,:,9:191] = data[:,:,9:209,:]
  assert final_volume.min() >= 0
  return final_volume 


def _safe_file_token(value):
    value = str(value).strip()
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in value
    )


def load_precomputed_kfold_split(
    full_df,
    split_dir,
    fold,
    prefix="dataset",
    subject_col="SUBJ_ID",
):
    """
    Load one precomputed K-fold assignment and apply it to full_df.
    """

    if split_dir is None:
        raise ValueError(
            "split_dir cannot be None when loading a precomputed fold."
        )

    if fold is None:
        raise ValueError(
            "fold must be specified when split_dir is provided."
        )

    fold = int(fold)

    if fold < 1:
        raise ValueError("fold must be indexed from 1.")

    if subject_col not in full_df.columns:
        raise KeyError(f"Missing subject column: {subject_col}")

    split_dir = Path(split_dir)
    prefix = _safe_file_token(prefix)

    stem = f"{prefix}_fold_{fold}_base"

    train_subject_path = (
        split_dir / f"{stem}_train_subjects.csv"
    )

    val_subject_path = (
        split_dir / f"{stem}_validation_subjects.csv"
    )

    missing_files = [
        str(path)
        for path in (train_subject_path, val_subject_path)
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Could not find the requested precomputed fold files:\n  "
            + "\n  ".join(missing_files)
        )

    train_subject_df = pd.read_csv(train_subject_path)
    val_subject_df = pd.read_csv(val_subject_path)

    for label, subject_df in (
        ("train", train_subject_df),
        ("validation", val_subject_df),
    ):
        if subject_col not in subject_df.columns:
            raise KeyError(
                f"{label} subject file does not contain "
                f"column '{subject_col}'."
            )

    train_subjects = set(
        train_subject_df[subject_col].astype(str)
    )

    val_subjects = set(
        val_subject_df[subject_col].astype(str)
    )

    # leakage check
    overlap = train_subjects.intersection(val_subjects)

    if overlap:
        raise ValueError(
            f"Fold {fold} contains subjects in both "
            f"train and validation: {list(overlap)[:10]}"
        )

    # Subjects available in current dataset
    current_subjects = set(
        full_df[subject_col].astype(str)
    )

    assigned_subjects = train_subjects | val_subjects

    unknown_saved_subjects = (
        assigned_subjects - current_subjects
    )

    unassigned_current_subjects = (
        current_subjects - assigned_subjects
    )

    if unknown_saved_subjects:
        raise ValueError(
            f"Fold {fold} contains subjects absent from "
            f"the current dataset. Examples: "
            f"{list(unknown_saved_subjects)[:10]}"
        )

    if unassigned_current_subjects:
        raise ValueError(
            f"The current dataset contains subjects not "
            f"assigned to fold {fold}. Examples: "
            f"{list(unassigned_current_subjects)[:10]}"
        )

    subject_as_str = full_df[subject_col].astype(str)

    train_df = (
        full_df[
            subject_as_str.isin(train_subjects)
        ]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        full_df[
            subject_as_str.isin(val_subjects)
        ]
        .copy()
        .reset_index(drop=True)
    )

    assert set(
        train_df[subject_col].astype(str)
    ).isdisjoint(
        set(val_df[subject_col].astype(str))
    )

    print("\nLoaded precomputed K-fold split")
    print(f"  Split directory: {split_dir}")
    print(f"  Prefix: {prefix}")
    print(f"  Fold: {fold}")

    print(
        f"  TRAIN: {len(train_df)} rows, "
        f"{train_df[subject_col].nunique()} subjects"
    )

    print(
        f"  VAL:   {len(val_df)} rows, "
        f"{val_df[subject_col].nunique()} subjects"
    )

    return train_df, val_df


def add_age_bins(
    df,
    age_col="real_age",
    num_age_bins=8,
    use_quantiles=True,
    age_bin_width=5,
    dynamic_bins=False,
):
    """
    Adds an 'age_bins' column.
    """

    df = df.copy()

    min_age = df[age_col].min()
    max_age = df[age_col].max()
    age_range = max_age - min_age

    print("\n=== Age Bin Diagnostics ===")
    print(f"Age range: {min_age:.2f}–{max_age:.2f}  (range {age_range:.2f} years)")

    if dynamic_bins:
        dynamic_n = int(np.ceil(age_range / age_bin_width))
        num_age_bins = max(3, min(dynamic_n, 20))
        print("\n=== Dynamic Age-Bin Mode Enabled ===")
        print(f"Target ~{age_bin_width}-year bins → suggested bins: {dynamic_n}")
        print(f"Using num_age_bins = {num_age_bins}")

    if use_quantiles:
        df["age_bins"], bin_edges = pd.qcut(
            df[age_col],
            q=num_age_bins,
            labels=False,
            retbins=True,
            duplicates="drop",
        )

        print("\nAge bin boundaries (quantiles):")
        for i in range(len(bin_edges) - 1):
            print(f"  Bin {i}: {bin_edges[i]:.2f} → {bin_edges[i+1]:.2f}")

    else:
        start = np.floor(min_age / age_bin_width) * age_bin_width
        stop = np.ceil(max_age / age_bin_width) * age_bin_width + age_bin_width

        bins = np.arange(start, stop, age_bin_width)

        df["age_bins"] = pd.cut(
            df[age_col],
            bins=bins,
            include_lowest=True,
            right=False,
        )

        print("\nFixed-width age bin boundaries:")
        for interval in df["age_bins"].cat.categories:
            print(f"  {interval}")

    print("\nAge-bin counts:")
    print(df["age_bins"].value_counts().sort_index())

    return df


def find_image_path(row, root):
    """
    Reconstruct the expected preprocessed T1w brain path.
    Falls back to glob search if the exact path is missing.
    """

    # Expected file path
    fpath = os.path.join(
        root, row["filename"]
    )

    if not os.path.exists(fpath):
        fpath = None 
    return fpath


def check_image_paths(df, split_name):
    """
    Verify that all img_path entries in a dataframe exist on disk.
    Prints basic stats and returns count of missing paths.
    """
    missing_paths = [p for p in df["img_path"] if not os.path.exists(p)]
    n_missing = len(missing_paths)
    n_total = len(df)

    if n_missing > 0:
        print(f"\n⚠️  {split_name}: {n_missing}/{n_total} image files missing ({100*n_missing/n_total:.1f}%)")
        print("Examples of missing files:")
        for p in missing_paths[:5]:
            print("   ", p)
    else:
        print(f"\n✅  {split_name}: All {n_total} image files verified.")

    return n_missing


def get_train_transform():
    """
    Training-time augmentations used in the thesis.
    """
    return mn.transforms.Compose([
        mn.transforms.Resize((128,128,128)),
        mn.transforms.RandAffine(prob=0.5, translate_range=(5,5,5)),
        mn.transforms.RandRotate(prob=0.5, range_x=0.3, range_y=0.3, range_z=0.3),
        mn.transforms.RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.1),
        mn.transforms.RandScaleIntensity(factors=0.2, prob=0.25),
        mn.transforms.RandShiftIntensity(offsets=0.2, prob=0.25),
        mn.transforms.RandAdjustContrast(),
        mn.transforms.RandGaussianNoise(prob=0.25),
        Normalize01(),
        mn.transforms.ToTensor(),
    ])


def get_val_transform():
    """
    Validation and test-time transforms.
    """
    return mn.transforms.Compose([
        mn.transforms.Resize((128,128,128)),
        Normalize01(),
        mn.transforms.ToTensor(),
    ])


def print_dataset_composition(
    df,
    name,
    age_bin_col="age_bins",
    age_col="AGE",
    sex_col="Sex",
    subject_col="SUBJ_ID",
):
    print("\n" + "=" * 80)
    print(f"{name} COMPOSITION")
    print("=" * 80)

    print(f"\nN rows: {len(df)}")

    if subject_col in df.columns:
        print(f"N unique subjects: {df[subject_col].nunique()}")

    print("\nAge summary:")
    print(df[age_col].describe())

    if age_bin_col in df.columns:
        print("\nAge-bin counts:")
        print(df[age_bin_col].value_counts().sort_index())

    if sex_col in df.columns:
        print("\nSex counts:")
        print(df[sex_col].value_counts(dropna=False))

    if age_bin_col in df.columns and sex_col in df.columns:
        print("\nAge-bin × sex counts:")
        print(pd.crosstab(df[age_bin_col], df[sex_col], margins=True))

    if "is_oversampled" in df.columns:
        print("\nOriginal vs oversampled rows:")
        print(df["is_oversampled"].value_counts())


def balance_train_by_age_bin_and_sex(
    train_df,
    age_bin_col="age_bins",
    sex_col="Sex",
    min_samples_per_bin=None,
    random_state=42,
    ):
    """
    Oversample training set so that all age bins reach a common target size.

    Target rule:
    - if min_samples_per_bin is None:
        target = size of the largest existing age bin
    - if min_samples_per_bin <= size of the largest existing age bin:
        target = size of the largest existing age bin
    - if min_samples_per_bin > size of the largest existing age bin:
        target = min_samples_per_bin

    Within each age bin, oversample the currently underrepresented sex as much
    as possible, without exceeding the age-bin target size.

    Constraints:
    - only oversampling, no downsampling
    - validation/test sets are unchanged
    - sex balance is made as close as possible under the chosen age-bin target
    """

    rng = np.random.RandomState(random_state)

    train_df = train_df.copy()
    train_df["is_oversampled"] = False

    if age_bin_col not in train_df.columns:
        raise ValueError(f"Missing age-bin column: {age_bin_col}")

    if sex_col not in train_df.columns:
        raise ValueError(f"Missing sex column: {sex_col}")

    age_counts = train_df[age_bin_col].value_counts().sort_index()

    if age_counts.empty:
        raise ValueError("Cannot balance an empty training dataframe.")

    max_bin_size = int(age_counts.max())

    if min_samples_per_bin is None:
        target_per_bin = max_bin_size
    else:
        min_samples_per_bin = int(min_samples_per_bin)

        if min_samples_per_bin <= 0:
            raise ValueError("min_samples_per_bin must be positive or None.")

        target_per_bin = max(max_bin_size, min_samples_per_bin)

    print("\n=== Balancing TRAIN set ===")
    print("Original age-bin counts:")
    print(age_counts)
    print(f"\nLargest existing age bin = {max_bin_size}")
    print(f"Requested min_samples_per_bin = {min_samples_per_bin}")
    print(f"Target samples per age bin = {target_per_bin}")

    balanced_parts = []

    for age_bin in age_counts.index:
        bin_df = train_df[train_df[age_bin_col] == age_bin].copy()
        current_total = len(bin_df)

        print("\n" + "-" * 60)
        print(f"Age bin: {age_bin}")
        print(f"Current total: {current_total} / target: {target_per_bin}")

        print("Current sex counts:")
        print(bin_df[sex_col].value_counts(dropna=False))

        if current_total >= target_per_bin:
            print("Already at or above target. No age-bin oversampling needed.")
            balanced_parts.append(bin_df)
            continue

        n_to_add = target_per_bin - current_total
        extra_rows = []

        available_sexes = list(bin_df[sex_col].dropna().unique())

        if len(available_sexes) == 0:
            print("WARNING: no valid SEX values in this age bin. Sampling from whole bin.")

            sampled = bin_df.sample(
                n=n_to_add,
                replace=True,
                random_state=int(rng.randint(0, 1_000_000)),
            ).copy()

            sampled["is_oversampled"] = True
            extra_rows.append(sampled)

        else:
            # Greedy balancing:
            # repeatedly add to the currently least represented available sex.
            simulated_counts = bin_df[sex_col].value_counts().to_dict()

            for _ in range(n_to_add):
                chosen_sex = min(
                    available_sexes,
                    key=lambda s: simulated_counts.get(s, 0),
                )

                sex_pool = bin_df[bin_df[sex_col] == chosen_sex]

                if len(sex_pool) == 0:
                    sex_pool = bin_df

                sampled_one = sex_pool.sample(
                    n=1,
                    replace=True,
                    random_state=int(rng.randint(0, 1_000_000)),
                ).copy()

                sampled_one["is_oversampled"] = True
                extra_rows.append(sampled_one)

                simulated_counts[chosen_sex] = simulated_counts.get(chosen_sex, 0) + 1

        if len(extra_rows) > 0:
            extra_df = pd.concat(extra_rows, ignore_index=True)
            bin_balanced = pd.concat([bin_df, extra_df], ignore_index=True)
        else:
            bin_balanced = bin_df

        print("\nFinal sex counts after balancing:")
        print(bin_balanced[sex_col].value_counts(dropna=False))
        print(f"Final total: {len(bin_balanced)}")

        balanced_parts.append(bin_balanced)

    balanced_train = pd.concat(balanced_parts, ignore_index=True)

    print("\n=== Final balanced TRAIN composition ===")
    print("Age-bin counts:")
    print(balanced_train[age_bin_col].value_counts().sort_index())

    print("\nSex counts:")
    print(balanced_train[sex_col].value_counts(dropna=False))

    print("\nAge-bin × sex counts:")
    print(
        pd.crosstab(
            balanced_train[age_bin_col],
            balanced_train[sex_col],
            margins=True,
        )
    )

    print("\nOversampling summary:")
    print(balanced_train["is_oversampled"].value_counts())

    return balanced_train.reset_index(drop=True)

def apply_ft_ablation(train_csv, ft_ratio, bin_width=5):
    """
    Apply proportional sub-sampling for transfer learning ablation.
    Automatically prints age distributions and age-bin counts BEFORE and AFTER ablation.

    Parameters
    ----------
    train_csv : pd.DataFrame
        The stratified training dataframe (after splitting).
    ft_ratio : float
        Fraction of training samples to retain (0 < ft_ratio <= 1).
    bin_width : int
        Width of temporary age bins for reporting only.

    Returns
    -------
    pd.DataFrame
        Ablated (or unchanged) training CSV.
    """

    # 1. Handle full-data case
    if ft_ratio >= 1.0 or ft_ratio <= 0:
        print("\n🔧 Fine-tune ablation disabled → Using 100% of stratified train set.")
        return train_csv

    # 2. Number of samples to keep
    n = int(len(train_csv) * ft_ratio)
    print(f"\n🔧 Fine-tune ablation ACTIVE")
    print(f"   Using {n}/{len(train_csv)} samples ({ft_ratio*100:.1f}%) for fine-tuning.")

    # Bin helper (for reporting)
    def compute_age_bins(df):
        min_age = df["AGE"].min()
        max_age = df["AGE"].max()
        bins = np.arange(min_age, max_age + bin_width, bin_width)
        return pd.cut(df["AGE"], bins=bins, include_lowest=True)

    # ---- BEFORE ablation: report stats ----
    print("\n📊 Age distribution BEFORE ablation:")
    print(train_csv['AGE'].describe())

    print("\n📊 Sex distribution BEFORE ablation:")
    print(train_csv['Sex'].value_counts())

    # Compute & print age-bin counts
    before_bins = compute_age_bins(train_csv)
    print("\n📊 Age-bin counts BEFORE ablation:")
    print(before_bins.value_counts().sort_index())

    # 3. Proportional random sampling
    train_csv_sub = train_csv.sample(n=n, random_state=42).reset_index(drop=True)

    # ---- AFTER ablation: report stats ----
    print("\n📊 Age distribution AFTER ablation:")
    print(train_csv_sub['AGE'].describe())

    print("\n📊 Sex distribution AFTER ablation:")
    print(train_csv_sub['Sex'].value_counts())

    # Compute & print bin counts AFTER
    after_bins = compute_age_bins(train_csv_sub)
    print("\n📊 Age-bin counts AFTER ablation:")
    print(after_bins.value_counts().sort_index())

    return train_csv_sub



def prepare_train_val_datasets(df, opt):

    df = df.copy()
    # Remove ignored subjects
    if hasattr(opt, "ignore_subjects") and len(opt.ignore_subjects) > 0:

        print(f"\n⚠️ Ignoring subjects: {opt.ignore_subjects}")

        df = df[
            ~df["SUBJ_ID"]
            .astype(str)
            .isin(opt.ignore_subjects)
        ].reset_index(drop=True)

    else:
        print("\nNo subjects ignored.")
    # Resolve image paths
    # BEFORE applying fold
    df["img_path"] = df.apply(
        lambda r: find_image_path(r, opt.imgPath),
        axis=1,
    )

    print(
        f"\nRows before image-path check: {len(df)}"
    )

    df = (
        df
        .dropna(subset=["img_path"])
        .reset_index(drop=True)
    )

    print(
        f"Rows after image-path check: {len(df)}"
    )
    # Load PRECOMPUTED K-fold
    train_df, val_df = load_precomputed_kfold_split(
        full_df=df,
        split_dir=opt.split_dir,
        fold=opt.fold,
        prefix=getattr(opt, "split_prefix", "dataset"),
        subject_col="SUBJ_ID",
    )
    # FT ablation: TRAIN ONLY
    train_df = apply_ft_ablation(
        train_df,
        opt.ft_ratio,
    )
    # Standardisation values
    # BEFORE oversampling
    if opt.standardize:

        opt.train_age_mean = train_df["AGE"].mean()
        opt.train_age_std = train_df["AGE"].std()

        print(
            f"\n🔧 Standardizing ages with mean: "
            f"{opt.train_age_mean:.2f}, "
            f"std: {opt.train_age_std:.2f}"
        )

    else:
        opt.train_age_mean = None
        opt.train_age_std = None

        print(
            "\n🔧 Not standardizing ages; "
            "using division by 100."
        )
    # Age bins AFTER ablation
    train_df = add_age_bins(
        train_df,
        age_col="AGE",
        num_age_bins=8,
        age_bin_width=5,
        dynamic_bins=True,
    )
    # Balance TRAIN only
    if getattr(opt, "balance_train", False):

        train_df = balance_train_by_age_bin_and_sex(
            train_df,
            age_bin_col="age_bins",
            sex_col="Sex",
            min_samples_per_bin=getattr(
                opt,
                "min_samples_per_bin",
                None,
            ),
            random_state=getattr(
                opt,
                "manualSeed",
                42,
            ),
        )
    # Verify paths
    check_image_paths(train_df, "TRAIN")
    check_image_paths(val_df, "VAL")
    # Transforms
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    # Dataset objects
    trainset = MyCustomDataset(
        train_transform,
        loader,
        train_df,
        opt.imgPath,
        opt.standardize,
        opt.train_age_mean,
        opt.train_age_std,
    )

    valset = MyCustomDataset(
        val_transform,
        loader,
        val_df,
        opt.imgPath,
        opt.standardize,
        opt.train_age_mean,
        opt.train_age_std,
    )

    return trainset, valset, train_df, val_df
