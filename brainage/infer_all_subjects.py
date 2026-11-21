"""Generate per-run development predictions from trained brain-age checkpoints.

This reconstructs the post-training development inference used in the thesis.
Each TL_summary.json supplies the selected checkpoint; for every run the script
saves development_predictions.csv next to that summary. The downstream merge
and validation-only selection are separate steps.

Run from the repository root, for example:

    python -m brainage.infer_all_subjects \\
        --runs-dir '<RUNS_DIR>' \\
        --development-csv '<DEVELOPMENT_CSV>' \\
        --image-root '<IMAGE_ROOT>' \\
        --pretrained-root '<PRETRAINED_WEIGHTS_ROOT>'

The pretrained root must contain TP1_White/best_weights.pth, TP2_Asian/..., etc.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

PRETRAINING_NAMES = ("TP1_White", "TP2_Asian", "TP2_Black", "TP3")


def identify_run(summary_path: Path) -> tuple[str, str]:
    """Return (TL run tag, pretraining paradigm) from TL_*/seedN/ft... path."""
    run_tags = [p.name for p in summary_path.parents if p.name.startswith("TL_")]
    if len(run_tags) != 1:
        raise ValueError(f"Expected exactly one TL_<run> directory: {summary_path}")
    tag = run_tags[0]
    matches = [name for name in PRETRAINING_NAMES if tag.startswith(f"TL_{name}_")]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify pretrained paradigm in run '{tag}'")
    return tag, matches[0]


def method_for_run(tag: str, last_stage: dict) -> str:
    if "_none_" in tag:
        return "Zero-shot"
    if "_PiSSA_" in tag:
        return "PiSSA"
    if "_FilteredLoRA_" in tag:
        return "FiltLoRA"
    if "_DoRA_" in tag:
        return "DoRA"
    if "_LoRA_" in tag:
        return "LoRA"
    mode = str(last_stage.get("mode", ""))
    if not mode:
        raise ValueError(f"Missing training mode in summary for {tag}")
    if mode in ("LoRA", "FilteredLoRA"):
        raise ValueError(f"Cannot determine adapter variant from run name {tag}")
    return "Traditional"


def extract_target_scope(tag: str) -> str:
    match = re.search(r"_tm(.+?)(?=_f\d+_sigmaS|$)", tag)
    if not match:
        raise ValueError(f"Missing adapter target scope (_tm...) in {tag}")
    scope = match.group(1)
    return "all" if scope in {"None", "null"} else scope


def _run_param(tag: str, pattern: str, parameter: str, cast):
    match = re.search(pattern, tag)
    if match is None:
        raise ValueError(f"Missing {parameter} in run tag '{tag}'")
    return cast(match.group(1))


def _checkpoint_candidates(summary_path: Path, raw_checkpoint: str | None) -> list[Path]:
    """Accommodate checkpoint paths saved before copying a run directory."""
    run_dir = summary_path.parent
    candidates: list[Path] = []
    if raw_checkpoint:
        cp = Path(raw_checkpoint).expanduser()
        candidates.append(cp)
        if not cp.is_absolute():
            candidates.append(run_dir / cp)
        candidates.append(run_dir / cp.name)
    return candidates


def resolve_checkpoint(summary_path: Path, summary: dict) -> Path:
    stages = summary.get("stages", [])
    if not stages:
        raise ValueError(f"No training stage in {summary_path}; cannot load a fine-tuned checkpoint")
    last = stages[-1]
    raw = last.get("checkpoint")
    candidates = _checkpoint_candidates(summary_path, raw)
    # Search only the run's own folder, to avoid loading a checkpoint from a different run.
    for mask in ("*_best_adapter", "*_best.pth", "*_best_peft_state_dict.pth"):
        candidates.extend(sorted(summary_path.parent.glob(mask)))
    for candidate in candidates:
        if candidate.exists() and (candidate.is_file() or candidate.is_dir()):
            return candidate
    raise FileNotFoundError(
        f"No checkpoint for {summary_path}. Stored value: {raw!r}. "
        "Expected a *_best.pth file or *_best_adapter directory/file next to the summary."
    )


def resolve_pretrained_weights(root: Path, paradigm: str, filename: str) -> Path:
    weights = root / paradigm / filename
    if not weights.is_file():
        raise FileNotFoundError(
            f"Pretrained weights not found at {weights}. "
            "Expected <PRETRAINED_WEIGHTS_ROOT>/<TP_NAME>/best_weights.pth."
        )
    return weights


def read_development_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"SUBJ_ID": str}, low_memory=False)
    required = {"SUBJ_ID", "AGE", "filename"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} is missing columns: {sorted(required - set(df.columns))}")
    if df["SUBJ_ID"].isna().any():
        raise ValueError(f"Missing SUBJ_ID values in {path}")
    # The thesis's original development inference discarded duplicate sessions.
    if "SESSION_ID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="SESSION_ID", keep="first").reset_index(drop=True)
        if before != len(df):
            print(f"Deduplicated SESSION_ID: {before} -> {len(df)} records")
    if df["SUBJ_ID"].duplicated().any():
        raise ValueError(
            "The development CSV still has repeated SUBJ_ID values. "
            "merge_predictions.py uses SUBJ_ID as a unique key; resolve repeated sessions first."
        )
    return df


def _read_state_dict(weights: Path, device: torch.device) -> dict:
    state = torch.load(weights, map_location=device, weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected a PyTorch state_dict in {weights}")
    return state


def reconstruct_custom_options(
    tag: str, stage: dict, method: str, filter_alpha: float | None
) -> SimpleNamespace:
    scope = extract_target_scope(tag)
    rank = int(stage.get("lora_r") or _run_param(tag, r"_r(\d+)(?:_|$)", "rank", int))
    alpha = stage.get("lora_alpha")
    if alpha is None:
        if method == "FiltLoRA":
            alpha = filter_alpha
            if alpha is None:
                raise ValueError(
                    f"FiltLoRA alpha is absent from the summary for {tag}. "
                    "Supply the actual training value via --filter-alpha; "
                    "do not assume it equals the rank."
                )
        else:
            raise ValueError(f"PiSSA alpha missing from training summary: {tag}")
    # Filter configuration is encoded in the run tag by brainage.main.run_name.
    if method == "FiltLoRA":
        filter_kernel_size = _run_param(tag, r"_f(\d+)_sigmaS", "filter kernel size", int)
        sigma = _run_param(tag, r"_sigmaS([0-9]+(?:\.[0-9]+)?)_scale", "spatial sigma", float)
        scale = _run_param(tag, r"_scale([0-9]+(?:\.[0-9]+)?)$", "filter scale", float)
    else:
        filter_kernel_size, sigma, scale = 3, 1.0, 1.0
    return SimpleNamespace(
        lora_r=rank,
        lora_alpha=float(alpha),
        lora_target_modules=scope,
        lora_dropout=0.0,  # Dropout is disabled in eval mode.
        lora_modules_to_save=None,
        pissa_svd_device=None,
        use_pissa=(method == "PiSSA"),
        use_dora=False,
        filter_kernel_size=filter_kernel_size,
        sigma_spatial=sigma,
        sigma_range=1.0,  # Unused legacy parameter in the custom filter.
        filter_scale=scale,
    )


def load_trained_model(
    tag: str,
    stage: dict,
    checkpoint: Path | None,
    pretrained_weights: Path,
    device: torch.device,
    filter_alpha: float | None,
):
    # Delayed imports let --help and dry-run work without a MONAI installation.
    from brainage.models.backbone import MyDenseNet
    from brainage.models.finetuning import set_trainable_layers

    method = method_for_run(tag, stage)
    net = MyDenseNet()
    net.load_state_dict(_read_state_dict(pretrained_weights, device), strict=True)

    if method == "Zero-shot":
        pass  # The base model is already loaded; no fine-tuning checkpoint exists.
    elif method in {"LoRA", "DoRA"}:
        if checkpoint is None or not checkpoint.is_dir() or not (checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"{method} requires a PEFT adapter directory with adapter_config.json: {checkpoint}"
            )
        from peft import PeftModel
        net = PeftModel.from_pretrained(net, str(checkpoint), is_trainable=False)
    else:
        if checkpoint is None or not checkpoint.is_file():
            raise ValueError(f"Expected full model state_dict file for {method}: {checkpoint}")
        if method in {"PiSSA", "FiltLoRA"}:
            options = reconstruct_custom_options(tag, stage, method, filter_alpha)
            net = set_trainable_layers(
                net, "LoRA" if method == "PiSSA" else "FilteredLoRA", opt=options
            )
        net.load_state_dict(_read_state_dict(checkpoint, device), strict=True)

    net.to(device)
    net.eval()
    return net


@torch.inference_mode()
def predict_development(model, df: pd.DataFrame, image_root: Path, device: torch.device,
                        allow_missing_images: bool = False) -> list[float]:
    from brainage.data.utils import find_image_path, get_val_transform, loader

    transform = get_val_transform()
    result: list[float] = []
    for _, row in df.iterrows():
        file = find_image_path(row, str(image_root))
        if file is None or not Path(file).is_file():
            if not allow_missing_images:
                raise FileNotFoundError(f"Image missing for SUBJ_ID={row['SUBJ_ID']}: {row['filename']}")
            result.append(float("nan"))
            continue
        image = transform(loader(file)).unsqueeze(0).to(device=device, dtype=torch.float32)
        result.append(float(model(image).detach().cpu().item() * 100.0))
    return result


def process_summary(
    summary_path: Path,
    dev: pd.DataFrame,
    image_root: Path,
    pretrained_root: Path,
    pretrained_filename: str,
    device: torch.device,
    filter_alpha: float | None,
    overwrite: bool,
    dry_run: bool,
    allow_missing_images: bool,
) -> Path:
    run_tag, paradigm = identify_run(summary_path)
    output = summary_path.parent / "development_predictions.csv"
    if output.exists() and not overwrite:
        print(f"SKIP already exists: {output}")
        return output
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    stage = summary.get("stages", [])[-1] if summary.get("stages") else {"mode": "none"}
    method = method_for_run(run_tag, stage)
    if method != "Zero-shot" and not summary.get("stages"):
        raise ValueError(f"No training stages in {summary_path}: {run_tag}")
    checkpoint = (None if method == "Zero-shot"
                  else resolve_checkpoint(summary_path, summary))
    pretrained = resolve_pretrained_weights(pretrained_root, paradigm, pretrained_filename)
    if method in {"PiSSA", "FiltLoRA"}:
        # Validate all required reconstruction metadata even for --dry-run.
        reconstruct_custom_options(run_tag, stage, method, filter_alpha)
    print(f"{run_tag} | {summary_path.parent.name} | {method}")
    print(f"  pretrained: {pretrained}")
    print(f"  checkpoint: {checkpoint if checkpoint is not None else '(pretrained weights: zero-shot)'}")
    if dry_run:
        print(f"  would write: {output}")
        return output
    model = load_trained_model(run_tag, stage, checkpoint, pretrained, device, filter_alpha)
    try:
        result = dev.copy()
        result[f"dl_tl_{run_tag}_pred_age"] = predict_development(
            model, dev, image_root, device, allow_missing_images=allow_missing_images
        )
        result.to_csv(output, index=False)
        print(f"  wrote {len(result)} development predictions: {output}")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, required=True, help="Root containing TL_*/seedN/ftXpY/TL_summary.json")
    parser.add_argument("--development-csv", type=Path, required=True, help="Development subject metadata")
    parser.add_argument("--image-root", type=Path, required=True, help="Root of preprocessed MRI files")
    parser.add_argument("--pretrained-root", type=Path, required=True,
                        help="Directory with <TP_NAME>/<pretrained-filename> for each paradigm")
    parser.add_argument("--pretrained-filename", default="best_weights.pth")
    parser.add_argument("--summary", type=Path, help="Process a single TL_summary.json; default: all under --runs-dir")
    parser.add_argument("--filter-alpha", type=float,
                        help="Actual FiltLoRA alpha if not stored in its TL_summary.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing development_predictions.csv")
    parser.add_argument("--dry-run", action="store_true", help="Validate run paths/configuration without inference")
    parser.add_argument("--allow-missing-images", action="store_true", help="Write NaNs for missing MRI images")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed run; print failed runs at end")
    args = parser.parse_args()

    if args.filter_alpha is not None and args.filter_alpha <= 0:
        parser.error("--filter-alpha must be strictly positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    dev = read_development_csv(args.development_csv)
    if args.summary:
        summaries = [args.summary]
    else:
        summaries = sorted(args.runs_dir.rglob("TL_summary.json"))
    if not summaries:
        parser.error(f"No TL_summary.json under {args.runs_dir}")
    print(f"Found {len(summaries)} run summaries; device: {device}")
    failures = []
    for index, summary in enumerate(summaries, 1):
        print(f"[{index}/{len(summaries)}] {summary}")
        try:
            process_summary(
                summary, dev, args.image_root, args.pretrained_root,
                args.pretrained_filename, device, args.filter_alpha,
                args.overwrite, args.dry_run, args.allow_missing_images,
            )
        except Exception as exc:
            failures.append((summary, str(exc)))
            print(f"ERROR: {summary}: {exc}")
            if not args.keep_going:
                raise
    if failures:
        details = "\n".join(f"  {path}: {error}" for path, error in failures)
        raise RuntimeError(f"Failed to process {len(failures)} runs:\n{details}")
    print("Development inference completed. Next: merge_predictions.py --mode development")


if __name__ == "__main__":
    main()
