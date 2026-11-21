# config.py
"""
Argument parsing and basic config utilities for the transfer learning script.
"""

import argparse
import os
import glob


def get_config():
    """
    Parse command-line arguments and auto-derive weights if not provided.

    Returns
    -------
    opt : argparse.Namespace
        Parsed configuration/options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=4, help='Training batch size')
    parser.add_argument('--nEpoch', type=int, default=60, help='number of epochs on entire dataset')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True, help='Enable CUDA if available')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    parser.add_argument('--loadModel', default=False, action='store_true', help='Load pretrained weights')
    
    parser.add_argument('--ft_ratio', type=float, default=1.0, help="Proportion of training data to use for fine-tuning (0–1).")
    parser.add_argument('--standardize', action='store_true', help='Unsupported with sigmoid output (kept only for explicit error message)')

    parser.add_argument('--weightsFolder', type=str, required=True,
        help='Folder containing pretrained .pth checkpoints')
    parser.add_argument('--weights', type=str, default=None,
        help='Pretrained state_dict (.pth). If omitted, select from weightsFolder')
    parser.add_argument('--output_dir', type=str, required=True,
        help='Folder to store model checkpoints and outputs.')

    parser.add_argument('--test', type=str, default=None,
        help='Held-out test metadata CSV')

    parser.add_argument('--csv_path', type=str,
        help='Development metadata CSV with SUBJ_ID, filename and AGE')
    parser.add_argument('--imgPath', type=str,
        help='Directory containing MRI files referenced by filename in the CSV')

    parser.add_argument(
        '--freeze_mode',
        type=str,
        default='head',
        choices=['head','dense4','dense3_4','dense2_4','dense1_4','encoder','all','none','progressive','freeze_bn','LoRA','FilteredLoRA'],
        help=(
            "Which layers to unfreeze during transfer learning:\n"
            "  head        = only regression/class_layers.out\n"
            "  dense4      = final dense block + head\n"
            "  dense3_4    = last two dense blocks + head\n"
            "  dense2_4    = last three dense blocks + head\n"
            "  dense1_4    = all dense blocks + head (same as encoder)\n"
            "  encoder     = train all except conv0/norm0\n"
            "  all         = unfreeze everything\n"
            "  none        = freeze everything (debug / inference-only)\n"
            "  progressive = progressively unfreeze layers starting with head up to all\n"
            "  freeze_bn   = freeze normalization layers, fine-tune other parameters\n"
            "  LoRA        = LoRA, DoRA or PiSSA\n"
            "  FilteredLoRA = custom filtered low-rank adaptation"
        )
    )

    parser.add_argument(
    '--ignore_subjects',
    type=str,
    nargs='*',
    default=[],
    help='Optional list of subject IDs to exclude'
)

    parser.add_argument("--balance_train", action="store_true", help="Balance training set by age bin and sex")
    parser.add_argument("--min_samples_per_bin", type=int,
        default=None,
        help=(
            "Minimum target number of samples per age bin after oversampling. "
            "If None or smaller than the largest age bin, balances to the largest bin."
        ),
    )
    parser.add_argument("--dataset_dir", type=str, help="Optional subfolder name for saving outputs (e.g., target_cohort). If not provided, outputs are saved directly under the base output folder.")
    
    # LoRA-specific arguments
    parser.add_argument("--pissa_svd_device", default=None, help="Device for PiSSA SVD (cpu or cuda)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default=None,
                        help="DenseNet scope: head, dense4, dense3_4, dense2_4, encoder or all")
    parser.add_argument("--lora_modules_to_save", type=str, default=None,
                        help="Comma-separated module names to fully unfreeze alongside LoRA")
    parser.add_argument("--use_dora", action="store_true", help="Use DoRA variant of LoRA")
    parser.add_argument("--use_pissa", action="store_true", help="Use PiSSA initialization for LoRA")

    # FiltLoRA uses the custom 3D trigonometric filter.
    parser.add_argument("--filter_kernel_size", type=int, default=3, help="Filter kernel size for CoLoRA")
    parser.add_argument("--sigma_spatial", type=float, default=1.0, help="Spatial sigma for CoLoRA filter")
    parser.add_argument("--sigma_range", type=float, default=1.0, help="Legacy unused argument; included for compatibility")
    parser.add_argument("--filter_scale", type=float, default=1.0, help="Filter scale for CoLoRA")
    
    
    

    # k-fold options
    parser.add_argument(
    "--split_dir",
    type=str,
    default=None,
    help="Directory containing the precomputed K-fold CSVs.",
    )

    parser.add_argument(
        "--split_prefix",
        type=str,
        default="dataset",
        help="Filename prefix used when the K-fold CSVs were generated.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Precomputed K-fold split to load.",
    )

    opt = parser.parse_args()
    if not opt.loadModel:
        parser.error('--loadModel is required for transfer learning from pretrained weights.')
    if opt.use_pissa and opt.use_dora:
        parser.error('--use_pissa and --use_dora cannot be combined.')
    if opt.freeze_mode == 'FilteredLoRA' and (opt.use_pissa or opt.use_dora):
        parser.error('FilteredLoRA cannot be combined with PiSSA or DoRA.')
    if opt.standardize:
        parser.error('--standardize is not supported by this sigmoid-output backbone.')
    if opt.fold is None or opt.split_dir is None or not opt.test or not opt.csv_path or not opt.imgPath:
        parser.error('--fold, --split_dir, --test, --csv_path and --imgPath are required.')
    if not 0 < opt.ft_ratio <= 1:
        parser.error('--ft_ratio must be in (0, 1].')

    # auto-derive weights if not provided
    if opt.weights is None:
        best = os.path.join(opt.weightsFolder, 'best_weights.pth')
        if os.path.isfile(best):
            opt.weights = best
        else:
            pths = sorted(glob.glob(f"{opt.weightsFolder}/*.pth"))
            if len(pths) == 0:
                raise FileNotFoundError(f"No .pth found in {opt.weightsFolder}")
            opt.weights = pths[0]

    # Special case: freeze everything → inference-only mode
    if opt.freeze_mode == "none":
        print("freeze_mode='none' → Running in inference-only mode.")
        opt.loadModel = True        # MUST load pretrained weights
        opt.skip_training = True    # used in main
    else:
        opt.skip_training = False

    return opt
