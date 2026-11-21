"""Train an adapted 3D DenseNet and evaluate the selected checkpoint."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from brainage.config import get_config
from brainage.data.utils import prepare_train_val_datasets
from brainage.inference import run_inference_and_metrics
from brainage.models.backbone import MyDenseNet
from brainage.training import run_training_stage


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_name(opt):
    pretrained = Path(opt.weightsFolder).name
    method = ('PiSSA' if opt.use_pissa else
              'DoRA' if opt.use_dora else opt.freeze_mode)
    tag = f'TL_{pretrained}_{method}_bt{opt.balance_train}_min{opt.min_samples_per_bin or ""}'
    if opt.freeze_mode in ('LoRA', 'FilteredLoRA'):
        tag += f'_r{opt.lora_r}_tm{opt.lora_target_modules or "all"}'
    if opt.freeze_mode == 'FilteredLoRA':
        tag += f'_f{opt.filter_kernel_size}_sigmaS{opt.sigma_spatial}_scale{opt.filter_scale}'
    return tag


def main():
    opt = get_config()
    set_seed(opt.manualSeed)
    device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Torch: {torch.__version__}')

    run_tag = run_name(opt)
    ft_tag = f'ft{opt.ft_ratio}'.replace('.', 'p')
    output_parts = [opt.output_dir]
    if opt.dataset_dir:
        output_parts.append(opt.dataset_dir)
    output_parts += [run_tag, f'seed{opt.manualSeed}', ft_tag]
    opt.savePath = os.path.join(*output_parts)
    os.makedirs(opt.savePath, exist_ok=True)
    summary_path = os.path.join(opt.savePath, 'TL_summary.json')

    if os.path.isfile(summary_path):
        with open(summary_path, encoding='utf8') as handle:
            previous = json.load(handle)
        if 'inference_metrics' in previous:
            print(f'Run already complete: {opt.savePath}')
            return
        raise RuntimeError(f'Incomplete run summary exists: {summary_path}')

    dev = pd.read_csv(opt.csv_path)
    test = pd.read_csv(opt.test)
    required = {'SUBJ_ID', 'AGE', 'filename'}
    for name, df in [('development', dev), ('test', test)]:
        if not required.issubset(df.columns):
            raise ValueError(f'{name} CSV is missing {required - set(df.columns)}')
    overlap = set(dev['SUBJ_ID'].astype(str)) & set(test['SUBJ_ID'].astype(str))
    if overlap:
        raise ValueError('The development and held-out test cohorts overlap.')

    trainset, valset, _, _ = prepare_train_val_datasets(dev, opt)
    model = MyDenseNet()
    state = torch.load(opt.weights, map_location='cpu', weights_only=True)
    # Checkpoint keys must match the pretrained backbone exactly.
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    summary = {'stages': []}
    best_val_loss = float('inf')
    if opt.skip_training:
        print('Inference only: using pretrained model.')
    else:
        if opt.freeze_mode == 'progressive':
            stages = [
                ('stage1', 'head', opt.lr),
                ('stage2', 'dense4', opt.lr * 0.5),
                ('stage3', 'dense3_4', opt.lr * 0.35),
                ('stage4', 'dense2_4', opt.lr * 0.25),
                ('stage5', 'dense1_4', opt.lr * 0.15),
                ('stage6', 'encoder', opt.lr * 0.1),
            ]
        else:
            stages = [(opt.freeze_mode, opt.freeze_mode, opt.lr)]
        for stage_tag, mode, learning_rate in stages:
            model, best_val_loss, summary = run_training_stage(
                net=model, trainset=trainset, valset=valset,
                age_loss=nn.L1Loss(), opt=opt, summary=summary,
                stage_tag=stage_tag, freeze_mode=mode, stage_lr=learning_rate,
                best_val_loss=best_val_loss, ft_tag=ft_tag,
            )

    # Keep the partial summary for inspection if test inference fails.
    with open(summary_path, 'w', encoding='utf8') as handle:
        json.dump(summary, handle, indent=2)
    _, summary = run_inference_and_metrics(model, test, opt, run_tag, summary, ft_tag)
    with open(summary_path, 'w', encoding='utf8') as handle:
        json.dump(summary, handle, indent=2)
    print(f'Completed run: {opt.savePath}')


if __name__ == '__main__':
    main()
