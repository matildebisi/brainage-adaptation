"""Run zero-shot brain-age inference for one pretrained DenseNet checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brainage.data.utils import find_image_path, get_val_transform, loader
from brainage.models.backbone import MyDenseNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--image-root', required=True, type=Path)
    parser.add_argument('--out-csv', required=True, type=Path)
    parser.add_argument('--model-name', required=True,
                        help='Model identifier, for example TP1_White')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    net = MyDenseNet()
    net.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True),
                        strict=True)
    net = net.to(device).eval()

    df = pd.read_csv(args.metadata)
    if 'filename' not in df.columns:
        raise ValueError('Metadata must include a filename column.')
    transform = get_val_transform()
    predictions = []
    for _, row in df.iterrows():
        path = find_image_path(row, args.image_root)
        if path is None:
            predictions.append(np.nan)
            continue
        x = transform(loader(path)).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            predictions.append(float(net(x).item() * 100.0))

    column = f'{args.model_name}_pred_age'
    df[column] = predictions
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f'Saved {len(df)} predictions ({column}) to {args.out_csv}')


if __name__ == '__main__':
    main()
