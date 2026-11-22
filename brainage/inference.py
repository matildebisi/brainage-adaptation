# inference.py
"""
Inference and evaluation logic
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from brainage.data.utils import find_image_path, get_val_transform, loader

def run_inference_and_metrics(net, df, opt, unique_run_tag, summary, ft_tag):
    """
    Run inference, compute metrics, and update the summary dict.

    Parameters
    ----------
    net : nn.Module
    df : pd.DataFrame
    opt : argparse.Namespace
    unique_run_tag : str
    summary : dict

    Returns
    -------
    df : pd.DataFrame
        Updated dataframe with prediction column.
    summary : dict
        Updated summary with inference_metrics.
    """
    
    pred_col = f"dl_tl_{unique_run_tag}_{ft_tag}_pred_age"
    net.eval()
    device = 'cuda' if torch.cuda.is_available() and opt.cuda else 'cpu'

    val_transform = get_val_transform()
    preds = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fpath = find_image_path(row, opt.imgPath)
        if not fpath or not os.path.exists(fpath):
            print(f"⚠️ Missing image for {row['SESSION_ID']}")
            preds.append(np.nan)
            continue

        img = loader(fpath)
        img = val_transform(img)
        img = img.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            if opt.standardize:
                preds.append((net(img).cpu().item() * opt.train_age_std) + opt.train_age_mean)
            else:
                preds.append(net(img).cpu().item() * 100.0)

    df[pred_col] = preds

    # tag training rows and insert predictions for all test subjects in the df
    merged_out = os.path.join(opt.savePath, f"test_{unique_run_tag}_{ft_tag}.csv")
    df.to_csv(merged_out, index=False)
    print(f"Saved → {merged_out}")

    # evaluation on wave 2 healthy controls only
    eval_df = df[(df['CDR']==0) & (df['SES_NUM']==2)].dropna(subset=[pred_col])
    if eval_df.empty:
        raise ValueError('No valid wave-2 healthy-control predictions to evaluate.')
    mse = np.mean((eval_df['AGE'] - eval_df[pred_col].astype(float))**2)
    mae = np.mean(np.abs(eval_df['AGE'] - eval_df[pred_col].astype(float)))
    corr = np.corrcoef(eval_df['AGE'].values, eval_df[pred_col].astype(float).values)[0,1]
    print(f"Wave 2 Test results → MSE={mse:.3f} | MAE={mae:.3f} | Corr={corr:.3f}")

    summary["inference_metrics"] = {
        "mse": float(mse),
        "mae": float(mae),
        "corr": float(corr),
        "n_samples": int(eval_df.shape[0]),
        "pred_col": pred_col,
        "output_csv": merged_out,
        "ft_tag": ft_tag,
        
    }

    return df, summary
