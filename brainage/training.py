# training.py
"""
Training-related utilities: single-stage loop and (optionally) progressive training.

Refactor is minimal:
Parameters and metrics are passed explicitly to each training stage.
"""

import os
import time
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from brainage.models.finetuning import set_trainable_layers


def train_loop(
    net,
    trainset,
    valset,
    best_val_loss,
    optimizer,
    criterion,
    save_path,
    opt,
    freeze_mode,
    ft_tag,
    stage_tag="stage1",
    patience=20,
    min_delta=1e-4,
    max_epochs=200,
):
    """
    Train with early stopping on validation MAE.

    Parameters
    ----------
    net : nn.Module
    trainset, valset : Dataset
    best_val_loss : float
        Best validation loss so far.
    optimizer : torch.optim.Optimizer
    criterion : loss function
    save_path : str
    opt : argparse.Namespace
    freeze_mode : str
        Mode for freezing layers.
    stage_tag : str
        Name for this training stage.
    ft_tag : str
    patience : int
    min_delta : float
    max_epochs : int

    Returns
    -------
    net : nn.Module
        Trained network from this stage.
    best_val_loss : float
        Updated best validation loss.
    """
    device = next(net.parameters()).device
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(save_path, f"tensorboard_{stage_tag}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging → {log_dir}")

    if freeze_mode == "LoRA":
        best_adapter_path = os.path.join(save_path, f"{stage_tag}_best_adapter")
        best_ckpt = best_adapter_path
    else:
        best_ckpt = os.path.join(save_path, f"{stage_tag}_best.pth")

    epochs_no_improve = 0
    best_epoch = -1
    best_r2 = np.nan
    best_corr = np.nan

    for epoch in range(max_epochs):
        net.train()

        for module in net.modules():
            if isinstance(module, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                module.eval()

        total_train_loss = 0.0
        for img, real_age in DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=4):
            img, real_age = img.to(device), real_age.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred_age = net(img)
            loss = criterion(pred_age, real_age)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * img.size(0)

        epoch_loss_train = total_train_loss / len(trainset)
        net.eval()
        total_val_loss, preds, reals = 0.0, [], []
        with torch.no_grad():
            for img, real_age in DataLoader(valset, batch_size=opt.batchSize, shuffle=False, num_workers=4):
                img, real_age = img.to(device), real_age.to(device).unsqueeze(1)
                pred_age = net(img)
                loss = criterion(pred_age, real_age)
                total_val_loss += loss.item() * img.size(0)
                preds.append(pred_age.cpu().numpy())
                reals.append(real_age.cpu().numpy())

        epoch_loss_val = total_val_loss / len(valset)

        reals_flat = np.concatenate(reals).ravel()
        preds_flat = np.concatenate(preds).ravel()

        if len(reals_flat) > 1 and np.std(reals_flat) > 0 and np.std(preds_flat) > 0:
            val_corr = np.corrcoef(reals_flat, preds_flat)[0, 1]
        else:
            val_corr = np.nan
            

        if len(reals_flat) > 0:
            ss_res = np.sum((reals_flat - preds_flat) ** 2)
            ss_tot = np.sum((reals_flat - np.mean(reals_flat)) ** 2)
            if ss_tot > 0:
                val_r2 = 1.0 - (ss_res / ss_tot)
            else:
                val_r2 = np.nan
        else:
            val_r2 = np.nan 

        print(f"[{stage_tag}] Epoch {epoch+1} | Train {epoch_loss_train:.4f} | Val {epoch_loss_val:.4f} | Corr {val_corr:.3f} | R2 {val_r2:.3f}")

        # Training and validation metrics.
        writer.add_scalar(f"{stage_tag}/Train_Loss", epoch_loss_train, epoch)
        writer.add_scalar(f"{stage_tag}/Val_Loss", epoch_loss_val, epoch)
        writer.add_scalar(f"{stage_tag}/Val_Corr", val_corr, epoch)

        # Check improvement
        if epoch_loss_val < best_val_loss - min_delta:
            best_val_loss = epoch_loss_val
            best_r2 = val_r2
            best_corr = val_corr
            best_epoch = epoch
            epochs_no_improve = 0
            if freeze_mode == "LoRA" and not opt.use_pissa:
                net.save_pretrained(best_adapter_path)
                torch.save(net.state_dict(), os.path.join(save_path, f"{stage_tag}_best_peft_state_dict.pth"))
            else:
                torch.save(net.state_dict(), best_ckpt)
        else:
            epochs_no_improve += 1

        # Early stopping conditions
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping — validation loss plateaued at {best_val_loss:.4f}")
        
            break

    writer.close()
    print(f'Stage {stage_tag} completed {epoch + 1} epochs; best validation loss: {best_val_loss:.4f}')
    return net, best_val_loss, best_r2, best_corr, best_epoch, epoch + 1


def run_training_stage(
    net,
    trainset,
    valset,
    age_loss,
    opt,
    summary,
    stage_tag,
    freeze_mode,
    stage_lr,
    best_val_loss,
    ft_tag,
    prev_ckpt=None,
):
    """
    Configure trainable layers, fit the model and reload the best checkpoint.
    """
    print(f"\n===== {stage_tag.upper()} → Unfreeze: {freeze_mode} | LR={stage_lr:.1e} =====")
    net_stage = set_trainable_layers(net, freeze_mode, opt=opt)

    if torch.cuda.is_available() and opt.cuda:
        net_stage = net_stage.cuda()

    trainable_params = sum(p.numel() for p in net_stage.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in net_stage.parameters())

    if prev_ckpt and os.path.exists(prev_ckpt):
        if freeze_mode == "LoRA":
            net_stage.load_adapter(prev_ckpt, adapter_name="default")
        else:
            net_stage.load_state_dict(torch.load(prev_ckpt))
        print(f"Loaded previous checkpoint → {prev_ckpt}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net_stage.parameters()),
        lr=stage_lr, betas=(opt.beta1, 0.999)
    )

    stage_start = time.time()
    net_stage, best_val_loss, best_R2, best_corr, best_epoch, epochs_trained = train_loop(
        net=net_stage,
        trainset=trainset,
        valset=valset,
        best_val_loss=best_val_loss,
        optimizer=optimizer,
        criterion=age_loss,
        save_path=opt.savePath,
        opt=opt,
        freeze_mode=freeze_mode,
        stage_tag=stage_tag,
        ft_tag=ft_tag,
        patience=20,
        min_delta=1e-5,
        max_epochs=opt.nEpoch,
    )
    stage_end = time.time()

    # Reload the checkpoint selected by validation MAE.
    if freeze_mode == "LoRA":
        best_ckpt = os.path.join(opt.savePath, f"{stage_tag}_best_adapter")
    else:
        best_ckpt = os.path.join(opt.savePath, f"{stage_tag}_best.pth")


    if os.path.exists(best_ckpt):
        val_loss = best_val_loss
    else:
        val_loss = None

    if freeze_mode == "LoRA" and not opt.use_pissa:
        best_ckpt = os.path.join(
            opt.savePath,
            f"{stage_tag}_best_adapter"
        )

        if os.path.exists(best_ckpt):
            # For PEFT LoRA/DoRA adapter:
            # reload best adapter weights into the model
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            adapter_file = os.path.join(
                best_ckpt,
                "adapter_model.safetensors"
            )

            if os.path.exists(adapter_file):
                adapter_state = load_file(adapter_file)
                set_peft_model_state_dict(
                    net_stage,
                    adapter_state,
                    adapter_name="default"
                )
            else:
                raise FileNotFoundError(
                    f"Best adapter not found: {adapter_file}"
                )

    else:
        if freeze_mode == "LoRA":
            # PiSSA
            best_ckpt = os.path.join(
                opt.savePath,
                f"{stage_tag}_best_adapter"
            )
        else:
            best_ckpt = os.path.join(
                opt.savePath,
                f"{stage_tag}_best.pth"
            )

        if os.path.exists(best_ckpt):
            state_dict = torch.load(
                best_ckpt,
                map_location='cpu', weights_only=True
            )
            net_stage.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(
                f"Best checkpoint not found: {best_ckpt}"
            )

    print(
        f"Reloaded best checkpoint "
        f"(epoch {best_epoch + 1}, val={best_val_loss:.6f})"
    )

    summary["stages"].append({
        "stage": stage_tag,
        "mode": freeze_mode,
        "learning_rate": stage_lr,
        "best_val_loss": val_loss,
        "best_corr": best_corr,
        "best_R2": best_R2,
        "best_epoch": best_epoch + 1,
        "epochs_trained": epochs_trained,
        "mean_seconds_per_epoch": round((stage_end - stage_start) / epochs_trained, 3),
        "duration_min": round((stage_end - stage_start) / 60, 2),
        "checkpoint": best_ckpt if os.path.exists(best_ckpt) else None,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_pct": round(100 * trainable_params / total_params, 4),
        **({"lora_r": opt.lora_r, "lora_alpha": opt.lora_alpha,
            "lora_dropout": opt.lora_dropout}
           if freeze_mode in ("LoRA", "FilteredLoRA") else {}),
        **({"filter_kernel_size": opt.filter_kernel_size,
            "sigma_spatial": opt.sigma_spatial,
            "filter_scale": opt.filter_scale}
           if freeze_mode == "FilteredLoRA" else {}),
    })


    # Record that this stage finished.
    stop_marker = os.path.join(opt.savePath, f"{stage_tag}_stopped.txt")
    with open(stop_marker, "w") as f:
        f.write(f"Stage {stage_tag} stopped at {datetime.datetime.now()}\n")

    print(f"Stage {stage_tag} complete in {(stage_end - stage_start)/60:.1f} min")



    return net_stage, best_val_loss, summary
