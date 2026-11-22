"""Choose which parameters to train without modifying pretrained weights."""

import torch.nn as nn
from .adapters import (
    inject_filtered_lora_3d, inject_pissa_3d,
    get_lora_target_modules, print_trainable_parameters_custom,
)

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def freeze_batch_norm(model):
    for layer in model.modules():
        if isinstance(layer, BN_TYPES):
            layer.eval()
            layer.requires_grad_(False)


def set_trainable_layers(model, mode, opt):
    """Configure full fine-tuning or an adapter method after loading pretrained weights."""
    scope = opt.lora_target_modules or 'all'
    if str(scope).lower() in ('none', 'null'):
        scope = 'all'

    if mode in ('LoRA', 'FilteredLoRA'):
        target_modules = get_lora_target_modules(model, scope)
        if mode == 'FilteredLoRA':
            if opt.use_pissa or opt.use_dora:
                raise ValueError('FiltLoRA is a separate method; do not combine it with PiSSA/DoRA.')
            model, replaced = inject_filtered_lora_3d(
                model, target_modules, r=opt.lora_r, alpha=opt.lora_alpha,
                dropout=opt.lora_dropout,
                filter_kernel_size=opt.filter_kernel_size,
                filter_sigma_spatial=opt.sigma_spatial,
                filter_sigma_range=opt.sigma_range,
                filter_scale=opt.filter_scale,
            )
        elif opt.use_pissa:
            if opt.use_dora:
                raise ValueError('PiSSA and DoRA cannot both be enabled.')
            model, replaced = inject_pissa_3d(
                model, target_modules, r=opt.lora_r, alpha=opt.lora_alpha,
                dropout=opt.lora_dropout, svd_device=opt.pissa_svd_device,
            )
        else:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=opt.lora_r, lora_alpha=opt.lora_alpha,
                target_modules=target_modules,
                modules_to_save=(opt.lora_modules_to_save.split(',')
                                 if opt.lora_modules_to_save else None),
                lora_dropout=opt.lora_dropout, bias='none',
                use_dora=opt.use_dora, init_lora_weights=True,
            )
            model = get_peft_model(model, cfg)
            replaced = target_modules
        freeze_batch_norm(model)
        print(f'Adapter modules: {len(replaced)}')
        print_trainable_parameters_custom(model)
        return model

    if mode == 'none':
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    if mode == 'freeze_bn':
        for param in model.parameters():
            param.requires_grad_(True)
        freeze_batch_norm(model)
        return model

    if mode not in ('head', 'dense4', 'dense3_4', 'dense2_4', 'dense1_4', 'encoder', 'all'):
        raise ValueError(f'Unknown fine-tuning mode: {mode}')

    for name, param in model.named_parameters():
        if mode == 'head':
            train = 'class_layers.out' in name
        elif mode == 'dense4':
            train = any(s in name for s in ('denseblock4', 'class_layers.out'))
        elif mode == 'dense3_4':
            train = any(s in name for s in ('denseblock3', 'denseblock4', 'class_layers.out'))
        elif mode == 'dense2_4':
            train = any(s in name for s in ('denseblock2', 'denseblock3', 'denseblock4', 'class_layers.out'))
        elif mode in ('dense1_4', 'encoder'):
            train = not any(s in name for s in ('conv0', 'norm0'))
        else:
            train = True
        param.requires_grad_(train)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError(f'No trainable parameters for mode {mode}.')

    print_trainable_parameters_custom(model)
    return model
