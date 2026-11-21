"""Custom 3D LoRA adapters and the PiSSA initialization used in the thesis.

PiSSA splits W into a frozen residual W - BA and a trainable BA branch,
so the initial wrapped layer computes the same function as the pretrained one.
FiltLoRA inserts a parameter-free 3D trigonometric spatial filter between
its low-rank projections.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def initialize_pissa_wrapper(
    wrapper,
    svd_device=None,
):
    base = wrapper.base
    lora_A = wrapper.lora_A
    lora_B = wrapper.lora_B
    scaling = float(wrapper.scaling)

    if not isinstance(base, (nn.Linear, nn.Conv3d)):
        raise TypeError(
            f"PiSSA does not support module type {type(base)}"
        )

    if isinstance(base, nn.Conv3d) and base.groups != 1:
        raise NotImplementedError(
            "PiSSA currently supports only Conv3d with groups=1."
        )

    if scaling <= 0:
        raise ValueError(
            f"LoRA scaling must be positive, received {scaling}."
        )

    original_device = base.weight.device
    original_dtype = base.weight.dtype
    original_shape = base.weight.shape

    if svd_device is None:
        svd_device = original_device

    # Linear:
    #   [out_features, in_features]
    #
    # Conv3d:
    #   [out_channels, in_channels, kD, kH, kW]
    #   -> [out_channels, in_channels * kD * kH * kW]
    weight_matrix = (
        base.weight
        .detach()
        .to(device=svd_device, dtype=torch.float32)
        .reshape(original_shape[0], -1)
    )

    rank = lora_A.weight.shape[0]
    max_rank = min(weight_matrix.shape)
    effective_rank = min(rank, max_rank)

    if effective_rank < rank:
        print(
            f"WARNING: PiSSA rank reduced from {rank} to {effective_rank} "
            f"for layer with matrix shape {tuple(weight_matrix.shape)}."
        )

    U, S, Vh = torch.linalg.svd(
        weight_matrix,
        full_matrices=False,
    )

    U_r = U[:, :effective_rank]
    S_r = S[:effective_rank]
    Vh_r = Vh[:effective_rank, :]

    sqrt_s = torch.sqrt(S_r / scaling)

    # Effective PiSSA components only up to effective_rank
    B_eff = U_r * sqrt_s.unsqueeze(0)
    A_eff = sqrt_s.unsqueeze(1) * Vh_r

    # Full LoRA matrices with requested rank
    A_matrix = torch.empty(
        rank,
        weight_matrix.shape[1],
        device=weight_matrix.device,
        dtype=weight_matrix.dtype,
    )

    B_matrix = torch.zeros(
        weight_matrix.shape[0],
        rank,
        device=weight_matrix.device,
        dtype=weight_matrix.dtype,
    )

    # PiSSA initialization for the available rank
    A_matrix[:effective_rank, :] = A_eff
    B_matrix[:, :effective_rank] = B_eff

    # Remaining components: standard LoRA initialization
    # A random, B zero -> no initial contribution
    if effective_rank < rank:
        nn.init.kaiming_uniform_(
            A_matrix[effective_rank:, :],
            a=np.sqrt(5),
        )

    # Shape checks before copying
    if A_matrix.numel() != lora_A.weight.numel():
        raise RuntimeError(
            f"PiSSA A has flat shape {tuple(A_matrix.shape)}, "
            f"but lora_A.weight has shape {tuple(lora_A.weight.shape)}. "
            "For Conv3d, lora_A must match the original convolution kernel."
        )

    if B_matrix.numel() != lora_B.weight.numel():
        raise RuntimeError(
            f"PiSSA B has flat shape {tuple(B_matrix.shape)}, "
            f"but lora_B.weight has shape {tuple(lora_B.weight.shape)}. "
            "For Conv3d, lora_B should usually be a 1x1x1 convolution."
        )

    # Copy PiSSA initialization into LoRA parameters
    lora_A.weight.copy_(
        A_matrix.reshape_as(lora_A.weight).to(
            device=lora_A.weight.device,
            dtype=lora_A.weight.dtype,
        )
    )

    lora_B.weight.copy_(
        B_matrix.reshape_as(lora_B.weight).to(
            device=lora_B.weight.device,
            dtype=lora_B.weight.dtype,
        )
    )

    if lora_A.bias is not None:
        lora_A.bias.zero_()

    if lora_B.bias is not None:
        lora_B.bias.zero_()

    # Important:
    # subtract only the true PiSSA component, not the full A_matrix,
    # because the extra A rows are random LoRA components.
    low_rank_component = scaling * (B_eff @ A_eff)

    residual_weight = (
        weight_matrix - low_rank_component
    ).reshape(original_shape)

    base.weight.copy_(
        residual_weight.to(
            device=original_device,
            dtype=original_dtype,
        )
    )

    # Base layer frozen
    base.weight.requires_grad = False

    if base.bias is not None:
        base.bias.requires_grad = False

    # LoRA parameters trainable
    lora_A.weight.requires_grad = True
    lora_B.weight.requires_grad = True

    if lora_A.bias is not None:
        lora_A.bias.requires_grad = True

    if lora_B.bias is not None:
        lora_B.bias.requires_grad = True

    del weight_matrix, U, S, Vh
    del U_r, S_r, Vh_r
    del A_eff, B_eff
    del A_matrix, B_matrix, low_rank_component, residual_weight

    return wrapper


def inject_pissa_3d(
    model,
    target_modules,
    r=4,
    alpha=8,
    dropout=0.0,
    svd_device=None,
):
    """
    Replace selected Conv3d and Linear modules with custom PiSSA wrappers.

    Parameters
    ----------
    model
        Original model.

    target_modules
        Exact module names to replace.

    r
        PiSSA rank.

    alpha
        Adapter scaling parameter.

    dropout
        Dropout applied before PiSSA A.

    svd_device
        Device used for SVD. None means the same device as the layer.
        Can be set to "cpu" to reduce GPU-memory use.
    """

    target_set = set(target_modules)
    replaced = []

    # Freeze the complete pretrained network first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    def replace_recursively(parent, prefix=""):
        for child_name, child in list(parent.named_children()):
            full_name = (
                f"{prefix}.{child_name}"
                if prefix
                else child_name
            )

            if full_name not in target_set:
                replace_recursively(
                    child,
                    full_name,
                )
                continue

            if isinstance(child, nn.Conv3d):
                new_child = PiSSAConv3d(
                    base_conv=child,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )

            elif isinstance(child, nn.Linear):
                new_child = LoRALinear(
                    base_linear=child,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )

            else:
                print(
                    f"Skipping {full_name}: "
                    f"unsupported module type {type(child)}"
                )
                continue

            # Move newly created A and B to the same device/dtype
            # as the pretrained layer before initialization.
            new_child = new_child.to(
                device=child.weight.device,
                dtype=child.weight.dtype,
            )

            # Initialize the LoRA branch and remove the same component from the frozen weight.
            initialize_pissa_wrapper(
                new_child,
                svd_device=svd_device,
            )

            setattr(
                parent,
                child_name,
                new_child,
            )

            replaced.append(full_name)

    replace_recursively(model)

    if len(replaced) == 0:
        raise RuntimeError(
            "No modules were replaced by PiSSA. "
            "Check the target module names."
        )

    return model, replaced


class CoLoRATrigFilter3d(nn.Module):
    """
    3D version of the CoLoRA parameter-free filtering idea.

    Input:
        x: [B, C, D, H, W]

    The spatial filter uses:
        range kernel via sin/cos features
        spatial kernel via fixed Gaussian kernel
        residual output: x + scale_factor * filtered
    """

    def __init__(self, kernel_size=3, sigma=1.0, scale_factor=1.0, eps=1e-5):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.eps = eps

        grid = torch.arange(kernel_size, dtype=torch.float32)
        grid = grid - (kernel_size - 1) / 2

        zz, yy, xx = torch.meshgrid(grid, grid, grid, indexing="ij")
        kernel = torch.exp(
            -0.5 * (xx ** 2 + yy ** 2 + zz ** 2) / (sigma ** 2)
        )
        kernel = kernel / kernel.sum()

        self.register_buffer(
            "kernel",
            kernel.view(1, 1, kernel_size, kernel_size, kernel_size),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        # CoLoRA-style per-sample, per-channel dynamic gamma
        x_flat = x.view(B, C, -1)
        xmax = x_flat.max(dim=-1, keepdim=True).values.view(B, C, 1, 1, 1)
        xmin = x_flat.min(dim=-1, keepdim=True).values.view(B, C, 1, 1, 1)

        T = (xmax - xmin).clamp_min(self.eps)
        gamma = torch.pi / (2.0 * T)

        sinx = torch.sin(gamma * x).detach().clamp_min(0.0)
        cosx = torch.cos(gamma * x).detach().clamp_min(0.0)

        x_pad = F.pad(
            x,
            (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad),
            mode="replicate",
        )
        sin_pad = F.pad(
            sinx,
            (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad),
            mode="replicate",
        )
        cos_pad = F.pad(
            cosx,
            (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad),
            mode="replicate",
        )

        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        kernel = kernel.repeat(C, 1, 1, 1, 1)

        cos_filt = cosx * F.conv3d(
            cos_pad * x_pad,
            weight=kernel,
            bias=None,
            stride=1,
            padding=0,
            groups=C,
        )

        sin_filt = sinx * F.conv3d(
            sin_pad * x_pad,
            weight=kernel,
            bias=None,
            stride=1,
            padding=0,
            groups=C,
        )

        cos_map = cosx * F.conv3d(
            cos_pad,
            weight=kernel,
            bias=None,
            stride=1,
            padding=0,
            groups=C,
        )

        sin_map = sinx * F.conv3d(
            sin_pad,
            weight=kernel,
            bias=None,
            stride=1,
            padding=0,
            groups=C,
        )

        filtered = (cos_filt + sin_filt) / (cos_map + sin_map + self.eps)

        return x + self.scale_factor * filtered


class PiSSAConv3d(nn.Module):
    """
    PiSSA wrapper for nn.Conv3d.

    Base branch:
        y_base = base(x)

    PiSSA branch:
        y_pissa = B(A(dropout(x)))

    Output:
        y = y_base + scaling * y_pissa
    """

    def __init__(
        self,
        base_conv,
        r=4,
        alpha=8,
        dropout=0.0,
    ):
        super().__init__()

        if not isinstance(base_conv, nn.Conv3d):
            raise TypeError("base_conv must be nn.Conv3d")

        if base_conv.groups != 1:
            raise NotImplementedError(
                "PiSSAConv3d currently supports only Conv3d with groups=1."
            )

        self.base = base_conv

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r

        self.dropout = (
            nn.Dropout(dropout)
            if dropout > 0
            else nn.Identity()
        )

        # Same spatial operation as the original convolution,
        # but projecting into r channels.
        self.lora_A = nn.Conv3d(
            in_channels=base_conv.in_channels,
            out_channels=self.r,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=1,
            bias=False,
            padding_mode=base_conv.padding_mode,
        )

        # Project from rank r back to output channels.
        self.lora_B = nn.Conv3d(
            in_channels=self.r,
            out_channels=base_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        base_output = self.base(x)

        adapter_output = self.lora_A(
            self.dropout(x)
        )
        adapter_output = self.lora_B(
            adapter_output
        )

        return base_output + self.scaling * adapter_output


class FilteredLoRAConv3d(nn.Module):
    """
    Custom LoRA wrapper for nn.Conv3d with filtering between A and B.

    Base branch:
        y_base = Conv3d(x)

    LoRA branch:
        y_lora = B( Filter( A(x) ) )

    Output:
        y = y_base + scaling * y_lora
    """

    def __init__(
        self,
        base_conv,
        r=4,
        alpha=8,
        dropout=0.0,
        filter_kernel_size=3,
        filter_sigma_spatial=1.0,
        filter_sigma_range=1.0,
        filter_scale=1.0,
    ):
        super().__init__()

        if not isinstance(base_conv, nn.Conv3d):
            raise TypeError("base_conv must be nn.Conv3d")

        self.base = base_conv

        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A has the same spatial kernel/stride/padding as the original conv.
        # This gives low-rank feature maps with the same spatial size as base output.
        self.lora_A = nn.Conv3d(
            in_channels=base_conv.in_channels,
            out_channels=r,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=1,
            bias=False,
            padding_mode=base_conv.padding_mode,
        )

        self.filter = CoLoRATrigFilter3d(
            kernel_size=filter_kernel_size,
            sigma=filter_sigma_spatial,
            scale_factor=filter_scale,
        )

        # B maps low-rank channels back to output channels.
        self.lora_B = nn.Conv3d(
            in_channels=r,
            out_channels=base_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base(x)

        z = self.lora_A(self.dropout(x))
        z = self.filter(z)
        lora_out = self.lora_B(z)

        return base_out + self.scaling * lora_out


class LoRALinear(nn.Module):
    """
    Simple PEFT-like LoRA wrapper for nn.Linear.
    No filtering here, because filtering is spatial and makes sense for Conv3d features only.
    """

    def __init__(self, base_linear, r=4, alpha=8, dropout=0.0):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.base = base_linear

        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(
            base_linear.in_features,
            r,
            bias=False,
        )

        self.lora_B = nn.Linear(
            r,
            base_linear.out_features,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(
            self.lora_A(self.dropout(x))
        )


def inject_filtered_lora_3d(
    model,
    target_modules,
    r=4,
    alpha=8,
    dropout=0.0,
    filter_kernel_size=3,
    filter_sigma_spatial=1.0,
    filter_sigma_range=1.0,
    filter_scale=1.0,
    use_pissa=False,
):
    """
    Replace selected nn.Conv3d and nn.Linear modules with PEFT-like wrappers.

    Conv3d:
        Conv3d -> FilteredLoRAConv3d

    Linear:
        Linear -> LoRALinear
    """

    target_set = set(target_modules)
    replaced = []

    # The pretrained branch remains frozen.
    for p in model.parameters():
        p.requires_grad = False

    def _replace(parent, prefix=""):
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if full_name in target_set:
                if isinstance(child, nn.Conv3d):
                    new_child = FilteredLoRAConv3d(
                        child,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                        filter_kernel_size=filter_kernel_size,
                        filter_sigma_spatial=filter_sigma_spatial,
                        filter_sigma_range=filter_sigma_range,
                        filter_scale=filter_scale,
                    )

                    new_child = new_child.to(
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )

                    setattr(parent, child_name, new_child)
                    replaced.append(full_name)

                    if use_pissa:
                            initialize_pissa_wrapper(
                                new_child
                        )

                elif isinstance(child, nn.Linear):
                    new_child = LoRALinear(
                        child,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                    )

                    new_child = new_child.to(
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )

                    if use_pissa:
                        initialize_pissa_wrapper(
                            new_child
                        )

                    setattr(parent, child_name, new_child)
                    replaced.append(full_name)

                else:
                    print(f"Skipping {full_name}: unsupported module type {type(child)}")

            else:
                _replace(child, full_name)

    _replace(model)

    if len(replaced) == 0:
        raise RuntimeError("No modules were replaced by FilteredLoRA. Check target names.")

    return model, replaced


def print_trainable_parameters_custom(model):
    trainable = 0
    total = 0

    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    pct = 100 * trainable / total if total > 0 else 0

    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({pct:.4f}%)"
    )


def get_lora_target_modules(net, mode):
    target_modules = []

    for name, module in net.named_modules():
        if not isinstance(module, (nn.Conv3d, nn.Linear)):
            continue

        if mode == "head":
            use_lora = ("class_layers.out" in name)

        elif mode == "dense4":
            use_lora = any(k in name for k in [
                "denseblock4",
                "class_layers.out"
            ])

        elif mode == "dense3_4":
            use_lora = any(k in name for k in [
                "denseblock3",
                "denseblock4",
                "class_layers.out"
            ])

        elif mode == "dense2_4":
            use_lora = any(k in name for k in [
                "denseblock2",
                "denseblock3",
                "denseblock4",
                "class_layers.out"
            ])

        elif mode in ["dense1_4", "encoder"]:
            use_lora = not any(k in name for k in ["conv0", "norm0"])

        elif mode == "all":
            use_lora = True

        else:
            raise ValueError(f"Unknown LoRA mode: {mode}")

        if use_lora:
            target_modules.append(name)

    print(f"\n[LoRA mode = {mode}] → {len(target_modules)} target modules")
    for n in target_modules[:30]:
        print("   ", n)

    if len(target_modules) == 0:
        raise RuntimeError("No LoRA target modules found! Check layer naming.")

    return target_modules
