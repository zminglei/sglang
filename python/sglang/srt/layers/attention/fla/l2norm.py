# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.utils import input_guard


# Original kernels for contiguous (T, D) input (standard path)
@triton.jit
def l2norm_fwd_kernel1(x, y, D, BD: tl.constexpr, eps):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.jit
def l2norm_fwd_kernel(x, y, eps,
                      NB: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                      BT: tl.constexpr, BD: tl.constexpr):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


# Optimized kernel for 4D non-contiguous input from conv1d transpose.
# Input: (1, T, H, K) with strides (*, 1, K*padded, padded) — T is fast dim.
# This kernel reads T-contiguous data (coalesced) and writes (T, H, K) contiguous output.
# Grid: (cdiv(T, BT), H). Each program normalizes BT time steps for one head.
@triton.jit
def l2norm_fwd_kernel_4d(
    x, y, eps,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    # Input strides (from the 4D tensor)
    S_X_T: tl.constexpr,   # stride along T dim (=1 for column-major)
    S_X_H: tl.constexpr,   # stride along H dim (=K*padded)
    S_X_K: tl.constexpr,   # stride along K dim (=padded)
):
    i_bt = tl.program_id(0)  # T block index
    i_h = tl.program_id(1)   # head index

    t_start = i_bt * BT
    t_off = tl.arange(0, BT)
    t_mask = (t_start + t_off) < T

    # Base pointer for this head
    x_base = x + i_h * S_X_H

    # Pass 1: accumulate sum of squares across K dim
    b_var = tl.zeros([BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        k_off = tl.arange(0, BK)
        k_start = i_k * BK
        k_mask = (k_start + k_off) < K
        # Load (BK, BT) block: K has stride S_X_K, T has stride S_X_T (=1, coalesced!)
        # Offsets: x_base + (k_start + k_off) * S_X_K + (t_start + t_off) * S_X_T
        offs = (k_start + k_off[:, None]) * S_X_K + (t_start + t_off[None, :]) * S_X_T
        mask = k_mask[:, None] & t_mask[None, :]
        b_x = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
        b_var += tl.sum(b_x * b_x, axis=0)

    b_rstd = 1.0 / tl.sqrt(b_var + eps)

    # Pass 2: normalize and store to contiguous output (T, H, K)
    # Output layout: y[(t * H + h) * K + k]
    y_base = y + i_h * K  # offset for this head

    for i_k in range(tl.cdiv(K, BK)):
        k_off = tl.arange(0, BK)
        k_start = i_k * BK
        k_mask = (k_start + k_off) < K
        # Reload input (same coalesced pattern)
        offs_x = (k_start + k_off[:, None]) * S_X_K + (t_start + t_off[None, :]) * S_X_T
        mask = k_mask[:, None] & t_mask[None, :]
        b_x = tl.load(x_base + offs_x, mask=mask, other=0.0).to(tl.float32)
        b_y = b_x * b_rstd[None, :]

        # Store to contiguous output: offset = (t_start + t_off) * H * K + i_h * K + k_start + k_off
        # = y_base + (t_start + t_off) * H * K + k_start + k_off
        offs_y = (t_start + t_off[None, :]) * (H * K) + (k_start + k_off[:, None])
        tl.store(y_base + offs_y, b_y.to(y.dtype.element_ty), mask=mask)


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape

    # Detect 4D non-contiguous input from conv1d transpose+split+reshape
    # Pattern: (1, T, H, K) with stride[-1] != 1 (column-major from transpose)
    if (x.dim() == 4 and x.stride(-1) != 1 and x.stride(1) == 1):
        # Use optimized 4D kernel that reads T-contiguous data
        B, T, H, K = x.shape
        assert B == 1, "4D l2norm only supports B=1"
        D = K

        y_flat = torch.empty(T * H, K, dtype=output_dtype or x.dtype, device=x.device)

        BT = min(32, triton.next_power_of_2(T))
        BK = min(64, triton.next_power_of_2(K))

        grid = (triton.cdiv(T, BT), H)
        l2norm_fwd_kernel_4d[grid](
            x, y_flat, eps,
            T=T, H=H, K=K, BT=BT, BK=BK,
            S_X_T=x.stride(1),
            S_X_H=x.stride(2),
            S_X_K=x.stride(3),
            num_warps=4,
            num_stages=2,
        )
        return y_flat.view(x_shape_og)

    # Standard path for contiguous input
    x = x.reshape(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x, y, eps,
            NB=NB, T=T, D=D, BD=BD, BT=16,
            num_warps=8, num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x, y, eps=eps, D=D, BD=BD,
            num_warps=8, num_stages=3,
        )

    return y.view(x_shape_og)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard(no_guard_contiguous=["x"])
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        return l2norm_fwd(x, eps, output_dtype)


def l2norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):

    def __init__(self, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)
