# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.utils import input_guard

BT_LIST = [8, 16, 32, 64, 128]


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
#     ],
#     key=["D"],
# )
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    S_X_ROW,
    S_X_HEAD,
    S_Y_ROW,
    S_Y_HEAD,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    S_X_COL = S_X_HEAD // D
    p_x = tl.make_block_ptr(
        x + i_h * S_X_HEAD, (T, D), (S_X_ROW, S_X_COL),
        (i_t * BT, 0), (BT, BD), (1, 0)
    )
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(
        y + i_h * S_Y_HEAD, (T, D), (S_Y_ROW, 1),
        (i_t * BT, 0), (BT, BD), (1, 0)
    )
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    # Non-contiguous path: read strides from the tensor, write contiguous output.
    # Handles channel-first as_strided views like (1, T, H, K) with non-standard strides.
    if not x.is_contiguous() and x.ndim >= 3:
        K = x.shape[-1]
        H = x.shape[-2]
        T = x.shape[-3]

        dtype = output_dtype if output_dtype is not None else x.dtype
        y = torch.empty(x.shape, dtype=dtype, device=x.device)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(K))
        if K > BD:
            raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

        BT = 16
        grid = (triton.cdiv(T, BT), H)
        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            T=T,
            D=K,
            BD=BD,
            BT=BT,
            S_X_ROW=x.stride(-3),
            S_X_HEAD=x.stride(-2),
            S_Y_ROW=y.stride(-3),
            S_Y_HEAD=y.stride(-2),
            num_warps=8,
            num_stages=3,
        )
        return y

    # Contiguous path (original)
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:
        # Standard path: 1D grid over all rows, H_grid=1
        l2norm_fwd_kernel[(triton.cdiv(T, 16), 1)](
            x,
            y,
            eps,
            T=T,
            D=D,
            BD=BD,
            BT=16,
            S_X_ROW=D,
            S_X_HEAD=D,
            S_Y_ROW=D,
            S_Y_HEAD=D,
            num_warps=8,
            num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    return y.view(x_shape_og)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
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
