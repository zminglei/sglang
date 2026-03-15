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


# @triton.autotune(
#     configs=[
#         triton.Config({"BT": BT}, num_warps=num_warps)
#         for num_warps in [1, 2, 4, 8, 16]
#         for BT in BT_LIST
#     ],
#     key=["D", "NB"],
# )
@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            NB=NB,
            T=T,
            D=D,
            BD=BD,
            BT=16,
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


# Fused l2norm for channel-first packed buffer from conv1d output.
# Input: (H*K, padded_T) contiguous — each row has T contiguous elements.
# Output: (1, T, H, K) contiguous.
# Reads T-coalesced (stride 1), reduces over K, writes row-major output.
@triton.jit
def l2norm_fwd_packed_kernel(
    x, y, eps,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    S_ROW,  # stride between rows in input (= padded_T), NOT constexpr to avoid recompilation
    BT: tl.constexpr,
    BD: tl.constexpr,     # >= K, power of 2
):
    # Grid: (cdiv(T, BT), H). Each program normalizes BT time steps for one head.
    i_bt = tl.program_id(0)
    i_h = tl.program_id(1)

    t_start = i_bt * BT
    t_off = tl.arange(0, BT)
    k_off = tl.arange(0, BD)
    t_mask = (t_start + t_off) < T
    k_mask = k_off < K

    # Input: head i_h occupies rows [i_h*K, (i_h+1)*K) of the (H*K, padded_T) buffer.
    # Element (k, t) at offset: (i_h*K + k) * S_ROW + t
    x_base = x + i_h * K * S_ROW
    # Load (BD, BT) tile: K rows of BT T-contiguous elements. T is stride 1 → coalesced.
    offs_x = k_off[:, None] * S_ROW + (t_start + t_off[None, :])
    mask = k_mask[:, None] & t_mask[None, :]
    b_x = tl.load(x_base + offs_x, mask=mask, other=0.0).to(tl.float32)

    # L2 norm: reduce over K (axis 0)
    b_var = tl.sum(b_x * b_x, axis=0)  # (BT,)
    b_rstd = 1.0 / tl.sqrt(b_var + eps)
    b_y = b_x * b_rstd[None, :]

    # Store to contiguous (T, H, K) output: element (t, h, k) at y[(t*H + h)*K + k]
    y_base = y + i_h * K
    offs_y = (t_start + t_off[None, :]) * (H * K) + k_off[:, None]
    tl.store(y_base + offs_y, b_y.to(y.dtype.element_ty), mask=mask)


def l2norm_fwd_packed(
    x: torch.Tensor, seq_len: int, num_heads: int, head_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused l2norm + transpose from channel-first packed buffer.

    Args:
        x: (H*K, padded_T) contiguous tensor from conv1d output dim-0 split.
        seq_len: actual sequence length (may be < padded_T).
        num_heads: number of heads (H).
        head_dim: head dimension (K).

    Returns:
        (1, seq_len, H, K) contiguous tensor with per-head L2-normalized vectors.
    """
    H, K = num_heads, head_dim
    T = seq_len
    S_ROW = x.stride(0)  # padded_T

    y = torch.empty(T * H, K, dtype=x.dtype, device=x.device)

    BD = triton.next_power_of_2(K)
    BT = 16

    grid = (triton.cdiv(T, BT), H)
    l2norm_fwd_packed_kernel[grid](
        x, y, eps,
        T=T, H=H, K=K, S_ROW=S_ROW,
        BT=BT, BD=BD,
        num_warps=4, num_stages=2,
    )
    return y.view(1, T, H, K)


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
