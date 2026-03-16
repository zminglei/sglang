# Fused l2norm and extract+transpose kernels for channel-first packed buffer
# from causal_conv1d output. These read T-coalesced data (stride 1) and write
# contiguous (T, H, K) output, eliminating the transpose copy overhead.

import torch
import triton
import triton.language as tl


@triton.jit
def l2norm_fwd_packed_kernel(
    x, y, eps,
    T,  # NOT constexpr: varies per chunked prefill batch
    H: tl.constexpr,
    K: tl.constexpr,
    S_ROW,  # NOT constexpr: = padded_T, varies
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
    S_ROW = x.stride(0)

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


@triton.jit
def extract_transpose_packed_kernel(
    x, y,
    T,  # NOT constexpr: varies per chunked prefill batch
    H: tl.constexpr,
    V: tl.constexpr,
    S_ROW,  # NOT constexpr: = padded_T, varies
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_bt = tl.program_id(0)
    i_h = tl.program_id(1)

    t_start = i_bt * BT
    t_off = tl.arange(0, BT)
    v_off = tl.arange(0, BD)
    t_mask = (t_start + t_off) < T
    v_mask = v_off < V

    x_base = x + i_h * V * S_ROW
    offs_x = v_off[:, None] * S_ROW + (t_start + t_off[None, :])
    mask = v_mask[:, None] & t_mask[None, :]
    b_x = tl.load(x_base + offs_x, mask=mask, other=0.0)

    y_base = y + i_h * V
    offs_y = (t_start + t_off[None, :]) * (H * V) + v_off[:, None]
    tl.store(y_base + offs_y, b_x.to(y.dtype.element_ty), mask=mask)


def extract_transpose_packed(
    x: torch.Tensor, seq_len: int, num_heads: int, head_dim: int,
) -> torch.Tensor:
    """Extract and transpose from channel-first packed buffer.

    Args:
        x: (H*V, padded_T) contiguous tensor from conv1d output dim-0 split.
        seq_len: actual sequence length.
        num_heads: H.
        head_dim: V.

    Returns:
        (1, seq_len, H, V) contiguous tensor.
    """
    H, V = num_heads, head_dim
    T = seq_len
    S_ROW = x.stride(0)

    y = torch.empty(T * H, V, dtype=x.dtype, device=x.device)

    BD = triton.next_power_of_2(V)
    BT = 16

    grid = (triton.cdiv(T, BT), H)
    extract_transpose_packed_kernel[grid](
        x, y,
        T=T, H=H, V=V, S_ROW=S_ROW,
        BT=BT, BD=BD,
        num_warps=4, num_stages=2,
    )
    return y.view(1, T, H, V)
