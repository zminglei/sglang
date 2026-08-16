# Fused GDN extend: reads q/k/v directly from conv1d channel-first output
# without split, transpose, or contiguous copies.

from typing import Optional

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.l2norm_fused import (
    extract_transpose_fused,
    l2norm_fwd_fused,
)
from sglang.srt.layers.attention.fla.solve_tril import solve_tril
from sglang.srt.layers.attention.fla.utils import SUPPRESS_LEVEL
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd


def chunk_gated_delta_rule_fwd_fused(
    conv_out: torch.Tensor,
    seq_len: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """Fused GDN chunk forward that reads from conv1d channel-first output.

    Args:
        conv_out: (total_dim, padded_T) contiguous tensor from causal_conv1d_fn.
        seq_len: actual sequence length (may be < padded_T).
        q_dim, k_dim, v_dim: dimension sizes for q, k, v in the packed buffer.
        num_*_heads, head_*_dim: head counts and dimensions.
        g, beta, scale, initial_state, ...: same as chunk_gated_delta_rule.

    Returns:
        (o, h) — same as chunk_gated_delta_rule.
    """
    # Step 1: Fused l2norm — reads T-contiguous from channel-first, writes
    # contiguous (1, T, H, K). No split, no transpose copy, no contiguous().
    q_raw = conv_out[:q_dim]
    k_raw = conv_out[q_dim:q_dim + k_dim]

    q = l2norm_fwd_fused(q_raw, seq_len, num_q_heads, head_q_dim)
    k = l2norm_fwd_fused(k_raw, seq_len, num_k_heads, head_k_dim)

    # Step 2: v — fused extract+transpose from channel-first to contiguous.
    v_raw = conv_out[q_dim + k_dim:]
    v = extract_transpose_fused(v_raw, seq_len, num_v_heads, head_v_dim)

    # Step 3: Standard chunk pipeline on contiguous q, k, v.
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=cu_seqlens,
    )
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens,
    )
    if SUPPRESS_LEVEL < 3:
        return o, h
    elif SUPPRESS_LEVEL >= 3:
        return o, h
