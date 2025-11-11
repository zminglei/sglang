# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.fla.solve_tril import solve_tril
from sglang.srt.layers.attention.fla.utils import (
    SUPPRESS_LEVEL,
    autocast_custom_fwd,
    input_guard,
)
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    import os
    enable_determinism_logging = os.environ.get("ENABLE_DETERMINISM_LOGGING", "0") == "1"
    log_file = os.environ.get("DETERMINISM_LOG_FILE", "/tmp/qwen3_determinism_2.log")
    
    def log_chunk_tensor(tensor, name, cu_seqlens):
        """Log differences among sequences in chunk_gated_delta_rule_fwd"""
        if not enable_determinism_logging or tensor is None:
            return
        
        try:
            # Special handling for final_state which has shape [N, H, K, V]
            # where N is number of sequences (not concatenated)
            if "final_state" in name and len(tensor.shape) == 4:
                num_seqs = tensor.shape[0]
                sequences = [tensor[i:i+1] for i in range(num_seqs)]
            # If cu_seqlens is provided, we have multiple sequences concatenated
            elif cu_seqlens is not None and len(cu_seqlens) > 2:
                num_seqs = len(cu_seqlens) - 1
                
                # Extract each sequence
                sequences = []
                for i in range(num_seqs):
                    start = cu_seqlens[i].item()
                    end = cu_seqlens[i + 1].item()
                    
                    if len(tensor.shape) == 3:  # [1, total_seq, H] or [1, total_seq, dim]
                        seq = tensor[:, start:end, :]
                    elif len(tensor.shape) == 4:  # [1, total_seq, H, D]
                        seq = tensor[:, start:end, :, :]
                    elif len(tensor.shape) == 5:  # [1, num_chunks, H, chunk_size, D]
                        # For chunked tensors, this is more complex
                        seq = tensor  # Just use as-is for now
                    else:
                        return
                    
                    sequences.append(seq)
            else:
                return
                
                # Compare all sequences against the first
                if len(sequences) > 1:
                    reference = sequences[0]
                    max_diff = 0.0
                    all_equal = True
                    
                    for i in range(1, len(sequences)):
                        if sequences[i].shape == reference.shape:
                            diff = (sequences[i] - reference).abs().max().item()
                            if diff > 0:
                                all_equal = False
                            max_diff = max(max_diff, diff)
                    
                    # Try to count unique sequences
                    unique_count = "N/A"
                    try:
                        stacked = torch.stack([s.reshape(-1) for s in sequences])
                        unique_count = len(torch.unique(stacked, dim=0))
                    except:
                        pass
                    
                    min_val = tensor.min().item()
                    max_val = tensor.max().item()
                    
                    with open(log_file, "a") as f:
                        f.write(f"  [chunk_internal:{name}] shape={tensor.shape}, num_seqs={num_seqs}\n")
                        f.write(f"    max_diff={max_diff:.10e}, min={min_val:.6f}, max={max_val:.6f}\n")
                        f.write(f"    unique_seqs={unique_count}/{num_seqs}, all_equal={all_equal}\n")
                        if not all_equal:
                            f.write(f"    ⚠️⚠️  NON-DETERMINISTIC IN CHUNK_GATED_DELTA_RULE!\n")
                        f.write("\n")
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"  [chunk_internal:{name}] Error logging: {str(e)}\n\n")
    
    if enable_determinism_logging:
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"CHUNK_GATED_DELTA_RULE_FWD - Internal Steps\n")
            f.write(f"{'='*60}\n\n")
    
    # Log inputs
    log_chunk_tensor(q, "input_q", cu_seqlens)
    log_chunk_tensor(k, "input_k", cu_seqlens)
    log_chunk_tensor(v, "input_v", cu_seqlens)
    log_chunk_tensor(g, "input_g", cu_seqlens)
    log_chunk_tensor(beta, "input_beta", cu_seqlens)
    
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    log_chunk_tensor(g, "after_cumsum_g", cu_seqlens)
    
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    log_chunk_tensor(A, "after_scaled_dot_kkt_A", cu_seqlens)
    
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    log_chunk_tensor(A, "after_solve_tril_A", cu_seqlens)
    
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    log_chunk_tensor(w, "after_recompute_w", cu_seqlens)
    log_chunk_tensor(u, "after_recompute_u", cu_seqlens)
    
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    log_chunk_tensor(h, "after_fwd_h_h", cu_seqlens)
    log_chunk_tensor(v_new, "after_fwd_h_v_new", cu_seqlens)
    log_chunk_tensor(final_state, "after_fwd_h_final_state", cu_seqlens)
    
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    log_chunk_tensor(o, "after_fwd_o_output", cu_seqlens)
    
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    # if not head_first and q.shape[1] < q.shape[2]:
    #     warnings.warn(
    #         f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
    #         "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
    #         "when head_first=False was specified. "
    #         "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
    #     )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, final_state
