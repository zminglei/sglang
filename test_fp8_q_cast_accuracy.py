"""
Verify that SGLang's raw Q cast to FP8 causes accuracy degradation
compared to vLLM's scaled_fp8_quant approach.

This simulates the attention score computation with:
1. BF16 baseline (ground truth)
2. Raw .to(fp8) cast (SGLang's current approach)
3. Scaled FP8 quant + descale (vLLM's approach)
"""

import torch
import torch.nn.functional as F
import math


def scaled_fp8_quant(tensor, scale=None, fp8_dtype=torch.float8_e4m3fn):
    """Mimics vLLM's scaled_fp8_quant: divide by scale, cast, return scale."""
    if scale is None:
        # Dynamic quantization: compute scale from tensor
        amax = torch.abs(tensor).max()
        fp8_max = torch.finfo(fp8_dtype).max
        scale = amax / fp8_max
        scale = scale.clamp(min=1e-12)
    fp8_tensor = (tensor / scale).clamp(
        -torch.finfo(fp8_dtype).max, torch.finfo(fp8_dtype).max
    ).to(fp8_dtype)
    return fp8_tensor, scale


def attention_score(q, k, v, head_dim):
    """Simple scaled dot-product attention."""
    scale = 1.0 / math.sqrt(head_dim)
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v.float())


def run_test(batch=2, n_heads=32, seq_len=2048, head_dim=128, seed=42):
    torch.manual_seed(seed)
    device = "cuda"

    # Simulate post-RoPE Q, K, V in BF16
    # RoPE can amplify magnitudes, especially at long positions
    q = torch.randn(batch, n_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    # Amplify some Q values to simulate RoPE at long positions
    # RoPE frequencies can cause values > 448 (E4M3 max)
    q_amplified = q.clone()
    q_amplified[:, :, :, :16] *= 5.0  # push some dims toward/beyond E4M3 max

    print(f"=== Test Config: batch={batch}, heads={n_heads}, seq_len={seq_len}, head_dim={head_dim} ===")
    print(f"Q range: [{q_amplified.min().item():.2f}, {q_amplified.max().item():.2f}]")
    print(f"Q abs max: {q_amplified.abs().max().item():.2f}")
    print(f"FP8 E4M3 max: {torch.finfo(torch.float8_e4m3fn).max}")
    num_overflow = (q_amplified.abs() > torch.finfo(torch.float8_e4m3fn).max).sum().item()
    total_elements = q_amplified.numel()
    print(f"Elements exceeding E4M3 max: {num_overflow}/{total_elements} ({100*num_overflow/total_elements:.2f}%)")
    print()

    # --- 1. BF16 Baseline (ground truth) ---
    out_bf16 = attention_score(q_amplified, k, v, head_dim)

    # --- 2. SGLang approach: raw .to(fp8) ---
    q_sglang = q_amplified.to(torch.float8_e4m3fn)  # raw cast, no scale
    # No q_descale, so we just use the raw FP8 values
    out_sglang = attention_score(q_sglang, k, v, head_dim)

    # --- 3. vLLM approach: scaled_fp8_quant + descale ---
    q_flat = q_amplified.reshape(-1, head_dim)
    q_vllm_fp8, q_scale = scaled_fp8_quant(q_flat)
    q_vllm_fp8 = q_vllm_fp8.reshape(batch, n_heads, 1, head_dim)
    # Descale Q before attention (simulating what the kernel does with q_descale)
    q_vllm_dequant = q_vllm_fp8.float() * q_scale
    out_vllm = attention_score(q_vllm_dequant.to(torch.bfloat16), k, v, head_dim)

    # --- Compare ---
    def compute_error(out_test, out_ref, label):
        # Max absolute error
        max_err = (out_test.float() - out_ref.float()).abs().max().item()
        # Relative error (cosine similarity)
        cos_sim = F.cosine_similarity(
            out_test.float().reshape(-1).unsqueeze(0),
            out_ref.float().reshape(-1).unsqueeze(0)
        ).item()
        # RMSE
        rmse = ((out_test.float() - out_ref.float()) ** 2).mean().sqrt().item()
        print(f"{label}:")
        print(f"  Max absolute error: {max_err:.6f}")
        print(f"  RMSE:               {rmse:.6f}")
        print(f"  Cosine similarity:  {cos_sim:.8f}")
        print()

    compute_error(out_sglang, out_bf16, "SGLang (raw .to(fp8), no q_descale)")
    compute_error(out_vllm, out_bf16, "vLLM   (scaled_fp8_quant + descale)")

    # --- Show the Q value corruption ---
    print("--- Q Value Roundtrip Check ---")
    q_raw_roundtrip = q_amplified.to(torch.float8_e4m3fn).to(torch.float32)
    q_scaled_roundtrip = (q_vllm_fp8.float() * q_scale)
    raw_q_err = (q_raw_roundtrip - q_amplified.float()).abs().max().item()
    scaled_q_err = (q_scaled_roundtrip - q_amplified.float()).abs().max().item()
    print(f"Raw cast Q roundtrip max error:    {raw_q_err:.4f}")
    print(f"Scaled quant Q roundtrip max error: {scaled_q_err:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("Test 1: Normal activations (should be small diff)")
    print("=" * 70)
    run_test(seed=42)

    print("\n" + "=" * 70)
    print("Test 2: Larger amplification (simulate extreme RoPE positions)")
    print("=" * 70)
    torch.manual_seed(123)
    device = "cuda"
    q = torch.randn(2, 32, 1, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 32, 2048, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 32, 2048, 128, dtype=torch.bfloat16, device=device)
    # Simulate extreme case: some Q values well beyond E4M3 range
    q[:, :, :, :32] *= 20.0  # push well beyond 448

    print(f"Q range: [{q.min().item():.2f}, {q.max().item():.2f}]")
    print(f"Q abs max: {q.abs().max().item():.2f}")
    num_overflow = (q.abs() > torch.finfo(torch.float8_e4m3fn).max).sum().item()
    print(f"Elements exceeding E4M3 max: {num_overflow}/{q.numel()} ({100*num_overflow/q.numel():.2f}%)")
    print()

    out_bf16 = attention_score(q, k, v, 128)
    q_sglang = q.to(torch.float8_e4m3fn)
    out_sglang = attention_score(q_sglang, k, v, 128)

    q_flat = q.reshape(-1, 128)
    q_fp8, q_scale = scaled_fp8_quant(q_flat)
    q_dequant = q_fp8.reshape(2, 32, 1, 128).float() * q_scale
    out_vllm = attention_score(q_dequant.to(torch.bfloat16), k, v, 128)

    def compute_error(out_test, out_ref, label):
        max_err = (out_test.float() - out_ref.float()).abs().max().item()
        cos_sim = F.cosine_similarity(
            out_test.float().reshape(-1).unsqueeze(0),
            out_ref.float().reshape(-1).unsqueeze(0)
        ).item()
        rmse = ((out_test.float() - out_ref.float()) ** 2).mean().sqrt().item()
        print(f"{label}:")
        print(f"  Max absolute error: {max_err:.6f}")
        print(f"  RMSE:               {rmse:.6f}")
        print(f"  Cosine similarity:  {cos_sim:.8f}")

    compute_error(out_sglang, out_bf16, "SGLang (raw .to(fp8), no q_descale)")
    compute_error(out_vllm, out_bf16, "vLLM   (scaled_fp8_quant + descale)")
