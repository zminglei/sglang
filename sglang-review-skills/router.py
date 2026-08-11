"""
SGLang PR Review Router

Analyzes a PR diff and determines which domain experts should review it.
Returns a list of expert names ranked by relevance.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ExpertRoute:
    name: str
    description: str
    # File path patterns that trigger this expert (regex)
    path_patterns: list[str]
    # Keywords in diff content that trigger this expert
    content_keywords: list[str] = field(default_factory=list)
    # Minimum confidence score (0-1) to include this expert
    threshold: float = 0.3


EXPERT_ROUTES: list[ExpertRoute] = [
    ExpertRoute(
        name="speculative_decoding",
        description="Speculative decoding: EAGLE, EAGLE v2, multi-layer EAGLE, ngram, MTP, draft models",
        path_patterns=[
            r"srt/speculative/",
            r"_eagle",
            r"_mtp",
            r"ngram",
            r"draft",
            r"spec_",
            r"speculative",
        ],
        content_keywords=[
            "speculative", "eagle", "draft_model", "ngram", "mtp",
            "multi_token_prediction", "accept_length", "draft_token",
            "spec_info", "target_model", "verify", "tree_attention",
            "draft_kv", "is_draft", "draft_worker",
        ],
    ),
    ExpertRoute(
        name="quantization",
        description="Quantization: FP8, INT8, AWQ, GPTQ, BitsAndBytes, Marlin, MXFP4, weight-only, KV cache quantization",
        path_patterns=[
            r"srt/layers/quantization/",
            r"sgl-kernel/csrc/quantization/",
            r"fp8",
            r"int8",
            r"awq",
            r"gptq",
            r"marlin",
            r"mxfp4",
            r"quant",
            r"bitsandbytes",
        ],
        content_keywords=[
            "quantize", "quantization", "fp8", "int8", "awq", "gptq",
            "marlin", "w8a8", "w4a16", "scale", "zero_point",
            "bitsandbytes", "calibration", "QuantizationConfig",
            "per_channel", "per_tensor", "blockwise", "mxfp4",
        ],
    ),
    ExpertRoute(
        name="attention",
        description="Attention backends: FlashInfer, FlashAttention, MLA, NSA, triton attention, TBO, vision attention",
        path_patterns=[
            r"srt/layers/attention/",
            r"sgl-kernel/csrc/attention/",
            r"_attn_",
            r"_attention",
            r"flash_?infer",
            r"flash_?attention",
            r"_mla_",
        ],
        content_keywords=[
            "attention", "flashinfer", "flashattention", "flash_attn",
            "mla", "gqa", "mha", "kv_head", "head_dim", "softmax",
            "causal_mask", "page_table", "paged_attention",
            "extend_attention", "decode_attention", "prefill_attention",
            "nsa", "sliding_window", "attn_backend",
        ],
    ),
    ExpertRoute(
        name="moe",
        description="Mixture of Experts: routing, expert parallelism (EP), EPLB, fused MoE, token dispatch",
        path_patterns=[
            r"srt/layers/moe/",
            r"srt/elastic_ep/",
            r"srt/eplb/",
            r"sgl-kernel/csrc/moe/",
            r"_moe",
            r"expert",
        ],
        content_keywords=[
            "moe", "mixture_of_experts", "expert", "top_k", "router",
            "gate", "ep_moe", "fused_moe", "token_dispatch",
            "expert_parallel", "eplb", "load_balance", "aux_loss",
            "capacity_factor", "num_experts", "experts_per_token",
        ],
    ),
    ExpertRoute(
        name="scheduler",
        description="Request scheduling, batch management, prefill/decode scheduling, overlap, DP controller",
        path_patterns=[
            r"srt/managers/scheduler",
            r"srt/managers/schedule_batch",
            r"srt/managers/schedule_policy",
            r"srt/managers/prefill_delayer",
            r"srt/managers/data_parallel",
            r"srt/managers/overlap",
            r"srt/batch_overlap/",
            r"srt/managers/tp_worker",
        ],
        content_keywords=[
            "scheduler", "schedule_batch", "schedule_policy", "prefill",
            "decode", "batch", "running_batch", "waiting_queue",
            "forward_batch", "continuous_batching", "chunked_prefill",
            "dp_controller", "data_parallel", "overlap", "pipeline",
        ],
    ),
    ExpertRoute(
        name="memory_cache",
        description="Radix cache, memory pool, KV cache management, eviction, chunk cache, HiRadix",
        path_patterns=[
            r"srt/mem_cache/",
            r"sgl-kernel/csrc/memory/",
            r"radix",
            r"memory_pool",
            r"kvcache",
        ],
        content_keywords=[
            "radix_cache", "radix_tree", "memory_pool", "kv_cache",
            "evict", "prefix_cache", "chunk_cache", "token_to_kv",
            "req_to_token", "free_slots", "alloc", "cache_hit",
            "hiradix", "session_aware", "flush_cache",
            "kv_data_ptrs", "kv_item_len", "kv_head_num",
            "contiguous_buf", "page_size",
        ],
    ),
    ExpertRoute(
        name="distributed",
        description="Tensor/Data/Pipeline parallelism, NCCL, disaggregated inference, weight sync, all-reduce",
        path_patterns=[
            r"srt/distributed/",
            r"srt/disaggregation/",
            r"srt/weight_sync/",
            r"sgl-kernel/csrc/allreduce/",
            r"_pp_mixin",
            r"_dp_attn",
        ],
        content_keywords=[
            "tensor_parallel", "data_parallel", "pipeline_parallel",
            "nccl", "all_reduce", "all_gather", "broadcast",
            "tp_size", "dp_size", "pp_size", "disaggregate",
            "prefill_decode_disagg", "weight_sync", "communicator",
            "pynccl", "custom_all_reduce",
            "attn_tp_size", "tp_rank", "mooncake_session",
            "bootstrap_info", "transfer_block",
        ],
    ),
    ExpertRoute(
        name="model_arch",
        description="Model implementations (LLaMA, DeepSeek, Qwen, Gemma, etc.), model registry, weight loading",
        path_patterns=[
            r"srt/models/",
            r"srt/model_loader/",
            r"srt/configs/",
        ],
        content_keywords=[
            "ForCausalLM", "DecoderLayer", "model_config", "weight_loader",
            "load_weights", "EntryClass", "model_registry",
            "hf_config", "num_hidden_layers", "hidden_size",
            "intermediate_size", "vocab_size",
        ],
    ),
    ExpertRoute(
        name="lora",
        description="LoRA adapters: management, triton/torch ops, memory pool, multi-adapter serving",
        path_patterns=[
            r"srt/lora/",
        ],
        content_keywords=[
            "lora", "adapter", "lora_rank", "lora_alpha", "sgmv",
            "lora_manager", "lora_config", "lora_a", "lora_b",
            "base_model", "adapter_id", "multi_lora",
        ],
    ),
    ExpertRoute(
        name="hybrid_models",
        description="Hybrid/linear attention models: Mamba, FLA, linear attention, GDN, Lightning attention",
        path_patterns=[
            r"srt/layers/attention/mamba/",
            r"srt/layers/attention/fla/",
            r"srt/layers/attention/linear/",
            r"srt/layers/attention/hybrid",
            r"srt/mem_cache/.*mamba",
            r"falcon_h1",
            r"nemotron_h",
            r"granitemoehybrid",
        ],
        content_keywords=[
            "mamba", "ssm", "state_space", "conv1d", "selective_scan",
            "linear_attention", "fla", "recurrent", "gdn", "lightning",
            "hybrid", "mixer", "delta_rule",
        ],
    ),
    ExpertRoute(
        name="constrained_decoding",
        description="Grammar-guided/structured generation: xgrammar, outlines, llguidance, JSON schema",
        path_patterns=[
            r"srt/constrained/",
            r"sgl-kernel/csrc/grammar/",
        ],
        content_keywords=[
            "grammar", "constrained", "xgrammar", "outlines",
            "llguidance", "json_schema", "regex", "bitmask",
            "structured_output", "jump_forward",
        ],
    ),
    ExpertRoute(
        name="api_serving",
        description="API serving: OpenAI/Anthropic/Ollama compatibility, HTTP server, protocols, function calling",
        path_patterns=[
            r"srt/entrypoints/",
            r"srt/function_call/",
            r"srt/grpc/",
            r"srt/managers/tokenizer_manager",
            r"srt/managers/detokenizer",
            r"srt/managers/io_struct",
        ],
        content_keywords=[
            "openai", "anthropic", "ollama", "chat_completion",
            "streaming", "sse", "http_server", "protocol",
            "function_call", "tool_use", "api_key", "endpoint",
            "v1/chat", "v1/completions",
        ],
    ),
    ExpertRoute(
        name="cuda_kernels",
        description="Custom CUDA/C++ kernels in sgl-kernel and jit_kernel: attention, MoE, memory, gemm ops",
        path_patterns=[
            r"sgl-kernel/",
            r"jit_kernel/",
            r"\.cu$",
            r"\.cuh$",
            r"csrc/",
        ],
        content_keywords=[
            "cuda", "kernel", "triton", "__global__", "blockDim",
            "threadIdx", "shared_memory", "warp", "tl.load",
            "tl.store", "tl.dot", "deep_gemm", "cutlass",
        ],
    ),
    ExpertRoute(
        name="sampling",
        description="Sampling strategies: temperature, top-k/p, penalties, logit processors, batch sampling",
        path_patterns=[
            r"srt/sampling/",
        ],
        content_keywords=[
            "sampling", "temperature", "top_k", "top_p", "penalty",
            "frequency_penalty", "presence_penalty", "min_new_tokens",
            "logit_processor", "sampling_params", "sampling_batch",
        ],
    ),
    ExpertRoute(
        name="multimodal",
        description="Multimodal: image/video/audio processing, vision encoders, multimodal processors",
        path_patterns=[
            r"srt/multimodal/",
            r"multimodal_gen/",
            r"srt/managers/.*mm",
            r"srt/models/.*(vl|vlm|audio|vision|clip|siglip|pixtral|whisper|ocr)",
        ],
        content_keywords=[
            "multimodal", "image", "video", "audio", "vision",
            "pixel_values", "image_processor", "vision_tower",
            "clip", "siglip", "whisper", "evs",
        ],
    ),
    ExpertRoute(
        name="compilation",
        description="torch.compile, CUDA graph capture/replay, piecewise compilation, graph runner",
        path_patterns=[
            r"srt/compilation/",
            r"srt/model_executor/",
            r"cuda_graph",
            r"graph_runner",
        ],
        content_keywords=[
            "torch.compile", "cuda_graph", "graph_runner", "capture",
            "replay", "inductor", "compile", "fx_graph",
            "piecewise", "compilation_config",
        ],
    ),
]


def route_pr(changed_files: list[str], diff_content: str) -> list[tuple[str, float]]:
    """
    Analyze changed files and diff content to determine which experts should review.

    Returns list of (expert_name, confidence_score) sorted by confidence descending.
    """
    scores: dict[str, float] = {}

    diff_lower = diff_content.lower()

    for expert in EXPERT_ROUTES:
        # Score based on file path matches (0 or 1+ files)
        file_match_count = 0
        for filepath in changed_files:
            for pattern in expert.path_patterns:
                if re.search(pattern, filepath, re.IGNORECASE):
                    file_match_count += 1
                    break  # Don't double-count same file

        # File score: any file match is a strong signal
        # 1 match = 0.5, 2+ matches = 0.7, 3+ = 0.85, 5+ = 1.0
        if file_match_count == 0:
            file_score = 0.0
        elif file_match_count == 1:
            file_score = 0.5
        elif file_match_count == 2:
            file_score = 0.7
        elif file_match_count <= 4:
            file_score = 0.85
        else:
            file_score = 1.0

        # Score based on content keywords
        keyword_hits = 0
        for keyword in expert.content_keywords:
            count = diff_lower.count(keyword.lower())
            if count > 0:
                keyword_hits += min(count, 10)  # Cap per-keyword

        # Keyword score: normalize by keyword list size
        keyword_ratio = keyword_hits / max(len(expert.content_keywords), 1)
        keyword_score = min(keyword_ratio / 2.0, 1.0)  # ratio of 2.0+ => 1.0

        # Combined confidence (file paths weighted more heavily)
        confidence = file_score * 0.7 + keyword_score * 0.3

        if confidence >= expert.threshold:
            scores[expert.name] = confidence

    # Sort by confidence
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def get_expert_prompt(expert_name: str) -> str:
    """Load the expert prompt file content."""
    import os
    skill_dir = os.path.dirname(os.path.abspath(__file__))
    expert_path = os.path.join(skill_dir, "experts", f"{expert_name}.md")
    if os.path.exists(expert_path):
        with open(expert_path) as f:
            return f.read()
    return f"No expert prompt found for {expert_name}"


def get_expert_description(expert_name: str) -> str:
    """Get the short description for an expert."""
    for expert in EXPERT_ROUTES:
        if expert.name == expert_name:
            return expert.description
    return expert_name


def format_routing_summary(routes: list[tuple[str, float]]) -> str:
    """Format the routing decision as a readable summary."""
    lines = ["## PR Review Routing\n"]
    for name, confidence in routes:
        desc = get_expert_description(name)
        bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        lines.append(f"- **{name}** [{bar}] {confidence:.0%} — {desc}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    test_files = [
        "python/sglang/srt/speculative/eagle_worker.py",
        "python/sglang/srt/layers/attention/flashinfer_backend.py",
        "python/sglang/srt/managers/scheduler.py",
    ]
    test_diff = "eagle speculative decoding attention flashinfer scheduler batch"
    results = route_pr(test_files, test_diff)
    print(format_routing_summary(results))
