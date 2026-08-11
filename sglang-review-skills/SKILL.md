# SGLang PR Review Skill

Review SGLang GitHub PRs by routing them to domain-expert reviewers, then aggregating their feedback into a unified review comment.

## Usage

```
/review-sglang-pr <PR_URL_or_NUMBER>
```

Example:
```
/review-sglang-pr https://github.com/sgl-project/sglang/pull/1234
/review-sglang-pr 1234
```

## How It Works

1. **Fetch PR**: Retrieves the PR diff, description, and changed files
2. **Route**: Analyzes which files changed and routes to relevant domain experts
3. **Expert Review**: Each expert reviews the PR from their domain perspective
4. **Aggregate**: Combines expert reviews into a single structured comment
5. **Post**: Posts the review as a PR comment (with user confirmation)

## Domain Experts

| Expert | Domain | Key Files/Dirs |
|--------|--------|----------------|
| `speculative_decoding` | EAGLE, ngram, draft models, MTP | `srt/speculative/`, `*_eagle*`, `*_mtp*` |
| `quantization` | FP8, INT8, AWQ, GPTQ, Marlin, MXFP4 | `srt/layers/quantization/`, `sgl-kernel/csrc/quantization/` |
| `attention` | FlashInfer, FlashAttention, MLA, NSA, triton attention ops | `srt/layers/attention/`, `sgl-kernel/csrc/attention/` |
| `moe` | MoE routing, expert parallelism, EPLB, fused MoE | `srt/layers/moe/`, `srt/elastic_ep/`, `srt/eplb/` |
| `scheduler` | Batch scheduling, prefill/decode, overlap, DP controller | `srt/managers/scheduler*.py`, `schedule_batch.py`, `schedule_policy.py` |
| `memory_cache` | Radix cache, memory pool, KV cache, eviction | `srt/mem_cache/`, `sgl-kernel/csrc/memory/` |
| `distributed` | TP/DP/PP, NCCL, disaggregated inference, weight sync | `srt/distributed/`, `srt/disaggregation/`, `srt/weight_sync/` |
| `model_arch` | Model implementations (100+ models), DeepSeek common | `srt/models/` |
| `lora` | LoRA adapters, triton ops, memory management | `srt/lora/` |
| `hybrid_models` | Mamba, linear attention, FLA, hybrid attention | `srt/layers/attention/mamba/`, `fla/`, `linear/` |
| `constrained_decoding` | Grammar backends, structured output (xgrammar, outlines) | `srt/constrained/` |
| `api_serving` | OpenAI/Anthropic/Ollama API, HTTP server, protocols | `srt/entrypoints/` |
| `cuda_kernels` | Custom CUDA/C++ kernels, sgl-kernel | `sgl-kernel/`, `jit_kernel/` |
| `sampling` | Sampling strategies, penalties, logit processors | `srt/sampling/` |
| `multimodal` | Vision, audio, video processing, multimodal models | `srt/multimodal/`, `multimodal_gen/` |
| `compilation` | torch.compile, CUDA graphs, piecewise compilation | `srt/compilation/`, `model_executor/` |

## Architecture

```
review-sglang-pr (main orchestrator)
├── router.py          — analyzes diff, selects experts
├── experts/           — domain expert prompt files
│   ├── speculative_decoding.md
│   ├── quantization.md
│   ├── attention.md
│   ├── moe.md
│   ├── scheduler.md
│   ├── memory_cache.md
│   ├── distributed.md
│   ├── model_arch.md
│   ├── lora.md
│   ├── hybrid_models.md
│   ├── constrained_decoding.md
│   ├── api_serving.md
│   ├── cuda_kernels.md
│   ├── sampling.md
│   ├── multimodal.md
│   └── compilation.md
└── review.py          — orchestrator: fetch PR, route, review, post
```
