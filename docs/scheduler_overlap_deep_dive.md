# SGLang Scheduler Deep Dive: How Overlap Scheduling Works

## Table of Contents

1. [Introduction: The Throughput Problem](#1-introduction)
2. [System Architecture: Three-Tier Batch Abstraction](#2-system-architecture)
3. [Two Execution Modes: Normal vs. Overlap](#3-two-execution-modes)
4. [CUDA Streams: The Foundation of Overlap](#4-cuda-streams)
5. [The FutureMap: Passing Results Across Batch Boundaries](#5-the-futuremap)
6. [The Overlap Event Loop Step by Step](#6-the-overlap-event-loop-step-by-step)
7. [run_batch: What Happens Inside](#7-run-batch)
8. [Scheduling Policy: Choosing the Next Batch](#8-scheduling-policy)
9. [Worked Example: Steady-State Decode Overlap](#9-worked-example)
10. [Edge Cases and Disable Conditions](#10-edge-cases-and-disable-conditions)
11. [Speculative Decoding with Overlap](#11-speculative-decoding-with-overlap)
12. [Summary: Why This Works](#12-summary)

---

## 1. Introduction: The Throughput Problem <a id="1-introduction"></a>

In LLM serving, there are two distinct workloads competing for time on a single CPU thread:

1. **GPU computation** — running the transformer forward pass (very expensive, but runs asynchronously on the GPU)
2. **CPU scheduling** — deciding the next batch: prefix cache matching, KV memory allocation, tensor preparation

In the naive design, these are sequential:

```
Step 1: [CPU] Schedule batch N
Step 2: [GPU] Run batch N forward pass
Step 3: [CPU] Process results of batch N (decode tokens, send outputs)
Step 4: [CPU] Schedule batch N+1
Step 5: [GPU] Run batch N+1 forward pass
...
```

The GPU sits idle during steps 1, 3, and 4. For small batches (common in decode-heavy workloads), scheduling overhead is proportionally large. SGLang's **overlap scheduler** eliminates this waste by pipelining:

```
[CPU] Schedule N+1 │ Process N-1  │ Schedule N+2 │ Process N   │ ...
[GPU]              │   Run N      │              │   Run N+1   │ ...
```

But this creates a **dependency problem**: to schedule batch N+1, we need the output tokens of batch N (what token was generated?). If the GPU is still computing batch N when we're preparing batch N+1, those tokens aren't ready yet. The FutureMap solves this.

---

## 2. System Architecture: Three-Tier Batch Abstraction <a id="2-system-architecture"></a>

SGLang uses three distinct batch representations as data flows from the scheduler down to the GPU:

```
┌──────────────────────────────────────────────────────────┐
│  ScheduleBatch  (scheduler.py)                           │
│  - Python-level request list                             │
│  - Prefix cache indices, sampling configs                │
│  - forward_mode: EXTEND/DECODE/MIXED                     │
│  - output_ids: may contain FUTURE INDICES (negative!)    │
└──────────────────────┬───────────────────────────────────┘
                       │ .get_model_worker_batch()
                       ▼
┌──────────────────────────────────────────────────────────┐
│  ModelWorkerBatch  (schedule_batch.py)                   │
│  - input_ids tensor (may contain negative future values) │
│  - seq_lens, sampling_info                               │
│  - spec_info (for speculative decoding)                  │
│  Owned by: TpModelWorker                                 │
└──────────────────────┬───────────────────────────────────┘
                       │ forward_batch_generation()
                       ▼
┌──────────────────────────────────────────────────────────┐
│  ForwardBatch  (model_executor)                          │
│  - Actual GPU kernel inputs                              │
│  - Attention metadata, KV cache locations                │
│  - Consumed by model layers (RadixAttention, etc.)       │
└──────────────────────────────────────────────────────────┘
```

The key insight: `ScheduleBatch.output_ids` can hold **negative integers** (future indices) instead of real token IDs. These placeholders are resolved to actual tokens on the GPU using the FutureMap — completely bypassing the CPU-side synchronization bottleneck.

---

## 3. Two Execution Modes: Normal vs. Overlap <a id="3-two-execution-modes"></a>

### Normal Mode (`event_loop_normal`, scheduler.py:1115)

```python
def event_loop_normal(self):
    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()   # [CPU] Schedule
        self.cur_batch = batch

        if batch:
            result = self.run_batch(batch)      # [GPU] Forward pass
            self.process_batch_result(batch, result)  # [CPU] Decode tokens, send output
        else:
            self.self_check_during_idle()

        self.last_batch = batch
```

This is simple but leaves the GPU idle whenever the CPU is doing scheduling or output processing.

### Overlap Mode (`event_loop_overlap`, scheduler.py:1142)

```python
def event_loop_overlap(self):
    self.result_queue = deque()  # Holds (batch, result) pairs

    def pop_and_process():
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)  # Process last batch

    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()   # [CPU] Schedule NEXT batch
        self.cur_batch = batch
        disable_overlap = self.is_disable_overlap_for_batch(batch)

        if disable_overlap:
            pop_and_process()  # Sync: process last result before launching

        if batch:
            batch_result = self.run_batch(batch)         # [GPU] Launch (async!)
            self.result_queue.append((batch.copy(), batch_result))

        if self.last_batch:
            if not disable_overlap:
                pop_and_process()  # [CPU] Process LAST result while GPU runs CURRENT

        if self.is_generation:
            self.launch_batch_sample_if_needed(batch_result)  # Deferred sampling

        self.last_batch = batch
```

The critical structural difference: `run_batch` **enqueues** the result rather than blocking on it. Then `pop_and_process` runs the previous result's CPU work **while the GPU is still computing the current batch**.

The `result_queue` acts as a one-slot pipeline buffer — at most one result is queued at any time (the just-launched batch).

---

## 4. CUDA Streams: The Foundation of Overlap <a id="4-cuda-streams"></a>

### What CUDA streams do

A CUDA stream is an ordered queue of GPU operations. Operations in **different streams can execute concurrently** (subject to hardware limits). The host (CPU) can enqueue work into a stream without waiting for it to complete.

SGLang uses two streams (`init_overlap`, scheduler.py:999):

```python
def init_overlap(self):
    self.device_module = torch.get_device_module(self.device)  # torch.cuda

    # Stream 1: where forward computation runs
    self.forward_stream = self.device_module.Stream()
    self.forward_stream_ctx = self.device_module.stream(self.forward_stream)

    # Stream 2: async copies (used for pipeline parallelism)
    self.copy_stream = self.device_module.Stream()
    self.copy_stream_ctx = self.device_module.stream(self.copy_stream)
```

There's also an **implicit default stream** — PyTorch operations use this unless you specify otherwise. This is the "schedule stream" where tensors are created during batch preparation.

### Stream synchronization

In `run_batch` (scheduler.py:2327):

```python
with self.forward_stream_ctx:
    self.forward_stream.wait_stream(self.schedule_stream)
    # ...GPU work here...
```

`forward_stream.wait_stream(schedule_stream)` inserts a **GPU-side fence**: the forward stream won't start executing until all previously enqueued schedule stream work is done. This is **not a CPU block** — the CPU returns immediately while the GPU handles ordering.

```
Schedule Stream:   [tensor alloc] [data prep] ─────────────────────────────
                                               ↑ GPU fence (wait_stream)
Forward Stream:                                └──→ [resolve_future] [forward pass]
```

### Why two streams?

| Aspect | Schedule Stream | Forward Stream |
|--------|----------------|----------------|
| What runs here | Tensor creation, memory copies during batch prep | Transformer forward pass |
| CPU synchronization | CPU waits for this implicitly (PyTorch default) | CPU does NOT wait — kernel runs async |
| Batch N+1 uses | All tensor allocation for next batch | Never (it's on forward stream) |
| Result availability | CPU-visible immediately after stream sync | CPU needs `.copy_to_cpu()` + `copy_done` event |

---

## 5. The FutureMap: Passing Results Across Batch Boundaries <a id="5-the-futuremap"></a>

This is the most novel component. The core problem: when preparing batch N+1, we need the output token IDs of batch N. But batch N is still running on the GPU.

### The solution: negative indices as placeholders

Instead of blocking until batch N finishes, SGLang:

1. **Allocates future slots** — reserved positions in a GPU circular buffer
2. **Puts negative indices** as input_ids for batch N+1 (`-index`)
3. **At the start of batch N+1's forward pass** (on the GPU), resolves those negatives to real token IDs from the buffer
4. **At the end of batch N's forward pass** (on the GPU), writes the results into the buffer

Because both happen on the same GPU stream (forward stream), they're ordered correctly without any CPU synchronization.

### FutureMap structure (`overlap_utils.py:35`)

```python
class FutureMap:
    def __init__(self, max_running_requests, chunked_prefill_size, context_len, device, spec_algo):
        # Circular buffer: large enough to hold in-flight futures
        max_num_chunks = context_len // chunked_prefill_size
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        self.future_buffer_len = self.future_limit + 2 * max_running_requests

        # The actual GPU buffer storing next token IDs
        self.token_ids_buf = torch.empty(
            (self.future_buffer_len,), dtype=torch.int64, device=device
        )
```

Why `3 + max_num_chunks` per request? The circular buffer must hold results for all in-flight batches:
- 1 slot for the current decode result
- 1 slot per prefill chunk (up to `max_num_chunks`)
- Extra margin (+2 * max_running_requests) for safety

### Step 1: Allocate future indices (`alloc_future_indices`, line 111)

Called on the CPU **before** launching batch N's forward pass:

```python
def alloc_future_indices(self, bs: int) -> FutureIndices:
    cur_future_ct = self.future_ct
    self.future_ct = (cur_future_ct + bs) % self.future_limit  # Advance circular pointer
    start = cur_future_ct + 1
    end   = cur_future_ct + 1 + bs
    indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
    return FutureIndices(indices=indices, interval=slice(start, end))
```

Returns `FutureIndices(indices=[3, 4, 5, ...])` for a batch of size 3.

These indices are then negated: `batch.output_ids = -future_indices.indices` = `[-3, -4, -5, ...]`.

When the next batch is prepared, it uses these negative values as `input_ids`. The scheduler passes them through into `ModelWorkerBatch.input_ids`.

### Step 2: Store results (`store_to_map`, line 151)

At the end of batch N's forward pass, **on the GPU (forward stream)**:

```python
def store_to_map(self, future_indices, batch_result):
    intv = future_indices.interval       # e.g., slice(3, 6)
    self.token_ids_buf[intv] = batch_result.next_token_ids  # GPU tensor write
```

This runs on the forward stream — after the forward pass sampled the next token but before batch N+1's forward pass starts.

### Step 3: Resolve futures (`resolve_future`, line 120)

At the **very start** of batch N+1's forward pass, still on the forward stream:

```python
@torch.compile(dynamic=True)
def _resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,                              # Is this a future placeholder?
        future_token_ids_map[torch.clamp(-input_ids, min=0)],  # Yes: look up actual token
        input_ids,                                  # No: use as-is (real token)
    )
```

This is a compiled GPU kernel. Because it's on the same stream as the `store_to_map` write, the ordering guarantee is:

```
Forward Stream timeline:
  Batch N:  [forward pass] → [sample tokens] → [store_to_map: write buf[3..6]]
  Batch N+1:                                                    → [resolve_future: read buf[3..6]] → [forward pass]
```

No CPU synchronization needed. The GPU enforces the ordering.

### Why this is elegant

The CPU never sees the actual token IDs. It only deals with indices. The GPU does all the real work of resolving them at the right time. From the CPU's perspective:

```python
# CPU code for preparing batch N+1:
future_indices_or_next_token_ids = -future_indices.indices  # Negative placeholders
batch.output_ids = future_indices_or_next_token_ids

# These negatives become input_ids for the next token in decode mode
# The GPU will fix them before the transformer ever sees them
```

---

## 6. The Overlap Event Loop Step by Step <a id="6-the-overlap-event-loop-step-by-step"></a>

Let's trace exactly what happens in each iteration, assuming steady-state decode (most common case):

### Iteration T (processing batch N, launching batch N+1)

```
╔══════════════════════════════════════════════════════════════════╗
║ CPU (schedule stream)                                            ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. recv_requests() — check ZMQ for new incoming requests         ║
║ 2. process_input_requests() — tokenize, add to waiting_queue     ║
║ 3. get_next_batch_to_run()                                       ║
║    ├─ update_running_batch() — remove finished reqs              ║
║    ├─ match_prefix() on waiting requests (radix cache)           ║
║    ├─ inc_lock_ref() on matched tree nodes                       ║
║    ├─ alloc_for_decode() — allocate KV slots for 1 new token    ║
║    └─ prepare_for_decode() — build seq_lens, sampling tensors    ║
║ 4. run_batch(batch N+1)                                          ║
║    ├─ get_model_worker_batch() — pack into ModelWorkerBatch      ║
║    ├─ record_batch_in_overlap() — prevent GC of GPU tensors      ║
║    ├─ alloc_future_indices(bs) → future_indices for batch N+1    ║
║    └─ Enqueue on forward stream (returns immediately to CPU)     ║
║         ↓ (GPU fence: wait for schedule stream)                  ║
║ 5. result_queue.append((batch_N+1_copy, batch_result))           ║
║ 6. pop_and_process()  ← Process batch N's result                 ║
║    ├─ Wait for batch_N.copy_done event                           ║
║    ├─ cache_finished_req() / cache_unfinished_req()              ║
║    ├─ Decode output tokens from CPU copy                         ║
║    └─ Send responses via ZMQ                                     ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║ GPU (forward stream) — running concurrently with CPU above       ║
╠══════════════════════════════════════════════════════════════════╣
║ [After GPU fence clears]                                         ║
║ 1. resolve_future(model_worker_batch_N+1)                        ║
║    └─ Replace negative input_ids with real tokens from buf       ║
║ 2. forward_batch_generation() for batch N+1                      ║
║    ├─ Flash attention using req_to_token_pool                    ║
║    └─ Sampling: argmax/topk over logits                          ║
║ 3. store_to_map(future_indices_N+1, result)                      ║
║    └─ Write next_token_ids into token_ids_buf[future_N+1_slots]  ║
║ 4. copy_to_cpu() — async D2H copy of token IDs + logprobs        ║
║ 5. copy_done.record() — CPU-observable event                     ║
╚══════════════════════════════════════════════════════════════════╝
```

### The key insight: what overlaps with what?

```
Time →
CPU: ┌──schedule N+1──┐┌──process N──┐┌──schedule N+2──┐┌──process N+1──┐
GPU:                   ┌─────forward N+1──────┐          ┌─────forward N+2──────┐
```

The CPU processes batch N's results **while** the GPU runs batch N+1's forward. This is the overlap.

### The `copy_done` event

To read GPU results on the CPU, SGLang uses a CUDA Event:

```python
# In run_batch, on forward stream:
batch_result.copy_done = self.device_module.Event()
batch_result.copy_to_cpu(...)     # Async D2H copy, on forward stream
# copy_done.record() happens after copy_to_cpu completes
```

In `process_batch_result`, before reading tokens:
```python
batch_result.copy_done.synchronize()  # CPU blocks until D2H copy completes
# Now safe to access batch_result.next_token_ids on CPU
```

This is the **only point where the CPU actually waits** for the GPU in the overlap path.

---

## 7. run_batch: What Happens Inside <a id="7-run-batch"></a>

The full sequence inside `run_batch` with overlap enabled (scheduler.py:2315):

```python
def run_batch(self, batch):
    model_worker_batch = batch.get_model_worker_batch()  # Pack to GPU-ready form
    self.record_batch_in_overlap(model_worker_batch)      # Prevent premature GC

    # Sampling info may be mutated during forward — take a snapshot
    model_worker_batch.sampling_info = model_worker_batch.sampling_info.copy_for_forward()

    bs = len(model_worker_batch.seq_lens)
    future_indices = self.future_map.alloc_future_indices(bs)  # Reserve buffer slots

    with self.forward_stream_ctx:                # Switch to forward stream
        self.forward_stream.wait_stream(self.schedule_stream)  # GPU fence

        # Step 1: Resolve any future placeholders in input_ids
        self.future_map.resolve_future(model_worker_batch)

        # Step 2: Run the actual transformer forward
        batch_result = self.model_worker.forward_batch_generation(model_worker_batch)

        # Step 3: Store THIS batch's results into the future map
        batch_result.copy_done = self.device_module.Event()
        if batch_result.delay_sample_func is None:
            self.future_map.store_to_map(future_indices, batch_result)
            batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
        else:
            # Delayed sampling (spec v2): sample happens after last batch is processed
            batch_result.future_indices = future_indices

    # Put NEGATIVE future indices into output_ids — these become next batch's input_ids
    batch.output_ids = -future_indices.indices

    return batch_result
```

### The double-buffer trick (`record_batch_in_overlap`)

```python
def record_batch_in_overlap(self, model_worker_batch):
    self.batch_record_ct = (self.batch_record_ct + 1) % 2
    self.batch_record_buf[self.batch_record_ct] = model_worker_batch
```

Tensors in `model_worker_batch` were created on the default stream (schedule stream) during `get_model_worker_batch()`. When they're used by the forward stream, Python might garbage-collect them before the GPU finishes. This double-buffer keeps a Python-level reference alive for at least 2 iterations, safely outlasting the GPU work.

---

## 8. Scheduling Policy: Choosing the Next Batch <a id="8-scheduling-policy"></a>

`get_next_batch_to_run()` runs on the CPU while the GPU is busy with the previous batch. Here's what it does:

### Priority: Prefill over Decode

```
if waiting_queue has requests:
    new_batch = get_new_batch_prefill()   # Try to schedule a prefill batch
    if new_batch is not None:
        return new_batch

# Otherwise, run decode for running requests
return update_running_batch(running_batch)
```

### `get_new_batch_prefill` — the scheduling decision

1. **Compute prefix matches** for all waiting requests:
   ```python
   for req in waiting_queue:
       match_result = tree_cache.match_prefix(MatchPrefixParams(...))
       req.prefix_indices = match_result.device_indices
       req.last_node = match_result.last_device_node
   ```

2. **Sort by policy**: `lpm` (longest prefix match), `dfs-weight`, `fcfs`, etc.

3. **Greedily add requests** until memory limit:
   ```python
   for req in sorted_waiting_queue:
       est_tokens_needed = seq_len - prefix_len
       if available_tokens >= est_tokens_needed:
           add to batch
   ```

4. **Lock cache nodes** for selected requests:
   ```python
   tree_cache.inc_lock_ref(req.last_node)
   ```

### `update_running_batch` — decode step prep

For an ongoing decode batch:
1. Filter out finished requests
2. `alloc_for_decode()` — allocate 1 new KV slot per running request
3. `prepare_for_decode()` — update `seq_lens` tensors, reset KV locations

All of this CPU work happens while the GPU runs the previous forward pass.

---

## 9. Worked Example: Steady-State Decode Overlap <a id="9-worked-example"></a>

Let's trace 4 consecutive decode iterations with 3 running requests.

### Setup

- 3 requests: R1, R2, R3 all in decode phase
- Token IDs after previous step: R1→42, R2→17, R3→88
- FutureMap: `token_ids_buf` is a GPU tensor of size 1000

### Iteration 1 — CPU prepares batch B1, launches, processes old batch B0

**CPU (schedule stream):**

```
1. get_next_batch_to_run()
   - Filter finished reqs (none)
   - alloc_for_decode(): allocate new KV slots [kv99, kv100, kv101]
   - prepare_for_decode(): seq_lens = [50, 30, 70]

2. run_batch(B1):
   - alloc_future_indices(3) → indices=[1,2,3], buf_slice=slice(1,4)

   [Enter forward_stream context]
   - forward_stream.wait_stream(schedule_stream)  ← GPU fence (not CPU block!)
   - Enqueue: resolve_future(B1.input_ids)        ← reads buf[prev_indices]
   - Enqueue: forward pass
   - Enqueue: store_to_map(indices, result)        ← writes buf[1..3]
   - Enqueue: copy_to_cpu(result)
   [Exit context — GPU starts executing async]

   - B1.output_ids = [-1, -2, -3]                ← negative future indices
   - result_queue.append((B1_copy, batch_result))

3. pop_and_process() ← Process B0's result
   - batch_result.copy_done.synchronize()         ← WAIT for D2H copy of B0
   - Read CPU copy of B0's next_token_ids = [42, 17, 88]
   - cache_unfinished_req(R1, R2, R3)
   - Send intermediate outputs to detokenizer
```

**GPU (forward stream, concurrent with step 3 above):**

```
- fence clears (schedule stream done with tensor prep)
- resolve_future(B1): B1.input_ids had [-prev1, -prev2, -prev3]
  → replace with token_ids_buf[prev1], token_ids_buf[prev2], token_ids_buf[prev3]
  → B1.input_ids = [42, 17, 88] (the actual tokens!)
- Transformer forward pass with these tokens
- Sample: next tokens = [55, 91, 7]
- store_to_map: token_ids_buf[1]=55, token_ids_buf[2]=91, token_ids_buf[3]=7
- async D2H copy: copy 55,91,7 to CPU pinned memory
- copy_done.record()
```

### Iteration 2 — Same pattern, but now input_ids = [-1, -2, -3]

**CPU (schedule stream):**

```
1. B1.output_ids = [-1, -2, -3] (from last iteration)
2. get_next_batch_to_run():
   - input_ids for B2 = B1.output_ids = [-1, -2, -3]  ← still future values!
   - alloc_for_decode(), prepare_for_decode()

3. run_batch(B2):
   - alloc_future_indices(3) → indices=[4,5,6]

   [forward stream]
   - resolve_future(B2.input_ids):
     input_ids = [-1, -2, -3]
     → [token_ids_buf[1], token_ids_buf[2], token_ids_buf[3]]
     → [55, 91, 7]  (B1's results, written by GPU in iteration 1)
   - forward pass with [55, 91, 7]
   - sample → [12, 34, 56]
   - store_to_map: token_ids_buf[4]=12, [5]=34, [6]=56
   ...

4. pop_and_process() ← Process B1's result
   - copy_done.synchronize() ← B1's D2H already done (or nearly done)
   - CPU reads tokens [55, 91, 7]
   - Send to detokenizer
```

The GPU resolved the future values itself, without the CPU ever seeing the intermediate tokens. The CPU only sees them when it's time to send them to the user.

---

## 10. Edge Cases and Disable Conditions <a id="10-edge-cases-and-disable-conditions"></a>

Not every batch can be overlapped. `is_disable_overlap_for_batch` (scheduler.py:1195) turns off overlap when:

### Case 1: Consecutive prefill batches

```python
disable_overlap = (
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP
    and batch.forward_mode.is_extend()
    and self.last_batch.forward_mode.is_extend()
)
```

When two prefill batches run back-to-back, overlapping the second with the first's result processing would **delay TTFT** (time-to-first-token) for the first batch. Since the result can't be sent until the second batch starts anyway, you'd rather process and send the first batch's response immediately.

### Case 2: Speculative decoding + grammar constraints

```python
need_grammar_sync = (
    batch.is_spec_v2 and batch.has_grammar
    and batch.forward_mode.is_decode()
    and len(self.result_queue) > 0
)
```

Grammar-constrained generation with speculative decoding requires verifying accepted tokens against the grammar **before** the next draft batch can proceed. This verification happens on CPU and needs the previous result — incompatible with overlap.

### When disabled: synchronous path

```python
if disable_overlap_for_batch:
    pop_and_process()   # Process last result NOW, before launching current batch

# Launch current batch
batch_result = self.run_batch(batch)
result_queue.append(...)

# Skip the normal pop_and_process at the end
if self.last_batch:
    if not disable_overlap_for_batch:  # False — we already processed above
        pop_and_process()
```

This degrades gracefully to normal (sequential) behavior.

### Delayed sampling (`launch_batch_sample_if_needed`)

For speculative decoding v2, sampling is the most expensive CPU-GPU synchronized part. SGLang defers it:

```python
def launch_batch_sample_if_needed(self, batch_result):
    if batch_result.delay_sample_func is None:
        return  # Not needed for standard decode

    with self.forward_stream_ctx:
        self.forward_stream.wait_stream(self.schedule_stream)
        # Run sampling AFTER last batch's CPU processing
        # (because grammar state is updated by process_batch_result)
        _batch_result = batch_result.delay_sample_func()
        self.future_map.store_to_map(batch_result.future_indices, batch_result)
        batch_result.copy_to_cpu(return_logprob=...)
```

This is called **after** `pop_and_process()`, so grammar state is already updated before sampling consults it.

---

## 11. Speculative Decoding with Overlap <a id="11-speculative-decoding-with-overlap"></a>

For EAGLE speculative decoding, the FutureMap stores more than just token IDs. After `_lazy_init_buf`, it holds:

```python
self.topk_p_buf       # Top-k probabilities for draft tree construction
self.topk_index_buf   # Top-k token indices
self.verified_id_buf  # Which draft tokens were accepted
self.new_seq_lens_buf # Updated sequence lengths after verification
self.hidden_states_buf  # Hidden states for draft model (optional)
```

The same three-step pattern (alloc → store → resolve) applies to all these tensors. The draft model for the next batch gets its inputs entirely from the GPU buffer, without any CPU round-trip.

---

## 12. Summary: Why This Works <a id="12-summary"></a>

The overlap scheduler is built on three interlocking ideas:

### 1. CUDA stream separation

```
Default stream (schedule): tensor allocation, batch prep
Forward stream:            forward pass, sampling, D2H copies
```

The CPU can continue scheduling on the default stream while the forward stream runs independently. `forward_stream.wait_stream(schedule_stream)` provides a lightweight GPU-side ordering fence.

### 2. Future indices as a zero-copy IPC mechanism

Instead of transferring token IDs from GPU to CPU to GPU again (the slow path), the FutureMap keeps everything on the GPU:

```
GPU batch N → writes token_ids_buf[1..bs]
GPU batch N+1 ← reads token_ids_buf[1..bs] (via resolve_future)
```

The CPU only touches the token IDs much later, when it needs to send them to the user (after `copy_done.synchronize()`).

### 3. Result queue as a one-slot pipeline

The `result_queue` always holds at most one item (the just-launched batch). The processing pattern:

```
[Launch N+1] → [process N] → [Launch N+2] → [process N+1] → ...
```

guarantees that at most one batch worth of GPU results is "in flight" at any time, keeping memory overhead predictable.

### Throughput impact

In a healthy decode-heavy steady state, the CPU scheduling overhead (typically 0.5–2ms) is fully hidden behind the GPU forward pass (typically 5–20ms for modern LLMs). This translates to near-100% GPU utilization versus 70–90% in non-overlap mode, depending on batch size and model architecture.

### Key files reference

| Component | File | Key Lines |
|-----------|------|-----------|
| Normal event loop | `scheduler.py` | 1115–1140 |
| Overlap event loop | `scheduler.py` | 1142–1193 |
| Disable overlap check | `scheduler.py` | 1195–1217 |
| Stream initialization | `scheduler.py` | 999–1022 |
| run_batch (overlap path) | `scheduler.py` | 2315–2361 |
| Delayed sampling | `scheduler.py` | 2433–2446 |
| FutureMap alloc | `overlap_utils.py` | 111–118 |
| FutureMap resolve | `overlap_utils.py` | 120–143 |
| FutureMap store | `overlap_utils.py` | 151–177 |
| Batch preparation | `schedule_policy.py` | 182–239 |
| Decode alloc | `common.py` | 423–462 |
