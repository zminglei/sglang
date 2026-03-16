# SGLang RadixCache Deep Dive: How KV Cache Sharing Works End-to-End

## Table of Contents

1. [Introduction: Why a Radix Tree for KV Cache?](#1-introduction)
2. [Core Data Structures](#2-core-data-structures)
3. [The Radix Tree: Insert, Match, Split](#3-the-radix-tree)
4. [End-to-End Request Lifecycle](#4-end-to-end-request-lifecycle)
5. [Prefix Matching During Scheduling](#5-prefix-matching-during-scheduling)
6. [Prefill Phase: Allocation and Cache Reuse](#6-prefill-phase)
7. [Decode Phase: Token-by-Token Growth](#7-decode-phase)
8. [Eviction: Reclaiming Memory Under Pressure](#8-eviction)
9. [Lock References: Protecting In-Flight KV Data](#9-lock-references)
10. [Worked Example: Two Requests Sharing a Prefix](#10-worked-example)
11. [Advanced Topics](#11-advanced-topics)

---

## 1. Introduction: Why a Radix Tree for KV Cache? <a id="1-introduction"></a>

In LLM serving, every token that passes through the model produces a key-value (KV) pair that must be stored for subsequent attention computations. When multiple requests share a common prefix (e.g., the same system prompt, few-shot examples, or shared conversation history), recomputing those KV pairs is wasteful.

SGLang solves this with a **Radix Tree** (also known as a compressed trie). The key insight:

> **Token sequences are like strings in a dictionary. A radix tree naturally shares common prefixes, so requests with overlapping token prefixes can share the same KV cache entries in GPU memory.**

For example, if Request A processes tokens `[1, 2, 3, 4, 5]` and Request B later arrives with tokens `[1, 2, 3, 6, 7]`, the KV cache for `[1, 2, 3]` is computed once and reused. Request B only needs to compute KV for `[6, 7]`.

### What makes this different from vLLM's PagedAttention?

While vLLM's PagedAttention focuses on efficient memory management through paging, SGLang's RadixCache goes further by enabling **automatic prefix sharing** across requests. The tree structure naturally identifies shared prefixes, while vLLM requires explicit prefix-caching mechanisms on top of paging.

---

## 2. Core Data Structures <a id="2-core-data-structures"></a>

### 2.1 TreeNode

Every node in the radix tree is a `TreeNode` (defined in `radix_cache.py:117`):

```
TreeNode
├── key: RadixKey          # The token subsequence stored at this edge
├── value: torch.Tensor    # KV cache pool indices (int64) for these tokens
├── children: dict         # Child nodes keyed by first token(s)
├── parent: TreeNode       # Back-pointer for upward traversal
├── lock_ref: int          # Reference count — protected from eviction when > 0
├── last_access_time       # For LRU eviction
├── creation_time          # For FIFO eviction
├── hit_count              # For LFU eviction
├── priority               # For priority-aware eviction
└── host_value             # For HiCache (CPU offloading)
```

Key properties:
- **`value` is a tensor of pool indices**, not the actual KV tensors. These indices point into the `TokenToKVPool`, which holds the real KV data on GPU.
- **`lock_ref > 0` means the node is "protected"** — an active request is using this prefix, so it cannot be evicted.
- A node is considered **evicted** when `value is None` (`evicted` property at line 149).

### 2.2 RadixKey

`RadixKey` wraps the token sequence with an optional namespace (`extra_key`):

```python
RadixKey(
    token_ids=[1, 2, 3, 4, 5],  # The actual token sequence
    extra_key="lora_adapter_1",  # Namespace isolation (LoRA, cache salt, etc.)
    is_bigram=False              # EAGLE speculative decoding flag
)
```

The `extra_key` ensures that **two requests with the same tokens but different LoRA adapters never share KV cache**. This is critical for multi-tenant serving.

### 2.3 Memory Pools

Two pools work together with the radix tree:

```
┌─────────────────────────────────────────────────┐
│               ReqToTokenPool                     │
│  req_to_token[req_pool_idx, position] → kv_idx  │
│  Maps (request, sequence position) → pool index  │
└─────────────────────────────────────────────────┘
                        │
                        ▼ kv_idx
┌─────────────────────────────────────────────────┐
│            TokenToKVPool (GPU)                   │
│  kv_data[layer][kv_idx] → actual K, V tensors   │
│  The physical KV cache storage on GPU            │
└─────────────────────────────────────────────────┘
                        ▲
                        │ value tensor (pool indices)
┌─────────────────────────────────────────────────┐
│               RadixCache (Tree)                  │
│  Tracks which pool indices belong to which       │
│  token prefix; enables sharing across requests   │
└─────────────────────────────────────────────────┘
```

- **`ReqToTokenPool`**: A 2D array `[max_batch, max_context_len]` mapping each request's sequence positions to KV pool indices.
- **`TokenToKVPoolAllocator`**: Manages free/used KV pool slots. When the tree evicts a node, the pool indices are returned here.

---

## 3. The Radix Tree: Insert, Match, Split <a id="3-the-radix-tree"></a>

### 3.1 Tree Structure

The radix tree is a **compressed trie** where each edge can store multiple tokens. Here's what a tree looks like after inserting several sequences:

```
Root (key=[], value=[])
├── [1,2,3] → value=[idx0, idx1, idx2]
│   ├── [4,5] → value=[idx3, idx4]
│   └── [6,7,8] → value=[idx5, idx6, idx7]
└── [8,9,10] → value=[idx8, idx9, idx10]
```

Children are keyed by their first token (or first `page_size` tokens for paged mode). The `get_child_key_fn` extracts this:
- Page size 1: child key = first token id (e.g., `1`, `8`)
- Page size > 1: child key = tuple of first N tokens (e.g., `(1, 2)`)
- With `extra_key`: child key = `(extra_key, first_token)` (e.g., `("lora_1", 1)`)

### 3.2 Prefix Matching (`match_prefix`)

**Purpose**: Given a token sequence, find the longest prefix already cached in the tree.

**Algorithm** (`_match_prefix_helper`, line 661):

```
function match_prefix(node=root, key=[1,2,3,6,7]):
    values = []

    while key is not empty:
        child_key = get_child_key(key)    # e.g., 1
        if child_key not in node.children:
            break                          # No match, stop

        child = node.children[child_key]
        prefix_len = key_match(child.key, key)  # How many tokens match?

        if prefix_len < len(child.key):
            # PARTIAL MATCH — need to split the node!
            new_node = split_node(child, prefix_len)
            values.append(new_node.value)
            break
        else:
            # FULL MATCH of this edge — continue down
            values.append(child.value)
            key = key[prefix_len:]   # Consume matched tokens
            node = child

    return concat(values), node   # KV indices + last matched node
```

**Key behaviors**:
- Updates `last_access_time` on every visited node (for LRU eviction).
- Returns the concatenated KV pool indices for the entire matched prefix.
- Returns the `last_node` — used later for lock reference management.

### 3.3 Node Splitting

Splitting is the most subtle operation. It happens when a match **ends in the middle of a node's key**.

**Before split** — matching `[1,2,3]` against a node with key `[1,2,3,4,5]`:

```
Parent
└── [1,2,3,4,5] → value=[a,b,c,d,e]   (child)
```

**After split** at position 3:

```
Parent
└── [1,2,3] → value=[a,b,c]     (new_node)
    └── [4,5] → value=[d,e]     (child, now shortened)
```

The split operation (`_split_node`, line 687):
1. Creates a `new_node` with the matched prefix portion
2. The original child is shortened to hold only the remaining suffix
3. The new_node takes the child's parent, lock_ref count, and priority
4. The child becomes a child of new_node
5. KV values are `.clone()`d to separate the tensors

This is essential because it creates a precise boundary where future insertions can branch.

### 3.4 Insertion (`insert`)

**Purpose**: Add a new token sequence and its KV indices to the tree.

**Algorithm** (`_insert_helper`, line 708):

```
function insert(node=root, key=[1,2,3,6,7], value=[a,b,c,d,e]):
    total_prefix_length = 0

    while key is not empty:
        child_key = get_child_key(key)
        if child_key not in node.children:
            break   # Need to create new node

        child = node.children[child_key]
        prefix_len = key_match(child.key, key)
        total_prefix_length += prefix_len

        # Consume matched portion
        key = key[prefix_len:]
        value = value[prefix_len:]

        if prefix_len < len(child.key):
            # Partial match — split first
            split_node(child, prefix_len)
            break

        node = child

    if key still has remaining tokens:
        # Create new leaf for unmatched suffix
        new_leaf = TreeNode()
        new_leaf.key = key
        new_leaf.value = value.clone()
        node.children[child_key] = new_leaf
        evictable_size += len(key)

    return total_prefix_length   # How much was already cached
```

**The return value `total_prefix_length` is crucial**: it tells the caller how many tokens were *already* in the tree, so their *duplicate* KV pool indices can be freed.

---

## 4. End-to-End Request Lifecycle <a id="4-end-to-end-request-lifecycle"></a>

Here's the complete journey of a request through the RadixCache:

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. REQUEST ARRIVES                                                  │
│     └─ Scheduler places request in waiting queue                     │
│                                                                      │
│  2. SCHEDULING DECISION (schedule_policy.py)                         │
│     ├─ match_prefix() on all waiting requests                        │
│     ├─ Compute prefix_len, estimate memory needed                    │
│     └─ Select which requests to run                                  │
│                                                                      │
│  3. PREFILL PREPARATION                                              │
│     ├─ inc_lock_ref(last_node) — protect cached prefix               │
│     ├─ alloc_for_extend() — allocate KV slots for NEW tokens         │
│     │   └─ May trigger evict() if memory is low                      │
│     └─ Write prefix indices + new indices to req_to_token_pool       │
│                                                                      │
│  4. PREFILL FORWARD PASS                                             │
│     ├─ Only compute attention for extend tokens (not prefix!)        │
│     └─ KV pairs written to TokenToKVPool at allocated indices        │
│                                                                      │
│  5. DECODE LOOP (repeated per token)                                 │
│     ├─ alloc_for_decode() — allocate 1 new KV slot                   │
│     │   └─ May trigger evict() if memory is low                      │
│     ├─ Forward pass — compute next token                             │
│     ├─ If request NOT finished:                                      │
│     │   └─ cache_unfinished_req()                                    │
│     │       ├─ insert() current tokens into tree                     │
│     │       ├─ Free duplicate KV indices                             │
│     │       ├─ dec_lock_ref(old_node)                                │
│     │       └─ inc_lock_ref(new_node)                                │
│     └─ If request FINISHED:                                          │
│         └─ release_kv_cache() → cache_finished_req()                 │
│             ├─ insert() final token sequence into tree               │
│             ├─ Free duplicate KV indices                             │
│             ├─ dec_lock_ref(last_node)                               │
│             └─ Free request pool slot                                │
│                                                                      │
│  6. MEMORY PRESSURE (anytime during alloc)                           │
│     └─ evict()                                                       │
│         ├─ Pick leaf with lowest priority (LRU/LFU/FIFO...)          │
│         ├─ Free its KV pool indices                                  │
│         ├─ Remove leaf from tree                                     │
│         └─ If parent now childless + unlocked → becomes evictable    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Prefix Matching During Scheduling <a id="5-prefix-matching-during-scheduling"></a>

When the scheduler decides which requests to run, it first computes how much of each request's input is already cached. This happens in `schedule_policy.py`:

```python
# For each waiting request:
match_result = tree_cache.match_prefix(
    MatchPrefixParams(key=RadixKey(token_ids=prefix_ids, extra_key=extra_key))
)
req.prefix_indices = match_result.device_indices   # Cached KV pool indices
req.last_node = match_result.last_device_node       # Tree position
```

This determines the **extend length** — how many new tokens need computation:

```
extend_len = total_input_len - prefix_len
```

If a request has a long cached prefix, it needs less compute, so the scheduler can fit more requests or choose to prioritize it (e.g., "locality-aware" scheduling sorts by prefix length descending, maximizing cache reuse).

### Why match during scheduling (not just at prefill)?

Matching early lets the scheduler make informed batching decisions. If Request A has 90% prefix hit and Request B has 0%, the scheduler knows A is cheap and can be batched alongside other work. It also avoids over-evicting: the scheduler can check if `available + evictable >= needed` before committing.

---

## 6. Prefill Phase: Allocation and Cache Reuse <a id="6-prefill-phase"></a>

Once requests are selected for prefill, the system:

### Step 1: Lock the prefix

```python
tree_cache.inc_lock_ref(req.last_node)
```

This walks from the matched node up to the root, incrementing `lock_ref` on every node. While locked, these nodes **cannot be evicted**. This is critical — we're about to use these KV cache entries for the attention computation, and evicting them would corrupt results.

### Step 2: Allocate new KV slots

`alloc_for_extend()` (in `common.py:328`) allocates GPU KV pool slots for the **extend tokens** (tokens not covered by the cached prefix):

```python
# Simplified flow:
evict_from_tree_cache(tree_cache, num_tokens)  # Make room if needed
out_cache_loc = allocator.alloc(num_tokens)    # Get pool indices
```

### Step 3: Write index mapping

The `req_to_token_pool` is populated with both:
1. **Prefix indices** (from the tree match) — positions 0..prefix_len
2. **New allocation indices** (from the allocator) — positions prefix_len..seq_len

This uses a Triton kernel (`write_req_to_token_pool_triton`) for GPU-side efficiency:

```
req_to_token[req_pool_idx, 0:prefix_len] = prefix_indices  (from cache)
req_to_token[req_pool_idx, prefix_len:seq_len] = out_cache_loc  (newly allocated)
```

### Step 4: Forward pass

The model's forward pass uses `RadixAttention` (in `layers/radix_attention.py`), which reads KV data from `TokenToKVPool` using the indices in `req_to_token_pool`. Only the **extend tokens** go through the full computation; the prefix tokens' KV is read directly from the pool.

---

## 7. Decode Phase: Token-by-Token Growth <a id="7-decode-phase"></a>

During autoregressive decoding, one new token is generated per step.

### Per-step allocation

```python
out_cache_loc = alloc_for_decode(batch, token_per_req=1)
# Writes: req_to_token[req_pool_idx, seq_len] = out_cache_loc
```

### Caching unfinished requests

After each decode step (for chunked prefill or ongoing decode), `cache_unfinished_req()` is called:

```python
def cache_unfinished_req(self, req, chunked=False):
    token_ids = req.fill_ids
    kv_indices = req_to_token_pool[req.req_pool_idx, :len(token_ids)]

    # Insert all tokens so far into the tree
    result = self.insert(InsertParams(key=radix_key, value=kv_indices))
    new_prefix_len = result.prefix_len

    # Free KV indices that are now tracked by the tree (duplicates)
    allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])

    # Re-match to get updated tree position
    match_result = self.match_prefix(MatchPrefixParams(key=radix_key))

    # Update lock references: release old, acquire new
    self.dec_lock_ref(req.last_node)
    self.inc_lock_ref(new_last_node)

    req.last_node = new_last_node
    req.cache_protected_len = len(new_indices)
```

**Why insert then re-match?** The insert may change the tree structure (splitting nodes, etc.), so the node references might be stale. Re-matching gives fresh, correct node pointers.

### Caching finished requests

When a request completes, `cache_finished_req()` is called (line 459):

```python
def cache_finished_req(self, req, is_insert=True):
    token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
    kv_indices = req_to_token_pool[req.req_pool_idx, :len(token_ids)]

    if is_insert:
        result = self.insert(InsertParams(key=radix_key, value=values))
        # Free duplicates (tokens already in tree)
        allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])
    else:
        # Don't cache — free everything
        allocator.free(kv_indices[req.cache_protected_len : len(keys)])

    # Release lock — this prefix is now available for eviction
    self.dec_lock_ref(req.last_node)
```

After this, the **entire token sequence** (input + output) is in the tree. Future requests with overlapping prefixes will benefit from it.

---

## 8. Eviction: Reclaiming Memory Under Pressure <a id="8-eviction"></a>

### When does eviction happen?

Eviction is triggered **lazily** — only when an allocation would fail:

```python
def evict_from_tree_cache(tree_cache, num_tokens):
    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator.available_size() < num_tokens:
        tree_cache.evict(EvictParams(num_tokens=num_tokens))
```

This is called from:
- `alloc_token_slots()` — before prefill token allocation
- `alloc_paged_token_slots_extend()` — before paged prefill allocation
- `alloc_paged_token_slots_decode()` — before decode allocation

### Eviction algorithm

The `evict()` method (line 578) uses a **min-heap** based on the configured eviction strategy:

```python
def evict(self, params):
    num_tokens = params.num_tokens

    # Build heap from evictable leaf nodes
    leaves = list(self.evictable_leaves)
    heap = [(strategy.get_priority(node), node) for node in leaves]
    heapify(heap)

    num_evicted = 0
    while num_evicted < num_tokens and heap:
        _, node = heappop(heap)             # Lowest priority leaf

        allocator.free(node.value)          # Return KV indices to pool
        num_evicted += len(node.value)
        delete_leaf(node)                   # Remove from tree

        # Cascade: if parent is now a childless, unlocked leaf → evictable
        if len(node.parent.children) == 0 and node.parent.lock_ref == 0:
            heappush(heap, (strategy.get_priority(node.parent), node.parent))
```

### What makes a node evictable?

A node is an "evictable leaf" (`_update_leaf_status`, line 781) when ALL of:
1. `value is not None` (not already evicted)
2. `lock_ref == 0` (no active request using it)
3. **All children are evicted** (or has no children) — i.e., it's a "logical leaf"

This ensures bottom-up eviction: you can only evict a node after all its descendants are gone.

### Eviction strategies

SGLang supports 6 eviction policies (defined in `evict_policy.py`):

| Policy | Priority Key | Evicts First |
|--------|-------------|--------------|
| **LRU** (default) | `last_access_time` | Least recently accessed |
| **LFU** | `(hit_count, last_access_time)` | Least frequently used |
| **FIFO** | `creation_time` | Oldest created |
| **MRU** | `-last_access_time` | Most recently accessed |
| **FILO** | `-creation_time` | Newest created |
| **Priority** | `(priority, last_access_time)` | Lowest priority, then LRU |

### Eviction cascade example

```
Before eviction:
Root
└── [1,2,3] (lock_ref=0)
    ├── [4,5] (lock_ref=0, leaf)    ← evicted first
    └── [6,7] (lock_ref=1, locked)

After evicting [4,5]:
Root
└── [1,2,3] (lock_ref=0)           ← NOT evictable yet (child [6,7] exists)
    └── [6,7] (lock_ref=1, locked)

Later, when [6,7] is unlocked and evicted:
Root
└── [1,2,3] (lock_ref=0, leaf)     ← NOW evictable
```

---

## 9. Lock References: Protecting In-Flight KV Data <a id="9-lock-references"></a>

Lock references are the mechanism that prevents the eviction system from pulling KV data out from under running requests.

### `inc_lock_ref(node)` — Protect

Called when a request is scheduled to run. Walks **from the matched node up to root**:

```python
def inc_lock_ref(self, node):
    while node != root:
        if node.lock_ref == 0:
            evictable_size -= len(node.key)    # No longer evictable
            protected_size += len(node.key)    # Now protected
        node.lock_ref += 1
        update_leaf_status(node)               # Remove from evictable_leaves
        node = node.parent
```

### `dec_lock_ref(node)` — Release

Called when a request finishes or its tree position changes. Reverse walk:

```python
def dec_lock_ref(self, node):
    while node != root:
        if node.lock_ref == 1:                 # About to reach 0
            evictable_size += len(node.key)    # Now evictable again
            protected_size -= len(node.key)
        node.lock_ref -= 1
        update_leaf_status(node)               # Maybe add to evictable_leaves
        node = node.parent
```

### Why lock the entire path to root?

If only the leaf node were locked, a parent node could be evicted, orphaning the child. Since eviction works bottom-up through leaves, locking the entire ancestor chain ensures structural integrity.

### Memory accounting

The cache tracks two sizes:
- **`evictable_size`**: Tokens that CAN be evicted (no request using them)
- **`protected_size`**: Tokens that CANNOT be evicted (active requests)

The scheduler uses these to determine if it can fit new requests:

```
total_available = allocator.available_size() + tree_cache.evictable_size()
```

---

## 10. Worked Example: Two Requests Sharing a Prefix <a id="10-worked-example"></a>

Let's trace through a concrete scenario with two requests sharing a system prompt.

### Setup

System prompt tokens: `[10, 20, 30, 40, 50]`
- Request A input: `[10, 20, 30, 40, 50, 60, 70]` (system + user A)
- Request B input: `[10, 20, 30, 40, 50, 80, 90]` (system + user B)

### Request A arrives (empty tree)

**1. match_prefix([10, 20, 30, 40, 50, 60, 70])**

Tree is empty → returns empty match (0 cached tokens).

```
Tree state:
  Root []
```

**2. Prefill Request A**

- Allocate KV pool indices for all 7 tokens: `[idx0..idx6]`
- Forward pass computes KV for all 7 tokens
- `req_to_token[A, 0..6] = [idx0, idx1, idx2, idx3, idx4, idx5, idx6]`

**3. Request A generates tokens `[100, 101]` then finishes**

**4. cache_finished_req(A)**

Inserts `[10, 20, 30, 40, 50, 60, 70, 100, 101]` with KV indices into tree:

```
Tree state:
  Root []
  └── [10,20,30,40,50,60,70,100,101] → [idx0..idx8]  (lock_ref=0)
```

### Request B arrives

**1. match_prefix([10, 20, 30, 40, 50, 80, 90])**

Walks the tree:
- Child key `10` found → match `[10,20,30,40,50,60,70,100,101]` against `[10,20,30,40,50,80,90]`
- `key_match` returns 5 (first 5 tokens match, diverge at position 5: `60 != 80`)
- **Partial match! Split the node at position 5:**

```
Tree state after split:
  Root []
  └── [10,20,30,40,50] → [idx0..idx4]  (new_node, lock_ref=0)
      └── [60,70,100,101] → [idx5..idx8]  (original, shortened)
```

- Returns: `device_indices=[idx0,idx1,idx2,idx3,idx4]`, `last_node=new_node`
- **Request B gets 5 tokens for FREE — no recomputation!**

**2. inc_lock_ref(new_node)**

Locks the `[10,20,30,40,50]` node (and ancestors up to root):

```
  Root []
  └── [10,20,30,40,50] → [idx0..idx4]  (lock_ref=1, PROTECTED)
      └── [60,70,100,101] → [idx5..idx8]  (lock_ref=0, evictable)
```

**3. Prefill Request B**

- Only need to allocate 2 new KV slots for tokens `[80, 90]`
- `req_to_token[B, 0..4] = [idx0..idx4]` (from cache!)
- `req_to_token[B, 5..6] = [idx7, idx8]` (newly allocated)
- Forward pass only computes KV for `[80, 90]` — 71% compute savings!

**4. Request B generates `[200]` and finishes**

**5. cache_finished_req(B)**

Inserts `[10,20,30,40,50,80,90,200]`:
- Walks tree: matches `[10,20,30,40,50]` (5 tokens, all in tree)
- Remaining `[80,90,200]` creates new leaf:

```
Tree state:
  Root []
  └── [10,20,30,40,50] → [idx0..idx4]  (lock_ref=0)
      ├── [60,70,100,101] → [idx5..idx8]  (lock_ref=0)
      └── [80,90,200] → [idx7,idx8,idx9]  (lock_ref=0)
```

Now a third request with the same system prompt would get 5 tokens cached immediately.

---

## 11. Advanced Topics <a id="11-advanced-topics"></a>

### 11.1 Page-Aligned Keys

When `page_size > 1`, keys are truncated to a multiple of `page_size` before any operation:

```python
page_aligned_len = len(key) // page_size * page_size
key = key[:page_aligned_len]
```

This means the last `len(key) % page_size` tokens are "orphaned" — they have KV computed but aren't tracked in the tree. The `cache_protected_len` field on requests ensures these orphaned indices are properly freed in subsequent operations.

### 11.2 EAGLE Speculative Decoding (Bigram Keys)

For EAGLE models, tokens are converted to **bigram keys** — pairs of consecutive tokens `(token[i], token[i+1])`. This captures two-token context in the tree key, improving cache hit rates for speculative decoding where draft tokens may partially match.

```python
def convert_to_bigram_key(token_ids):
    # [a, b, c, d] → [(a,b), (b,c), (c,d)]
    return list(zip(token_ids[:-1], token_ids[1:]))
```

### 11.3 HiRadixCache (Hierarchical Cache)

The `HiRadixCache` extends `RadixCache` with a **host (CPU) memory tier**. When GPU memory is under pressure, instead of evicting KV data entirely, it's moved to CPU memory first. On a cache hit for host-resident data, it's loaded back to GPU — slower than a GPU hit but much faster than recomputation.

### 11.4 SWA RadixCache (Sliding Window Attention)

For models with sliding window attention (like Mistral), `SWARadixCache` manages two separate KV pools — one for full attention layers and one for sliding window layers. The sliding window pool only keeps the last `window_size` tokens, so eviction is handled differently.

### 11.5 The `cache_protected_len` Subtlety

When a request's prefix is matched and locked, the `cache_protected_len` tracks how many tokens are "owned" by the tree (vs. the request's own allocation). During `cache_unfinished_req`:

```python
# Free duplicates between cache_protected_len and new_prefix_len
allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])
```

This prevents double-freeing: tokens in `[0, cache_protected_len)` are shared with the tree (via lock_ref), tokens in `[cache_protected_len, new_prefix_len)` are duplicates that the tree already owns, and tokens in `[new_prefix_len, end)` are new and belong to the request.

### 11.6 Evictable Leaves Set

Rather than scanning the entire tree on every eviction, `RadixCache` maintains a **set of evictable leaves** (`self.evictable_leaves`). This set is updated incrementally:

- **Added** when: a node becomes a leaf (no non-evicted children) AND has `lock_ref == 0`
- **Removed** when: a node gets locked, gains a child, or is evicted

This makes eviction O(k log n) where k = tokens to evict and n = evictable leaves, rather than O(tree_size).

---

## Summary

SGLang's RadixCache is an elegant solution to KV cache sharing in LLM serving:

| Operation | When | Complexity |
|-----------|------|------------|
| **match_prefix** | Scheduling + prefill prep | O(prefix_len) tree walk |
| **insert** | After prefill/decode, on request finish | O(seq_len) tree walk |
| **split_node** | During match/insert on partial overlap | O(1) pointer surgery |
| **inc/dec_lock_ref** | Request start/finish | O(tree_depth) walk to root |
| **evict** | Before allocation when memory low | O(k log n) heap operations |

The key insight is that by structuring KV cache metadata as a radix tree, prefix sharing becomes **automatic and transparent**. The scheduler doesn't need to explicitly track which requests share prompts — the tree structure handles it naturally. Combined with reference counting (lock_ref) and lazy eviction, this creates a system that maximizes GPU memory utilization while keeping shared prefixes safely cached.
