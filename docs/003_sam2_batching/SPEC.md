# SAM2 Batched Inference Problem

## Current Situation

We're using SAM2 for "SST-style" inference: given a few labeled reference frames with masks, predict masks for thousands of unlabeled target frames. All frames within a group share the same reference masks.

### Current Pipeline

```
1. Pre-compute vision features (batched, efficient)
   ┌─────────────────────────────────────────────────────┐
   │  Images ──► ViT ──► Vision Features                 │
   │  [B, 3, H, W]      [HW, B, C] (3 scale levels)      │
   │                                                     │
   │  Using DataLoader with batch_size=32, num_workers=8 │
   │  ~30 fps throughput                                 │
   └─────────────────────────────────────────────────────┘

2. Build memory bank from reference frames (once per group)
   ┌─────────────────────────────────────────────────────┐
   │  For each ref frame:                                │
   │    Vision Features + Mask ──► Memory Encoder        │
   │                                                     │
   │  Result: Shared memory bank for the group           │
   │  - memory features: [N_tokens, 1, mem_dim]          │
   │  - memory pos enc:  [N_tokens, 1, mem_dim]          │
   └─────────────────────────────────────────────────────┘

3. Predict masks for each target (SEQUENTIAL - the bottleneck)
   ┌─────────────────────────────────────────────────────┐
   │  For each target frame:                             │
   │    Vision Features ──► Memory Attention ──► Decoder │
   │                        (uses memory bank)           │
   │                                                     │
   │  Currently: ~5 fps, one frame at a time             │
   │  GPU utilization: ~50%                              │
   │  Memory utilization: ~50%                           │
   └─────────────────────────────────────────────────────┘
```

### The Problem

Step 3 processes targets **one at a time**, leaving GPU underutilized. We want to process B targets simultaneously:

```
Current (sequential):
  Target 1 ──► Memory Attention ──► Mask
  Target 2 ──► Memory Attention ──► Mask
  Target 3 ──► Memory Attention ──► Mask
  ...

Desired (batched):
  ┌─ Target 1 ─┐                    ┌─ Mask 1 ─┐
  │  Target 2  │ ──► Mem Attention ──► │  Mask 2  │
  └─ Target 3 ─┘     (shared bank)  └─ Mask 3 ─┘
```

## Why Batching is Difficult

### SAM2's Session Abstraction

The HuggingFace SAM2 API uses a "video session" abstraction designed for sequential video processing:

```python
session = processor.init_video_session(video=images, ...)
for frame_idx in range(len(images)):
    output = model(inference_session=session, frame_idx=frame_idx)
```

This assumes frames are processed in order, with each frame potentially influencing the memory for subsequent frames. Our use case is different: targets are **independent** and all share the same fixed memory bank from reference frames.

### Memory Attention Internals

When we tried to bypass the session and call memory attention directly:

```python
# Batch B target vision features
target_feats = torch.cat([cache[pk]["vision_feats"][-1] for pk in batch_pks], dim=1)
# Shape: [HW, B, C]

# Broadcast memory bank to batch
memory_bank_batched = memory_bank.expand(-1, B, -1)
# Shape: [N_tokens, B, mem_dim]

# Call memory attention directly
output = model.memory_attention(
    curr=target_feats,
    memory=memory_bank_batched,
    ...
)
```

This failed with rotary position embedding errors:
```
RuntimeError: The size of tensor a (100) must match the size of tensor b (0)
```

The memory attention layer has internal assumptions about:
- Rotary position embeddings that expect specific memory structure
- The relationship between "current" tokens and "memory" tokens
- How object pointers are handled across the batch

### Mask Decoder Batching

Even if memory attention worked, the mask decoder has its own complexity:
- Multi-scale features from the vision encoder
- IoU prediction heads
- Object pointer updates

## Potential Solutions

### Option 1: Multiple Independent Sessions (Simple, Memory-Heavy)

Create B sessions, each with the same reference frames but different target:

```python
sessions = []
for target in batch_targets:
    session = create_session(ref_frames + [target])
    sessions.append(session)

# Run in parallel (requires careful GPU memory management)
with ThreadPoolExecutor() as executor:
    results = executor.map(run_session, sessions)
```

**Pros**: Uses existing API, guaranteed correctness
**Cons**: B× memory overhead, session creation overhead

### Option 2: Extended Session with Multiple Target Slots

Create one session with B target slots:

```python
session = processor.init_video_session(
    video=ref_images + [dummy] * B,  # B placeholder slots
    ...
)

# Inject B targets' features
for i, target in enumerate(batch_targets):
    inject_cached_features(session, n_refs + i, features_cache[target.pk])

# Run forward for each slot (still sequential internally)
for i in range(B):
    output = model(inference_session=session, frame_idx=n_refs + i)
```

**Pros**: Reuses session infrastructure, shares memory bank
**Cons**: Forward passes still sequential, limited speedup

### Option 3: Direct Component Batching (Complex, Highest Potential)

Bypass the session abstraction entirely and batch through individual components:

```python
# 1. Batch vision features for B targets
target_feats = batch_vision_features(targets, features_cache)  # [HW, B, C]

# 2. Get memory bank (already built from refs)
memory = memory_bank  # [N_tokens, 1, mem_dim]

# 3. Carefully construct inputs for memory attention
#    - Handle rotary embeddings correctly
#    - Broadcast memory to batch dimension
#    - Set up object pointers

# 4. Run batched memory attention
attended = model.memory_attention(...)  # Need to figure out correct args

# 5. Run batched mask decoder
masks = model.mask_decoder(...)  # [B, N_obj, H, W]
```

**Pros**: Maximum throughput, full GPU utilization
**Cons**: Requires deep understanding of SAM2 internals, fragile to API changes

### Option 4: Custom CUDA Kernel (Nuclear Option)

Write a custom fused kernel that combines memory attention + mask decoding for our specific use case.

**Pros**: Maximum performance
**Cons**: Significant engineering effort, maintenance burden

## Recommended Approach

Try **Option 3** with careful study of SAM2's memory attention:

1. Read `transformers/models/sam2/modeling_sam2.py` to understand:
   - How `memory_attention` handles the `curr` and `memory` inputs
   - What the rotary embeddings expect
   - How object pointers are propagated

2. Start with batch size B=2 to debug shape mismatches

3. Key tensors to understand:
   - `vision_feats`: [HW, B, C] - current frame features
   - `vision_pos_embeds`: [HW, B, C] - position embeddings
   - `memory`: [N_mem_tokens, 1, mem_dim] - encoded memories
   - `memory_pos_enc`: [N_mem_tokens, 1, mem_dim] - memory positions

4. The memory attention call signature (from SAM2 code):
   ```python
   def forward(
       self,
       curr: torch.Tensor,  # Current frame tokens
       memory: torch.Tensor,  # Memory bank tokens
       curr_pos: torch.Tensor,  # Current position embeddings
       memory_pos: torch.Tensor,  # Memory position embeddings
       num_obj_ptr_tokens: int = 0,  # Object pointer count
   )
   ```

## Current Performance

| Stage | Throughput | GPU Util | Notes |
|-------|------------|----------|-------|
| Vision features | ~30 fps | High | Already batched |
| Memory encoding | N/A | N/A | Done once per group |
| Mask prediction | ~5 fps | ~50% | **Bottleneck** |

With successful batching (B=8), we could potentially reach 20-40 fps for mask prediction.

## Actual Results (Session-Based Approach)

Full inference on 29,066 frames completed:

| Metric | Value |
|--------|-------|
| Total time | 6,697 seconds (~112 min / 1.87 hours) |
| Average speed | 4.34 frames/sec |
| GPU utilization | ~50% |

**Comparison to naive approach:**
- Original (rebuild session per target): ~0.19 fps → 42+ hours
- Session reuse approach: 4.34 fps → 1.87 hours
- **Speedup: ~23x**

Still leaving performance on the table due to sequential mask prediction.

---

## Gemini's Review & Recommendations

### Evaluation of Proposed Solutions

1. **Option 1 (Multiple Sessions)**: Reliable but inefficient. Duplicating the memory bank B times will explode VRAM usage (`B * Memory_Size`), severely limiting batch size.

2. **Option 2 (Extended Session)**: Does not solve the core issue. The internal loop in `Sam2VideoModel` will still process the "dummy" target slots sequentially.

3. **Option 3 (Direct Component Batching)**: **This is the correct approach.** By manually managing the memory bank and calling the attention layers, you can treat multiple target frames as a batch.
   - **Challenge:** SAM2 usually batches over *objects* (N objects per frame). You want to batch over *frames* (B frames, each with N objects).
   - **Strategy:** Flatten the batch dimensions: `Effective_Batch = B_frames * N_objects`.

4. **Option 4 (Custom Kernel)**: Unnecessary. Standard PyTorch operations (with Flash Attention) are sufficient if the data is shaped correctly.

### Key Technical Insights

#### 1. Batching Strategy (Dimension Flattening)

SAM2's internal components expect a batch dimension `B`:
- **Current:** `B = N_objects` (for 1 frame)
- **Target:** `B = N_targets * N_objects`

Implementation:
1. Pre-compute image features for `N_targets` frames: `[N_targets, C, H, W]`
2. Expand/repeat for `N_objects` if necessary
3. Flatten to `[N_targets * N_objects, ...]` for attention

#### 2. Memory Bank Broadcasting (Critical Optimization)

The memory bank (Keys/Values) `[N_memories, N_objects, C]` is shared across all `N_targets`:

**Naive approach (bad):**
```python
# Repeat Memory Bank N_targets times - high VRAM usage
memory_batched = memory_bank.repeat(N_targets, 1, 1)
```

**Optimized approach (good):**
```python
# Reshape to enable broadcasting
# Query (Current Frames): [N_objects, N_targets, L, C]
# Key/Value (Memory):      [N_objects, 1, S, C]
# F.scaled_dot_product_attention broadcasts on N_targets dimension
```

Standard `F.scaled_dot_product_attention` supports broadcasting if batch dimensions align. Reshape to `[N_objects, N_targets, L, C]` vs `[N_objects, 1, S, C]` to exploit broadcasting.

#### 3. KV-Cache Analogy

Treat the Memory Bank as a **KV Cache** in a Large Language Model:
- Fixed context = reference frames (the "prompt")
- Multiple parallel decoders = target frames (multiple "users")

The mechanisms used for batched LLM inference (paging, broadcasting) apply here.

#### 4. Positional Embeddings (RoPE) Caution

Be careful with Rotary Positional Embeddings:
- SAM2's `RoPEAttention` assumes specific shapes
- Since all targets are independent, they can all be treated as "frame T+1" relative to the memory
- Their absolute positions might not matter if they don't attend to each other

### Suggested Implementation Plan

1. **Create `predict_batch_direct` function:**
   - Input: batch of target frame pks, pre-built memory bank
   - Output: masks for all targets

2. **Implement batched memory attention:**
   ```python
   def batched_memory_attention(
       target_feats,      # [HW, B, C] - B target frames
       memory_bank,       # [N_mem, 1, C] - shared memory
       memory_pos,        # [N_mem, 1, C] - shared positions
   ):
       # Reshape for broadcasting
       # Query: [B, HW, C]
       # Key/Value: [1, N_mem, C] - broadcasts to [B, N_mem, C]

       # Use F.scaled_dot_product_attention with broadcasting
       output = F.scaled_dot_product_attention(
           query=target_feats,
           key=memory_bank.expand(B, -1, -1),  # or rely on broadcasting
           value=memory_bank.expand(B, -1, -1),
       )
       return output
   ```

3. **Handle multi-object case:**
   - Flatten: `[B_frames * N_objects, HW, C]`
   - Process attention
   - Unflatten: `[B_frames, N_objects, HW, C]`

4. **Run batched mask decoder:**
   - Feed attended features to mask decoder
   - Output: `[B, N_objects, H, W]`

### Expected Performance Gains

With proper batching (B=8-16):
- Current: 4.34 fps (sequential)
- Expected: 15-30 fps (batched)
- Potential speedup: 3-7x additional improvement
- Total time: ~15-30 minutes instead of 112 minutes
