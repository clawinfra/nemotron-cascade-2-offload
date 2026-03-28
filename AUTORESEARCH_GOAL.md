# AUTORESEARCH_GOAL.md — Nemotron-Cascade-2-30B-A3B on RTX 3070 8GB

## Model
- **Name**: Nemotron-Cascade-2-30B-A3B
- **Architecture**: NemotronH — SSM (Mamba2-style) + Attention hybrid
- **Parameters**: 30B total / 3B active (Mixture-of-Experts)
- **Quantisation**: IQ2_XXS GGUF
- **File size**: ~18.1 GB
- **llama.cpp support**: Native via `llm_build_nemotron_h`

## Hardware
- **GPU**: NVIDIA RTX 3070 (8 GB VRAM)
- **RAM**: 16 GB DDR4
- **CPU**: AMD Ryzen (XPS 8940)
- **Storage**: NVMe SSD

## Challenge
IQ2_XXS = **18.1 GB** — more than double Qwen3.5-35B-A3B's 9.6 GB for the same MoE architecture class. This is because:
1. SSM state transition matrices (A, B, C, D) resist aggressive quantisation — recurrent dynamics amplify quantisation error across time steps
2. Mamba2-style selective scan requires higher-precision intermediates than standard attention
3. The expert routing + SSM combination means both expert weights AND SSM states compete for VRAM

Only **8 GB VRAM** available. Must offload aggressively while keeping generation speed competitive.

## Baseline Comparison
- **Qwen3.5-35B-A3B** (pure MoE transformer): **29.899 tok/s** at IQ2_XXS, n_gpu=27, RTX 3070
- NemotronH's SSM layers change the VRAM scaling law — the sweet spot for n_gpu_layers will be different
- Target: match or exceed Qwen baseline despite 2x larger model file

## Compression Stack (ALL FOUR must be explored)

### 1. Apple Flash-Memory MoE Expert Offloading
- Page inactive expert layers between CPU RAM and GPU VRAM on demand
- Only active experts (3B worth) need to be GPU-resident at inference time
- Key parameter: `n_gpu_layers` controls the CPU↔GPU split
- SSM layers vs attention layers may have different offloading priorities

### 2. Google TurboQuant KV Compression
- Quantise KV cache to q4_0, q8_0, or keep fp16
- Reduces VRAM footprint of attention layers' KV cache
- For SSM layers: applies to the state buffers (SSM's equivalent of KV cache)
- Key parameter: `--cache-type-k` and `--cache-type-v` in llama-bench

### 3. PolarQuant — Polar Coordinate KV Quantisation
- Decomposes KV cache tensors into magnitude (r) and angle (θ) components
- Quantises each component independently — angles tolerate lower precision
- Already implemented in `crates/polarquant/` in the qwen35-moe-offload repo
- **Port to this repo** and integrate with the benchmark harness
- Particularly relevant for SSM state matrices: polar decomposition of A matrices preserves eigenvalue structure

### 4. QJL — Johnson-Lindenstrauss Projection on KV
- Projects high-dimensional KV vectors into lower-dimensional space using random JL projections
- Provably preserves pairwise distances (within ε) with high probability
- Already implemented in `crates/qjl/` in the qwen35-moe-offload repo
- **Port to this repo** and integrate with the benchmark harness
- For SSM: reduces dimensionality of hidden state vectors, compressing the recurrent state

## Primary Tools
- **Benchmark binary**: `/tmp/llama-cpp-build/llama-bench` (llama.cpp CUDA build)
- **Alternative**: `./target/release/bench` (Rust binary, if built)
- **Python**: fallback only — use `uv run python` always

## Model Path
```
/tmp/nemotron-models/Nemotron-Cascade-2-30B-A3B-IQ2_XXS.gguf
```

## Metric
- **Primary**: tok/s (tokens per second) for text generation
- **Secondary**: VRAM usage (MB), prompt processing speed (pp tok/s)

## Results Format
File: `results.tsv` — TAB-separated columns:
```
exp_id	model	quant	n_gpu	n_ctx	batch_size	ubatch_size	n_threads	kv_type	flash_attn	op_offload	tok_s	notes
```

## Experimental Design Rules

### Self-Direction
The agent owns experimental design. No pre-specified matrices. **Follow the data.**

Axes to explore (non-exhaustive):
- `n_gpu_layers`: 0 to 99 (find the VRAM cliff for NemotronH)
- `batch_size`: 64, 128, 256, 512
- `ubatch_size`: 64, 128, 256
- `n_threads`: 4, 8, 12, 16
- `kv_type`: q4_0, q8_0, fp16
- `flash_attn`: on/off
- `n_ctx`: 512, 1024, 2048, 4096
- Expert offloading strategies
- PolarQuant integration
- QJL integration

### SSM-Specific Considerations
- SSM layers have different VRAM scaling than attention layers
- The n_gpu_layers sweet spot will NOT be the same as Qwen
- SSM state accumulates over sequence length — context length affects VRAM differently
- Flash attention only applies to the attention layers, not SSM layers
- Monitor for OOM at different context lengths — SSM state may grow unexpectedly

### Loop Rule
**NEVER STOP.** Run experiments continuously. When one axis is exhausted, find a new axis. The loop ends only when hardware is physically shut down.

### Rust Binary Rule
Primary benchmark is `./target/release/bench` OR `/tmp/llama-cpp-build/llama-bench`. Python is only a fallback orchestrator.

## First Experiment (after model download completes)
```bash
/tmp/llama-cpp-build/llama-bench \
  -m /tmp/nemotron-models/Nemotron-Cascade-2-30B-A3B-IQ2_XXS.gguf \
  -ngl 0 -t 8 -n 128 -p 512
```
Then sweep n_gpu_layers upward until OOM to find the VRAM cliff.
