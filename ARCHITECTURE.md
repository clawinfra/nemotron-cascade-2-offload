# ARCHITECTURE.md — NemotronH (Nemotron-Cascade-2-30B-A3B)

## Overview

Nemotron-Cascade-2-30B-A3B uses the **NemotronH** architecture — a hybrid that interleaves:
- **SSM layers** (Mamba2-style selective state space models)
- **Attention layers** (standard multi-head attention with KV cache)

Within a Mixture-of-Experts (MoE) framework:
- **30B total parameters** across all experts
- **3B active parameters** per token (sparse routing)

This is fundamentally different from Qwen3.5-35B-A3B (pure transformer MoE) and creates unique challenges for our compression stack.

## Layer Structure

```
┌─────────────────────────────────────────┐
│         NemotronH Block (repeated)       │
│                                          │
│  ┌──────────┐    ┌────────────────────┐  │
│  │ SSM Layer │    │ Attention Layer    │  │
│  │ (Mamba2)  │    │ (Multi-Head Attn)  │  │
│  │           │    │                    │  │
│  │ A,B,C,D   │    │ Q, K, V, O        │  │
│  │ matrices   │    │ + KV Cache         │  │
│  │ + state    │    │                    │  │
│  └──────────┘    └────────────────────┘  │
│         │                  │              │
│         └──────┬───────────┘              │
│                ▼                          │
│  ┌──────────────────────────┐            │
│  │    MoE Feed-Forward      │            │
│  │  (sparse expert routing)  │            │
│  │  N experts, top-k active  │            │
│  └──────────────────────────┘            │
└─────────────────────────────────────────┘
```

## SSM (Mamba2) Internals

### State Space Model Equations
```
h(t) = A · h(t-1) + B · x(t)    # state update (recurrent)
y(t) = C · h(t) + D · x(t)       # output projection
```

### Key Dimensions
- **State size** (`d_state`): Typically 16-128 per head
- **Hidden size** (`d_model`): Model's hidden dimension
- **Number of SSM heads**: Parallel state machines
- **Selective scan**: Input-dependent A, B, C matrices (not static)

### SSM State vs Attention KV Cache
| Property | Attention KV Cache | SSM State |
|---|---|---|
| Growth with seq len | Linear (stores all K,V) | Constant (fixed-size state) |
| VRAM per layer | O(seq_len × d_head × n_heads) | O(d_state × n_heads) |
| Quantisation tolerance | Moderate (redundancy in K,V) | Low (recurrent error accumulates) |
| Information density | Sparse (not all K,V equally important) | Dense (compressed history) |

## Why SSM States Resist Quantisation

### 1. Recurrent Error Amplification
In attention, quantisation error in one K/V vector affects only that token's attention score. In SSM, quantisation error in the state matrix **compounds across every subsequent time step**:
```
h(t) = A · h(t-1) + B · x(t)
     = A · (A · h(t-2) + B · x(t-1)) + B · x(t)
     = A² · h(t-2) + A · B · x(t-1) + B · x(t)
```
Error in A at step t propagates through A^n for all future steps.

### 2. Eigenvalue Sensitivity
The A matrix's eigenvalues control memory decay. Small quantisation errors can:
- Push stable eigenvalues (|λ| < 1) to unstable (|λ| > 1) → divergence
- Shift eigenvalue phases → frequency response distortion
- Destroy carefully learned decay rates

### 3. Selective Scan Precision
Mamba2's input-dependent parameterisation means A, B, C matrices are **computed per-token**, requiring the projection weights to maintain high precision to preserve the input-conditional dynamics.

## Compression Stack Applied to NemotronH

### 1. Expert Offloading (Apple Flash-Memory Style)
**Priority ordering for GPU residency:**
1. **SSM layers first** — SSM state updates are sequential and latency-sensitive; CPU↔GPU transfer per token would be catastrophic
2. **Attention layers second** — can tolerate slightly higher latency since they process in parallel
3. **Inactive MoE experts** — offload to CPU RAM, page in on demand

**Strategy:** Keep ALL SSM layers on GPU (they're smaller per-layer), offload attention KV cache and inactive experts to CPU.

### 2. TurboQuant KV Compression (Google)
**For attention layers:** Standard KV cache quantisation (q4_0, q8_0, fp16)
**For SSM layers:** Quantise the state buffer — but with higher precision floors:
- Attention KV → q4_0 is usually fine
- SSM state → minimum q8_0, ideally fp16 (due to recurrent error amplification)

The "KV cache" flags in llama-bench (`--cache-type-k`, `--cache-type-v`) affect attention layers. SSM state precision is controlled separately in the model implementation.

### 3. PolarQuant — Polar Coordinate Decomposition
**For attention KV:** Decompose K, V tensors into (magnitude, angle) pairs, quantise independently.
**For SSM state matrices (A):**
- Decompose A = R · Θ (polar form) where R = magnitudes, Θ = phases
- Magnitudes (eigenvalue magnitudes) → quantise to q8_0 (controls memory decay rate)
- Phases (eigenvalue angles) → quantise to q4_0 (controls oscillation frequency, more tolerant)
- **Key insight:** Polar decomposition preserves the eigenvalue structure that matters for stability

**Implementation:** Port from `crates/polarquant/` in qwen35-moe-offload repo.

### 4. QJL — Johnson-Lindenstrauss Projection
**For attention KV:** Project K, V vectors from d_head dimensions to k << d_head dimensions using random projection matrices. JL lemma guarantees distance preservation within (1±ε) with k = O(log(n)/ε²).
**For SSM hidden states:**
- Project h(t) from d_state to k dimensions
- Reduces memory footprint of the recurrent state
- **Key insight:** SSM states are dense but often low-rank in practice — JL projection exploits this

**Implementation:** Port from `crates/qjl/` in qwen35-moe-offload repo.

## VRAM Budget Analysis (RTX 3070, 8 GB)

```
Total VRAM:              8,192 MB
CUDA overhead:            ~500 MB
Available for model:    ~7,692 MB

Model file (IQ2_XXS):  18,100 MB  ← doesn't fit!

Must offload at least:  10,408 MB to CPU RAM

Strategy:
- Active expert weights (3B):    ~2,000 MB (GPU)
- SSM state buffers:               ~500 MB (GPU, high priority)
- Attention KV cache (quantised):  ~500 MB (GPU)
- Routing/embedding:               ~200 MB (GPU)
- CUDA workspace:                  ~500 MB
                          Total: ~3,700 MB on GPU
                          Headroom: ~4,000 MB for more layers
```

## Key Architectural Insight: NemotronH vs Qwen3.5

**Qwen3.5-35B-A3B** is a pure transformer MoE:
- VRAM scales linearly with n_gpu_layers + linearly with context length (KV cache)
- Sweet spot: n_gpu=27 was optimal on RTX 3070
- All layers are attention → uniform compression behaviour

**Nemotron-Cascade-2-30B-A3B** is a hybrid SSM+Attention MoE:
- VRAM has TWO components: layer weights + SSM state (fixed) + attention KV (context-dependent)
- SSM layers are smaller per-layer but need higher precision
- The n_gpu_layers sweet spot will depend on the SSM/attention layer ratio
- Context length affects VRAM less than pure transformer (SSM state is fixed-size)
- **Prediction:** Optimal n_gpu may be HIGHER than Qwen despite larger model, because SSM layers are smaller and context-dependent VRAM is lower
