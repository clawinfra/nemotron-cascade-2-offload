#!/usr/bin/env python3
"""
Agent-editable benchmark script for Nemotron-Cascade-2-30B-A3B.

This file is the main entry point for the autonomous benchmark loop.
The agent modifies this file each iteration to explore new parameter axes.

Current phase: Initial grid sweep — find n_gpu_layers VRAM cliff and optimal thread count.
"""

import sys
import os
import time

# Ensure we can import harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness


def phase_1_gpu_layer_sweep():
    """
    Phase 1: Sweep n_gpu_layers to find VRAM cliff.
    
    NemotronH has SSM+Attention hybrid layers — the VRAM scaling law
    will differ from pure transformer (Qwen). Need to find:
    1. Max n_gpu before OOM
    2. tok/s curve shape (may not be monotonic due to SSM/attention mix)
    """
    print("\n" + "=" * 70)
    print("PHASE 1: n_gpu_layers sweep (finding VRAM cliff)")
    print("=" * 70)
    
    n_gpu_values = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 40]
    best_tok_s = 0.0
    best_n_gpu = 0
    
    for n_gpu in n_gpu_values:
        exp_id = f"p1_ngl{n_gpu}"
        result = harness.run_bench(
            {"n_gpu": n_gpu, "n_threads": 8, "n_ctx": 512, "batch_size": 128},
            exp_id=exp_id,
            notes="phase1: gpu layer sweep"
        )
        
        if result is None:
            print(f"[bench] n_gpu={n_gpu} failed — likely OOM. Cliff found at n_gpu={n_gpu}")
            break
        
        if result > best_tok_s:
            best_tok_s = result
            best_n_gpu = n_gpu
        
        print(f"[bench] n_gpu={n_gpu}: {result:.3f} tok/s (best so far: {best_tok_s:.3f} at n_gpu={best_n_gpu})")
    
    print(f"\n[bench] Phase 1 result: best n_gpu={best_n_gpu} at {best_tok_s:.3f} tok/s")
    return best_n_gpu, best_tok_s


def phase_2_thread_sweep(best_n_gpu: int):
    """
    Phase 2: Sweep thread count at optimal n_gpu.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 2: Thread sweep at n_gpu={best_n_gpu}")
    print("=" * 70)
    
    thread_values = [4, 6, 8, 10, 12, 16]
    best_tok_s = 0.0
    best_threads = 8
    
    for threads in thread_values:
        exp_id = f"p2_ngl{best_n_gpu}_t{threads}"
        result = harness.run_bench(
            {"n_gpu": best_n_gpu, "n_threads": threads, "n_ctx": 512, "batch_size": 128},
            exp_id=exp_id,
            notes="phase2: thread sweep"
        )
        
        if result is not None and result > best_tok_s:
            best_tok_s = result
            best_threads = threads
    
    print(f"\n[bench] Phase 2 result: best threads={best_threads} at {best_tok_s:.3f} tok/s")
    return best_threads, best_tok_s


def phase_3_batch_sweep(best_n_gpu: int, best_threads: int):
    """
    Phase 3: Sweep batch sizes.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 3: Batch sweep at n_gpu={best_n_gpu}, threads={best_threads}")
    print("=" * 70)
    
    batch_values = [64, 128, 256, 512]
    best_tok_s = 0.0
    best_batch = 128
    
    for batch in batch_values:
        exp_id = f"p3_ngl{best_n_gpu}_t{best_threads}_b{batch}"
        result = harness.run_bench(
            {
                "n_gpu": best_n_gpu,
                "n_threads": best_threads,
                "n_ctx": 512,
                "batch_size": batch,
                "ubatch_size": min(batch, 256),
            },
            exp_id=exp_id,
            notes="phase3: batch sweep"
        )
        
        if result is not None and result > best_tok_s:
            best_tok_s = result
            best_batch = batch
    
    print(f"\n[bench] Phase 3 result: best batch={best_batch} at {best_tok_s:.3f} tok/s")
    return best_batch, best_tok_s


def phase_4_kv_quant_sweep(best_n_gpu: int, best_threads: int, best_batch: int):
    """
    Phase 4: KV cache quantisation — TurboQuant exploration.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 4: KV quant sweep (TurboQuant)")
    print("=" * 70)
    
    kv_types = ["f16", "q8_0", "q4_0"]
    best_tok_s = 0.0
    best_kv = "f16"
    
    for kv in kv_types:
        # Test with and without flash attention
        for fa in [False, True]:
            exp_id = f"p4_ngl{best_n_gpu}_kv{kv}_fa{'on' if fa else 'off'}"
            result = harness.run_bench(
                {
                    "n_gpu": best_n_gpu,
                    "n_threads": best_threads,
                    "n_ctx": 512,
                    "batch_size": best_batch,
                    "kv_type": kv,
                    "flash_attn": fa,
                },
                exp_id=exp_id,
                notes=f"phase4: kv={kv} fa={'on' if fa else 'off'}"
            )
            
            if result is not None and result > best_tok_s:
                best_tok_s = result
                best_kv = kv
    
    print(f"\n[bench] Phase 4 result: best kv={best_kv} at {best_tok_s:.3f} tok/s")
    return best_kv, best_tok_s


def phase_5_context_length_sweep(best_n_gpu: int, best_threads: int, best_batch: int, best_kv: str):
    """
    Phase 5: Context length sweep — SSM state is fixed-size, so this may
    behave very differently from pure transformer.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 5: Context length sweep (SSM vs Transformer scaling)")
    print("=" * 70)
    
    ctx_values = [256, 512, 1024, 2048, 4096]
    
    for ctx in ctx_values:
        exp_id = f"p5_ngl{best_n_gpu}_ctx{ctx}"
        result = harness.run_bench(
            {
                "n_gpu": best_n_gpu,
                "n_threads": best_threads,
                "n_ctx": ctx,
                "batch_size": best_batch,
                "kv_type": best_kv,
            },
            exp_id=exp_id,
            notes=f"phase5: ctx={ctx} — SSM scaling test"
        )
        
        if result is not None:
            print(f"[bench] ctx={ctx}: {result:.3f} tok/s")


def main():
    """Main benchmark loop."""
    print("=" * 70)
    print("Nemotron-Cascade-2-30B-A3B Autoresearch Benchmark")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Pre-flight check
    if not harness.check_ready():
        print("\n[bench] Model or binary not ready. Waiting...")
        print("[bench] Check: ls -la /tmp/nemotron-models/")
        print("[bench] Check: ls -la /tmp/llama-cpp-build/llama-bench")
        sys.exit(1)
    
    # Phase 1: GPU layer sweep
    best_n_gpu, _ = phase_1_gpu_layer_sweep()
    
    # Phase 2: Thread sweep
    best_threads, _ = phase_2_thread_sweep(best_n_gpu)
    
    # Phase 3: Batch sweep
    best_batch, _ = phase_3_batch_sweep(best_n_gpu, best_threads)
    
    # Phase 4: KV quant sweep (TurboQuant)
    best_kv, _ = phase_4_kv_quant_sweep(best_n_gpu, best_threads, best_batch)
    
    # Phase 5: Context length sweep
    phase_5_context_length_sweep(best_n_gpu, best_threads, best_batch, best_kv)
    
    print("\n" + "=" * 70)
    print("Initial grid complete. Agent should now:")
    print("1. Analyse results.tsv for patterns")
    print("2. Fine-grain sweep around the best config")
    print("3. Integrate PolarQuant and QJL")
    print("4. Explore expert offloading strategies")
    print("5. NEVER STOP — find new axes to explore")
    print("=" * 70)


if __name__ == "__main__":
    main()
