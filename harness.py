#!/usr/bin/env python3
"""
Benchmark harness for Nemotron-Cascade-2-30B-A3B on RTX 3070 8GB.
Adapted from clawinfra/qwen35-moe-offload harness.

Usage:
    import harness
    result = harness.run_bench({"n_gpu": 16, "n_ctx": 512, "batch_size": 128})
    print(f"tok/s: {result}")
"""

import subprocess
import re
import time
import os
from typing import Optional

# === Configuration ===
MODEL_PATH = "/tmp/nemotron-models/Nemotron-Cascade-2-30B-A3B-IQ2_XXS.gguf"
LLAMA_BENCH = "/tmp/llama-cpp-build/llama-bench"
RUST_BENCH = "./target/release/bench"
TIMEOUT_SECONDS = 600  # SSM models can be slower to initialise
RESULTS_FILE = "results.tsv"

# === Defaults ===
DEFAULT_CONFIG = {
    "n_gpu": 0,
    "n_ctx": 512,
    "batch_size": 128,
    "ubatch_size": 128,
    "n_threads": 8,
    "kv_type": "f16",       # f16, q8_0, q4_0
    "flash_attn": False,
    "op_offload": False,
    "n_gen": 128,
}


def _find_bench_binary() -> str:
    """Find the benchmark binary. Prefer Rust, fallback to llama-bench."""
    if os.path.isfile(RUST_BENCH) and os.access(RUST_BENCH, os.X_OK):
        return RUST_BENCH
    if os.path.isfile(LLAMA_BENCH) and os.access(LLAMA_BENCH, os.X_OK):
        return LLAMA_BENCH
    raise FileNotFoundError(
        f"No benchmark binary found. Checked:\n"
        f"  {RUST_BENCH}\n"
        f"  {LLAMA_BENCH}"
    )


def _build_command(config: dict) -> list[str]:
    """Build the llama-bench command from config dict."""
    cfg = {**DEFAULT_CONFIG, **config}
    bench = _find_bench_binary()

    cmd = [
        bench,
        "-m", MODEL_PATH,
        "-ngl", str(cfg["n_gpu"]),
        "-c", str(cfg["n_ctx"]),
        "-b", str(cfg["batch_size"]),
        "-ub", str(cfg["ubatch_size"]),
        "-t", str(cfg["n_threads"]),
        "-n", str(cfg["n_gen"]),
    ]

    # KV cache type
    kv = cfg["kv_type"]
    if kv != "f16":
        cmd.extend(["-ctk", kv, "-ctv", kv])

    # Flash attention
    if cfg["flash_attn"]:
        cmd.extend(["-fa", "1"])

    return cmd


def _parse_tok_s(output: str) -> Optional[float]:
    """
    Parse tokens/second from llama-bench output.
    
    llama-bench outputs a markdown table with columns including t/s (tokens per second).
    We look for the generation (tg) row's tok/s value.
    
    Example line:
    | ... | tg 128 | ... | 12.34 ± 0.56 |
    """
    # Pattern 1: llama-bench markdown table — tg row with tok/s
    # The last numeric column before the ± is tok/s
    for line in output.split('\n'):
        if 'tg' in line.lower() and '|' in line:
            # Extract all numbers from the line
            numbers = re.findall(r'(\d+\.\d+)\s*±', line)
            if numbers:
                return float(numbers[-1])  # Last match is typically tok/s
            # Fallback: look for standalone floats
            numbers = re.findall(r'(\d+\.\d+)', line)
            if len(numbers) >= 2:
                return float(numbers[-2])  # Second-to-last is often tok/s

    # Pattern 2: "X.XX tokens per second" or "X.XX tok/s"
    m = re.search(r'(\d+\.?\d*)\s*(?:tokens?\s*per\s*second|tok/s)', output, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Pattern 3: grep for any line with "tg" and a float
    for line in output.split('\n'):
        if 'tg' in line.lower():
            floats = re.findall(r'(\d+\.\d+)', line)
            if floats:
                return float(floats[-1])

    return None


def run_bench(config: dict, exp_id: str = "", notes: str = "") -> Optional[float]:
    """
    Run a single benchmark experiment.
    
    Args:
        config: Dict with keys matching DEFAULT_CONFIG
        exp_id: Experiment identifier for results.tsv
        notes: Free-text notes for results.tsv
    
    Returns:
        float: tokens/second for generation, or None on failure
    """
    cfg = {**DEFAULT_CONFIG, **config}
    cmd = _build_command(config)
    
    print(f"\n{'='*60}")
    print(f"[harness] Experiment: {exp_id or 'unnamed'}")
    print(f"[harness] Config: n_gpu={cfg['n_gpu']}, n_ctx={cfg['n_ctx']}, "
          f"batch={cfg['batch_size']}, ub={cfg['ubatch_size']}, "
          f"threads={cfg['n_threads']}, kv={cfg['kv_type']}, "
          f"fa={'on' if cfg['flash_attn'] else 'off'}")
    print(f"[harness] Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        elapsed = time.time() - start
        
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            print(f"[harness] FAILED (exit {result.returncode}, {elapsed:.1f}s)")
            print(f"[harness] stderr: {stderr[:500]}")
            _append_result(exp_id, cfg, None, f"FAIL: exit {result.returncode} — {notes}")
            return None
        
        tok_s = _parse_tok_s(stdout)
        
        if tok_s is None:
            print(f"[harness] WARNING: Could not parse tok/s from output")
            print(f"[harness] stdout (last 500 chars): {stdout[-500:]}")
            _append_result(exp_id, cfg, None, f"PARSE_FAIL — {notes}")
            return None
        
        print(f"[harness] SUCCESS: {tok_s:.3f} tok/s ({elapsed:.1f}s)")
        _append_result(exp_id, cfg, tok_s, notes)
        return tok_s
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"[harness] TIMEOUT after {elapsed:.1f}s")
        _append_result(exp_id, cfg, None, f"TIMEOUT ({TIMEOUT_SECONDS}s) — {notes}")
        return None
    except Exception as e:
        print(f"[harness] ERROR: {e}")
        _append_result(exp_id, cfg, None, f"ERROR: {e} — {notes}")
        return None


def _append_result(exp_id: str, cfg: dict, tok_s: Optional[float], notes: str):
    """Append a result row to results.tsv."""
    row = "\t".join([
        exp_id or f"exp_{int(time.time())}",
        "Nemotron-Cascade-2-30B-A3B",
        "IQ2_XXS",
        str(cfg["n_gpu"]),
        str(cfg["n_ctx"]),
        str(cfg["batch_size"]),
        str(cfg["ubatch_size"]),
        str(cfg["n_threads"]),
        cfg["kv_type"],
        "on" if cfg["flash_attn"] else "off",
        "on" if cfg.get("op_offload") else "off",
        f"{tok_s:.3f}" if tok_s is not None else "FAIL",
        notes,
    ])
    
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_FILE)
    with open(results_path, "a") as f:
        f.write(row + "\n")


def check_ready() -> bool:
    """Check if model file and benchmark binary exist."""
    issues = []
    if not os.path.isfile(MODEL_PATH):
        issues.append(f"Model not found: {MODEL_PATH}")
    try:
        _find_bench_binary()
    except FileNotFoundError as e:
        issues.append(str(e))
    
    if issues:
        print("[harness] NOT READY:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("[harness] Ready.")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Bench: {_find_bench_binary()}")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_ready()
    else:
        # Quick test with CPU-only
        result = run_bench({"n_gpu": 0, "n_threads": 8}, exp_id="test_cpu_only", notes="initial test")
        print(f"\nResult: {result}")
