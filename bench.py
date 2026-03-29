#!/usr/bin/env python3
"""
bench.py — Self-directing autoresearch for Nemotron-Cascade-2-30B-A3B.

AGENT RULES (read this):
- Detect hardware reality immediately — don't assume GPU works
- If all-CPU: don't timeout, run fast CPU-optimised experiments
- If CUDA available: push GPU layers hard
- No fixed phases — follow the data, keep experimenting
- NEVER STOP — keep looping until hardware shuts down
"""
import os, subprocess, re, time, sys, csv, json, random
from pathlib import Path

# === Paths ===
MODEL = "/tmp/nemotron-models/Nemotron-Cascade-2-30B-A3B-IQ2_XXS.gguf"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
RESULTS_HEADER = ["exp_id","tok_s","pp_tok_s","vram_mb","status","wall_time_s",
                  "n_gpu","n_threads","batch","ubatch","n_ctx","kv_type","flash_attn","notes"]

# === Candidate bench binaries (try in order) ===
BENCH_CANDIDATES = [
    "/tmp/llama-cpp-src/build/bin/llama-bench",
    "/tmp/llama-cpp-build/llama-bench",
]

# === CUDA plugin path ===
CUDA13 = "/usr/local/lib/ollama/mlx_cuda_v13"
CUDA_PLUGIN = "/tmp/llama-cpp-build/libggml-cuda.so"
LD_PATH = f"{CUDA13}:/tmp/llama-cpp-build:{os.environ.get('LD_LIBRARY_PATH','')}"

CUDA_ENV = os.environ.copy()
CUDA_ENV["LD_LIBRARY_PATH"] = LD_PATH
if os.path.exists(CUDA_PLUGIN):
    CUDA_ENV["GGML_BACKEND_PATH"] = CUDA_PLUGIN


def find_bench():
    for p in BENCH_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No llama-bench found. Tried: {BENCH_CANDIDATES}")


def probe_hardware():
    """Quick hardware probe: returns {'cuda': bool, 'free_vram_mb': int, 'ram_gb': float}"""
    info = {'cuda': False, 'free_vram_mb': 0, 'ram_gb': 0}
    try:
        r = subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            info['cuda'] = True
            info['free_vram_mb'] = int(r.stdout.strip().split('\n')[0])
    except Exception:
        pass
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    info['ram_gb'] = int(line.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass
    return info


def ensure_header():
    if not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0:
        with open(RESULTS_FILE, 'w') as f:
            f.write('\t'.join(RESULTS_HEADER) + '\n')


def load_results():
    if not RESULTS_FILE.exists():
        return []
    rows = []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                row['_tg'] = float(row.get('tok_s', 0) or 0)
            except ValueError:
                row['_tg'] = 0.0
            rows.append(row)
    return rows


def already_ran(exp_id):
    for r in load_results():
        if r.get('exp_id') == exp_id:
            return True
    return False


def record(exp_id, tok_s, pp_tok_s, vram_mb, status, wall_time,
           n_gpu, n_threads, batch, ubatch, n_ctx, kv_type, flash_attn, notes=""):
    ensure_header()
    row = [
        exp_id,
        f"{tok_s:.3f}" if tok_s is not None else "0.000",
        f"{pp_tok_s:.2f}" if pp_tok_s is not None else "0.00",
        str(vram_mb or 0),
        status,
        f"{wall_time:.1f}",
        str(n_gpu), str(n_threads), str(batch), str(ubatch),
        str(n_ctx), kv_type, str(flash_attn), notes
    ]
    with open(RESULTS_FILE, 'a') as f:
        f.write('\t'.join(row) + '\n')
    print(f"  → recorded: {status} | tg={tok_s:.3f} tok/s | {notes}")


def run_bench(bench, n_gpu, n_threads=8, batch=128, ubatch=128,
              n_ctx=512, n_gen=128, n_prompt=512,
              kv_type="f16", flash_attn=0, timeout=120, exp_id="", notes=""):
    """Run one experiment. Returns (tg_tok_s, pp_tok_s, vram_mb, backend)."""

    if already_ran(exp_id):
        print(f"  [skip] {exp_id} already in results")
        return None, None, None, None

    cmd = [bench,
           "-m", MODEL,
           "-ngl", str(n_gpu),
           "-b", str(batch),
           "-ub", str(ubatch),
           "-t", str(n_threads),
           "-n", str(n_gen),
           "-p", str(n_prompt),
           "--flash-attn", str(flash_attn)]
    if kv_type != "f16":
        cmd += ["-ctk", kv_type, "-ctv", kv_type]

    print(f"\n[{exp_id}] ngl={n_gpu} t={n_threads} b={batch}/{ubatch} "
          f"ctx={n_ctx} kv={kv_type} fa={flash_attn} timeout={timeout}s")

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=CUDA_ENV)
        elapsed = time.time() - t0
        out = proc.stdout

        if proc.returncode != 0:
            record(exp_id, 0, 0, 0, "fail", elapsed,
                   n_gpu, n_threads, batch, ubatch, n_ctx, kv_type, flash_attn,
                   f"exit={proc.returncode}")
            return None, None, None, None

        tg = _parse_tg(out)
        pp = _parse_pp(out)
        vram = _parse_vram(out)
        backend = "CUDA" if "CUDA" in out else "CPU"

        print(f"  backend={backend} tg={tg} pp={pp} vram={vram}MB ({elapsed:.1f}s)")
        record(exp_id, tg, pp, vram, "ok" if tg else "parse_fail", elapsed,
               n_gpu, n_threads, batch, ubatch, n_ctx, kv_type, flash_attn,
               f"backend={backend} {notes}")
        return tg, pp, vram, backend

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  TIMEOUT after {elapsed:.0f}s")
        record(exp_id, 0, 0, 0, "timeout", elapsed,
               n_gpu, n_threads, batch, ubatch, n_ctx, kv_type, flash_attn,
               f"timed_out_after_{timeout}s")
        return None, None, None, "timeout"
    except Exception as e:
        elapsed = time.time() - t0
        record(exp_id, 0, 0, 0, "error", elapsed,
               n_gpu, n_threads, batch, ubatch, n_ctx, kv_type, flash_attn, str(e))
        return None, None, None, None


def _parse_tg(out):
    for line in out.split('\n'):
        if '| tg' in line or ('tg' in line.lower() and '|' in line):
            m = re.search(r'(\d+\.\d+)\s*[±|]', line)
            if m: return float(m.group(1))
    # fallback: any tok/s number
    m = re.search(r'(\d+\.\d+)\s*tok/s', out)
    return float(m.group(1)) if m else None


def _parse_pp(out):
    for line in out.split('\n'):
        if '| pp' in line or ('pp' in line.lower() and '|' in line):
            m = re.search(r'(\d+\.\d+)\s*[±|]', line)
            if m: return float(m.group(1))
    return None


def _parse_vram(out):
    m = re.search(r'(\d+)\s*MiB', out)
    return int(m.group(1)) if m else 0


def best_result():
    rows = load_results()
    goods = [r for r in rows if r['_tg'] > 0 and r.get('status') == 'ok']
    if not goods: return None, 0
    best = max(goods, key=lambda r: r['_tg'])
    return best, best['_tg']


def timeout_for_mode(cuda_available):
    """Pick a sane timeout. CPU: 90s (model responds slowly). CUDA: 60s."""
    return 90 if not cuda_available else 60


# ============================================================
# STRATEGY SELECTION — core of self-direction
# ============================================================
def choose_strategy(hw, rows):
    """
    Analyse existing results and hardware to decide next strategy.
    Returns strategy string.
    """
    if not rows:
        return "probe_cuda"  # First run ever

    statuses = [r.get('status','') for r in rows]
    total = len(rows)
    timeouts = statuses.count('timeout')
    oks = statuses.count('ok')
    cuda_oks = sum(1 for r in rows if r.get('status') == 'ok' and 'CUDA' in r.get('notes',''))

    timeout_rate = timeouts / total if total > 0 else 0

    # If CUDA never worked and most timeout → CPU-only mode
    if cuda_oks == 0 and timeout_rate > 0.5:
        return "cpu_only_optimise"

    # If some CUDA results exist → explore GPU space
    if cuda_oks > 0:
        return "cuda_optimise"

    # If we have CPU baseline but no CUDA → test GPU
    if oks > 0 and cuda_oks == 0:
        return "probe_cuda"

    return "cpu_only_optimise"


# ============================================================
# STRATEGY: Probe CUDA
# ============================================================
def strategy_probe_cuda(bench, hw):
    """Quick ngl sweep to see if GPU helps at all."""
    print("\n[strategy] PROBE_CUDA — testing if GPU layers improve speed")

    results_by_ngl = {}
    # Try a range of ngl values with SHORT timeout to detect GPU vs CPU quickly
    for ngl in [0, 5, 10, 15, 20, 25, 30]:
        eid = f"probe_ngl{ngl}"
        tg, pp, vram, backend = run_bench(
            bench, n_gpu=ngl, n_threads=8, batch=128, ubatch=128,
            n_gen=64, n_prompt=256, timeout=90, exp_id=eid,
            notes="probe:cuda_test")
        if tg and tg > 0:
            results_by_ngl[ngl] = (tg, backend)
            print(f"  ngl={ngl}: {tg:.2f} tok/s backend={backend}")
        elif backend == "timeout":
            print(f"  ngl={ngl}: TIMEOUT — likely OOM or CPU-only stall")

    # Analyse: did any ngl give CUDA backend?
    cuda_wins = {ngl: v for ngl, v in results_by_ngl.items() if v[1] == 'CUDA'}
    if cuda_wins:
        print(f"[probe] CUDA confirmed at ngl values: {list(cuda_wins.keys())}")
        return "cuda_confirmed"
    else:
        print("[probe] No CUDA backend detected — model exceeds VRAM, switching to CPU-only mode")
        return "cpu_only"


# ============================================================
# STRATEGY: CPU-only optimisation (runs forever)
# ============================================================
def strategy_cpu_only(bench):
    """
    Model is CPU-only (IQ2_XXS 17GB > 8GB VRAM).
    Optimise threads, batch, context for maximum CPU throughput.
    This loop NEVER ENDS — keeps finding new axes to explore.
    """
    print("\n[strategy] CPU_ONLY_OPTIMISE — model is CPU-resident, maximising CPU throughput")
    print("IQ2_XXS = 17GB > 8GB VRAM. GPU not usable. Goal: max CPU tok/s.")

    exp_counter = [len(load_results())]

    def next_id(tag):
        exp_counter[0] += 1
        return f"cpu_{exp_counter[0]:04d}_{tag}"

    # Load existing CPU results to seed Bayesian-style search
    rows = load_results()
    cpu_rows = [r for r in rows if r.get('status') == 'ok' and r['_tg'] > 0]

    best_cfg = {
        'n_threads': 8, 'batch': 128, 'ubatch': 128,
        'n_ctx': 512, 'kv_type': 'q8_0', 'flash_attn': 0,
        'n_gen': 128, 'n_prompt': 512
    }

    # Seed from existing best if available
    if cpu_rows:
        br = max(cpu_rows, key=lambda r: r['_tg'])
        best_cfg.update({
            'n_threads': int(br.get('n_threads', 8)),
            'batch': int(br.get('batch', 128)),
            'ubatch': int(br.get('ubatch', 128)),
            'kv_type': br.get('kv_type', 'q8_0'),
        })
        print(f"[cpu] Seeding from existing best: {br['_tg']:.3f} tok/s")
        print(f"[cpu] Best config: {best_cfg}")

    iteration = 0
    axes = [
        # (axis_name, param_name, candidate_values)
        ("threads",  "n_threads", [2, 4, 6, 8, 10, 12]),
        ("batch",    "batch",     [64, 128, 256, 512]),
        ("ubatch",   "ubatch",    [32, 64, 128, 256]),
        ("kv_type",  "kv_type",   ["f16", "q8_0", "q4_0"]),
        ("ctx",      "n_ctx",     [256, 512, 1024, 2048]),
        ("n_gen",    "n_gen",     [64, 128, 256, 512]),
    ]

    while True:  # NEVER STOP
        iteration += 1
        print(f"\n{'='*60}")
        print(f"CPU OPTIMISATION — iteration {iteration}")
        _, current_best_tg = best_result()
        print(f"Current best: {current_best_tg:.3f} tok/s")
        print(f"Config: {best_cfg}")
        print('='*60)

        improved = False

        # Cycle through axes, optimise one at a time (coordinate descent)
        ax_idx = (iteration - 1) % len(axes)
        axis_name, param, candidates = axes[ax_idx]

        print(f"\n[iter {iteration}] Sweeping axis: {axis_name}")
        axis_results = {}

        for val in candidates:
            cfg = best_cfg.copy()
            cfg[param] = val
            eid = next_id(f"{axis_name}{val}")

            tg, pp, vram, backend = run_bench(
                bench,
                n_gpu=0,  # always CPU
                n_threads=cfg['n_threads'],
                batch=cfg['batch'],
                ubatch=cfg['ubatch'],
                n_ctx=cfg['n_ctx'],
                n_gen=cfg['n_gen'],
                n_prompt=cfg['n_prompt'],
                kv_type=cfg['kv_type'],
                flash_attn=cfg['flash_attn'],
                timeout=120,  # CPU: generous timeout, it's slow but steady
                exp_id=eid,
                notes=f"cpu_opt axis={axis_name} val={val}"
            )
            if tg and tg > 0:
                axis_results[val] = tg

        if axis_results:
            best_val = max(axis_results, key=axis_results.get)
            best_val_tg = axis_results[best_val]
            print(f"  Axis {axis_name}: best={best_val} → {best_val_tg:.3f} tok/s")

            if best_val_tg > (current_best_tg or 0) * 1.005:  # >0.5% improvement
                best_cfg[param] = best_val
                print(f"  ✓ Updated best_cfg[{param}] = {best_val}")
                improved = True
            else:
                print(f"  ✗ No improvement (best_cfg[{param}] stays {best_cfg[param]})")

        # Every 3 iterations, try a random perturbation to escape local optima
        if iteration % 3 == 0:
            print(f"\n[iter {iteration}] Random perturbation to escape local optima")
            perturb_cfg = best_cfg.copy()
            # Randomly vary 2 params
            for _ in range(2):
                ax_name, p, cands = random.choice(axes)
                perturb_cfg[p] = random.choice(cands)
            eid = next_id("perturb")
            tg, _, _, _ = run_bench(
                bench, n_gpu=0,
                n_threads=perturb_cfg['n_threads'],
                batch=perturb_cfg['batch'],
                ubatch=perturb_cfg['ubatch'],
                n_ctx=perturb_cfg['n_ctx'],
                n_gen=perturb_cfg['n_gen'],
                kv_type=perturb_cfg['kv_type'],
                flash_attn=0,
                timeout=120, exp_id=eid,
                notes=f"perturb cfg={perturb_cfg}")
            if tg and tg > (current_best_tg or 0):
                best_cfg = perturb_cfg
                print(f"  ✓ Perturbation improved to {tg:.3f} tok/s!")

        # Every 10 iterations, print a summary
        if iteration % 10 == 0:
            all_rows = load_results()
            ok_rows = [r for r in all_rows if r['_tg'] > 0]
            if ok_rows:
                top5 = sorted(ok_rows, key=lambda r: r['_tg'], reverse=True)[:5]
                print(f"\n[summary] Top 5 results so far:")
                for r in top5:
                    print(f"  {r['exp_id']}: {r['_tg']:.3f} tok/s "
                          f"t={r['n_threads']} b={r['batch']}/{r['ubatch']} kv={r['kv_type']}")


# ============================================================
# STRATEGY: CUDA optimisation (runs forever)
# ============================================================
def strategy_cuda(bench, best_ngl):
    """GPU-assisted optimisation — model partially on GPU."""
    print(f"\n[strategy] CUDA_OPTIMISE — ngl={best_ngl} confirmed working")

    exp_counter = [len(load_results())]

    def next_id(tag):
        exp_counter[0] += 1
        return f"cuda_{exp_counter[0]:04d}_{tag}"

    best_cfg = {
        'n_gpu': best_ngl, 'n_threads': 8, 'batch': 128, 'ubatch': 128,
        'n_ctx': 512, 'kv_type': 'q8_0', 'flash_attn': 1,
        'n_gen': 128, 'n_prompt': 512
    }

    axes = [
        ("ngl",     "n_gpu",    list(range(max(0, best_ngl-5), min(52, best_ngl+8)))),
        ("threads", "n_threads",[4, 6, 8, 10, 12]),
        ("batch",   "batch",    [64, 128, 256, 512]),
        ("ubatch",  "ubatch",   [32, 64, 128, 256]),
        ("kv",      "kv_type",  ["f16", "q8_0", "q4_0"]),
        ("ctx",     "n_ctx",    [512, 1024, 2048]),
        ("fa",      "flash_attn",[0, 1]),
    ]

    iteration = 0
    while True:
        iteration += 1
        _, current_best = best_result()
        print(f"\n[cuda iter {iteration}] best={current_best:.3f} tok/s | cfg={best_cfg}")

        ax_idx = (iteration - 1) % len(axes)
        axis_name, param, candidates = axes[ax_idx]
        axis_results = {}

        for val in candidates:
            cfg = best_cfg.copy()
            cfg[param] = val
            eid = next_id(f"{axis_name}{val}")
            tg, _, _, backend = run_bench(
                bench, n_gpu=cfg['n_gpu'],
                n_threads=cfg['n_threads'],
                batch=cfg['batch'], ubatch=cfg['ubatch'],
                n_ctx=cfg['n_ctx'], n_gen=cfg['n_gen'],
                n_prompt=cfg['n_prompt'],
                kv_type=cfg['kv_type'], flash_attn=cfg['flash_attn'],
                timeout=60, exp_id=eid,
                notes=f"cuda_opt axis={axis_name} val={val}")
            if tg and tg > 0:
                axis_results[val] = tg

        if axis_results:
            best_val = max(axis_results, key=axis_results.get)
            if axis_results[best_val] > (current_best or 0) * 1.005:
                best_cfg[param] = best_val
                print(f"  ✓ {axis_name}={best_val}")

        if iteration % 5 == 0:
            # Random perturbation
            perturb = best_cfg.copy()
            ax_name, p, cands = random.choice(axes)
            perturb[p] = random.choice(cands)
            eid = next_id("perturb")
            tg, _, _, _ = run_bench(bench, n_gpu=perturb['n_gpu'],
                n_threads=perturb['n_threads'], batch=perturb['batch'],
                ubatch=perturb['ubatch'], kv_type=perturb['kv_type'],
                flash_attn=perturb['flash_attn'],
                timeout=60, exp_id=eid, notes="perturb")
            if tg and tg > (current_best or 0):
                best_cfg = perturb
                print(f"  ✓ Perturbation: {tg:.3f} tok/s")


# ============================================================
# MAIN — hardware-aware self-directing loop
# ============================================================
def main():
    print("=" * 65)
    print("Nemotron-Cascade-2-30B-A3B — Self-Directing Autoresearch")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    bench = find_bench()
    print(f"Bench binary: {bench}")

    hw = probe_hardware()
    print(f"Hardware: CUDA={hw['cuda']} free_vram={hw['free_vram_mb']}MB ram={hw['ram_gb']:.1f}GB")
    print(f"Model: {MODEL} ({Path(MODEL).stat().st_size/1e9:.1f}GB)")

    ensure_header()
    rows = load_results()
    print(f"Existing results: {len(rows)} experiments")
    best_row, best_tg = best_result()
    if best_tg > 0:
        print(f"Current best: {best_tg:.3f} tok/s")

    strategy = choose_strategy(hw, rows)
    print(f"Initial strategy: {strategy}")

    # --- Hardware reality check ---
    # IQ2_XXS = ~17GB, RTX 3070 = 8GB VRAM
    # Model CANNOT fit in GPU — always CPU-only
    # But still probe first to confirm (maybe partial GPU layers help)
    if strategy == "probe_cuda" and hw['free_vram_mb'] < 17000:
        print(f"\n[reality] Model is {Path(MODEL).stat().st_size/1e9:.1f}GB, "
              f"VRAM free={hw['free_vram_mb']/1024:.1f}GB — probing for partial GPU layers")
        result = strategy_probe_cuda(bench, hw)
        if result == "cuda_confirmed":
            # Find best ngl from probe
            probe_rows = [r for r in load_results()
                          if r.get('exp_id','').startswith('probe_') and r['_tg'] > 0]
            if probe_rows:
                best_probe = max(probe_rows, key=lambda r: r['_tg'])
                best_ngl = int(best_probe.get('n_gpu', 0))
                print(f"[main] CUDA partial offload works at ngl={best_ngl}. Optimising GPU path.")
                strategy_cuda(bench, best_ngl)
            else:
                strategy_cpu_only(bench)
        else:
            strategy_cpu_only(bench)
    elif strategy == "cpu_only_optimise" or (hw['free_vram_mb'] < 17000 and not hw['cuda']):
        strategy_cpu_only(bench)
    elif strategy in ("cuda_optimise", "cuda_confirmed"):
        best_cuda_rows = [r for r in rows if 'CUDA' in r.get('notes','') and r['_tg'] > 0]
        if best_cuda_rows:
            best_ngl = int(max(best_cuda_rows, key=lambda r: r['_tg']).get('n_gpu', 10))
        else:
            best_ngl = 10
        strategy_cuda(bench, best_ngl)
    else:
        # Default: probe then decide
        result = strategy_probe_cuda(bench, hw)
        if result == "cpu_only":
            strategy_cpu_only(bench)
        else:
            strategy_cuda(bench, 10)


if __name__ == "__main__":
    main()
