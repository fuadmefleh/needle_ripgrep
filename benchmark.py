"""Steady-state speed benchmark for the finetuned NGT checkpoint.

Separates one-time cost (process start, checkpoint load, JAX JIT trace/
compile) from steady-state per-query latency by doing a warmup call before
timing anything. Reports both single-query latency (what a CLI user
actually feels) and batched throughput.
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "needle"))

from schema import TOOLS_JSON  # noqa: E402

_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

SAMPLE_QUERIES = [
    "find TODO comments",
    "search for DB_PORT=5432",
    "where do we handle retry logic",
    "find code handling payment failures",
    "find email addresses",
    "find async python functions starting with fetch or pull",
    "look for functions named get_x or set_x",
    "find all instances of API_KEY",
    "search for authentication errors logic",
    "find UUID values",
]


def default_checkpoint():
    candidates = sorted(glob.glob(os.path.join(_CHECKPOINT_DIR, "needle_finetuned_*_best.pkl")))
    if not candidates:
        raise FileNotFoundError(f"No finetuned checkpoint found in {_CHECKPOINT_DIR}.")
    return candidates[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n", type=int, default=20, help="Number of single-query timing runs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for the throughput test")
    args = p.parse_args()

    checkpoint_path = args.checkpoint or default_checkpoint()

    t0 = time.perf_counter()
    from needle import load_checkpoint, SimpleAttentionNetwork, get_tokenizer, generate_batch
    from fast_generate import generate_fixed_shape
    params, config = load_checkpoint(checkpoint_path)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()
    t_load = time.perf_counter() - t0
    print(f"Checkpoint + tokenizer load: {t_load:.2f}s (one-time, includes JAX/import cost)")

    # Warmup: triggers JIT trace + XLA compile for this (model, max_gen_len,
    # encoder bucket shape) combination.
    t0 = time.perf_counter()
    generate_fixed_shape(model, params, tokenizer, SAMPLE_QUERIES[0], TOOLS_JSON)
    t_warmup = time.perf_counter() - t0
    print(f"First call (cold JIT compile): {t_warmup:.2f}s (one-time per process)")

    # Steady-state single-query latency, using a fixed encoder input shape
    # (see fast_generate.py) -- without it, every differently-sized query
    # would trigger its own ~5-6s XLA recompile instead of reusing the one
    # compiled graph, which would measure recompilation cost, not the model.
    latencies = []
    for i in range(args.n):
        q = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        t0 = time.perf_counter()
        generate_fixed_shape(model, params, tokenizer, q, TOOLS_JSON)
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    n = len(latencies)
    p50 = latencies[n // 2]
    p95 = latencies[min(n - 1, int(n * 0.95))]
    avg = sum(latencies) / n
    print(f"\nSteady-state single-query latency over {n} calls (varying lengths, fixed-shape encoding):")
    print(f"  avg={avg*1000:.1f}ms  p50={p50*1000:.1f}ms  p95={p95*1000:.1f}ms  "
          f"min={latencies[0]*1000:.1f}ms  max={latencies[-1]*1000:.1f}ms")

    # Batched throughput (separate compile cache entry keyed by batch shape/max_gen_len,
    # so warm it up once before timing).
    batch = [SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)] for i in range(args.batch_size)]
    tools_list = [TOOLS_JSON] * len(batch)
    generate_batch(model, params, tokenizer, batch, tools_list)  # warmup

    t0 = time.perf_counter()
    results = generate_batch(model, params, tokenizer, batch, tools_list)
    t_batch = time.perf_counter() - t0
    total_tokens = sum(len(tokenizer.encode(r)) for r in results)
    print(f"\nBatched throughput ({len(batch)} queries in one call):")
    print(f"  wall time={t_batch*1000:.1f}ms  "
          f"{len(batch)/t_batch:.1f} queries/s  "
          f"{total_tokens/t_batch:.1f} tok/s")


if __name__ == "__main__":
    main()
