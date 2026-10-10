#!/usr/bin/env python3
"""Compare old/new hopsy coordinate hit-and-run on [-1,1]^100.

Run from a terminal (the __main__ guard is required for multiprocessing):
  PYTHONPATH=build/bindings python benchmark_old_vs_new.py
  python benchmark_old_vs_new.py --new-module hopsy.experimental._hopsy_core
  python benchmark_old_vs_new.py --samples 100000 --thinning 10 --output comparison.csv

Four cases: old single-process, new single-thread, old 4 processes through
hopsy.sample(n_procs=4), new 4 threads with one owning native chain per thread.
All cases use cycling coordinate selection and a uniform target. Timings include
sample allocation/storage and parallel pool startup/shutdown/communication, but
exclude geometry/kernel construction, warm-up, and computation of statistics.
Each parallel case generates --samples PER CHAIN (four times the single-chain work).
This is an end-to-end public-API benchmark, not an isolated C++ kernel benchmark.
"""

import os

# Set before importing numeric libraries, including in spawned workers.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import importlib
import json
import multiprocessing
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DIMENSION = 100


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--old-module", default="hopsy")
    p.add_argument("--new-module", default="_hopsy_core")
    p.add_argument(
        "--samples",
        type=positive,
        default=10000,
        help="stored samples per chain per repeat (default: 100000)",
    )
    p.add_argument(
        "--thinning",
        type=positive,
        default=100,
        help="attempted transitions per stored sample",
    )
    p.add_argument(
        "--warmup", type=positive, default=1000, help="untimed transitions per chain"
    )
    p.add_argument("--repeats", type=positive, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("old_vs_new.csv"))
    args = p.parse_args()
    if not 0 <= args.seed < 2**64 - 4 * args.repeats:
        p.error("--seed plus all repeat/chain offsets must fit uint64")
    return args


def sample_statistics(np, matrices):
    # Each matrix is (dimension, samples); old output is transposed as a view.
    # Run only after the timing ends. Avoid stacking all four new sample arrays.
    chain_means = [x.mean(axis=1) for x in matrices]
    coordinate_means = np.mean(chain_means, axis=0)
    for x in matrices:
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 1 + 1e-9:
            raise RuntimeError("nonfinite or infeasible sample in benchmark output")
    second = sum(float(np.einsum("ij,ij->", x, x)) for x in matrices) / sum(
        x.size for x in matrices
    )
    return dict(
        mean=float(coordinate_means.mean()),
        max_abs_coordinate_mean=float(np.max(np.abs(coordinate_means))),
        rms_coordinate_mean=float(np.sqrt(np.mean(coordinate_means**2))),
        second_moment=second,
    ), [x.tolist() for x in chain_means]


def run_old(np, old, problem, count, args, seed):
    chains = [
        old.MarkovChain(
            problem=problem,
            proposal=old.UniformHitAndRunProposal,
            starting_point=np.zeros(DIMENSION),
        )
        for _ in range(count)
    ]
    rngs = [old.RandomNumberGenerator(seed=seed + i) for i in range(count)]
    # Warm up in this process; hopsy transfers the warmed chain/RNG states to workers.
    old.sample(chains, rngs, n_samples=1, thinning=args.warmup, n_procs=1)
    start = time.perf_counter_ns()
    acceptance, samples = old.sample(
        chains, rngs, n_samples=args.samples, thinning=args.thinning, n_procs=count
    )
    seconds = (time.perf_counter_ns() - start) / 1e9
    samples = np.asarray(samples)
    expected = (count, args.samples, DIMENSION)
    if samples.shape != expected:
        raise RuntimeError(f"old API output shape {samples.shape}; expected {expected}")
    stats, means = sample_statistics(np, [samples[i].T for i in range(count)])
    return seconds, stats, means, float(np.mean(acceptance))


def run_new(np, new, geometry, count, args, seed):
    chains = [
        new.UniformHitAndRunKernel(geometry, np.zeros(DIMENSION), seed=seed + i)
        for i in range(count)
    ]
    for k in chains:
        k.draw(1, thinning=args.warmup)
    start = time.perf_counter_ns()
    if count == 1:
        samples = [chains[0].draw(args.samples, thinning=args.thinning)]
    else:
        # Native uniform draw releases the GIL. Each task owns a distinct chain.
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [
                pool.submit(k.draw, args.samples, thinning=args.thinning)
                for k in chains
            ]
            samples = [future.result() for future in futures]
    seconds = (time.perf_counter_ns() - start) / 1e9
    for x in samples:
        if x.shape != (DIMENSION, args.samples):
            raise RuntimeError(f"unexpected new API output shape: {x.shape}")
    stats, means = sample_statistics(np, samples)
    return (
        seconds,
        stats,
        means,
        None,
    )  # New draw deliberately does not track acceptance statistics.


def main():
    args = parse_args()
    import numpy as np

    old = importlib.import_module(args.old_module)
    new = importlib.import_module(args.new_module)
    required = (
        "Problem",
        "MarkovChain",
        "UniformCoordinateHitAndRunProposal",
        "sample",
    )
    if not all(hasattr(old, name) for name in required):
        raise RuntimeError(
            f"{args.old_module} does not expose the expected old hopsy API"
        )
    if not hasattr(new, "UniformCoordinateHitAndRunKernel"):
        raise RuntimeError(
            f"{args.new_module} does not expose the new coordinate kernel"
        )
    A = np.vstack((np.eye(DIMENSION), -np.eye(DIMENSION)))
    b = np.ones(2 * DIMENSION)
    problem, geometry = old.Problem(A, b), new.LinearPolytope(A, b)
    cases = [
        ("old_1_process", "old", 1),
        ("new_1_thread", "new", 1),
        ("old_4_processes", "old", 4),
        ("new_4_threads", "new", 4),
    ]
    print(f"Box [-1,1]^{DIMENSION}; uniform cycling-coordinate hit-and-run")
    print(
        f"{args.samples:,} samples/chain x {args.thinning:,} steps/sample; "
        f"{args.repeats} repeats; {args.warmup:,} warm-up steps/chain"
    )
    print(
        f'Old: {getattr(old, "__version__", "unknown version")} ({getattr(old, "__file__", None)})'
    )
    print(f'New: {getattr(new, "__file__", None)}')
    print("Expected coordinate means: 0; expected second moment: 1/3.")
    print("Parallel timings include pool startup, shutdown, and data transfer.\n")
    rows, means_detail = [], []
    for repeat in range(args.repeats):
        shift = repeat % len(cases)
        for label, api, count in cases[shift:] + cases[:shift]:
            seed = args.seed + 4 * repeat
            if api == "old":
                elapsed, stats, means, acceptance = run_old(
                    np, old, problem, count, args, seed
                )
            else:
                elapsed, stats, means, acceptance = run_new(
                    np, new, geometry, count, args, seed
                )
            attempts = count * args.samples * args.thinning
            row = dict(
                case=label,
                repeat=repeat + 1,
                chains=count,
                samples_per_chain=args.samples,
                thinning=args.thinning,
                total_steps=attempts,
                seconds=elapsed,
                steps_per_second=attempts / elapsed,
                stored_samples_per_second=count * args.samples / elapsed,
                old_acceptance_rate=acceptance,
                **stats,
            )
            rows.append(row)
            means_detail.append(
                dict(case=label, repeat=repeat + 1, per_chain_coordinate_means=means)
            )
            print(
                f"{label:18s} repeat {repeat+1}: {elapsed:9.4f} s | "
                f'{attempts/elapsed:12,.0f} steps/s | mean={stats["mean"]:+.5f} | '
                f'max|coord mean|={stats["max_abs_coordinate_mean"]:.5f}',
                flush=True,
            )
    print(
        "\nMedian wall times and aggregate throughput (four-chain runs do 4x the work):"
    )
    summaries = {}
    for label, _, count in cases:
        records = [r for r in rows if r["case"] == label]
        elapsed = statistics.median(r["seconds"] for r in records)
        rate = count * args.samples * args.thinning / elapsed
        summaries[label] = rate
        print(
            f"{label:18s} {elapsed:9.4f} s | {rate:12,.0f} steps/s | "
            f'mean={statistics.mean(r["mean"] for r in records):+.5f} | '
            f'E[x²]={statistics.mean(r["second_moment"] for r in records):.5f}'
        )
    print(
        f'\nNew/old throughput, one chain: {summaries["new_1_thread"]/summaries["old_1_process"]:.2f}x'
    )
    print(
        f'New/old throughput, four chains: {summaries["new_4_threads"]/summaries["old_4_processes"]:.2f}x'
    )
    print(
        f'Old four/one aggregate throughput: {summaries["old_4_processes"]/summaries["old_1_process"]:.2f}x'
    )
    print(
        f'New four/one aggregate throughput: {summaries["new_4_threads"]/summaries["new_1_thread"]:.2f}x'
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        dimension=DIMENSION,
        constraints=2 * DIMENSION,
        python=sys.version,
        numpy=np.__version__,
        platform=platform.platform(),
        logical_cpus=os.cpu_count(),
        old_version=getattr(old, "__version__", None),
        old_file=getattr(old, "__file__", None),
        new_file=getattr(new, "__file__", None),
        mean_details=means_detail,
        timing="sampling API calls including parallel startup, storage and IPC; no warmup or statistics",
        notes="Same seeds do not imply identical RNG sequences across APIs. Means are diagnostics, not ESS.",
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f'\nSaved {args.output} and {args.output.with_suffix(".json")}')


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
