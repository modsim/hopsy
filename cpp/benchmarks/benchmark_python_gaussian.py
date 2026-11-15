#!/usr/bin/env python3
"""Actual hopsy.sample multiprocessing versus nanobind batched native threads.

Four chains in every case; report aggregate attempted steps/s and sum of each
chain's minimum-coordinate ESS / elapsed seconds. Setup and 20k/chain warm-up
are excluded. Legacy hopsy.sample includes spawn, pickling/IPC, Python sampling
loops and shutdown. New ThreadPoolExecutor includes thread startup/shutdown and
sample allocation but no concatenation of the four native arrays. Statistics
and validation are outside the sampling timer. Both use dense native Gaussian
models, not Python density callbacks. Threading libraries are pinned to one.
"""

import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import argparse
import csv
import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from run_gaussian_family import ess


def old_run(old, family, n_samples, thinning, processes, seed):
    n = 10
    problem = old.Problem(
        np.vstack((np.eye(n), -np.eye(n))),
        np.full(2 * n, 3.0),
        old.Gaussian(np.zeros(n), np.eye(n)),
        starting_point=np.zeros(n),
    )
    scale = {"grw": 0.7, "dikin": n * 0.3**2, "csmala": np.sqrt(n) * 0.3}[family]
    cls = getattr(
        old,
        {
            "grw": "GaussianProposal",
            "dikin": "DikinWalkProposal",
            "csmala": "CSmMALAProposal",
        }[family],
    )
    options = {"fisher_weight": 0.0} if family == "csmala" else {}
    chains = [
        old.MarkovChain(
            problem=problem,
            proposal=cls(
                problem, starting_point=np.zeros(n), stepsize=scale, **options
            ),
        )
        for _ in range(4)
    ]
    rngs = [old.RandomNumberGenerator(seed=seed + i) for i in range(4)]
    old.sample(chains, rngs, n_samples=1, thinning=20000, n_procs=1)
    start = time.perf_counter()
    acceptance, samples = old.sample(
        chains,
        rngs,
        n_samples=n_samples,
        thinning=thinning,
        n_procs=processes,
        progress_bar=False,
    )
    elapsed = time.perf_counter() - start
    samples = np.asarray(samples)
    if samples.shape != (4, n_samples, n):
        raise RuntimeError(f"unexpected legacy sample shape {samples.shape}")
    return elapsed, [samples[i] for i in range(4)], float(np.mean(acceptance))


def new_run(new, family, n_samples, thinning, threads, seed):
    n = 10
    geometry = new.LinearPolytope(
        np.vstack((np.eye(n), -np.eye(n))), np.full(2 * n, 3.0)
    )
    target = new.GaussianTarget(np.zeros(n), np.eye(n))
    cls = getattr(
        new,
        {
            "grw": "GaussianRandomWalkKernel",
            "dikin": "DikinWalkKernel",
            "csmala": "CSmMALAKernel",
        }[family],
    )
    problem = new.Problem(geometry, target)
    chains = [
        new.bind_chain(
            problem,
            cls(np.zeros(n), step_size=0.7 if family == "grw" else 0.3),
            np.zeros(n),
        )
        for _ in range(4)
    ]
    rngs = [new.RandomNumberGenerator(seed + i) for i in range(4)]
    for chain, rng in zip(chains, rngs):
        chain.draw(rng, 1, thinning=20000)
    start = time.perf_counter()
    if threads == 1:
        samples = [
            chain.draw(rng, n_samples, thinning=thinning)
            for chain, rng in zip(chains, rngs)
        ]
    else:
        with ThreadPoolExecutor(4) as pool:
            futures = [
                pool.submit(chain.draw, rng, n_samples, thinning)
                for chain, rng in zip(chains, rngs)
            ]
            samples = [future.result() for future in futures]
    elapsed = time.perf_counter() - start
    if any(x.shape != (n, n_samples) for x in samples):
        raise RuntimeError("unexpected native sample shape")
    return elapsed, [x.T for x in samples], None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--thinning", type=int, nargs="+", default=[1, 20])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["grw", "dikin", "csmala"],
        default=["grw", "dikin", "csmala"],
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["old_serial", "old_processes", "new_serial", "new_threads"],
        default=["old_serial", "old_processes", "new_serial", "new_threads"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cpp/benchmarks/results/python_gaussian")
    )
    args = parser.parse_args()
    if args.samples < 100 or min(args.thinning) < 1 or args.repeats < 1:
        parser.error("need samples>=100, positive thinning and repeats")
    import _hopsy_core as new

    import hopsy as old

    metadata = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        old_module=old.core.__file__,
        new_module=new.__file__,
        samples_per_chain=args.samples,
        chains=4,
        warmup=20000,
        repeats=args.repeats,
        target="N(0,I_10) on [-3,3]^10",
        csmala_fisher_weight=0.0,
        legacy_clipping="norm > 1",
    )
    print(json.dumps(metadata), flush=True)
    rows = []
    csv_path = args.output.with_suffix(".csv")
    for thinning in args.thinning:
        for family in args.families:
            for repeat in range(args.repeats):
                for case in args.cases[:: (-1 if repeat % 2 else 1)]:
                    seed = 42 + 10 * repeat
                    elapsed, samples, acceptance = (
                        old_run(
                            old,
                            family,
                            args.samples,
                            thinning,
                            4 if case == "old_processes" else 1,
                            seed,
                        )
                        if case.startswith("old_")
                        else new_run(
                            new,
                            family,
                            args.samples,
                            thinning,
                            4 if case == "new_threads" else 1,
                            seed,
                        )
                    )
                    if any(
                        not np.isfinite(x).all() or np.max(np.abs(x)) > 3.0
                        for x in samples
                    ):
                        raise RuntimeError(f"invalid samples in {case}/{family}")
                    effective = sum(ess(x) for x in samples)
                    row = dict(
                        family=family,
                        case=case,
                        repeat=repeat,
                        seed=seed,
                        samples_per_chain=args.samples,
                        thinning=thinning,
                        seconds=elapsed,
                        steps_per_second=4 * args.samples * thinning / elapsed,
                        ess_per_sample=effective / (4 * args.samples),
                        ess_per_second=effective / elapsed,
                        acceptance=acceptance,
                        max_abs_mean=max(
                            float(np.max(np.abs(x.mean(axis=0)))) for x in samples
                        ),
                        second_moment=float(np.mean([np.mean(x * x) for x in samples])),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    with csv_path.open("w") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=list(rows[0]), lineterminator="\n"
                        )
                        writer.writeheader()
                        writer.writerows(rows)
    summary = []
    for thinning in args.thinning:
        for family in args.families:
            for case in args.cases:
                group = [
                    r
                    for r in rows
                    if r["family"] == family
                    and r["case"] == case
                    and r["thinning"] == thinning
                ]
                summary.append(
                    dict(
                        family=family,
                        case=case,
                        thinning=thinning,
                        **{
                            key: statistics.median(r[key] for r in group)
                            for key in (
                                "seconds",
                                "steps_per_second",
                                "ess_per_sample",
                                "ess_per_second",
                            )
                        },
                    )
                )
    args.output.with_suffix(".json").write_text(
        json.dumps(dict(metadata=metadata, summary=summary), indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
