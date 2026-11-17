#!/usr/bin/env python3
"""Bilinear/linear geometry throughput, identical native Hit-and-Run implementations.

Four independent chains/threads versus one, uniform target, 1k warm-up transitions,
fixed seeds. Includes output and thread-pool setup; reports attempted transitions/s.
Select the current extension with PYTHONPATH.
"""

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _hopsy_core as core
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-local", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 20 or args.steps % 20 or args.repeats < 1:
        parser.error("positive repeats and steps divisible by 20 required")
    rows = []
    for n in [2, 10]:
        split = n // 2
        A = np.vstack((np.eye(n), -np.eye(n)))
        b = np.ones(2 * n)
        B = np.zeros((split, n - split))
        B[0, 0] = 1.0
        cases = [("linear", core.LinearPolytope(A, b))]
        dense = np.random.default_rng(13).normal(size=B.shape) / np.sqrt(B.size)
        cases.extend(
            [
                (
                    "vector_only",
                    core.BilinearPolytope(A, b, split, [], np.zeros(0)),
                ),
                ("opposite", core.BilinearPolytope(A, b, split, [B], np.zeros(1))),
                (
                    "dense_cross",
                    core.BilinearPolytope(A, b, split, [dense], np.array([0.2])),
                ),
            ]
        )
        for region, g in cases:
            for name in ["UniformHitAndRunKernel", "UniformCoordinateHitAndRunKernel"]:
                if region == "opposite" and name == "UniformCoordinateHitAndRunKernel":
                    continue  # This direction schedule is reducible on the sign components.
                for workers in [1, 4]:
                    timings, checksums = [], []
                    for _ in range(args.repeats):
                        problem = core.Problem(g, core.UniformTarget())
                        initial = np.zeros(n)
                        if region == "opposite":
                            initial[0], initial[split] = -0.5, 0.5

                        def setup(i, barrier=None):
                            k = getattr(core, name)(initial)
                            c = core.bind_chain(problem, k, initial)
                            r = core.RandomNumberGenerator(42 + i)
                            c.draw(r, 1, thinning=1000)
                            if barrier is not None:
                                barrier.wait(timeout=30)
                            return c, r

                        if args.worker_local and workers == 4:
                            barrier = threading.Barrier(4)
                            with ThreadPoolExecutor(4) as pool:
                                prepared = [
                                    pool.submit(setup, i, barrier) for i in range(4)
                                ]
                                pairs = [f.result() for f in prepared]
                                start = time.perf_counter()
                                futures = [
                                    pool.submit(c.draw, r, args.steps // 20, 20)
                                    for c, r in pairs
                                ]
                                samples = [f.result() for f in futures]
                                elapsed = time.perf_counter() - start
                        else:
                            pairs = [setup(i) for i in range(workers)]
                            start = time.perf_counter()
                            if workers == 1:
                                samples = [
                                    pairs[0][0].draw(
                                        pairs[0][1], args.steps // 20, thinning=20
                                    )
                                ]
                            else:
                                with ThreadPoolExecutor(4) as pool:
                                    futures = [
                                        pool.submit(c.draw, r, args.steps // 20, 20)
                                        for c, r in pairs
                                    ]
                                    samples = [f.result() for f in futures]
                            elapsed = time.perf_counter() - start
                        timings.append(elapsed)
                        checksums.append([float(x[:, -1].sum()) for x in samples])
                        assert all(np.max(np.abs(x)) <= 1 + 1e-10 for x in samples)
                        if region == "opposite":
                            assert all(
                                np.max(x[0] * x[split]) <= 1e-10 for x in samples
                            )
                    row = dict(
                        region=region,
                        dimension=n,
                        kernel=name,
                        workers=workers,
                        steps_per_chain=args.steps,
                        thinning=20,
                        worker_local=args.worker_local,
                        seconds=timings,
                        steps_per_second=args.steps
                        * workers
                        / statistics.median(timings),
                        checksums=checksums,
                    )
                    rows.append(row)
                    print(
                        region,
                        n,
                        name,
                        workers,
                        round(row["steps_per_second"]),
                        flush=True,
                    )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
