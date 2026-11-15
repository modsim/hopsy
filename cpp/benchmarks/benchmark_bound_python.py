#!/usr/bin/env python3
"""Compare frozen compatibility bindings with persistent direct bindings.

Select the extension using PYTHONPATH and --api before/after. Native Gaussian on
[-3,3]^10, fixed seeds and scales. Times include Python calls and sample output,
exclude setup/warm-up. Use --output for JSON results. Never run beside a build.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _hopsy_core as core
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", choices=["before", "after"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=["gaussian", "uniform"], default="gaussian")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--only-parallel", action="store_true")
    args = parser.parse_args()
    n = 10
    geometry = core.LinearPolytope(
        np.vstack((np.eye(n), -np.eye(n))), np.full(2 * n, 3.0)
    )
    density = (
        core.UniformTarget()
        if args.target == "uniform"
        else core.GaussianTarget(np.zeros(n), np.eye(n))
    )
    names = [
        "UniformCoordinateHitAndRunKernel",
        "UniformHitAndRunKernel",
        "GaussianRandomWalkKernel",
        "DikinWalkKernel",
        "CSmMALAKernel",
    ]
    if args.target == "uniform":
        names = names[:2]
    rows = []
    cases = (
        [(args.steps, 4)]
        if args.only_parallel
        else [(1, 1), (16, 1), (args.steps, 1), (args.steps, 4)]
    )
    if (
        args.steps < 1
        or args.thinning < 1
        or any(batch % args.thinning or args.steps % batch for batch, _ in cases)
    ):
        parser.error(
            "steps and batches must be divisible by thinning, and steps by batch"
        )
    for name in names:
        for batch, workers in cases:
            timings = []
            checksums = []
            for repeat in range(3):
                draws = []
                for i in range(workers):
                    options = (
                        {
                            "step_size": (
                                0.7 if name == "GaussianRandomWalkKernel" else 0.3
                            )
                        }
                        if "HitAndRun" not in name
                        else {}
                    )
                    if args.api == "before":
                        chain = getattr(core, name)(
                            geometry, np.zeros(n), seed=42 + i, **options
                        )
                        draw = lambda count, thinning=1, chain=chain: chain.draw(
                            count, thinning=thinning, target=density
                        )
                    else:
                        problem = core.Problem(geometry, density)
                        initial = np.zeros(n)
                        kernel = getattr(core, name)(initial, **options)
                        chain = core.bind_chain(problem, kernel, initial)
                        rng = core.RandomNumberGenerator(42 + i)
                        draw = (
                            lambda count, thinning=1, chain=chain, rng=rng: chain.draw(
                                rng, count, thinning=thinning
                            )
                        )
                    draw(1000)
                    draws.append(draw)

                def run(draw):
                    last = None
                    for _ in range(args.steps // batch):
                        last = draw(batch // args.thinning, thinning=args.thinning)
                    return float(last[:, -1].sum())

                start = time.perf_counter()
                if workers == 1:
                    checksum = [run(draws[0])]
                else:
                    with ThreadPoolExecutor(workers) as pool:
                        checksum = list(pool.map(run, draws))
                timings.append(time.perf_counter() - start)
                checksums.append(checksum)
            seconds = statistics.median(timings)
            rows.append(
                dict(
                    kernel=name,
                    batch=batch,
                    workers=workers,
                    seconds=timings,
                    steps_per_second=args.steps * workers / seconds,
                    steps=args.steps,
                    thinning=args.thinning,
                    checksums=checksums,
                )
            )
            print(name, batch, workers, round(rows[-1]["steps_per_second"]), flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
