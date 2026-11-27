#!/usr/bin/env python3
"""Paired Release timings of public sampling operations. See README.md."""

import argparse
import hashlib
import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class Case:
    mode: str
    dimension: int = 8
    batch: int = 32768
    thinning: int = 1
    chains: int = 1
    kernel: str = "UniformCoordinateHitAndRunKernel"
    target: str = "uniform"
    traced: bool = False

    @property
    def steps_per_batch(self):
        return self.batch * self.thinning * self.chains


CASES = {
    "cpp-small": Case("cpp-regular", batch=1),
    "cpp-regular": Case("cpp-regular"),
    "cpp-thin16": Case("cpp-regular", thinning=16),
    "cpp-nrpt": Case(
        "cpp-nrpt",
        batch=4096,
        chains=8,
        kernel="UniformRandomCoordinateHitAndRunKernel",
        target="quadratic",
    ),
    "cpp-nrpt-gaussian": Case(
        "cpp-nrpt-gaussian", batch=4096, chains=8, target="gaussian"
    ),
    "py-small": Case("regular", batch=1),
    "py-regular": Case("regular"),
    "py-thin16": Case("regular", thinning=16),
    "py-independent": Case("regular", batch=8192, chains=4),
    "py-csmmala": Case(
        "regular", dimension=16, batch=2048, kernel="CSmMALAKernel", target="gaussian"
    ),
    "py-gaussian": Case("regular", batch=32768, target="gaussian"),
    "py-gaussian-traced": Case("regular", batch=32768, target="gaussian", traced=True),
    "py-independent-gaussian": Case("regular", batch=8192, chains=4, target="gaussian"),
    "py-independent-traced": Case(
        "regular", batch=8192, chains=4, target="gaussian", traced=True
    ),
    "nrpt-serial": Case("nrpt", batch=4096, chains=8, target="gaussian"),
    "nrpt-parallel": Case("nrpt", batch=4096, chains=8, target="gaussian"),
    "nrpt-small": Case("nrpt", batch=1, chains=8, target="gaussian"),
    "nrpt-traced": Case("nrpt", batch=4096, chains=8, target="gaussian", traced=True),
    "nrpt-thin16": Case("nrpt", batch=1024, thinning=16, chains=4, target="gaussian"),
    "flux-nrpt": Case(
        "nrpt",
        dimension=27,
        batch=8192,
        chains=8,
        kernel="UniformHitAndRunKernel",
        target="flux",
        traced=True,
    ),
}
AFTER_ONLY = {"py-gaussian-traced", "py-independent-traced"}


def bind_flux(case, seed):
    import _hopsy_core as core

    benchmarks = Path(__file__).resolve().parent.parent
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(core.__file__).parent / "matplotlib")
    )
    sys.path.insert(0, str(benchmarks))
    from flux_benchmark_common import bind_tempering, interior_start, load_model

    args = SimpleNamespace(
        fml=benchmarks / "Ecoli_NatProt_GOX.fml",
        configuration="default",
        unbounded_flux_caps=None,
    )
    simulator, matrix, bound, _ = load_model(args)
    if matrix.shape[1] != case.dimension:
        raise ValueError("Unexpected FluxML dimension")
    ladder = core.TemperatureLadder([0, 0.005, 0.02, 0.06, 0.16, 0.36, 0.65, 1])
    return bind_tempering(
        simulator,
        core.LinearPolytope(matrix, bound),
        None,
        interior_start(matrix, bound),
        ladder,
        getattr(core, case.kernel),
        seed,
    )


def python_timing(name, loops, seed):
    import _hopsy_core as core
    import numpy as np

    case = CASES[name]
    before_setup = time.perf_counter()
    if case.target != "flux":
        initial = np.zeros(case.dimension)
        identity = np.eye(case.dimension)
        box = core.LinearPolytope(
            np.vstack((identity, -identity)), np.full(2 * case.dimension, 3.0)
        )
        target = (
            core.UniformTarget()
            if case.target == "uniform"
            else core.GaussianTarget(initial, identity)
        )
        problem = core.Problem(box, target)
        kernels = [getattr(core, case.kernel)(initial) for _ in range(case.chains)]
        rngs = [core.RandomNumberGenerator(seed, stream=i) for i in range(case.chains)]
    pool = None
    if case.mode == "nrpt":
        if case.target == "flux":
            sampler = bind_flux(case, seed)
        else:
            sampler = core.bind_parallel_tempering(
                problem,
                core.TemperatureLadder.linear(case.chains),
                kernels,
                [initial] * case.chains,
                rngs,
                core.RandomNumberGenerator(seed, stream=case.chains),
            )
        if case.traced:

            def draw():
                return sampler.draw_traced(
                    case.batch, case.thinning, case.thinning, parallel=True
                )

        elif name == "nrpt-serial":

            def draw():
                return (sampler.draw(case.batch, case.thinning, case.thinning),)

        else:

            def draw():
                return (
                    sampler.draw_parallel(case.batch, case.thinning, case.thinning),
                )

    else:
        chains = [core.bind_chain(problem, kernel, initial) for kernel in kernels]
        draws = [chain.draw_traced if case.traced else chain.draw for chain in chains]

        def draw_one(i):
            result = draws[i](rngs[i], case.batch, case.thinning)
            return result if case.traced else (result,)

        if case.chains > 1:
            pool = ThreadPoolExecutor(case.chains)

            def draw():
                return tuple(
                    array
                    for group in pool.map(draw_one, range(case.chains))
                    for array in group
                )

        else:

            def draw():
                return draw_one(0)

    setup = time.perf_counter() - before_setup
    if case.target == "flux":
        sampler.draw_parallel(1024, case.thinning, case.thinning)
    elif case.mode == "nrpt":
        sampler.draw(512, case.thinning, case.thinning)
        draw()
    else:
        for chain, rng in zip(chains, rngs):
            chain.draw(rng, 4096)
        draw()
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    before = time.perf_counter()
    for _ in range(loops):
        output = draw()
    seconds = time.perf_counter() - before
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if pool:
        pool.shutdown()
    digest = hashlib.sha256()
    for array in output:
        digest.update(array.tobytes())
    return {
        "seconds": seconds,
        "cpu_seconds": usage.ru_utime
        + usage.ru_stime
        - start_usage.ru_utime
        - start_usage.ru_stime,
        "context_switches": usage.ru_nvcsw
        + usage.ru_nivcsw
        - start_usage.ru_nvcsw
        - start_usage.ru_nivcsw,
        "setup_seconds": setup,
        "digest": digest.hexdigest(),
    }


def run_timing(path, name, loops, seed, cpus):
    case = CASES[name]
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(path / "build" / "bindings"),
    )
    if case.mode.startswith("cpp-"):
        cmd = [
            str(path / "native"),
            case.mode[4:],
            str(case.dimension),
            str(case.batch),
            str(case.thinning),
            str(loops),
            str(seed),
        ]
    else:
        cmd = [sys.executable, __file__, "once", name, str(loops), str(seed)]
    return json.loads(subprocess.check_output(["taskset", "-c", cpus, *cmd], env=env))


def record(args):
    variants = dict(
        (name, Path(path).resolve())
        for name, path in (variant.split("=", 1) for variant in args.variant)
    )
    names = args.cases or [name for name in CASES if name != "flux-nrpt"]
    rng = random.Random(815)
    with args.output.open("x") as output:

        def save(row):
            print(json.dumps(row), file=output, flush=True)

        save(
            {
                "kind": "metadata",
                "variants": {k: str(v) for k, v in variants.items()},
                "cpus": args.cpus,
                "minimum_seconds": args.seconds,
                "reps": args.reps,
                "cases": {name: vars(CASES[name]) for name in names},
            }
        )
        for name in names:
            active = {
                k: v
                for k, v in variants.items()
                if name not in AFTER_ONLY or k != "baseline"
            }
            if not active:
                continue
            loops = 1
            while True:
                calibration = run_timing(
                    next(iter(active.values())), name, loops, 42, args.cpus
                )
                if calibration["seconds"] >= 0.15:
                    break
                loops *= 8
            loops = max(1, math.ceil(loops * args.seconds / calibration["seconds"]))
            print(f"{name}: {loops} batches per repetition", flush=True)
            for i in range(args.reps):
                order = list(active)
                if i % 2:
                    order.reverse()
                if len(order) > 2:
                    order = order[i % len(order) :] + order[: i % len(order)]
                digests = set()
                for variant in order:
                    data = run_timing(active[variant], name, loops, 1100 + i, args.cpus)
                    save(
                        dict(
                            data,
                            kind="timing",
                            case=name,
                            variant=variant,
                            repetition=i,
                            seed=1100 + i,
                            loops=loops,
                            transitions=CASES[name].steps_per_batch * loops,
                        )
                    )
                    digests.add(data["digest"])
                if len(digests) != 1:
                    raise RuntimeError(f"Seeded output mismatch in {name} pair {i}")
                # Vary the gap without favoring either variant in the pair.
                time.sleep(rng.uniform(0.01, 0.05))


def summarize(args):
    import numpy as np

    rows = [
        json.loads(line)
        for path in args.input
        for line in path.read_text().splitlines()
    ]
    groups = {}
    for row in rows:
        if row["kind"] == "timing":
            groups.setdefault(row["case"], {}).setdefault(row["variant"], {})[
                row["repetition"]
            ] = row
    rng = np.random.default_rng(611)
    print(
        "case,variant,reps,Mtransitions/s,CV_pct,delta_pct,CI_low_pct,CI_high_pct,setup_us,cpu_per_wall"
    )
    for name, versions in groups.items():
        reference = versions.get(args.reference, {})
        for variant, timings in versions.items():
            rates = [r["transitions"] / r["seconds"] for r in timings.values()]
            paired = [
                math.log(
                    r["transitions"]
                    / r["seconds"]
                    / (reference[i]["transitions"] / reference[i]["seconds"])
                )
                for i, r in timings.items()
                if i in reference
            ]
            effect = [""] * 3
            if paired and variant != args.reference:
                samples = rng.choice(
                    paired, size=(20000, len(paired)), replace=True
                ).mean(axis=1)
                low, high = np.quantile(np.expm1(samples) * 100, [0.025, 0.975])
                effect = [
                    f"{x:.2f}" for x in (math.expm1(np.mean(paired)) * 100, low, high)
                ]
            setup = statistics.median(r["setup_seconds"] for r in timings.values())
            cpu = statistics.median(
                r["cpu_seconds"] / r["seconds"] for r in timings.values()
            )
            print(
                ",".join(
                    [
                        name,
                        variant,
                        str(len(rates)),
                        f"{statistics.geometric_mean(rates) / 1e6:.5f}",
                        f"{100 * np.std(rates) / np.mean(rates):.2f}",
                        *effect,
                        f"{setup * 1e6:.1f}",
                        f"{cpu:.3f}",
                    ]
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    executor = commands.add_parser("record")
    executor.add_argument(
        "--variant", action="append", required=True, metavar="NAME=PATH"
    )
    executor.add_argument("--output", type=Path, required=True)
    executor.add_argument("--cases", nargs="+", choices=CASES)
    executor.add_argument("--cpus", required=True)
    executor.add_argument(
        "--seconds", type=float, default=2.5, help="target seconds per measured arm"
    )
    executor.add_argument("--reps", type=int, default=10)
    worker = commands.add_parser("once")
    worker.add_argument("case", choices=CASES)
    worker.add_argument("loops", type=int)
    worker.add_argument("seed", type=int)
    analysis = commands.add_parser("summarize")
    analysis.add_argument("input", type=Path, nargs="+")
    analysis.add_argument("--reference", default="baseline")
    args = parser.parse_args()
    if args.command == "record":
        record(args)
    elif args.command == "once":
        print(json.dumps(python_timing(args.case, args.loops, args.seed)))
    else:
        summarize(args)


if __name__ == "__main__":
    main()
