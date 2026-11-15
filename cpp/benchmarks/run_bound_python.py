#!/usr/bin/env python3
"""Alternate frozen and direct extension runs; verify exact final-state checksums."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before", type=Path, default=Path("cpp/build/bindings-before")
    )
    parser.add_argument(
        "--after", type=Path, default=Path("cpp/build-python/core/bindings")
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--cpus", default="0,1,2,3")
    parser.add_argument(
        "--output", type=Path, default=Path("cpp/benchmarks/results/bound_python_final")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    os.sched_setaffinity(0, {int(x) for x in args.cpus.split(",")})
    records = {}
    expected = {}
    for repeat in range(args.repeats):
        variants = [("before", args.before), ("after", args.after)]
        if repeat % 2:
            variants.reverse()
        for target in ["gaussian", "uniform"]:
            for api, path in variants:
                output = (
                    args.output.parent
                    / f"{args.output.name}_{target}_{api}_{repeat}.json"
                )
                command = [
                    sys.executable,
                    "cpp/benchmarks/benchmark_bound_python.py",
                    "--api",
                    api,
                    "--target",
                    target,
                    "--output",
                    str(output),
                ]
                if args.parallel:
                    command += [
                        "--steps",
                        "500000",
                        "--thinning",
                        "100",
                        "--only-parallel",
                    ]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(path)
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(command, env=env, stdout=log, check=True)
                for row in json.loads(output.read_text()):
                    key = (target, row["kernel"], row["batch"], row["workers"])
                    if key in expected and expected[key] != row["checksums"]:
                        raise RuntimeError(f"changed seeded trajectory: {key}, {api}")
                    expected[key] = row["checksums"]
                    records.setdefault((api, key), []).extend(row["seconds"])
                print("completed", repeat, target, api, flush=True)
    summary = []
    for (target, kernel, batch, workers), _ in expected.items():
        key = (target, kernel, batch, workers)
        steps = 500000 if args.parallel else 20000
        rates = {
            api: steps * workers / statistics.median(records[(api, key)])
            for api in ["before", "after"]
        }
        summary.append(
            dict(
                target=target,
                kernel=kernel,
                batch=batch,
                workers=workers,
                steps=steps,
                thinning=100 if args.parallel else 1,
                **rates,
                speedup=rates["after"] / rates["before"],
            )
        )
    modules = {
        name: next(path.glob("_hopsy_core*.so"))
        for name, path in [("before", args.before), ("after", args.after)]
    }
    result = dict(
        python=sys.version,
        platform=platform.platform(),
        cpus=args.cpus,
        repeats=args.repeats,
        samples_per_case=3 * args.repeats,
        module_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in modules.items()
        },
        summary=summary,
    )
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
