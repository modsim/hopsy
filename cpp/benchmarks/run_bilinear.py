#!/usr/bin/env python3
"""Paired frozen-linear and current geometry runs; summarize overhead and scaling."""

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
        "--before", type=Path, default=Path("cpp/build/bilinear-before")
    )
    parser.add_argument(
        "--after", type=Path, default=Path("cpp/build-python/core/bindings")
    )
    parser.add_argument("--worker-local", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument(
        "--output", type=Path, default=Path("cpp/benchmarks/results/bilinear_final")
    )
    args = parser.parse_args()
    all_rows = {}
    checksums = {}
    for repeat in range(args.rounds):
        variants = [("before", args.before), ("after", args.after)]
        if repeat % 2:
            variants.reverse()
        for variant, module in variants:
            output = args.output.parent / f"{args.output.name}_{variant}_{repeat}.json"
            command = [
                sys.executable,
                "cpp/benchmarks/benchmark_bilinear.py",
                "--steps",
                str(args.steps),
                "--output",
                str(output),
            ]
            if args.worker_local:
                command.append("--worker-local")
            if variant == "before":
                command.append("--baseline")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(module)
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, check=True)
            for row in json.loads(output.read_text()):
                key = (row["region"], row["dimension"], row["kernel"], row["workers"])
                all_rows.setdefault((variant, key), []).extend(row["seconds"])
                if row["region"] == "linear":
                    if key in checksums and checksums[key] != row["checksums"]:
                        raise RuntimeError(f"changed linear trajectory: {key}")
                    checksums[key] = row["checksums"]
            print("completed", repeat, variant, flush=True)
    rows = []
    for variant, key in all_rows:
        region, n, kernel, workers = key
        rate = args.steps * workers / statistics.median(all_rows[(variant, key)])
        rows.append(
            dict(
                variant=variant,
                region=region,
                dimension=n,
                kernel=kernel,
                workers=workers,
                steps_per_second=rate,
            )
        )
    for row in rows:
        if row["variant"] == "after":
            one = next(
                r["steps_per_second"]
                for r in rows
                if r["variant"] == "after"
                and r["region"] == row["region"]
                and r["dimension"] == row["dimension"]
                and r["kernel"] == row["kernel"]
                and r["workers"] == 1
            )
            row["scaling"] = row["steps_per_second"] / one
            base = next(
                r["steps_per_second"]
                for r in rows
                if r["variant"] == "before"
                and r["dimension"] == row["dimension"]
                and r["kernel"] == row["kernel"]
                and r["workers"] == row["workers"]
            )
            row["relative_to_frozen_linear"] = row["steps_per_second"] / base
    hashes = {
        name: hashlib.sha256(
            next(path.glob("_hopsy_core*.so")).read_bytes()
        ).hexdigest()
        for name, path in [("before", args.before), ("after", args.after)]
    }
    result = dict(
        platform=platform.platform(),
        python=sys.version,
        cpus="0,1,2,3",
        steps=args.steps,
        thinning=20,
        worker_local=args.worker_local,
        repetitions_per_case=args.rounds * 5,
        module_sha256=hashes,
        rows=rows,
    )
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
