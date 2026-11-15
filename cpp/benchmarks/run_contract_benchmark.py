#!/usr/bin/env python3
"""Paired frozen/direct/bound contract measurements; run pinned with taskset.

Each executable internally reports median-of-five timings with 100k transitions
and 1k warm-up. Repeat executable order alternately to expose timing variation.
Checksums must match exactly: these identity-coordinate changes preserve RNG
consumption and trajectories. No speed assertion is made from noisy timings.
"""

import argparse
import csv
import io
import json
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("cpp/benchmarks/results/contract_paired")
    )
    args = parser.parse_args()
    variants = [
        ("before", "cpp/build/contract_before"),
        ("direct", "cpp/build/contract_direct"),
        ("bound", "cpp/build/contract_bound"),
    ]
    rows = []
    expected = {}
    for repeat in range(args.repeats):
        for name, exe in variants[:: (-1 if repeat % 2 else 1)]:
            output = subprocess.check_output([exe], text=True)
            for row in csv.DictReader(io.StringIO(output)):
                key = (row["kernel"], row["dimension"])
                if key in expected and expected[key] != row["checksum"]:
                    raise RuntimeError(f"changed trajectory: {name}, {key}")
                expected[key] = row["checksum"]
                rows.append(dict(variant=name, repeat=repeat, **row))
            print(f"completed {name}, repeat {repeat}", flush=True)
    with args.output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for (kernel, dimension), checksum in expected.items():
        timings = {
            name: statistics.median(
                float(r["median_ns"])
                for r in rows
                if r["variant"] == name
                and r["kernel"] == kernel
                and r["dimension"] == dimension
            )
            for name, _ in variants
        }
        summary.append(
            dict(
                kernel=kernel,
                dimension=int(dimension),
                median_ns=timings,
                bound_over_before=timings["bound"] / timings["before"],
                checksum=checksum,
            )
        )
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
