#!/usr/bin/env python3
"""Run native legacy/new kernels sequentially; FFT ESS is outside timed sampling.

Standard N(0,I_10) on [-3,3]^10; 20k burn-in, no thinning. Outputs individual
repeat metrics and medians. ESS uses the minimum over coordinate Geyer initial
positive, monotone paired autocorrelation estimates. Retained rejections count
as samples. ESS/s = min_coordinate_ESS / timed sampling seconds. Timings include
state storage and exclude setup, warm-up, file I/O, and ESS computation.
"""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


def ess(samples):
    n = samples.shape[0]
    centered = samples - samples.mean(axis=0)
    fft = np.fft.rfft(centered, n=1 << (2 * n - 1).bit_length(), axis=0)
    acov = np.fft.irfft(fft.conj() * fft, axis=0)[:n] / n
    if np.any(acov[0] <= 0):
        raise RuntimeError("zero-variance chain")
    rho = acov / acov[0]
    result = []
    for col in range(samples.shape[1]):
        paired = rho[: 2 * (n // 2), col].reshape(-1, 2).sum(axis=1)
        nonpositive = np.flatnonzero(paired <= 0)
        if len(nonpositive):
            paired = paired[: nonpositive[0]]
        paired = np.minimum.accumulate(paired)
        tau = max(1.0, -1.0 + 2.0 * paired.sum())
        result.append(n / tau)
    return float(min(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path, default=Path("cpp/build/gaussian_family_after")
    )
    parser.add_argument("--samples", type=int, default=300000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("cpp/benchmarks/results/gaussian_family")
    )
    args = parser.parse_args()
    if args.samples < 100 or args.repeats < 1:
        parser.error("samples must be >=100 and repeats positive")
    rows = []
    binary = args.output.with_suffix(".bin")
    modes = ["old_grw", "new_grw", "old_dikin", "new_dikin", "old_csmala", "new_csmala"]
    for repeat in range(args.repeats):
        # Reverse order on alternating repeats to reduce systematic timing bias.
        for mode in modes[:: (-1 if repeat % 2 else 1)]:
            sigma = 0.7 if mode.endswith("grw") else 0.3
            output = subprocess.check_output(
                [
                    str(args.executable),
                    mode,
                    str(42 + repeat),
                    str(args.samples),
                    str(sigma),
                    str(binary),
                ],
                text=True,
            ).strip()
            _, seed, count, scale, rate, acceptance = output.split(",")
            samples = np.fromfile(binary, dtype=np.float64).reshape(-1, 10)
            if (
                samples.shape != (args.samples, 10)
                or not np.isfinite(samples).all()
                or np.max(np.abs(samples)) > 3
            ):
                raise RuntimeError(f"invalid samples: {mode}")
            effective = ess(samples)
            row = dict(
                mode=mode,
                seed=int(seed),
                samples=int(count),
                sigma=float(scale),
                steps_per_second=float(rate),
                acceptance=float(acceptance),
                ess_per_sample=effective / args.samples,
                ess_per_second=effective / args.samples * float(rate),
                max_abs_mean=float(np.max(np.abs(samples.mean(axis=0)))),
                min_second_moment=float(np.min(np.mean(samples * samples, axis=0))),
                max_second_moment=float(np.max(np.mean(samples * samples, axis=0))),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    with args.output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for mode in modes:
        group = [r for r in rows if r["mode"] == mode]
        summary.append(
            dict(
                mode=mode,
                **{
                    key: statistics.median(r[key] for r in group)
                    for key in (
                        "steps_per_second",
                        "ess_per_sample",
                        "ess_per_second",
                        "acceptance",
                    )
                },
            )
        )
    metadata = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        samples=args.samples,
        repeats=args.repeats,
        warmup=20000,
        box_radius=3,
        dimension=10,
        summary=summary,
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    # The raw file is scratch storage; retain metrics rather than a single chain.
    binary.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
