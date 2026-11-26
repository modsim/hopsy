#!/usr/bin/env python3
"""Benchmark dense Hit-and-Run or CSmMALA with and without NRPT on x3cflux."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import _hopsy_core as core
import numpy as np
from flux_benchmark_common import (
    BATCH_SIZE,
    CSMMALA_STEP_SIZE,
    add_model_arguments,
    bind_tempering,
    exchange_counts,
    interior_start,
    load_model,
    make_target,
    plot_flux_posteriors,
    reconstruct_fluxes,
    replica_diagnostics,
    round_geometry,
    sampling_report,
    validate_model_arguments,
    varying_flux_indices,
)

NUM_SAMPLES = 100_000
KERNELS = ("hit-and-run", "csmmala")
DEFAULT_FML = Path(__file__).with_name("Ecoli_NatProt_GOX.fml").resolve()


def run_kernel(args, kernel, simulator, geometry, coordinates, initial, ladder, A, b):
    samples, traces = {}, {}
    reference_basis = None
    if args.reference_kernel != "same":
        _, reference_coordinates, _ = round_geometry(A, b, initial)
        reference_basis = reference_coordinates.matrix

    default_step = CSMMALA_STEP_SIZE if kernel == "csmmala" else 1.0
    step_size = default_step if args.step_size is None else args.step_size

    def make_kernel(state):
        if kernel == "csmmala":
            return core.CSmMALAKernel(state, step_size=step_size)
        if kernel == "gaussian-hit-and-run":
            return core.GaussianHitAndRunKernel(state, standard_deviation=step_size)
        return core.UniformHitAndRunKernel(state)

    chain = core.bind_chain(
        core.Problem(geometry, make_target(simulator), coordinates),
        make_kernel(initial),
        initial,
    )
    rng = core.RandomNumberGenerator(args.seed, 0)
    chain.draw(rng, args.warmup, thinning=args.standalone_thinning)
    start = perf_counter()
    cold = chain.draw(rng, args.num_samples, thinning=args.standalone_thinning)
    baseline = sampling_report(
        "standalone",
        cold,
        perf_counter() - start,
        args.num_samples * args.standalone_thinning,
        A,
        b,
    )
    baseline["local_steps"] = args.standalone_thinning
    if args.posterior_plots is not None:
        samples[kernel, "standalone"] = cold
    del chain, cold

    def run_tempering(mode, reference_kind):
        replicas = len(ladder)
        rounded_reference = reference_kind == "rounded-coordinate"
        qualifier = "_rounded_reference" if rounded_reference else ""
        print(
            f"{kernel}: {mode} NRPT, reference={reference_kind}",
            file=sys.stderr,
            flush=True,
        )
        sampler = bind_tempering(
            simulator,
            geometry,
            coordinates,
            initial,
            ladder,
            make_kernel,
            args.seed,
            reference_basis=reference_basis if rounded_reference else None,
            reference_max_beta=args.reference_max_beta if rounded_reference else 0.0,
        )
        draw = sampler.draw_parallel if mode == "parallel" else sampler.draw
        steps_per_scan = (replicas - 1) * args.local_steps + args.reference_steps
        draw(
            args.warmup,
            local_steps=args.local_steps,
            reference_steps=args.reference_steps,
        )
        sampler.reset_statistics()
        start = perf_counter()
        if args.replica_trace is None:
            cold = draw(args.num_samples, args.local_steps, args.reference_steps)
        else:
            cold, replica_ids, log_densities = sampler.draw_traced(
                args.num_samples,
                args.local_steps,
                args.reference_steps,
                parallel=mode == "parallel",
            )
        seconds = perf_counter() - start
        tempered = sampling_report(
            f"nrpt_{mode}{qualifier}",
            cold,
            seconds,
            args.num_samples * steps_per_scan,
            A,
            b,
        )
        if args.posterior_plots is not None:
            samples[kernel, tempered["mode"]] = cold
        rejections = sampler.mean_exchange_rejections()
        trips = sampler.round_trips
        exchanges = exchange_counts(sampler)
        tempered.update(
            workers=replicas if mode == "parallel" else 1,
            reference_kernel=reference_kind,
            reference_max_beta=args.reference_max_beta if rounded_reference else 0.0,
            local_steps=args.local_steps,
            reference_steps=args.reference_steps,
            inverse_temperatures=ladder.inverse_temperatures,
            exchange_statistics=exchanges,
            mean_exchange_rejections=rejections,
            global_communication_barrier=sum(rejections),
            round_trips=trips,
            total_round_trips=sum(trips),
            round_trips_per_second=sum(trips) / seconds,
        )
        if args.replica_trace is not None:
            tempered["replica_diagnostics"] = replica_diagnostics(
                replica_ids.T, log_densities.T, exchanges, trips
            )
            prefix = (
                f"{kernel.replace('-', '_')}_{mode}" if args.kernel == "both" else mode
            )
            prefix += qualifier
            traces[f"{prefix}_replica_ids"] = replica_ids.T
            traces[f"{prefix}_log_densities"] = log_densities.T
        return tempered

    modes = ("serial", "parallel") if args.nrpt_mode == "both" else (args.nrpt_mode,)
    references = (
        ("same", "rounded-coordinate")
        if args.reference_kernel == "both"
        else (args.reference_kernel,)
    )
    results = [
        baseline,
        *(run_tempering(mode, ref) for mode in modes for ref in references),
    ]
    report = {
        "fml": str(args.fml),
        "configuration": args.configuration,
        "kernel": kernel,
        "dimension": A.shape[1],
        "seed": args.seed,
        "warmup": args.warmup,
        "results": results,
    }
    if args.round:
        report.update(rounded=True, sampling_dimension=initial.size)
    if kernel != "hit-and-run":
        report["step_size"] = step_size
    return report, samples, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, default_fml=DEFAULT_FML)
    parser.add_argument(
        "--kernel",
        choices=(*KERNELS, "gaussian-hit-and-run", "both"),
        default="hit-and-run",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        help=f"proposal scale (CSmMALA default: {CSMMALA_STEP_SIZE}; Gaussian Hit-and-Run default: 1)",
    )
    parser.add_argument(
        "--round", action="store_true", help="round the constraints with legacy hopsy"
    )
    parser.add_argument(
        "--reference-kernel",
        choices=("same", "rounded-coordinate", "both"),
        default="same",
        help="use rounded random-coordinate Hit-and-Run in the hottest slots for CSmMALA",
    )
    parser.add_argument(
        "--reference-max-beta",
        type=float,
        default=0.0,
        help="largest beta using the rounded reference kernel (default: 0)",
    )
    ladder_args = parser.add_mutually_exclusive_group()
    ladder_args.add_argument("--num-replicas", type=int, default=8)
    ladder_args.add_argument("--betas", type=float, nargs="+")
    parser.add_argument(
        "--nrpt-mode", choices=("parallel", "serial", "both"), default="parallel"
    )
    parser.add_argument(
        "--replica-trace",
        type=Path,
        help="save replica IDs and log densities by scan as an NPZ",
    )
    parser.add_argument(
        "--posterior-plots",
        type=Path,
        metavar="DIR",
        help="save net and exchange flux SVGs",
    )
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument(
        "--local-steps",
        type=int,
        default=1,
        help="local steps per positive beta per scan",
    )
    parser.add_argument(
        "--reference-steps",
        type=int,
        default=1,
        help="local steps at beta zero per scan",
    )
    parser.add_argument(
        "--standalone-thinning",
        type=int,
        default=1,
        help="standalone steps per retained draw",
    )
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    validate_model_arguments(parser, args)
    if not 0 <= args.reference_max_beta <= 1:
        parser.error("--reference-max-beta must be in [0, 1]")
    if args.reference_kernel == "same" and args.reference_max_beta:
        parser.error("--reference-max-beta requires the rounded reference kernel")
    if args.reference_kernel != "same" and (args.kernel != "csmmala" or args.round):
        parser.error("a rounded reference requires --kernel csmmala without --round")
    if args.step_size is not None:
        if args.kernel == "hit-and-run":
            parser.error("--step-size requires csmmala, gaussian-hit-and-run, or both")
        if not np.isfinite(args.step_size) or args.step_size <= 0:
            parser.error("--step-size must be finite and positive")
    if args.num_replicas < 2 or args.warmup < 0 or args.seed < 0:
        parser.error(
            "at least two replicas and nonnegative warm-up and seed are required"
        )
    if args.local_steps < 1 or args.reference_steps < 1:
        parser.error("local and reference steps must be positive")
    if args.standalone_thinning < 1:
        parser.error("standalone thinning must be positive")
    if args.num_samples < 2 * BATCH_SIZE:
        parser.error(
            f"at least {2 * BATCH_SIZE} samples are required for the batch-means ESS"
        )
    try:
        ladder = (
            core.TemperatureLadder(args.betas)
            if args.betas is not None
            else core.TemperatureLadder.linear(args.num_replicas)
        )
    except ValueError as error:
        parser.error(str(error))

    simulator, A, b, added_bounds = load_model(args)
    initial = interior_start(A, b)
    if args.round:
        geometry, coordinates, initial = round_geometry(A, b, initial)
    else:
        geometry, coordinates = core.LinearPolytope(A, b), None
    runs = []
    samples, traces = {}, {}
    kernels = KERNELS if args.kernel == "both" else (args.kernel,)
    for kernel in kernels:
        run, kernel_samples, kernel_traces = run_kernel(
            args, kernel, simulator, geometry, coordinates, initial, ladder, A, b
        )
        runs.append(run)
        samples.update(kernel_samples)
        traces.update(kernel_traces)
    report = {"kernel": "both", "runs": runs} if args.kernel == "both" else runs[0]
    if added_bounds:
        report["added_flux_bounds"] = added_bounds
    if args.posterior_plots is not None:
        indices = varying_flux_indices(simulator)
        fluxes = {
            series: reconstruct_fluxes(
                simulator.parameter_space, indices, draws[None, ...]
            )
            for series, draws in samples.items()
        }
        report["posterior_plots"] = plot_flux_posteriors(
            simulator,
            fluxes,
            next(iter(samples.values()))[:, 0],
            args.posterior_plots,
            reference_max_beta=args.reference_max_beta,
        )
    if args.replica_trace is not None:
        path = args.replica_trace
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        np.savez(path, inverse_temperatures=ladder.inverse_temperatures, **traces)
        report["replica_trace"] = str(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
