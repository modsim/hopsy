#!/usr/bin/env python3
"""Compare rounded Hit-and-Run and mixed NRPT with independent flux chains."""

import argparse
import json
import re
from functools import partial
from pathlib import Path
from time import perf_counter

import _hopsy_core as core
import numpy as np
import x3cflux
from flux_benchmark_common import (
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

MODES = ("standalone", "nrpt_parallel", "nrpt_parallel_rounded_reference")
ESS_TARGETS = {"full": 100, "first_half": 50, "last_half": 50}
TRACE_BATCH_SIZE = 2_000


def write_json(path, value):
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def draw_traces(sampler, args, mode, steps, dimension):
    """Keep one cold sample per scan, and a continuous post-swap trace."""
    count = args.num_samples
    samples = np.empty((dimension, count), order="F")
    ids = np.empty((count + 1, len(steps)), dtype=np.intp)
    densities = np.empty(ids.shape)
    seconds = 0.0
    for offset in range(0, count, TRACE_BATCH_SIZE):
        end = min(offset + TRACE_BATCH_SIZE, count)
        start = perf_counter()
        draws, batch_ids, batch_densities = sampler.draw_traced(
            end - offset, parallel=True, workers=args.workers, steps_per_slot=steps
        )
        seconds += perf_counter() - start
        # Native batches include their starting state. The previous batch
        # already wrote that snapshot, so only append the new scans.
        if offset == 0:
            ids[0], densities[0] = batch_ids[:, 0], batch_densities[:, 0]
        else:
            np.testing.assert_array_equal(ids[offset], batch_ids[:, 0])
        samples[:, offset:end] = draws
        ids[offset + 1 : end + 1] = batch_ids[:, 1:].T
        densities[offset + 1 : end + 1] = batch_densities[:, 1:].T
        print(
            f"seed {args.seed} {mode} {end} draws, {sum(sampler.round_trips)} trips",
            flush=True,
        )
    return samples, ids, densities, seconds


def run_replicate(args):
    replicate_start = perf_counter()
    simulator, A, b, added_bounds = load_model(args)
    center = interior_start(A, b)
    initial = center + args.fit_weight * (simulator.initial_parameters - center)
    if np.min(b - A @ initial) <= 0:
        raise ValueError("Interpolated initial point must be strictly feasible")
    rounded, coordinates, _ = round_geometry(A, b, center)
    basis, shift = coordinates.matrix, coordinates.shift
    rounded_start = np.linalg.solve(basis, initial - shift)
    np.testing.assert_allclose(
        coordinates.to_target(rounded_start), initial, atol=1e-10
    )
    geometry = core.LinearPolytope(A, b)
    ladder = core.TemperatureLadder(args.betas)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}

    def save(mode, samples, result, warmup_seconds):
        np.save(args.output / f"{mode}.npy", samples)
        result["warmup_seconds"] = warmup_seconds
        result["sampling_seconds"] = warmup_seconds + result["seconds"]
        results[mode] = result
        print(f"seed {args.seed} {mode} complete", flush=True)

    modes = MODES
    if args.method is not None:
        modes = (args.method,)
    elif args.mixed_only:
        modes = MODES[-1:]

    if "standalone" in modes:
        print(f"seed {args.seed} standalone starts", flush=True)
        chain = core.bind_chain(
            core.Problem(rounded, make_target(simulator), coordinates),
            core.UniformRandomCoordinateHitAndRunKernel(rounded_start),
            rounded_start,
        )
        rng = core.RandomNumberGenerator(args.seed, 0)
        start = perf_counter()
        chain.draw(rng, args.warmup)
        warmup_seconds = perf_counter() - start
        start = perf_counter()
        samples = chain.draw(rng, args.num_samples)
        seconds = perf_counter() - start
        save(
            "standalone",
            samples,
            sampling_report("standalone", samples, seconds, args.num_samples, A, b)
            | {"sampling_wall_seconds": warmup_seconds + seconds},
            warmup_seconds,
        )
        del chain, samples

    for mode in modes:
        if mode == "standalone":
            continue
        mixed = mode == "nrpt_parallel_rounded_reference"
        point = initial if mixed else rounded_start
        make_kernel = (
            partial(core.CSmMALAKernel, step_size=args.step_size)
            if mixed
            else core.UniformRandomCoordinateHitAndRunKernel
        )
        sampler = bind_tempering(
            simulator,
            geometry if mixed else rounded,
            None if mixed else coordinates,
            point,
            ladder,
            make_kernel,
            args.seed,
            reference_basis=basis if mixed else None,
            reference_max_beta=args.reference_max_beta if mixed else 0.0,
        )
        steps = [
            args.hit_and_run_steps if mixed and beta <= args.reference_max_beta else 1
            for beta in args.betas
        ]
        print(
            f"seed {args.seed} {mode} warm-up starts with step counts {steps}",
            flush=True,
        )
        sampling_start = perf_counter()
        sampler.draw_parallel(args.warmup, workers=args.workers, steps_per_slot=steps)
        warmup_seconds = perf_counter() - sampling_start
        sampler.reset_statistics()
        samples, ids, densities, seconds = draw_traces(
            sampler, args, mode, steps, initial.size
        )
        sampling_wall_seconds = perf_counter() - sampling_start
        result = sampling_report(
            mode, samples, seconds, args.num_samples * sum(steps), A, b
        )
        exchanges = exchange_counts(sampler)
        round_trips = sampler.round_trips
        result.update(
            workers=args.workers,
            sampling_wall_seconds=sampling_wall_seconds,
            steps_per_slot=steps,
            exchange_statistics=exchanges,
            global_communication_barrier=sum(sampler.mean_exchange_rejections()),
            round_trips=round_trips,
            total_round_trips=sum(round_trips),
            replica_diagnostics=replica_diagnostics(
                ids, densities, exchanges, round_trips
            ),
        )
        save(mode, samples, result, warmup_seconds)
        np.savez(args.output / f"{mode}_replicas.npz", ids=ids, log_densities=densities)

    write_json(
        args.output / "report.json",
        {
            "fml": str(args.fml.resolve()),
            "configuration": args.configuration,
            "seed": args.seed,
            "fit_weight": args.fit_weight,
            "initial_state": initial.tolist(),
            "warmup": args.warmup,
            "draws": args.num_samples,
            "inverse_temperatures": args.betas,
            "step_size": args.step_size,
            "reference_max_beta": args.reference_max_beta,
            "hit_and_run_steps": args.hit_and_run_steps,
            "elapsed_seconds": perf_counter() - replicate_start,
            "added_flux_bounds": added_bounds,
            "results": results,
        },
    )


def rhat_values(arviz, chains, names):
    values = {}
    half = chains.shape[-1] // 2
    for name, values_for_flux in zip(names, chains.transpose(1, 0, 2)):
        first, last = values_for_flux[:, :half], values_for_flux[:, -half:]
        if not np.any(np.ptp(first, axis=1)) and not np.any(np.ptp(last, axis=1)):
            # Exactly zero within-chain variance can look nonzero after the
            # floating-point mean subtraction in a variance calculation.
            split_levels = np.r_[first[:, 0], last[:, 0]]
            values[name] = "inf" if np.ptp(split_levels) else None
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            value = float(arviz.rhat(values_for_flux, method="rank"))
        values[name] = (
            "inf" if np.isposinf(value) else value if np.isfinite(value) else None
        )
    return values


def rhat_summary(values):
    ranked = {key: float(value) for key, value in values.items() if value is not None}
    median = float(np.median(list(ranked.values()))) if ranked else None
    maximum = max(ranked.values(), default=None)
    return {
        "median": "inf" if median == np.inf else median,
        "max": "inf" if maximum == np.inf else maximum,
        "worst": max(ranked, key=ranked.get) if ranked else None,
        "below_1_01": sum(value < 1.01 for value in ranked.values()),
        "total": len(values),
        "undefined": len(values) - len(ranked),
        "infinite": sum(value == np.inf for value in ranked.values()),
    }


def bulk_ess_values(arviz, chains, names):
    values = {}
    for name, values_for_flux in zip(names, chains.transpose(1, 0, 2)):
        # A set of frozen chains supplies no information about variation.
        if not np.any(np.ptp(values_for_flux, axis=1)):
            values[name] = 0.0
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ess = float(arviz.ess(values_for_flux, method="bulk"))
        values[name] = ess if np.isfinite(ess) and ess >= 0.0 else None
    return values


def minimum_ess(values):
    return None if None in values.values() else min(values.values(), default=None)


def chain_ess_diagnostics(arviz, chain, names):
    half = chain.shape[-1] // 2
    windows = {
        "full": chain,
        "first_half": chain[:, :half],
        "last_half": chain[:, -half:],
    }
    ess = {
        window: bulk_ess_values(arviz, data[None, ...], names)
        for window, data in windows.items()
    }
    failures = {
        window: [
            name
            for name, value in ess[window].items()
            if value is None or value < limit
        ]
        for window, limit in ESS_TARGETS.items()
    }
    return {
        "bulk_ess": ess,
        "minimum_bulk_ess": {w: minimum_ess(v) for w, v in ess.items()},
        "fluxes_below_target": failures,
    }


def load_reports(replicates):
    reports = [json.loads((p / "report.json").read_text()) for p in replicates]
    reference = reports[0]
    for item in reports[1:]:
        for key in (
            "fml",
            "configuration",
            "warmup",
            "draws",
            "inverse_temperatures",
            "step_size",
            "reference_max_beta",
            "added_flux_bounds",
            "hit_and_run_steps",
        ):
            if item.get(key) != reference.get(key):
                raise ValueError(f"Replicate setting differs for {key}")
        if item["results"].keys() != reference["results"].keys():
            raise ValueError("Replicate methods differ")
    if len({item["seed"] for item in reports}) != len(reports) or len(reports) < 2:
        raise ValueError("R-hat requires at least two independently seeded replicates")
    return reports


def sequential_summary(directories, reports, mode, diagnostics, analysis_seconds):
    replicates = []
    for directory, report in zip(directories, reports, strict=True):
        result = report["results"][mode]
        trips = result["total_round_trips"]
        wall = float((directory / "wall-seconds.txt").read_text())
        replicates.append(
            {
                "seed": report["seed"],
                "wall_seconds": wall,
                "sampling_wall_seconds": result["sampling_wall_seconds"],
                "generated_cold_draws": report["warmup"] + result["cold_samples"],
                "retained_cold_draws": result["cold_samples"],
                "retained_local_transitions": result["local_transitions"],
                "round_trips": trips,
                "round_trips_per_wall_second": trips / wall,
                "round_trips_by_physical_replica": result["round_trips"],
            }
        )
    sampling_wall = sum(r["sampling_wall_seconds"] for r in replicates)
    replicate_wall = sum(r["wall_seconds"] for r in replicates)
    total_wall = replicate_wall + analysis_seconds
    trips = sum(r["round_trips"] for r in replicates)
    ess = diagnostics["flux_pooled_bulk_ess"]
    defined = [value for value in ess.values() if value is not None]
    total = sum(defined) if defined else None
    ess_stats = {
        "minimum": min(defined, default=None),
        "mean": total / len(defined) if defined else None,
        "total": total,
    }
    return {
        "replicates": replicates,
        "sampling_wall_seconds": sampling_wall,
        "replicate_wall_seconds_total": replicate_wall,
        "analysis_wall_seconds": analysis_seconds,
        "total_wall_seconds": total_wall,
        "generated_cold_draws": sum(r["generated_cold_draws"] for r in replicates),
        "retained_cold_draws": sum(r["retained_cold_draws"] for r in replicates),
        "round_trips": trips,
        "round_trips_per_wall_second": trips / total_wall,
        "round_trips_per_sampling_second": trips / sampling_wall,
        "flux_ess_defined": len(defined),
        "varying_fluxes": len(ess),
        "flux_ess": ess_stats,
        "flux_ess_per_sampling_second": {
            name: value / sampling_wall if value is not None else None
            for name, value in ess_stats.items()
        },
        "max_flux_rhat": diagnostics["flux_rhat_summary"]["max"],
    }


def write_sequential_markdown(path, tag, summary, ess_method, rhat_method):
    def display(value):
        if value is None:
            return "undefined"
        if isinstance(value, float):
            return f"{value:,.4g}"
        return f"{value:,}" if isinstance(value, int) else str(value)

    lines = [
        f"# {tag} NRPT benchmark",
        "",
        "Sampling wall covers warm-up and retained draws. Total wall adds replicate setup,"
        " output, and diagnostics. Round trips start after warm-up.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for label, key in (
        ("Total wall (s)", "total_wall_seconds"),
        ("Sampling wall (s)", "sampling_wall_seconds"),
        ("Diagnostics and plots (s)", "analysis_wall_seconds"),
        ("Generated cold draws, including warm-up", "generated_cold_draws"),
        ("Retained cold draws", "retained_cold_draws"),
        ("Round trips", "round_trips"),
        ("Round trips per total wall second", "round_trips_per_wall_second"),
        ("Round trips per sampling second", "round_trips_per_sampling_second"),
        ("Maximum varying-flux R-hat", "max_flux_rhat"),
    ):
        lines.append(f"| {label} | {display(summary[key])} |")
    lines.extend(
        (
            "",
            "| Bulk ESS across varying fluxes | ESS | ESS per sampling second |",
            "|---|---:|---:|",
        )
    )
    for name in ("minimum", "mean", "total"):
        value = display(summary["flux_ess"][name])
        rate = display(summary["flux_ess_per_sampling_second"][name])
        lines.append(f"| {name.capitalize()} | {value} | {rate} |")
    lines.extend(
        (
            "",
            f"Defined ESS for {summary['flux_ess_defined']} of {summary['varying_fluxes']} varying fluxes.",
            "Total ESS sums marginal estimates across fluxes.",
            f"{ess_method}. {rhat_method}. Both use independent retained cold chains.",
            "",
            "| Seed | Wall (s) | Sampling wall (s) | Retained draws | Trips | Trips per wall second | Trips by physical replica |",
            "|---:|---:|---:|---:|---:|---:|---|",
        )
    )
    for replicate in summary["replicates"]:
        cells = [
            display(replicate[key])
            for key in (
                "seed",
                "wall_seconds",
                "sampling_wall_seconds",
                "retained_cold_draws",
                "round_trips",
                "round_trips_per_wall_second",
            )
        ]
        cells.append(str(replicate["round_trips_by_physical_replica"]))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def aggregate(args):
    analysis_start = perf_counter()
    import arviz

    reports = load_reports(args.replicates)
    reference = reports[0]
    if args.sequential_model and (
        len(reference["results"]) != 1 or "standalone" in reference["results"]
    ):
        raise ValueError("Sequential benchmark summaries require one NRPT method")
    simulator = x3cflux.create_simulator_from_fml(
        reference["fml"], reference["configuration"]
    )
    space = simulator.parameter_space
    names = space.parameter_names
    indices = varying_flux_indices(simulator)
    plots = {}
    modes = {}
    first_draw = None
    for mode in reference["results"]:
        draws = np.stack([np.load(p / f"{mode}.npy") for p in args.replicates])
        if first_draw is None:
            first_draw = draws[0, :, 0].copy()
        free_rhat = rhat_values(arviz, draws, space.free_parameter_names)
        tail_rhat = rhat_values(
            arviz, draws[..., draws.shape[-1] // 2 :], space.free_parameter_names
        )
        fluxes = reconstruct_fluxes(space, indices, draws)
        flux_rhat = rhat_values(arviz, fluxes, [names[i] for i in indices])
        free_ess = bulk_ess_values(arviz, draws, space.free_parameter_names)
        flux_ess = bulk_ess_values(arviz, fluxes, [names[i] for i in indices])
        individual_ess = [
            {
                "seed": report["seed"],
                **chain_ess_diagnostics(arviz, chain, space.free_parameter_names),
            }
            for report, chain in zip(reports, draws)
        ]
        failed_ess = {
            name
            for diagnostic in individual_ess
            for names_below in diagnostic["fluxes_below_target"].values()
            for name in names_below
        }
        per_run = [r["results"][mode] for r in reports]
        elapsed = sum(r["seconds"] for r in per_run)
        warmup_seconds = [r.get("warmup_seconds") for r in per_run]
        warmup_total = None if None in warmup_seconds else sum(warmup_seconds)
        total_sampling = None if warmup_total is None else elapsed + warmup_total
        passing = len(free_ess) - len(failed_ess)
        lowest = minimum_ess(free_ess)
        individual_minima = {
            window: minimum_ess(
                {
                    item["seed"]: item["minimum_bulk_ess"][window]
                    for item in individual_ess
                }
            )
            for window in ESS_TARGETS
        }
        modes[mode] = {
            "seconds_total": elapsed,
            "warmup_seconds_total": warmup_total,
            "sampling_seconds_total": total_sampling,
            "min_coordinate_ess_sum": sum(r["min_coordinate_ess"] for r in per_run),
            "min_coordinate_ess_per_second": sum(
                r["min_coordinate_ess"] for r in per_run
            )
            / elapsed,
            "free_pooled_bulk_ess": free_ess,
            "flux_pooled_bulk_ess": flux_ess,
            "min_free_pooled_bulk_ess": lowest,
            "min_free_pooled_bulk_ess_per_second": (
                None if lowest is None else lowest / elapsed
            ),
            "min_flux_pooled_bulk_ess": minimum_ess(flux_ess),
            "free_ess_per_replicate": individual_ess,
            "min_free_bulk_ess_across_chains": individual_minima,
            "free_fluxes_meeting_ess_targets_in_all_chains": passing,
            "round_trips_per_replicate": [r.get("total_round_trips") for r in per_run],
            "free_rhat_summary": rhat_summary(free_rhat),
            "free_tail_rhat_summary": rhat_summary(tail_rhat),
            "flux_rhat_summary": rhat_summary(flux_rhat),
            "free_rhat": free_rhat,
            "free_tail_rhat": tail_rhat,
            "flux_rhat": flux_rhat,
        }
        plot_kernel = "csmmala" if mode == MODES[-1] else "coordinate-hit-and-run"
        plots[plot_kernel, mode] = fluxes
        timing = (
            f"{elapsed:.1f}s retained"
            if total_sampling is None
            else f"{total_sampling:.1f}s including warm-up"
        )
        lowest_text = "undefined" if lowest is None else f"{lowest:.1f}"
        max_rhat = modes[mode]["free_rhat_summary"]["max"]
        max_rhat_text = (
            f"{max_rhat:.3f}"
            if isinstance(max_rhat, float)
            else max_rhat or "undefined"
        )
        checkpoints = "/".join(
            "undefined" if value is None else f"{value:.1f}"
            for value in individual_minima.values()
        )
        print(
            f"{mode}: {timing}, maximum free-flux R-hat {max_rhat_text}, "
            f"minimum pooled free-flux ESS {lowest_text}, "
            f"minimum ESS full/first/last {checkpoints}, "
            f"{passing}/{len(free_ess)} free fluxes reach the ESS targets in every chain",
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    durations = [report.get("elapsed_seconds") for report in reports]
    plots = plot_flux_posteriors(
        simulator,
        plots,
        first_draw,
        args.output,
        reference_max_beta=reference["reference_max_beta"],
        replicate_seeds=[r["seed"] for r in reports] if args.sequential_model else None,
        model_tag=args.sequential_model,
    )
    summary = {
        "rhat_method": f"ArviZ {arviz.__version__} rank-normalized and folded split R-hat",
        "ess_method": f"ArviZ {arviz.__version__} bulk ESS",
        "per_chain_ess_targets": ESS_TARGETS,
        "replicate_elapsed_seconds_total": (
            None if None in durations else sum(durations)
        ),
        "modes": modes,
        "replicates": reports,
        "posterior_plots": plots,
    }
    if args.sequential_model:
        mode, diagnostics = next(iter(modes.items()))
        summary["sequential_run"] = sequential_summary(
            args.replicates, reports, mode, diagnostics, perf_counter() - analysis_start
        )
        write_sequential_markdown(
            args.output / "summary.md",
            args.sequential_model,
            summary["sequential_run"],
            summary["ess_method"],
            summary["rhat_method"],
        )
    write_json(args.output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    run = commands.add_parser(
        "run", help="run one or more sampling modes for one replicate"
    )
    run.set_defaults(action=run_replicate)
    add_model_arguments(run)
    run.add_argument("--betas", type=float, nargs="+", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--fit-weight", type=float, required=True)
    run.add_argument("--num-samples", type=int, default=20_000)
    run.add_argument("--warmup", type=int, default=2_000)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--step-size", type=float, default=CSMMALA_STEP_SIZE)
    run.add_argument("--reference-max-beta", type=float, default=0.5)
    run.add_argument(
        "--hit-and-run-steps",
        type=int,
        default=1,
        help="local steps per rounded Hit-and-Run slot in mixed NRPT",
    )
    selection = run.add_mutually_exclusive_group()
    selection.add_argument(
        "--method", choices=MODES, help="run only the selected sampling mode"
    )
    selection.add_argument(
        "--mixed-only", action="store_true", help="skip the two Hit-and-Run controls"
    )
    run.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser(
        "summarize", help="plot retained draws and compute R-hat"
    )
    summary.set_defaults(action=aggregate)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument(
        "--sequential-model",
        metavar="TAG",
        help="report sequential wall times and save model-specific replicate overlays",
    )
    summary.add_argument("replicates", type=Path, nargs="+")
    args = parser.parse_args()
    if (
        args.action is aggregate
        and args.sequential_model
        and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.sequential_model)
    ):
        parser.error("--sequential-model must be a filename-safe model tag")
    if args.action is run_replicate:
        validate_model_arguments(parser, args)
        if args.seed < 0:
            parser.error("--seed must be nonnegative")
        if not 0 <= args.fit_weight < 1:
            parser.error("--fit-weight must be in [0, 1)")
        if not 0 <= args.reference_max_beta <= 1:
            parser.error("--reference-max-beta must be in [0, 1]")
        if args.hit_and_run_steps < 1:
            parser.error("--hit-and-run-steps must be positive")
        if not np.isfinite(args.step_size) or args.step_size <= 0:
            parser.error("--step-size must be finite and positive")
        if not 1 <= args.workers <= len(args.betas):
            parser.error("--workers must be between one and the number of temperatures")
        if args.num_samples < 1000 or args.warmup < 0:
            parser.error(
                "at least 1000 retained draws and nonnegative warm-up are required"
            )
    args.action(args)


if __name__ == "__main__":
    main()
