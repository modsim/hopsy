#!/usr/bin/env python3
"""Compare rounded Hit-and-Run and mixed NRPT with independent flux chains."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import _hopsy_core as core
import numpy as np
import x3cflux
from benchmark_ecoli_nrpt import (
    cap_unbounded_fluxes,
    interior_start,
    plot_flux_posteriors,
    replica_diagnostics,
    round_geometry,
    summarize,
)

MODES = ("standalone", "nrpt_parallel", "nrpt_parallel_rounded_reference")
ESS_TARGETS = {"full": 100, "first_half": 50, "last_half": 50}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_replicate(args):
    replicate_start = perf_counter()
    simulator = x3cflux.create_simulator_from_fml(str(args.fml), args.configuration)
    space = simulator.parameter_space
    A = np.asarray(space.inequality_system.matrix)
    b = np.asarray(space.inequality_system.bound).reshape(-1)
    added_bounds = []
    if args.unbounded_flux_caps is not None:
        A, b, added_bounds = cap_unbounded_fluxes(
            A, b, space.free_parameter_names, args.unbounded_flux_caps
        )
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

    def make_target():
        return core.CapsuleTarget(simulator.make_hopsy_target())

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
            core.Problem(rounded, make_target(), coordinates),
            core.UniformRandomCoordinateHitAndRunKernel(rounded_start),
            rounded_start,
        )
        rng = core.RandomNumberGenerator(args.seed, 0)
        start = perf_counter()
        chain.draw(rng, args.warmup)
        warmup_seconds = perf_counter() - start
        start = perf_counter()
        samples = chain.draw(rng, args.num_samples)
        save(
            "standalone",
            samples,
            summarize(
                "standalone", samples, perf_counter() - start, args.num_samples, A, b
            ),
            warmup_seconds,
        )
        del chain, samples

    for mode in modes:
        if mode == "standalone":
            continue
        mixed = mode == "nrpt_parallel_rounded_reference"
        point = initial if mixed else rounded_start
        targets = [make_target() for _ in args.betas]
        problem = core.Problem(
            geometry if mixed else rounded, targets[0], None if mixed else coordinates
        )
        kernels = (
            [core.CSmMALAKernel(point, step_size=args.step_size) for _ in targets]
            if mixed
            else [core.UniformRandomCoordinateHitAndRunKernel(point) for _ in targets]
        )
        sampler = core.bind_parallel_tempering(
            problem,
            ladder,
            kernels,
            [point] * len(targets),
            [core.RandomNumberGenerator(args.seed, i) for i in range(len(targets))],
            core.RandomNumberGenerator(args.seed, len(targets)),
            replica_targets=targets,
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
        start = perf_counter()
        sampler.draw_parallel(args.warmup, workers=args.workers, steps_per_slot=steps)
        warmup_seconds = perf_counter() - start
        sampler.reset_statistics()
        draws, ids, densities = [], [], []
        timed = 0.0
        for offset in range(0, args.num_samples, 2000):
            count = min(2000, args.num_samples - offset)
            start = perf_counter()
            samples, batch_ids, batch_densities = sampler.draw_traced(
                count,
                parallel=True,
                workers=args.workers,
                steps_per_slot=steps,
            )
            timed += perf_counter() - start
            if ids:
                np.testing.assert_array_equal(ids[-1][-1], batch_ids[:, 0])
            draws.append(samples)
            cut = int(offset > 0)
            ids.append(batch_ids.T[cut:])
            densities.append(batch_densities.T[cut:])
            print(
                f"seed {args.seed} {mode} {offset + count} draws, "
                f"{sum(sampler.round_trips)} trips",
                flush=True,
            )
        samples, ids, densities = map(
            np.concatenate, (draws, ids, densities), (1, 0, 0)
        )
        result = summarize(mode, samples, timed, args.num_samples * sum(steps), A, b)
        exchanges = [
            {"attempts": e.attempts, "accepts": e.accepts}
            for e in sampler.exchange_statistics
        ]
        result.update(
            workers=args.workers,
            steps_per_slot=steps,
            exchange_statistics=exchanges,
            global_communication_barrier=sum(sampler.mean_exchange_rejections()),
            round_trips=sampler.round_trips,
            total_round_trips=sum(sampler.round_trips),
            replica_diagnostics=replica_diagnostics(
                ids, densities, exchanges, sampler.round_trips
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


def aggregate(args):
    import arviz

    reports = [json.loads((p / "report.json").read_text()) for p in args.replicates]
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
    simulator = x3cflux.create_simulator_from_fml(
        reference["fml"], reference["configuration"]
    )
    space = simulator.parameter_space
    names = space.parameter_names
    indices = [
        names.index(name)
        for name in x3cflux.get_non_constant_names(simulator)
        if name.endswith((".n", ".x"))
    ]
    plots = {}
    modes = {}
    for mode in reference["results"]:
        draws = np.stack([np.load(p / f"{mode}.npy") for p in args.replicates])
        pooled = draws.transpose(1, 0, 2).reshape(draws.shape[1], -1)
        free_rhat = rhat_values(arviz, draws, space.free_parameter_names)
        tail_rhat = rhat_values(
            arviz, draws[..., draws.shape[-1] // 2 :], space.free_parameter_names
        )
        fluxes = np.empty((len(reports), len(indices), draws.shape[-1]))
        for out, chain in zip(fluxes, draws):
            for draw in range(chain.shape[1]):
                out[:, draw] = space.compute_parameters(chain[:, draw])[indices]
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
        plots[("csmmala" if mode == MODES[-1] else "coordinate-hit-and-run", mode)] = (
            pooled
        )
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
    write_json(
        args.output / "summary.json",
        {
            "rhat_method": f"ArviZ {arviz.__version__} rank-normalized and folded split R-hat",
            "ess_method": f"ArviZ {arviz.__version__} bulk ESS",
            "per_chain_ess_targets": ESS_TARGETS,
            "replicate_elapsed_seconds_total": (
                None if None in durations else sum(durations)
            ),
            "modes": modes,
            "replicates": reports,
            "posterior_plots": plot_flux_posteriors(
                simulator,
                plots,
                args.output,
                reference_max_beta=reference["reference_max_beta"],
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    run = commands.add_parser(
        "run", help="run one or more sampling modes for one replicate"
    )
    run.set_defaults(action=run_replicate)
    run.add_argument("--fml", type=Path, required=True)
    run.add_argument("--configuration", default="default")
    run.add_argument("--unbounded-flux-caps", type=float, nargs=2)
    run.add_argument("--betas", type=float, nargs="+", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--fit-weight", type=float, required=True)
    run.add_argument("--num-samples", type=int, default=20_000)
    run.add_argument("--warmup", type=int, default=2_000)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--step-size", type=float, default=0.15)
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
    summary.add_argument("replicates", type=Path, nargs="+")
    args = parser.parse_args()
    if args.action is run_replicate:
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
        if args.unbounded_flux_caps is not None and any(
            not np.isfinite(cap) or cap <= 0 for cap in args.unbounded_flux_caps
        ):
            parser.error("unbounded flux caps must be finite and positive")
        if not 1 <= args.workers <= len(args.betas):
            parser.error("--workers must be between one and the number of temperatures")
        if args.num_samples < 1000 or args.warmup < 0:
            parser.error(
                "at least 1000 retained draws and nonnegative warm-up are required"
            )
    args.action(args)


if __name__ == "__main__":
    main()
