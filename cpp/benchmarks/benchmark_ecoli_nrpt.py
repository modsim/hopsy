#!/usr/bin/env python3
"""Benchmark dense Hit-and-Run or CSmMALA with and without NRPT on x3cflux."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import _hopsy_core as core
import numpy as np
import x3cflux
from scipy.optimize import linprog

NUM_SAMPLES = 100_000
BATCH_SIZE = 500
CSMMALA_STEP_SIZE = 0.15
KERNELS = ("hit-and-run", "csmmala")
DEFAULT_FML = (
    Path(__file__).resolve().parents[1] / "benchmark-python" / "Ecoli_NatProt_GOX.fml"
)


def unbounded_directions(A, b):
    bounds = [(None, None)] * A.shape[1]
    for i, axis in enumerate(np.eye(A.shape[1])):
        for sign in (-1, 1):
            extent = linprog(
                -sign * axis, A_ub=A, b_ub=b, bounds=bounds, method="highs"
            )
            if extent.status == 3:
                yield i, sign, sign * axis
            elif not extent.success:
                raise ValueError(f"could not check the FluxML bounds: {extent.message}")


def cap_unbounded_fluxes(A, b, names, caps):
    """Apply the requested caps only to unbounded sides of free fluxes."""
    net_cap, exchange_cap = caps
    rows, added_bounds = [], []
    for i, sign, row in unbounded_directions(A, b):
        kind = names[i][-2:]
        if kind not in (".n", ".x"):
            raise ValueError(f"no cap for unbounded parameter {names[i]}")
        cap = net_cap if kind == ".n" else exchange_cap
        rows.append(row)
        added_bounds.append(
            {
                "flux": names[i],
                "side": "upper" if sign == 1 else "lower",
                "value": sign * cap,
            }
        )
    if rows:
        A = np.vstack((A, rows))
        b = np.r_[b, [abs(entry["value"]) for entry in added_bounds]]
    return A, b, added_bounds


def interior_start(A, b):
    """Find a strict interior point and check that the uniform reference exists."""
    dimension = A.shape[1]
    bounds = [(None, None)] * dimension
    norms = np.linalg.norm(A, axis=1)
    result = linprog(
        np.r_[np.zeros(dimension), -1.0],
        A_ub=np.column_stack((A, norms)),
        b_ub=b,
        bounds=bounds + [(0.0, None)],
        method="highs",
    )
    if not result.success or result.x[-1] <= 0:
        raise ValueError(f"could not find a strict interior point: {result.message}")

    if next(unbounded_directions(A, b), None) is not None:
        raise ValueError(
            "FluxML constraints must define a bounded polytope for a uniform reference"
        )
    return result.x[:-1]


def round_geometry(A, b, initial):
    """Return native geometry, affine coordinates and the original start in rounded space."""
    import hopsy as legacy_hopsy

    # Center first for PolyRound's numerical stability. Translate its result
    # back, then pull back all the original constraints with the composed map.
    centered = legacy_hopsy.Problem(
        A, b - A @ initial, starting_point=np.zeros_like(initial)
    )
    rounded = legacy_hopsy.round(centered)
    transform = np.asarray(rounded.transformation)
    if transform.shape != (initial.size, initial.size):
        raise RuntimeError(
            "legacy rounding reduced the dimension of the full-dimensional polytope"
        )
    shift = initial + np.asarray(rounded.shift).reshape(-1)
    start = np.asarray(rounded.starting_point).reshape(-1)
    coordinates = core.AffineCoordinates(transform, shift)
    if not np.allclose(coordinates.to_target(start), initial, rtol=1e-10, atol=1e-10):
        raise RuntimeError("legacy rounding did not preserve the initial point")
    geometry = core.LinearPolytope(A @ transform, b - A @ shift)
    return geometry, coordinates, start


def min_effective_sample_size(samples):
    # Sample columns are ordered in time. Each dimension has its own batch means.
    dimension, count = samples.shape
    num_batches = count // BATCH_SIZE
    complete = samples[:, : num_batches * BATCH_SIZE]
    batches = complete.reshape(dimension, num_batches, BATCH_SIZE).mean(axis=2)
    marginal = complete.var(axis=1, ddof=1)
    long_run = BATCH_SIZE * batches.var(axis=1, ddof=1)
    ess = np.divide(
        count * marginal, long_run, out=np.zeros(dimension), where=long_run > 0
    )
    return float(ess.min())


def summarize(mode, samples, seconds, transitions, A, b):
    if not np.isfinite(samples).all() or np.any(
        A @ samples > b[:, None] + 1e-9 * (1 + np.abs(b[:, None]))
    ):
        raise RuntimeError(f"{mode} produced invalid samples")
    ess = min_effective_sample_size(samples)
    return {
        "mode": mode,
        "cold_samples": samples.shape[1],
        "local_transitions": transitions,
        "seconds": seconds,
        "min_coordinate_ess": ess,
        "min_coordinate_ess_per_second": ess / seconds,
    }


def replica_diagnostics(replica_ids, log_densities, exchange_statistics, round_trips):
    """Analyze post-exchange snapshots in scan order, including the initial snapshot."""
    cold_slot = replica_ids.shape[1] - 1
    positions = np.argsort(replica_ids, axis=1)
    trips_from_trace = []
    progress = []
    for replica, track in enumerate(positions.T):
        endpoints = track[(track == 0) | (track == cold_slot)]
        endpoint_changes = endpoints[1:] != endpoints[:-1]
        hot_to_cold = int(
            np.count_nonzero(endpoint_changes & (endpoints[1:] == cold_slot))
        )
        cold_to_hot = int(np.count_nonzero(endpoint_changes & (endpoints[1:] == 0)))
        # A first leg from cold to hot cannot complete a trip in this window.
        trips_from_trace.append(
            cold_to_hot - int(cold_to_hot > 0 and endpoints[0] == cold_slot)
        )
        hot_visits = np.flatnonzero(track == 0)
        progress.append(
            {
                "replica": replica,
                "hottest_slot": int(track.min()),
                "coldest_slot": int(track.max()),
                "coldest_slot_after_hot_visit": (
                    int(track[hot_visits[0] :].max()) if hot_visits.size else None
                ),
                "hot_to_cold": hot_to_cold,
                "cold_to_hot": cold_to_hot,
            }
        )
    if trips_from_trace != round_trips:
        raise RuntimeError("replica trace disagrees with the native round-trip counts")

    prior = replica_ids[:-1]
    after = replica_ids[1:]
    swapped = (prior[:, :-1] == after[:, 1:]) & (prior[:, 1:] == after[:, :-1])
    if swapped.sum(axis=0).tolist() != [
        edge["accepts"] for edge in exchange_statistics
    ]:
        raise RuntimeError("replica trace disagrees with the native exchange counts")
    # Each edge is offered every other scan. Detect immediate reversals of
    # the same two physical replicas on consecutive offers at that edge.
    same_pair_reversed = (prior[2:, :-1] == prior[:-2, 1:]) & (
        prior[2:, 1:] == prior[:-2, :-1]
    )
    swap_backs = (swapped[2:] & swapped[:-2] & same_pair_reversed).sum(axis=0)

    # Follow the original occupant of each slot through its local steps and
    # then recover its explored log density from the post-exchange snapshot.
    explored_slots = np.take_along_axis(positions[1:], prior, axis=1)
    explored = np.take_along_axis(log_densities[1:], explored_slots, axis=1)
    start = log_densities[:-1]
    finite = np.isfinite(start) & np.isfinite(explored)
    changes = np.subtract(explored, start, out=np.zeros_like(explored), where=finite)
    local_changes = []
    for values, valid in zip(changes.T, finite.T):
        abs_changes = np.abs(values[valid])
        local_changes.append(
            {
                "median_abs_change": (
                    float(np.median(abs_changes)) if abs_changes.size else None
                ),
                "unchanged_fraction": (
                    float(np.mean(abs_changes == 0)) if abs_changes.size else None
                ),
            }
        )

    return {
        "per_replica": progress,
        "immediate_swap_backs_by_edge": swap_backs.tolist(),
        "local_log_density_changes_by_slot": local_changes,
    }


def plot_flux_posteriors(simulator, samples, directory, *, reference_max_beta=0.0):
    """Overlay marginal densities with shared bins for each named flux."""
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    space = simulator.parameter_space
    names = space.parameter_names
    variable = set(x3cflux.get_non_constant_names(simulator))
    groups = {}
    for group, suffix in (("net", ".n"), ("exchange", ".x")):
        groups[group] = [i for i, name in enumerate(names) if name.endswith(suffix)]
    flux_indices = [
        i for indices in groups.values() for i in indices if names[i] in variable
    ]

    # Reconstruct dependent fluxes with the same parameter map used by x3cflux.
    flux_samples = {}
    for series, draws in samples.items():
        expanded = np.empty((len(flux_indices), draws.shape[1]))
        for column, draw in enumerate(draws.T):
            expanded[:, column] = space.compute_parameters(draw)[flux_indices]
        if not np.isfinite(expanded).all():
            raise RuntimeError(f"nonfinite reconstructed fluxes for {series}")
        flux_samples[series] = expanded

    kernel_styles = {
        "hit-and-run": ("Hit-and-Run", "#0072B2"),
        "coordinate-hit-and-run": ("Rounded coordinate Hit-and-Run", "#009E73"),
        "gaussian-hit-and-run": ("Gaussian Hit-and-Run", "#009E73"),
        "csmmala": ("CSmMALA", "#D55E00"),
    }
    mode_styles = {
        "standalone": ("standalone", "-"),
        "nrpt_serial": ("serial NRPT", ":"),
        "nrpt_parallel": ("parallel NRPT", "--"),
        "nrpt_serial_rounded_reference": ("serial NRPT · rounded hot", "-."),
        "nrpt_parallel_rounded_reference": ("parallel NRPT · rounded hot", "-."),
    }
    handles = {}
    for kernel, mode in samples:
        kernel_name, color = kernel_styles[kernel]
        mode_name, linestyle = mode_styles[mode]
        if mode.endswith("_rounded_reference"):
            color = "#0072B2"
            if reference_max_beta > 0:
                execution = "parallel" if mode.startswith("nrpt_parallel") else "serial"
                mode_name = (
                    f"{execution} NRPT · Hit-and-Run at β ≤ {reference_max_beta:g}"
                )
        handles[kernel, mode] = Line2D(
            [], [], color=color, ls=linestyle, label=f"{kernel_name} · {mode_name}"
        )

    directory.mkdir(parents=True, exist_ok=True)
    fixed = space.compute_parameters(next(iter(samples.values()))[:, 0])
    plots = {}
    rows_by_flux = {index: row for row, index in enumerate(flux_indices)}
    for group, indices in groups.items():
        varying = [i for i in indices if i in rows_by_flux]
        info = {
            "histograms": len(varying),
            "fixed_fluxes": {
                names[i]: float(fixed[i]) for i in indices if i not in rows_by_flux
            },
        }
        plots[group] = info
        if not varying:
            continue
        columns = min(4, len(varying))
        rows = (len(varying) + columns - 1) // columns
        height = 1 + 1.85 * rows
        figure = Figure(figsize=(3.25 * columns + 0.4, height))
        axes = figure.subplots(rows, columns, squeeze=False).ravel()
        figure.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.5 / height,
            top=1 - 0.95 / height,
            wspace=0.37,
            hspace=0.66,
        )
        figure.suptitle(f"{group.capitalize()} flux posteriors", y=1 - 0.12 / height)
        figure.legend(
            handles=list(handles.values()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1 - 0.43 / height),
            ncols=min(4, len(handles)),
            fontsize=9,
            frameon=False,
        )
        figure.supxlabel("Flux", y=0.11 / height)
        figure.supylabel("Posterior density", x=0.012)
        for ax, index in zip(axes, varying):
            row = rows_by_flux[index]
            lower = min(values[row].min() for values in flux_samples.values())
            upper = max(values[row].max() for values in flux_samples.values())
            bins = 40
            # A chain can stay nearly constant along a varying flux. Keep bin
            # edges distinct even if the observed span is only a few ulps.
            min_span = bins * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
            if upper - lower < min_span:
                lower -= min_span
                upper += min_span
            edges = np.linspace(lower, upper, bins + 1)
            for style, values in flux_samples.items():
                handle = handles[style]
                ax.hist(
                    values[row],
                    bins=edges,
                    density=True,
                    histtype="step",
                    color=handle.get_color(),
                    linestyle=handle.get_linestyle(),
                    linewidth=1.4,
                )
            ax.set_title(names[index], fontsize=10, parse_math=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[len(varying) :]:
            ax.set_visible(False)
        path = directory / f"{group}_flux_posteriors.svg"
        with rc_context({"svg.fonttype": "none"}):
            figure.savefig(path)
        info["svg"] = str(path)
    return plots


def run_kernel(args, kernel, simulator, geometry, coordinates, initial, ladder, A, b):
    samples, traces = {}, {}
    reference_basis = None
    if args.reference_kernel != "same":
        _, reference_coordinates, _ = round_geometry(A, b, initial)
        reference_basis = reference_coordinates.matrix

    def make_target():
        return core.CapsuleTarget(simulator.make_hopsy_target())

    default_step = CSMMALA_STEP_SIZE if kernel == "csmmala" else 1.0
    step_size = default_step if args.step_size is None else args.step_size

    def make_kernel():
        if kernel == "csmmala":
            return core.CSmMALAKernel(initial, step_size=step_size)
        if kernel == "gaussian-hit-and-run":
            return core.GaussianHitAndRunKernel(initial, standard_deviation=step_size)
        return core.UniformHitAndRunKernel(initial)

    chain = core.bind_chain(
        core.Problem(geometry, make_target(), coordinates), make_kernel(), initial
    )
    rng = core.RandomNumberGenerator(args.seed, 0)
    chain.draw(rng, args.warmup, thinning=args.standalone_thinning)
    start = perf_counter()
    cold = chain.draw(rng, args.num_samples, thinning=args.standalone_thinning)
    baseline = summarize(
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
        targets = [make_target() for _ in range(replicas)]
        rounded_reference = reference_kind == "rounded-coordinate"
        qualifier = "_rounded_reference" if rounded_reference else ""
        print(
            f"{kernel}: {mode} NRPT, reference={reference_kind}",
            file=sys.stderr,
            flush=True,
        )
        sampler = core.bind_parallel_tempering(
            core.Problem(geometry, targets[0], coordinates),
            ladder,
            [make_kernel() for _ in targets],
            [initial] * replicas,
            [core.RandomNumberGenerator(args.seed, slot) for slot in range(replicas)],
            core.RandomNumberGenerator(args.seed, replicas),
            replica_targets=targets,
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
        tempered = summarize(
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
        exchanges = [
            {"attempts": edge.attempts, "accepts": edge.accepts}
            for edge in sampler.exchange_statistics
        ]
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
    parser.add_argument("--fml", type=Path, default=DEFAULT_FML)
    parser.add_argument("--configuration", default="default")
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
    parser.add_argument(
        "--unbounded-flux-caps",
        type=float,
        nargs=2,
        metavar=("NET", "EXCHANGE"),
        help="explicitly cap unbounded free-flux sides for a finite uniform reference",
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
    if not 0 <= args.reference_max_beta <= 1:
        parser.error("--reference-max-beta must be in [0, 1]")
    if args.reference_kernel == "same" and args.reference_max_beta:
        parser.error("--reference-max-beta requires the rounded reference kernel")
    if args.reference_kernel != "same" and (args.kernel != "csmmala" or args.round):
        parser.error("a rounded reference requires --kernel csmmala without --round")
    if args.unbounded_flux_caps is not None and any(
        not np.isfinite(cap) or cap <= 0 for cap in args.unbounded_flux_caps
    ):
        parser.error("unbounded flux caps must be finite and positive")
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

    simulator = x3cflux.create_simulator_from_fml(str(args.fml), args.configuration)
    inequalities = simulator.parameter_space.inequality_system
    A = np.asarray(inequalities.matrix)
    b = np.asarray(inequalities.bound).reshape(-1)
    added_bounds = []
    if args.unbounded_flux_caps is not None:
        A, b, added_bounds = cap_unbounded_fluxes(
            A,
            b,
            simulator.parameter_space.free_parameter_names,
            args.unbounded_flux_caps,
        )
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
        report["posterior_plots"] = plot_flux_posteriors(
            simulator,
            samples,
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
