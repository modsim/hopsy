"""Common model preparation, native sampling, and diagnostics for FluxML benchmarks."""

from pathlib import Path

import _hopsy_core as core
import numpy as np
import x3cflux
from scipy.optimize import linprog

BATCH_SIZE = 500
CSMMALA_STEP_SIZE = 0.15


def add_model_arguments(parser, default_fml=None):
    parser.add_argument(
        "--fml", type=Path, default=default_fml, required=default_fml is None
    )
    parser.add_argument("--configuration", default="default")
    parser.add_argument(
        "--unbounded-flux-caps",
        type=float,
        nargs=2,
        metavar=("NET", "EXCHANGE"),
        help="cap only unbounded free-flux sides for a finite uniform reference",
    )


def validate_model_arguments(parser, args):
    caps = args.unbounded_flux_caps
    if caps is not None and any(not np.isfinite(cap) or cap <= 0 for cap in caps):
        parser.error("unbounded flux caps must be finite and positive")


def load_model(args):
    simulator = x3cflux.create_simulator_from_fml(str(args.fml), args.configuration)
    space = simulator.parameter_space
    A = np.asarray(space.inequality_system.matrix)
    b = np.asarray(space.inequality_system.bound).reshape(-1)
    added_bounds = []
    if args.unbounded_flux_caps is not None:
        A, b, added_bounds = cap_unbounded_fluxes(
            A, b, space.free_parameter_names, args.unbounded_flux_caps
        )
    return simulator, A, b, added_bounds


def make_target(simulator):
    return core.CapsuleTarget(simulator.make_hopsy_target())


def bind_tempering(
    simulator,
    geometry,
    coordinates,
    initial,
    ladder,
    make_kernel,
    seed,
    *,
    reference_basis=None,
    reference_max_beta=0.0,
):
    slots = len(ladder)
    targets = [make_target(simulator) for _ in range(slots)]
    return core.bind_parallel_tempering(
        core.Problem(geometry, targets[0], coordinates),
        ladder,
        [make_kernel(initial) for _ in targets],
        [initial] * slots,
        [core.RandomNumberGenerator(seed, slot) for slot in range(slots)],
        core.RandomNumberGenerator(seed, slots),
        replica_targets=targets,
        reference_basis=reference_basis,
        reference_max_beta=reference_max_beta,
    )


def exchange_counts(sampler):
    return [
        {"attempts": edge.attempts, "accepts": edge.accepts}
        for edge in sampler.exchange_statistics
    ]


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


def sampling_report(mode, samples, seconds, transitions, A, b):
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


def varying_flux_indices(simulator):
    names = simulator.parameter_space.parameter_names
    variable = set(x3cflux.get_non_constant_names(simulator))
    return [
        i
        for i, name in enumerate(names)
        if name in variable and name.endswith((".n", ".x"))
    ]


def reconstruct_fluxes(space, indices, chains):
    """Expand free-parameter draws into named fluxes, keeping chain identity."""
    fluxes = np.empty((len(chains), len(indices), chains.shape[-1]))
    for out, chain in zip(fluxes, chains):
        for column, draw in enumerate(chain.T):
            out[:, column] = space.compute_parameters(draw)[indices]
    if not np.isfinite(fluxes).all():
        raise RuntimeError("nonfinite reconstructed fluxes")
    return fluxes


def plot_flux_posteriors(
    simulator,
    flux_samples,
    initial,
    directory,
    *,
    reference_max_beta=0.0,
    replicate_seeds=None,
    model_tag=None,
):
    """Overlay reconstructed marginal densities with shared bins for each flux."""
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    space = simulator.parameter_space
    names = space.parameter_names
    groups = {}
    for group, suffix in (("net", ".n"), ("exchange", ".x")):
        groups[group] = [i for i, name in enumerate(names) if name.endswith(suffix)]
    flux_indices = varying_flux_indices(simulator)

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
    if replicate_seeds is not None:
        if len(flux_samples) != 1:
            raise ValueError("Replicate overlays require exactly one sampling method")
        values = next(iter(flux_samples.values()))
        if len(replicate_seeds) != len(values):
            raise ValueError("Expected one seed per cold chain")
        flux_samples = {
            seed: values[i : i + 1] for i, seed in enumerate(replicate_seeds)
        }
        handles = {
            seed: Line2D([], [], color=f"C{i % 10}", label=f"Seed {seed}")
            for i, seed in enumerate(replicate_seeds)
        }
    else:
        for kernel, mode in flux_samples:
            kernel_name, color = kernel_styles[kernel]
            mode_name, linestyle = mode_styles[mode]
            if mode.endswith("_rounded_reference"):
                color = "#0072B2"
                if reference_max_beta > 0:
                    execution = (
                        "parallel" if mode.startswith("nrpt_parallel") else "serial"
                    )
                    mode_name = (
                        f"{execution} NRPT · Hit-and-Run at β ≤ {reference_max_beta:g}"
                    )
            handles[kernel, mode] = Line2D(
                [], [], color=color, ls=linestyle, label=f"{kernel_name} · {mode_name}"
            )

    directory.mkdir(parents=True, exist_ok=True)
    fixed = space.compute_parameters(initial)
    plots = {}
    rows_by_flux = {index: row for row, index in enumerate(flux_indices)}

    def save_page(title, panels, path):
        columns = min(4, len(panels))
        rows = (len(panels) + columns - 1) // columns
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
        figure.suptitle(title, y=1 - 0.12 / height)
        figure.legend(
            handles=list(handles.values()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1 - 0.43 / height),
            ncols=min(4, len(handles)),
            fontsize=9,
            frameon=False,
        )
        figure.supxlabel("Flux (model units)", y=0.11 / height)
        figure.supylabel("Posterior density", x=0.012)
        for ax, index in zip(axes, panels):
            if index not in rows_by_flux:
                ax.axvline(fixed[index], color="0.3", linewidth=1.4)
                ax.set_yticks([])
                ax.text(0.04, 0.85, "fixed", transform=ax.transAxes, fontsize=8)
            else:
                row = rows_by_flux[index]
                lower = min(values[:, row, :].min() for values in flux_samples.values())
                upper = max(values[:, row, :].max() for values in flux_samples.values())
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
                        values[:, row, :].ravel(),
                        bins=edges,
                        density=True,
                        histtype="step",
                        color=handle.get_color(),
                        linestyle=handle.get_linestyle(),
                        linewidth=1.4,
                    )
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_title(names[index], fontsize=10, parse_math=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[len(panels) :]:
            ax.set_visible(False)
        with rc_context({"svg.fonttype": "none"}):
            figure.savefig(path)

    for group, indices in groups.items():
        varying = [i for i in indices if i in rows_by_flux]
        info = {
            "histograms": len(varying),
            "fixed_fluxes": {
                names[i]: float(fixed[i]) for i in indices if i not in rows_by_flux
            },
        }
        plots[group] = info
        title = f"{group.capitalize()} flux posteriors"
        if replicate_seeds is None:
            if varying:
                path = directory / f"{group}_flux_posteriors.svg"
                save_page(title, varying, path)
                info["svg"] = str(path)
            continue

        # Put fixed fluxes on the same pages as varying fluxes. A point mass
        # is shown as a line because it has no finite density histogram.
        panels_per_page = 20
        info["figures"] = []
        for page, offset in enumerate(range(0, len(indices), panels_per_page), start=1):
            panels = indices[offset : offset + panels_per_page]
            path = directory / f"{model_tag}_{group}_flux_posteriors_{page:02}.svg"
            save_page(f"{model_tag} · {title} · {page}", panels, path)
            info["figures"].append(
                {"svg": str(path), "fluxes": [names[i] for i in panels]}
            )
    return plots
