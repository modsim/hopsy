"""Parallel LP redundancy removal for the exp2 HiGHS backend.

This follows ParallelRedundancyRemoval.pdf for PolyRound's Ax <= b convention:
use one slack LP to detect possible hidden linearities, promote independent
hidden rows to equalities, eliminate equalities, remove duplicate reduced rows,
then test the rest in parallel.  Each Python worker owns its mutable HiGHS model.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import linalg

from hopsy._polyround.default_settings import default_solver_timeout
from hopsy._polyround.polytope import Polytope

from . import lp_interfacing as lp

UNBOUNDED_STATUSES = {"unbounded", "infeasible_or_unbounded"}
RECOVERABLE_STATUSES = {
    "undefined",
    "unknown",
    "solver_error",
    "presolve_error",
    "postsolve_error",
    "time_limit",
    "iteration_limit",
    "objective_bound",
    "objective_target",
    "tolerance_violation",
}
FATAL_STATUSES = {"load_error", "model_error"}


def _set_highs_option(model, name, value):
    """Set an option directly; retry code needs options not exposed elsewhere."""
    highs_status = model.setOptionValue(name, value)
    lp._check_highs_status(highs_status, "set option " + name)


def _run_lp_once(model, settings):
    """Run HiGHS once and report both the API return status and model status."""
    highs_status = model.run()
    run_status = lp._highs_status_name(highs_status)
    model_status = lp.status(model)
    if lp._highs_run_failed(highs_status):
        if model_status in FATAL_STATUSES:
            raise RuntimeError(f"HiGHS failed to run optimizer: {highs_status}")
        if model_status in {"undefined", "unknown"}:
            model_status = "solver_error"
        return run_status, model_status, np.nan

    if model_status == "optimal":
        try:
            if getattr(settings, "check_lps", False):
                lp.check_tolerances(model)
        except ValueError:
            return run_status, "tolerance_violation", np.nan
        return run_status, model_status, lp.objective_value(model)
    return run_status, model_status, np.nan


def _lp_value(model, settings, context):
    """Run a row LP, retry likely numerical failures, and return status/value."""
    option_names = ("presolve", "solver", "simplex_strategy")
    original_options = {
        name: lp._highs_option(model, name, None) for name in option_names
    }
    attempts = [{}]
    if original_options.get("presolve") != "on":
        attempts.append({"presolve": "on"})
    attempts.extend(
        [
            {"presolve": "on", "solver": "simplex", "simplex_strategy": 1},
            {"presolve": "on", "solver": "simplex", "simplex_strategy": 4},
            {"presolve": "on", "solver": "ipm"},
        ]
    )
    best_optimal = None
    last_status = ("undefined", np.nan)
    last_run_status = "kNotset"
    changed_options = False

    try:
        for attempt_index, options in enumerate(attempts):
            if attempt_index > 0:
                changed_options = True
                model.clearSolver()
                for name, value in options.items():
                    _set_highs_option(model, name, value)
                if attempt_index == 1:
                    print(
                        f"[exp2 constraint_removal] {context}: retrying after "
                        f"{last_run_status}/{last_status[0]}",
                        flush=True,
                    )

            run_status, model_status, value = _run_lp_once(model, settings)
            last_run_status = run_status
            last_status = (model_status, value)

            # A clean optimal/unbounded solve is enough for classification.
            if run_status == "kOk" and model_status in {"optimal", *UNBOUNDED_STATUSES}:
                return model_status, value

            # kWarning plus optimal still gives a usable value, but first try
            # to get a cleaner solve from a safer HiGHS configuration.
            if model_status == "optimal":
                best_optimal = (model_status, value)
            elif model_status in FATAL_STATUSES:
                raise ValueError(
                    f"{context} LP failed with solver status: {model_status}"
                )
            elif model_status not in RECOVERABLE_STATUSES | UNBOUNDED_STATUSES:
                raise ValueError(
                    f"{context} LP failed with solver status: {model_status}"
                )

        if best_optimal is not None:
            return best_optimal
        return last_status
    finally:
        if changed_options:
            model.clearSolver()
            for name, value in original_options.items():
                if value is not None:
                    _set_highs_option(model, name, value)


def _make_model(A, b, S, h, settings, column_names, presolve=None):
    """Create the reusable worker LP: inequalities Ax <= b plus Sx = h."""
    model = lp.make_model([str(name) for name in column_names], settings)
    if presolve is None:
        presolve = getattr(settings, "presolve", False)
    lp.set_presolve(model, presolve)
    lp.set_time_limit(model, default_solver_timeout)
    rows = lp.add_constraint_system(model, A, b, equality=False)
    if S is not None and S.shape[0] > 0:
        lp.add_constraint_system(model, S, h, equality=True)
    return model, rows


def _parallel_row_checks(row_count, settings, build_state, check_row, label):
    """Evaluate row LPs with one lazily-built HiGHS model per Python worker."""
    if row_count == 0:
        return []

    requested = int((getattr(settings, "hp_flags", {}) or {}).get("Threads", 1) or 1)
    workers = max(1, min(row_count, requested, os.cpu_count() or 1))
    print(
        f"[exp2 constraint_removal] {label}: solving {row_count} LPs "
        f"with {workers} worker(s)",
        flush=True,
    )

    local = threading.local()
    progress_lock = threading.Lock()
    progress_step = max(1, min(100, row_count // 20 or 1))
    started = time.perf_counter()
    completed = 0

    def state():
        worker_state = getattr(local, "state", None)
        if worker_state is None:
            worker_state = build_state()
            local.state = worker_state
        return worker_state

    def run_row(row):
        nonlocal completed
        try:
            result = check_row(state(), row)
        except Exception as exc:
            print(
                f"[exp2 constraint_removal] {label}: row {row} failed after "
                f"{time.perf_counter() - started:.1f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        with progress_lock:
            completed += 1
            if completed == row_count or completed % progress_step == 0:
                print(
                    f"[exp2 constraint_removal] {label}: completed "
                    f"{completed}/{row_count} LPs in "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        return result

    if workers == 1:
        return [run_row(row) for row in range(row_count)]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run_row, range(row_count)))


def _interior_slack(A, b, S, h, settings, column_names, presolve=None):
    """LP (7) from the paper: maximize the minimum inequality slack."""
    if A.shape[0] == 0:
        return np.inf

    slack_A = np.hstack((A, np.ones((A.shape[0], 1), dtype=np.float64)))
    slack_S = None
    slack_h = None
    if S is not None and S.shape[0] > 0:
        slack_S = np.hstack((S, np.zeros((S.shape[0], 1), dtype=np.float64)))
        slack_h = h

    model, _rows = _make_model(
        slack_A,
        b,
        slack_S,
        slack_h,
        settings,
        [*column_names, "__polyround_slack"],
        presolve=presolve,
    )
    slack_column = (np.asarray([A.shape[1]], dtype=np.int64), np.asarray([1.0]))
    lp.set_linear_objective(model, slack_column, maximize=True)
    model_status, value = _lp_value(model, settings, "Interior slack")
    if model_status == "optimal":
        return float(value)
    if model_status in UNBOUNDED_STATUSES:
        return np.inf
    if model_status in RECOVERABLE_STATUSES:
        print(
            "[exp2 constraint_removal] interior slack: unresolved after retries; "
            "checking hidden linearities conservatively",
            flush=True,
        )
        return 0.0
    raise ValueError("Interior slack LP failed with solver status: " + model_status)


def _relaxed_row_mask(
    A, b, S, h, settings, column_names, maximize, accept, label, presolve=None
):
    """Run independent row LPs: relax one row, optimize, restore, classify."""
    expressions = lp.build_row_expressions(A)
    inf = float(lp.highspy.kHighsInf)

    def build_state():
        return _make_model(A, b, S, h, settings, column_names, presolve=presolve)

    def check_row(state, index):
        model, rows = state
        row = int(rows[index])
        try:
            lp.change_row_bounds(model, row, -inf, inf)
            lp.set_linear_objective(model, expressions[index], maximize=maximize)
            model_status, value = _lp_value(model, settings, f"{label} row {index}")
        finally:
            lp.set_row_upper(model, row, float(b[index]))

        if model_status == "optimal":
            return bool(accept(value, b[index]))
        if model_status in UNBOUNDED_STATUSES:
            return False
        if model_status in RECOVERABLE_STATUSES:
            print(
                f"[exp2 constraint_removal] {label}: row {index} unresolved "
                f"({model_status}); keeping row",
                flush=True,
            )
            return False
        raise ValueError(f"{label} LP for row {index} failed: {model_status}")

    checks = _parallel_row_checks(A.shape[0], settings, build_state, check_row, label)
    return np.asarray(checks, dtype=bool)


def _independent_equalities(
    S, h, S_names, hidden_indices, A, b, A_names, tol, rank_tol
):
    """Keep existing equalities first, then independent hidden linearities."""
    rows = []
    rhs_values = []
    names = []
    selected_hidden = set()
    rank = 0
    total = S.shape[0] + len(hidden_indices)
    dimension = A.shape[1]
    basis = np.zeros((total, dimension), dtype=np.float64)
    progress_step = max(1, min(100, total // 10 or 1))
    started = time.perf_counter()
    processed = 0

    def report_progress():
        if processed == total or processed % progress_step == 0:
            print(
                f"[exp2 constraint_removal] equality selection: processed "
                f"{processed}/{total} rows in {time.perf_counter() - started:.1f}s; "
                f"rank={rank}",
                flush=True,
            )

    def add(row, rhs, name, hidden_index=None):
        nonlocal rank
        row = np.asarray(row, dtype=np.float64).reshape(-1)
        rhs = float(rhs)

        # Zero rows either encode no information or prove inconsistency.
        row_norm = float(np.linalg.norm(row))
        if row_norm <= rank_tol:
            if abs(rhs) > tol:
                raise ValueError("Inconsistent zero equality encountered.")
            return

        # Modified Gram-Schmidt keeps a streaming row-space basis.  This avoids
        # recomputing a full matrix rank after every candidate equality.
        residual = row.copy()
        if rank:
            active_basis = basis[:rank]
            residual -= (active_basis @ residual) @ active_basis
            residual -= (active_basis @ residual) @ active_basis

        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= rank_tol * max(1.0, row_norm):
            return

        basis[rank] = residual / residual_norm
        rows.append(row)
        rhs_values.append(rhs)
        names.append(str(name))
        rank += 1
        if hidden_index is not None:
            selected_hidden.add(int(hidden_index))

    for index, row in enumerate(np.asarray(S, dtype=np.float64)):
        add(row, h[index], S_names[index])
        processed += 1
        report_progress()
    for hidden_index in hidden_indices:
        add(A[hidden_index], b[hidden_index], A_names[hidden_index], hidden_index)
        processed += 1
        report_progress()

    if not rows:
        empty_S = np.zeros((0, A.shape[1]), dtype=np.float64)
        return empty_S, np.zeros(0, dtype=np.float64), [], selected_hidden
    return np.vstack(rows), np.asarray(rhs_values), names, selected_hidden


def _affine_reduction(S, h, dimension, tol, rank_tol):
    """Represent Sx = h as x = x0 + N y for the later row LPs."""
    if S.shape[0] == 0:
        return np.zeros(dimension, dtype=np.float64), np.eye(dimension)

    x0, *_ = np.linalg.lstsq(S, h, rcond=None)
    residual = S @ x0 - h
    if residual.size and np.max(np.abs(residual)) > tol:
        raise ValueError("Equality system is infeasible within tolerance.")
    null_basis = np.asarray(linalg.null_space(S, rcond=rank_tol))
    return x0, null_basis


def _clean_projected_system(A, b, zero_tol):
    """Remove nullspace-projection roundoff before building a sparse HiGHS LP."""
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).reshape(-1).copy()
    A[np.abs(A) <= zero_tol] = 0.0
    b[np.abs(b) <= zero_tol] = 0.0
    return A, b


def _unique_reduced_rows(A, b, tol):
    """Drop zero rows, exact duplicates, and looser parallel halfspaces."""
    duplicate = []
    best_by_direction = {}
    decimals = max(0, int(np.ceil(-np.log10(max(tol, 1e-15)))) + 2)

    for index, (row, rhs) in enumerate(zip(A, b, strict=True)):
        if np.linalg.norm(row, ord=np.inf) <= tol:
            if rhs < -tol:
                raise ValueError("Infeasible reduced zero inequality encountered.")
            duplicate.append(index)
            continue

        # Positive scalar multiples define the same normal direction.  Keep the
        # tightest rhs for that direction and mark looser rows redundant now.
        scale = float(np.max(np.abs(row)))
        normalized_row = row / scale
        normalized_rhs = float(rhs) / scale
        key = tuple(np.round(normalized_row, decimals).tolist())
        previous = best_by_direction.get(key)
        if previous is None:
            best_by_direction[key] = (index, normalized_rhs)
        elif normalized_rhs < previous[1] - tol:
            duplicate.append(previous[0])
            best_by_direction[key] = (index, normalized_rhs)
        else:
            duplicate.append(index)

    unique = sorted(index for index, _rhs in best_by_direction.values())
    return np.asarray(unique, dtype=np.int64), np.asarray(duplicate, dtype=np.int64)


def _sequential_redundancy_mask(A, b, S, h, settings, column_names, tol):
    """Classic safe deletion pass used when simplify_only forbids refunctioning."""
    active = np.ones(A.shape[0], dtype=bool)
    if A.shape[0] == 0:
        return active, 0

    print(
        f"[exp2 constraint_removal] sequential redundancy: solving {A.shape[0]} LPs",
        flush=True,
    )
    model, rows = _make_model(A, b, S, h, settings, column_names)
    expressions = lp.build_row_expressions(A)
    inf = float(lp.highspy.kHighsInf)
    removed = 0
    started = time.perf_counter()
    progress_step = max(1, min(100, A.shape[0] // 20 or 1))

    for index, row in enumerate(rows):
        # Leave a redundant row relaxed for all later LPs.  This sequential
        # update avoids the duplicate/weak-redundancy race described in the
        # paper when hidden linearities are deliberately kept as inequalities.
        lp.change_row_bounds(model, int(row), -inf, inf)
        lp.set_linear_objective(model, expressions[index], maximize=True)
        model_status, value = _lp_value(
            model, settings, f"Sequential redundancy row {index}"
        )

        if model_status == "optimal" and value <= b[index] + tol:
            active[index] = False
            removed += 1
            if (index + 1) == A.shape[0] or (index + 1) % progress_step == 0:
                print(
                    f"[exp2 constraint_removal] sequential redundancy: completed "
                    f"{index + 1}/{A.shape[0]} LPs in "
                    f"{time.perf_counter() - started:.1f}s; removed={removed}",
                    flush=True,
                )
            continue

        lp.set_row_upper(model, int(row), float(b[index]))
        if model_status in {"optimal", *UNBOUNDED_STATUSES}:
            if (index + 1) == A.shape[0] or (index + 1) % progress_step == 0:
                print(
                    f"[exp2 constraint_removal] sequential redundancy: completed "
                    f"{index + 1}/{A.shape[0]} LPs in "
                    f"{time.perf_counter() - started:.1f}s; removed={removed}",
                    flush=True,
                )
            continue
        if model_status in RECOVERABLE_STATUSES:
            print(
                f"[exp2 constraint_removal] sequential redundancy: row {index} "
                f"unresolved ({model_status}); keeping row",
                flush=True,
            )
            if (index + 1) == A.shape[0] or (index + 1) % progress_step == 0:
                print(
                    f"[exp2 constraint_removal] sequential redundancy: completed "
                    f"{index + 1}/{A.shape[0]} LPs in "
                    f"{time.perf_counter() - started:.1f}s; removed={removed}",
                    flush=True,
                )
            continue
        raise ValueError(f"Redundancy LP for row {index} failed: {model_status}")

    return active, removed


def constraint_removal(polytope, settings):
    """Remove redundant inequalities and promote hidden linearities."""
    lp.require_package()
    total_started = time.perf_counter()

    A_df = polytope.A.copy()
    b_series = polytope.b.copy()
    if polytope.S is None:
        S_df = pd.DataFrame(columns=A_df.columns, dtype=np.float64)
        h_series = pd.Series(dtype=np.float64)
    else:
        S_df = polytope.S.copy()
        h_series = polytope.h.copy()

    A = np.asarray(A_df.values, dtype=np.float64)
    b = np.asarray(b_series.values, dtype=np.float64).reshape(-1)
    S = np.asarray(S_df.values, dtype=np.float64)
    h = np.asarray(h_series.values, dtype=np.float64).reshape(-1)
    columns = list(A_df.columns)
    A_names = [str(name) for name in b_series.index]
    S_names = [str(name) for name in h_series.index]
    tol = float(getattr(settings, "thresh", 1e-7))
    rank_tol = float(max(getattr(settings, "numerics_threshold", 1e-12), 1e-14))
    reduce = getattr(settings, "reduce", True)
    simplify_only = getattr(settings, "simplify_only", False)

    active = np.ones(A.shape[0], dtype=bool)
    equality = np.zeros(A.shape[0], dtype=bool)
    hidden = np.zeros(A.shape[0], dtype=bool)
    requested_threads = int(
        (getattr(settings, "hp_flags", {}) or {}).get("Threads", 1) or 1
    )
    print(
        f"[exp2 constraint_removal] start: A={A.shape}, S={S.shape}, "
        f"reduce={reduce}, simplify_only={simplify_only}, requested_threads={requested_threads}",
        flush=True,
    )

    # First remove dependent existing equalities and project the expensive
    # hidden-linearity LPs into that affine space.  The final output still uses
    # original-space rows, but HiGHS sees a much smaller problem.
    print(
        f"[exp2 constraint_removal] existing equalities: selecting independent "
        f"rows from {S.shape[0]} input rows",
        flush=True,
    )
    S_existing, h_existing, S_existing_names, _ = _independent_equalities(
        S,
        h,
        S_names,
        [],
        A,
        b,
        A_names,
        tol,
        rank_tol,
    )
    stage_started = time.perf_counter()
    print(
        "[exp2 constraint_removal] existing equalities: computing nullspace",
        flush=True,
    )
    existing_x0, existing_null_basis = _affine_reduction(
        S_existing, h_existing, A.shape[1], tol, rank_tol
    )
    if S.shape[0] > S_existing.shape[0]:
        residual = S @ existing_x0 - h
        if residual.size and np.max(np.abs(residual)) > tol:
            raise ValueError("Dependent existing equalities are inconsistent.")
    print(
        f"[exp2 constraint_removal] existing equalities: reduced dimension="
        f"{existing_null_basis.shape[1]} in {time.perf_counter() - stage_started:.1f}s",
        flush=True,
    )

    # The expensive hidden-linearity row LPs are only needed when the single
    # interior-slack LP says the inequality system has no strict relative
    # interior point.  These LPs are solved after existing equality elimination.
    if A.shape[0] > 0:
        hidden_A = A @ existing_null_basis
        hidden_b = b - A @ existing_x0
        hidden_A, hidden_b = _clean_projected_system(hidden_A, hidden_b, rank_tol)
        hidden_columns = [f"y{j}" for j in range(hidden_A.shape[1])]
        stage_started = time.perf_counter()
        print("[exp2 constraint_removal] interior slack: solving 1 LP", flush=True)
        slack = _interior_slack(
            hidden_A,
            hidden_b,
            None,
            None,
            settings,
            hidden_columns,
            presolve=True,
        )
        hidden_may_exist = slack <= tol
        print(
            f"[exp2 constraint_removal] interior slack: value={slack:.6g}, "
            f"hidden_may_exist={hidden_may_exist} in "
            f"{time.perf_counter() - stage_started:.1f}s",
            flush=True,
        )
        if hidden_may_exist and simplify_only:
            if reduce:
                active, removed = _sequential_redundancy_mask(
                    A, b, S_existing, h_existing, settings, columns, tol
                )
            else:
                removed = 0
            final_rows = np.flatnonzero(active)
            final_A = A_df.iloc[final_rows].copy()
            final_b = b_series.iloc[final_rows].copy()
            print(
                f"[exp2 constraint_removal] done: final_A={final_A.shape}, "
                f"removed={removed}, refunctioned=0 in "
                f"{time.perf_counter() - total_started:.1f}s",
                flush=True,
            )
            if polytope.S is None:
                return Polytope(final_A, final_b), removed, 0
            return Polytope(final_A, final_b, S_df, h_series), removed, 0

        if hidden_may_exist:
            # For Ax <= b, row i is a hidden equality iff P without row i
            # still implies a_i x >= b_i, so we minimize a_i x.
            hidden = _relaxed_row_mask(
                hidden_A,
                hidden_b,
                None,
                None,
                settings,
                hidden_columns,
                False,
                lambda value, rhs: value >= rhs - tol,
                "Hidden-linearity",
                presolve=True,
            )
            print(
                f"[exp2 constraint_removal] hidden linearities: found "
                f"{int(np.sum(hidden))}/{A.shape[0]} candidate rows",
                flush=True,
            )

    hidden_indices = np.flatnonzero(hidden)
    print(
        f"[exp2 constraint_removal] equality selection: checking "
        f"{S_existing.shape[0]} independent existing equalities and "
        f"{hidden_indices.size} hidden candidates",
        flush=True,
    )
    S_min, h_min, S_min_names, selected_hidden = _independent_equalities(
        S_existing,
        h_existing,
        S_existing_names,
        hidden_indices,
        A,
        b,
        A_names,
        tol,
        rank_tol,
    )

    # Dependent hidden linearities are deleted; independent ones move from A to S.
    for index in hidden_indices:
        if int(index) in selected_hidden:
            equality[index] = True
        else:
            active[index] = False

    refunctioned = int(equality.sum())
    removed = int(hidden_indices.size - refunctioned)
    candidates = np.flatnonzero(active & ~equality)
    print(
        f"[exp2 constraint_removal] equality selection: existing={S_existing.shape[0]}, "
        f"hidden_candidates={hidden_indices.size}, promoted={refunctioned}, "
        f"dependent_hidden_removed={removed}, candidate_inequalities={candidates.size}",
        flush=True,
    )

    if reduce and candidates.size > 0:
        # Equality elimination from the paper's step (d): write every feasible
        # original-space point as x = x0 + N y before duplicate/redundancy tests.
        stage_started = time.perf_counter()
        print(
            "[exp2 constraint_removal] equality elimination: computing nullspace",
            flush=True,
        )
        x0, null_basis = _affine_reduction(S_min, h_min, A.shape[1], tol, rank_tol)
        print(
            f"[exp2 constraint_removal] equality elimination: reduced dimension="
            f"{null_basis.shape[1]} in {time.perf_counter() - stage_started:.1f}s",
            flush=True,
        )

        candidate_A = A[candidates]
        reduced_A = candidate_A @ null_basis
        reduced_b = b[candidates] - candidate_A @ x0
        reduced_A, reduced_b = _clean_projected_system(reduced_A, reduced_b, rank_tol)

        unique, duplicate = _unique_reduced_rows(reduced_A, reduced_b, tol)
        drop = np.zeros(candidates.size, dtype=bool)
        drop[duplicate] = True
        print(
            f"[exp2 constraint_removal] duplicate reduced rows: unique={unique.size}, "
            f"dropped={duplicate.size}",
            flush=True,
        )

        if unique.size > 0:
            unique_A = reduced_A[unique]
            unique_b = reduced_b[unique]
            if unique_A.shape[1] == 0:
                if np.any(unique_b < -tol):
                    raise ValueError(
                        "Reduced zero-dimensional inequalities are infeasible."
                    )
                redundant = np.ones(unique.size, dtype=bool)
            else:
                # Row i is redundant iff max a_i y over all other rows still
                # satisfies a_i y <= b_i.  Unbounded means a violating point exists.
                redundant = _relaxed_row_mask(
                    unique_A,
                    unique_b,
                    None,
                    None,
                    settings,
                    [f"y{j}" for j in range(unique_A.shape[1])],
                    True,
                    lambda value, rhs: value <= rhs + tol,
                    "Redundancy",
                    presolve=True,
                )
            print(
                f"[exp2 constraint_removal] redundancy: found "
                f"{int(np.sum(redundant))}/{unique.size} redundant unique rows",
                flush=True,
            )
            drop[unique[redundant]] = True

        if np.any(drop):
            active[candidates[drop]] = False
            removed += int(np.sum(drop))

    final_rows = np.flatnonzero(active & ~equality)
    final_A = A_df.iloc[final_rows].copy()
    final_b = b_series.iloc[final_rows].copy()

    if S_min.shape[0] == 0:
        print(
            f"[exp2 constraint_removal] done: final_A={final_A.shape}, "
            f"final_S=(0, {A.shape[1]}), "
            f"removed={removed}, refunctioned={refunctioned} in "
            f"{time.perf_counter() - total_started:.1f}s",
            flush=True,
        )
        return Polytope(final_A, final_b), removed, refunctioned

    final_S = pd.DataFrame(S_min, index=S_min_names, columns=columns)
    final_h = pd.Series(h_min, index=S_min_names, dtype=np.float64)
    print(
        f"[exp2 constraint_removal] done: final_A={final_A.shape}, final_S={final_S.shape}, "
        f"removed={removed}, refunctioned={refunctioned} in "
        f"{time.perf_counter() - total_started:.1f}s",
        flush=True,
    )
    return Polytope(final_A, final_b, final_S, final_h), removed, refunctioned


def null_space(S, eps=1e-10):
    """
    Returns the null space of a matrix.

    This compatibility helper is imported by exp2.backend for equality-system
    transformations; constraint_removal itself uses scipy.linalg.null_space.
    """
    S = np.asarray(S, dtype=np.float64)
    _u, s, vh = np.linalg.svd(S)
    null_mask = np.concatenate((s <= eps, np.ones(1, dtype=bool)))
    null_ind = int(np.argmax(null_mask))
    return np.transpose(vh[null_ind:, :])
