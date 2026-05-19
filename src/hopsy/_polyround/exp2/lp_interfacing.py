import numpy as np
import pandas as pd
from scipy import sparse as sp

try:
    import highspy
except Exception:
    highspy = None

from hopsy._polyround.default_settings import (
    default_accepted_tol_violation,
    default_hp_flags,
)
from hopsy._polyround.polytope import Polytope


def require_package():
    if highspy is None:
        raise ImportError("This backend requires highspy.")


def _highs_inf():
    if highspy is None:
        return np.inf
    return float(highspy.kHighsInf)


def _check_highs_status(status, action):
    if not str(status).endswith("kOk"):
        raise RuntimeError(f"HiGHS failed to {action}: {status}")


def _highs_status_name(status):
    return getattr(status, "name", str(status)).rsplit(".", 1)[-1]


def _highs_run_failed(status):
    return _highs_status_name(status) not in {"kOk", "kWarning"}


def _highs_option(model, name, default=None):
    status, value = model.getOptionValue(name)
    if str(status).endswith("kOk"):
        return value
    return default


def _tolerances(settings):
    legacy_options = getattr(settings, "hp_flags", {}) or {}
    feasibility = legacy_options.get(
        "FeasibilityTol",
        default_hp_flags.get("FeasibilityTol", 1e-9),
    )
    optimality = legacy_options.get(
        "OptimalityTol",
        default_hp_flags.get("OptimalityTol", 1e-8),
    )
    return float(feasibility), float(optimality)


def _finite(value):
    return np.isfinite(float(value))


def _complete_start(start, major_dimension, nnz):
    start = np.asarray(list(start), dtype=np.int32)
    if start.size == major_dimension + 1:
        return start
    if start.size == major_dimension:
        return np.concatenate((start, np.asarray([nnz], dtype=np.int32)))
    if major_dimension == 0 and start.size == 0:
        return np.zeros(1, dtype=np.int32)
    raise ValueError("Unexpected HiGHS sparse matrix start array length.")


def _lp_matrix_to_csr(lp):
    rows = int(lp.num_row_)
    cols = int(lp.num_col_)
    matrix = lp.a_matrix_
    index = np.asarray(list(matrix.index_), dtype=np.int32)
    value = np.asarray(list(matrix.value_), dtype=np.float64)
    if rows == 0 or cols == 0 or value.size == 0:
        return sp.csr_matrix((rows, cols), dtype=np.float64)

    format_name = getattr(matrix.format_, "name", str(matrix.format_)).rsplit(".", 1)[
        -1
    ]
    if format_name == "kRowwise":
        indptr = _complete_start(matrix.start_, rows, value.size)
        return sp.csr_matrix((value, index, indptr), shape=(rows, cols))
    if format_name == "kColwise":
        indptr = _complete_start(matrix.start_, cols, value.size)
        return sp.csc_matrix((value, index, indptr), shape=(rows, cols)).tocsr()
    raise ValueError(f"Unsupported HiGHS matrix format: {matrix.format_}")


def _col_names(model):
    names = []
    for col in range(int(model.getNumCol())):
        status, name = model.getColName(col)
        if not str(status).endswith("kOk") or name == "":
            name = str(col)
        names.append(str(name))
    return names


def _row_name(model, row):
    status, name = model.getRowName(int(row))
    if not str(status).endswith("kOk") or name == "":
        name = f"constraint_{row}"
    return str(name)


def _append_named_row(rows, duplicate_counts, raw_name, coefficients, rhs):
    duplicate_counts.setdefault(raw_name, 0)
    count = duplicate_counts[raw_name]
    duplicate_counts[raw_name] += 1
    row_name = raw_name if count == 0 else f"{raw_name}__{count}"
    rows.append((row_name, coefficients, float(rhs)))


def status(model_or_status):
    if hasattr(model_or_status, "getModelStatus"):
        model_or_status = model_or_status.getModelStatus()
    name = getattr(model_or_status, "name", str(model_or_status)).rsplit(".", 1)[-1]
    mapping = {
        "kNotset": "undefined",
        "kLoadError": "load_error",
        "kModelError": "model_error",
        "kPresolveError": "presolve_error",
        "kSolveError": "solver_error",
        "kPostsolveError": "postsolve_error",
        "kModelEmpty": "empty",
        "kOptimal": "optimal",
        "kInfeasible": "infeasible",
        "kUnboundedOrInfeasible": "infeasible_or_unbounded",
        "kUnbounded": "unbounded",
        "kObjectiveBound": "objective_bound",
        "kObjectiveTarget": "objective_target",
        "kTimeLimit": "time_limit",
        "kIterationLimit": "iteration_limit",
        "kUnknown": "unknown",
    }
    return mapping.get(name, name)


def objective_direction(model):
    require_package()
    highs_status, sense = model.getObjectiveSense()
    _check_highs_status(highs_status, "get objective sense")
    if sense == highspy.ObjSense.kMaximize:
        return "max"
    return "min"


def objective_value(model):
    if status(model) == "optimal":
        return float(model.getObjectiveValue())
    return np.nan


def configure_model(model, settings):
    require_package()
    feasibility_tol, optimality_tol = _tolerances(settings)
    presolve = "on" if getattr(settings, "presolve", False) else "off"

    options = {
        "output_flag": False,
        "log_to_console": False,
        "presolve": presolve,
        "solver": "simplex",
        "parallel": "off",
        "threads": 1,
        "primal_feasibility_tolerance": feasibility_tol,
        "dual_feasibility_tolerance": optimality_tol,
        "optimality_tolerance": optimality_tol,
    }

    legacy_options = getattr(settings, "hp_flags", {}) or {}
    apply_legacy_option_aliases(options, legacy_options)
    options["threads"] = 1

    for key, value in options.items():
        highs_status = model.setOptionValue(key, value)
        _check_highs_status(highs_status, "set option " + key)


def apply_legacy_option_aliases(options, legacy_options):
    if "TimeLimit" in legacy_options:
        options["time_limit"] = float(legacy_options["TimeLimit"])
    if "Threads" in legacy_options:
        options["threads"] = int(legacy_options["Threads"])
    if "Presolve" in legacy_options:
        options["presolve"] = "on" if bool(legacy_options["Presolve"]) else "off"
    if "Method" in legacy_options:
        method = legacy_options["Method"]
        if method == 0:
            options["solver"] = "simplex"
            options["simplex_strategy"] = 4
        elif method == 1:
            options["solver"] = "simplex"
            options["simplex_strategy"] = 1
        elif method == 2:
            options["solver"] = "ipm"


def set_time_limit(model, value):
    highs_status = model.setOptionValue("time_limit", float(value))
    _check_highs_status(highs_status, "set time limit")


def set_presolve(model, value):
    if value is True:
        option = "on"
    elif value is False:
        option = "off"
    else:
        option = "choose"
    highs_status = model.setOptionValue("presolve", option)
    _check_highs_status(highs_status, "set presolve")


def make_model(variable_names, settings):
    require_package()
    model = highspy.Highs()
    configure_model(model, settings)

    variable_names = [str(name) for name in variable_names]
    if variable_names:
        inf = _highs_inf()
        lower = np.full(len(variable_names), -inf, dtype=np.float64)
        upper = np.full(len(variable_names), inf, dtype=np.float64)
        highs_status = model.addVars(int(len(variable_names)), lower, upper)
        _check_highs_status(highs_status, "add variables")

    for index, name in enumerate(variable_names):
        highs_status = model.passColName(int(index), name)
        _check_highs_status(highs_status, "set column name")

    return model


def build_row_expressions(matrix, variables=None):
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    expressions = []
    for row in range(matrix.shape[0]):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        expressions.append(
            (
                matrix.indices[start:end].astype(np.int64, copy=True),
                matrix.data[start:end].astype(np.float64, copy=True),
            )
        )
    return expressions


def constraint_names(names):
    if names is None:
        return None
    return [str(name) for name in names]


def add_constraint_system(model, matrix, rhs, names=None, equality=False):
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if matrix.shape[0] != rhs.size:
        raise ValueError("Constraint matrix and rhs dimensions do not match.")
    if matrix.shape[1] != int(model.getNumCol()):
        raise ValueError("Constraint matrix and model column dimensions do not match.")

    row_names = constraint_names(names)
    if row_names is None:
        row_names = [
            f"constraint_{int(model.getNumRow()) + i}" for i in range(rhs.size)
        ]
    elif len(row_names) != rhs.size:
        raise ValueError("Constraint name count does not match rhs length.")

    first_row = int(model.getNumRow())
    row_count = int(rhs.size)
    if row_count == 0:
        return np.empty(0, dtype=np.int64)

    inf = _highs_inf()
    lower = rhs.copy() if equality else np.full(row_count, -inf, dtype=np.float64)
    upper = rhs.copy()
    highs_status = model.addRows(
        row_count,
        lower,
        upper,
        int(matrix.nnz),
        matrix.indptr[:-1].astype(np.int32, copy=False),
        matrix.indices.astype(np.int32, copy=False),
        matrix.data.astype(np.float64, copy=False),
    )
    _check_highs_status(highs_status, "add rows")

    rows = np.arange(first_row, first_row + row_count, dtype=np.int64)
    for row, name in zip(rows, row_names, strict=True):
        highs_status = model.passRowName(int(row), name)
        _check_highs_status(highs_status, "set row name")
    return rows


def linexpr(coefficients, variables=None):
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    nonzero = np.flatnonzero(coefficients)
    return nonzero.astype(np.int64), coefficients[nonzero].astype(np.float64)


def clear_quadratic_objective(model):
    require_package()
    dimension = int(model.getNumCol())
    start = np.zeros(dimension + 1, dtype=np.int32)
    highs_status = model.passHessian(
        dimension,
        0,
        int(highspy.HessianFormat.kTriangular.value),
        start,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    _check_highs_status(highs_status, "clear quadratic objective")


def set_linear_objective(model, expression, maximize=False):
    require_package()
    clear_quadratic_objective(model)

    if isinstance(expression, tuple) and len(expression) == 2:
        index = np.asarray(expression[0], dtype=np.int64).reshape(-1)
        value = np.asarray(expression[1], dtype=np.float64).reshape(-1)
    else:
        coefficients = np.asarray(expression, dtype=np.float64).reshape(-1)
        if coefficients.size != int(model.getNumCol()):
            raise ValueError("Objective dimension does not match model column count.")
        index = np.flatnonzero(coefficients)
        value = coefficients[index]

    if index.size != value.size:
        raise ValueError("Objective index and value arrays differ in length.")

    dimension = int(model.getNumCol())
    if np.any(index < 0) or np.any(index >= dimension):
        raise ValueError("Objective column index out of bounds.")

    sense = highspy.ObjSense.kMaximize if maximize else highspy.ObjSense.kMinimize
    highs_status = model.changeObjectiveSense(sense)
    _check_highs_status(highs_status, "change objective sense")

    old_costs = np.asarray(list(model.getLp().col_cost_), dtype=np.float64)
    changed = np.union1d(np.flatnonzero(old_costs), index).astype(np.int32)
    if changed.size == 0:
        return
    costs = np.zeros(changed.size, dtype=np.float64)
    if index.size:
        order = np.argsort(index)
        sorted_index = index[order]
        sorted_value = value[order]
        positions = np.searchsorted(changed, sorted_index)
        costs[positions] = sorted_value
    highs_status = model.changeColsCost(int(changed.size), changed, costs)
    _check_highs_status(highs_status, "change column costs")


def set_quadratic_identity_objective(model):
    require_package()
    zero_objective = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
    )
    set_linear_objective(model, zero_objective, maximize=False)

    dimension = int(model.getNumCol())
    start = np.arange(dimension + 1, dtype=np.int32)
    index = np.arange(dimension, dtype=np.int32)
    value = np.full(dimension, 2.0, dtype=np.float64)
    highs_status = model.passHessian(
        dimension,
        dimension,
        int(highspy.HessianFormat.kTriangular.value),
        start,
        index,
        value,
    )
    _check_highs_status(highs_status, "set quadratic objective")


def optimize(model, settings=None):
    highs_status = model.run()
    if _highs_run_failed(highs_status):
        _check_highs_status(highs_status, "run optimizer")
    if (
        settings is not None
        and status(model) != "optimal"
        and getattr(settings, "presolve", None) == "auto"
    ):
        model.clearSolver()
        set_presolve(model, True)
        highs_status = model.run()
        if _highs_run_failed(highs_status):
            _check_highs_status(highs_status, "run optimizer with presolve")
        set_presolve(model, "auto")
    return highs_status


def solution(model, size=None):
    if size is None:
        size = int(model.getNumCol())
    if status(model) != "optimal":
        result = np.empty(size, dtype=np.float64)
        result[:] = np.nan
        return result
    values = np.asarray(list(model.getSolution().col_value), dtype=np.float64)
    if values.size == size:
        return values
    return values[:size]


def reduced_costs(model, size=None):
    if size is None:
        size = int(model.getNumCol())
    if status(model) != "optimal":
        result = np.empty(size, dtype=np.float64)
        result[:] = np.nan
        return result
    values = np.asarray(list(model.getSolution().col_dual), dtype=np.float64)
    if values.size == size:
        return values
    return values[:size]


def change_row_bounds(model, row, lower, upper):
    highs_status = model.changeRowBounds(int(row), float(lower), float(upper))
    _check_highs_status(highs_status, "change row bounds")


def set_row_upper(model, row, rhs):
    change_row_bounds(model, row, -_highs_inf(), float(rhs))


def set_row_equality(model, row, rhs):
    change_row_bounds(model, row, float(rhs), float(rhs))


def delete_row(model, row):
    highs_status = model.deleteRows(1, np.asarray([int(row)], dtype=np.int32))
    _check_highs_status(highs_status, "delete row")


def shift_row_indices_after_delete(row_indices, deleted_row):
    rows = np.asarray(row_indices, dtype=np.int64).copy()
    rows[rows == int(deleted_row)] = -1
    rows[rows > int(deleted_row)] -= 1
    return rows


def solve(obj, A, b, settings, S=None, h=None):
    variable_names = [str(row) for row in range(A.shape[1])]
    model = make_model(variable_names, settings)
    set_linear_objective(model, linexpr(obj), maximize=False)
    add_constraint_system(model, A, b, equality=False)
    if S is not None:
        assert h is not None
        add_constraint_system(model, S, h, equality=True)
    optimize(model, settings)
    return solution(model, A.shape[1]), model


def solve_model(obj, model):
    set_linear_objective(model, linexpr(obj), maximize=False)
    optimize(model)
    return solution(model), model


def regularize_chebyshev_center(obj_val, model):
    lower_bound = float(np.squeeze(obj_val)) / 2.0
    if not np.isfinite(lower_bound):
        raise ValueError("Cannot regularize Chebyshev center with non-finite radius.")

    lp = model.getLp()
    radius_column = int(model.getNumCol()) - 1
    upper = float(np.asarray(list(lp.col_upper_), dtype=np.float64)[radius_column])
    highs_status = model.changeColBounds(radius_column, lower_bound, upper)
    _check_highs_status(highs_status, "set Chebyshev radius lower bound")
    set_quadratic_identity_objective(model)
    optimize(model)
    return solution(model), model


def polytope_to_model(polytope, settings):
    model = make_model(polytope.A.columns, settings)
    set_presolve(model, settings.presolve)
    add_constraint_system(
        model,
        polytope.A.values,
        polytope.b.values,
        names=polytope.b.index,
        equality=False,
    )
    if polytope.S is not None:
        add_constraint_system(
            model,
            polytope.S.values,
            polytope.h.values,
            names=polytope.h.index,
            equality=True,
        )
    return model


def model_to_polytope(model):
    A, b = constraints_as_mat(model, sense="<")
    S, h = constraints_as_mat(model, sense="=")
    if S.size > 0:
        return Polytope(A, b, S, h)
    return Polytope(A, b)


def constraints_as_mat(model, sense="<"):
    if sense not in {"<", "="}:
        raise ValueError

    lp = model.getLp()
    matrix = _lp_matrix_to_csr(lp)
    lower = np.asarray(list(lp.row_lower_), dtype=np.float64)
    upper = np.asarray(list(lp.row_upper_), dtype=np.float64)
    columns = _col_names(model)
    rows = []
    duplicate_counts = {}

    for row in range(int(lp.num_row_)):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        row_indices = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        coefficients = {
            columns[int(index)]: float(value)
            for index, value in zip(row_indices, row_values, strict=True)
        }
        raw_name = _row_name(model, row)
        has_lower = _finite(lower[row])
        has_upper = _finite(upper[row])
        is_equality = has_lower and has_upper and lower[row] == upper[row]

        if sense == "=":
            if is_equality:
                _append_named_row(
                    rows,
                    duplicate_counts,
                    raw_name,
                    coefficients,
                    upper[row],
                )
            continue

        if is_equality:
            continue
        if has_upper:
            _append_named_row(
                rows,
                duplicate_counts,
                raw_name,
                coefficients,
                upper[row],
            )
        if has_lower:
            lower_coefficients = {name: -value for name, value in coefficients.items()}
            _append_named_row(
                rows,
                duplicate_counts,
                raw_name,
                lower_coefficients,
                -lower[row],
            )

    if not rows:
        return (
            pd.DataFrame(columns=columns, dtype=np.float64),
            pd.Series(dtype=np.float64),
        )

    rhs = pd.Series(
        [row_rhs for _, _, row_rhs in rows],
        index=[row_name for row_name, _, _ in rows],
        dtype=np.float64,
    )
    coefficient_frame = pd.DataFrame(
        0.0,
        index=rhs.index,
        columns=columns,
        dtype=np.float64,
    )
    for row_name, coefficients, _ in rows:
        for variable_name, value in coefficients.items():
            coefficient_frame.loc[row_name, variable_name] = value
    return coefficient_frame, rhs


def check_tolerances(model):
    feasibility = float(_highs_option(model, "primal_feasibility_tolerance", 1e-9))
    optimality = float(_highs_option(model, "dual_feasibility_tolerance", 1e-8))
    feasibility_threshold = feasibility * default_accepted_tol_violation
    optimality_threshold = optimality * default_accepted_tol_violation

    lp = model.getLp()
    matrix = _lp_matrix_to_csr(lp)
    lower = np.asarray(list(lp.row_lower_), dtype=np.float64)
    upper = np.asarray(list(lp.row_upper_), dtype=np.float64)
    x = solution(model)
    activity = matrix @ x

    worst_violation = 0.0
    for row in range(int(lp.num_row_)):
        if _finite(upper[row]):
            worst_violation = max(worst_violation, activity[row] - upper[row])
        if _finite(lower[row]):
            worst_violation = max(worst_violation, lower[row] - activity[row])

    if worst_violation > feasibility_threshold:
        raise ValueError("Feasibility tolerance violated")

    costs = reduced_costs(model)
    if costs.size == 0:
        return
    objective_sense = float(objective_direction(model) == "max") * 2 - 1
    opt_violation = np.max(costs * objective_sense)
    if opt_violation > optimality_threshold:
        raise ValueError("Optimality tolerance violated")


def get_opt(model, settings):
    model_status = status(model)
    if model_status == "optimal":
        if settings.check_lps:
            check_tolerances(model)
        return objective_value(model)
    if model_status in {"infeasible", "undefined", "unknown"}:
        model.clearSolver()
        optimize(model, settings)
        if status(model) == "optimal":
            return objective_value(model)
        raise ValueError(
            "Optimization fails despite resetting; solver status: " + str(status(model))
        )
    raise ValueError("Optimization failed with solver status: " + str(model_status))
