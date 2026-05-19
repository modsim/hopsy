# ©2020-​2021 ETH Zurich, Axel Theorell
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, optimize
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

_CHEBYSHEV_REGULARIZATION_RADIUS_FRACTION = 0.5
_CHEBYSHEV_REGULARIZATION_RCOND = 1e-12
_CHEBYSHEV_REGULARIZATION_FTOL = 1e-12


def verbose_print(settings, backend, message):
    if getattr(settings, "verbose", bool(settings)):
        print(f"[{backend}] {message}")


def _highs_option(problem, name, default=None):
    status, value = problem.getOptionValue(name)
    if str(status).endswith("kOk"):
        return value
    return default


def _highs_inf():
    if highspy is None:
        return np.inf
    return float(highspy.kHighsInf)


def _check_highs_status(status, action):
    if not str(status).endswith("kOk"):
        raise RuntimeError(f"HiGHS failed to {action}: {status}")


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


class Configuration:
    def __init__(self, problem, tolerances=None, presolve=False, lp_method="primal"):
        self.problem = problem
        self.tolerances = tolerances or self
        self.lp_method = lp_method
        self.presolve = presolve

    @property
    def feasibility(self):
        return float(_highs_option(self.problem, "primal_feasibility_tolerance", 1e-9))

    @feasibility.setter
    def feasibility(self, value):
        self.problem.setOptionValue("primal_feasibility_tolerance", float(value))

    @property
    def optimality(self):
        return float(_highs_option(self.problem, "dual_feasibility_tolerance", 1e-8))

    @optimality.setter
    def optimality(self, value):
        value = float(value)
        self.problem.setOptionValue("dual_feasibility_tolerance", value)
        self.problem.setOptionValue("optimality_tolerance", value)

    @property
    def lp_method(self):
        solver = _highs_option(self.problem, "solver", "simplex")
        if solver == "ipm":
            return "barrier"
        strategy = int(_highs_option(self.problem, "simplex_strategy", 1))
        if strategy == 4:
            return "primal"
        if strategy == 1:
            return "dual"
        return "auto"

    @lp_method.setter
    def lp_method(self, value):
        if value == "barrier":
            self.problem.setOptionValue("solver", "ipm")
            return
        if value in {"auto", "primal", "dual"}:
            self.problem.setOptionValue("solver", "simplex")
            if value == "primal":
                self.problem.setOptionValue("simplex_strategy", 4)
            elif value == "dual":
                self.problem.setOptionValue("simplex_strategy", 1)
            else:
                self.problem.setOptionValue("simplex_strategy", 0)
            return
        if value in {"concurrent", "deterministic_concurrent"}:
            self.problem.setOptionValue("solver", "choose")
            return
        raise ValueError("Invalid LP method.")

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        if value is True:
            self.problem.setOptionValue("presolve", "on")
        elif value is False:
            self.problem.setOptionValue("presolve", "off")
        else:
            self.problem.setOptionValue("presolve", "choose")
        self._presolve = value


class Objective:
    def __init__(self, model):
        self._model = model

    @property
    def direction(self):
        if self._model._maximize:
            return "max"
        return "min"

    @property
    def value(self):
        if self._model.status == "optimal":
            return float(self._model.problem.getInfo().objective_function_value)
        return np.nan


class Variable:
    def __init__(self, model, index, name):
        self._model = model
        self.index = int(index)
        self.VarName = str(name)

    @property
    def X(self):
        return float(self._model.solution_vector()[self.index])

    @property
    def RC(self):
        return float(self._model.reduced_cost_vector()[self.index])


@dataclass(frozen=True)
class RowExpression:
    index: np.ndarray
    value: np.ndarray


class Constraint:
    def __init__(self, model, index, name, coefficients, rhs, sense):
        self._model = model
        self.index = int(index)
        self.ConstrName = str(name)
        self.coefficients = dict(coefficients)
        self._rhs = float(rhs)
        self._sense = Interfacer.normalize_sense(sense)
        self.active = True

    @property
    def RHS(self):
        return self._rhs

    @RHS.setter
    def RHS(self, value):
        self._rhs = float(value)
        self._model.apply_constraint_bounds(self)

    @property
    def Sense(self):
        return self._sense

    @Sense.setter
    def Sense(self, value):
        self._sense = Interfacer.normalize_sense(value)
        self._model.apply_constraint_bounds(self)


class ConstraintList(list):
    def tolist(self):
        return list(self)


class Model:
    def __init__(self, problem, variables, configuration, source_polytope=None):
        self.problem = problem
        self.variables = variables
        self.configuration = configuration
        self._source_polytope = source_polytope
        self.constraints = []
        self._objective_index = np.empty(0, dtype=np.int64)
        self._objective_value = np.empty(0, dtype=np.float64)
        self._maximize = False
        self._has_quadratic_objective = False

    @property
    def status(self):
        return Interfacer.status(self.problem.getModelStatus())

    @property
    def primal_values(self):
        solution = self.solution_vector()
        return {var.VarName: solution[var.index] for var in self.variables}

    @property
    def reduced_costs(self):
        reduced_costs = self.reduced_cost_vector()
        return {var.VarName: reduced_costs[var.index] for var in self.variables}

    @property
    def objective(self):
        return Objective(self)

    def optimize(self):
        self.update()
        self.problem.run()
        if self.status != "optimal" and self.configuration.presolve == "auto":
            self.problem.clearSolver()
            self.configuration.presolve = True
            self.problem.run()
            self.configuration.presolve = "auto"

    def update(self):
        return None

    def set_time_limit(self, value):
        self.problem.setOptionValue("time_limit", float(value))

    def add_variables(self, names, lower=None, upper=None):
        names = [str(name) for name in names]
        if not names:
            return []
        first_index = int(self.problem.getNumCol())
        count = len(names)
        inf = _highs_inf()
        if lower is None:
            lower = np.full(count, -inf, dtype=np.float64)
        else:
            lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        if upper is None:
            upper = np.full(count, inf, dtype=np.float64)
        else:
            upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        if lower.size != count or upper.size != count:
            raise ValueError("Variable bound count does not match variable count.")
        status = self.problem.addVars(int(count), lower, upper)
        _check_highs_status(status, "add variables")
        variables = [
            Variable(self, first_index + index, name)
            for index, name in enumerate(names)
        ]
        self.variables.extend(variables)
        return variables

    def basis(self):
        try:
            return self.problem.getBasis()
        except Exception:
            return None

    def set_basis(self, basis):
        if basis is None:
            return
        try:
            self.problem.setBasis(basis)
        except Exception:
            return

    def set_sparse_objective(self, index, value, maximize=False):
        self.clear_quadratic_objective()
        index = np.asarray(index, dtype=np.int64).reshape(-1)
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if index.size != value.size:
            raise ValueError("Objective index and value arrays differ in length.")
        if index.size:
            order = np.argsort(index)
            index = index[order]
            value = value[order]

        if self._maximize != bool(maximize):
            sense = (
                highspy.ObjSense.kMaximize if maximize else highspy.ObjSense.kMinimize
            )
            self.problem.changeObjectiveSense(sense)
            self._maximize = bool(maximize)

        changed = np.union1d(self._objective_index, index)
        if changed.size:
            costs = np.zeros(changed.size, dtype=np.float64)
            if index.size:
                costs[np.searchsorted(changed, index)] = value
            self.problem.changeColsCost(
                int(changed.size),
                changed.astype(np.int32, copy=False),
                costs,
            )
        self._objective_index = index.copy()
        self._objective_value = value.copy()

    def set_dense_objective(self, coefficients, maximize=False):
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coefficients.size != len(self.variables):
            raise ValueError("Objective dimension does not match variable count.")
        index = np.flatnonzero(coefficients)
        self.set_sparse_objective(index, coefficients[index], maximize=maximize)

    def clear_quadratic_objective(self):
        if not self._has_quadratic_objective:
            return
        dimension = len(self.variables)
        start = np.zeros(dimension + 1, dtype=np.int32)
        status = self.problem.passHessian(
            int(dimension),
            0,
            int(highspy.HessianFormat.kTriangular.value),
            start,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
        _check_highs_status(status, "clear quadratic objective")
        self._has_quadratic_objective = False

    def set_quadratic_identity_objective(self):
        self.set_sparse_objective(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            maximize=False,
        )
        dimension = len(self.variables)
        start = np.arange(dimension + 1, dtype=np.int32)
        index = np.arange(dimension, dtype=np.int32)
        value = np.full(dimension, 2.0, dtype=np.float64)
        status = self.problem.passHessian(
            int(dimension),
            int(dimension),
            int(highspy.HessianFormat.kTriangular.value),
            start,
            index,
            value,
        )
        _check_highs_status(status, "set quadratic objective")
        self._has_quadratic_objective = True

    def apply_constraint_bounds(self, constr):
        if not constr.active:
            return
        inf = _highs_inf()
        if constr.Sense == "=":
            lower, upper = constr.RHS, constr.RHS
        elif constr.Sense == ">":
            lower, upper = constr.RHS, inf
        else:
            lower, upper = -inf, constr.RHS
        status = self.problem.changeRowBounds(
            int(constr.index), float(lower), float(upper)
        )
        _check_highs_status(status, "change row bounds")

    def remove_constraint(self, constr):
        if not constr.active:
            return
        deleted_index = int(constr.index)
        status = self.problem.deleteRows(
            1,
            np.asarray([deleted_index], dtype=np.int32),
        )
        _check_highs_status(status, "delete row")
        constr.active = False
        constr.index = -1
        for other in self.constraints:
            if other.active and other.index > deleted_index:
                other.index -= 1

    def solution_vector(self, size=None):
        if size is None:
            size = len(self.variables)
        if self.status != "optimal":
            solution = np.empty(size, dtype=np.float64)
            solution[:] = np.nan
            return solution
        return np.asarray(list(self.problem.getSolution().col_value), dtype=np.float64)

    def reduced_cost_vector(self, size=None):
        if size is None:
            size = len(self.variables)
        if self.status != "optimal":
            reduced = np.empty(size, dtype=np.float64)
            reduced[:] = np.nan
            return reduced
        return np.asarray(list(self.problem.getSolution().col_dual), dtype=np.float64)


class Interfacer:
    @staticmethod
    def require_package():
        if highspy is None:
            raise ImportError("This backend requires highspy.")

    @staticmethod
    def status(status):
        name = getattr(status, "name", str(status)).rsplit(".", 1)[-1]
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

    @staticmethod
    def configuration(problem):
        return Configuration(problem)

    @staticmethod
    def make_model(variable_names, settings):
        Interfacer.require_package()
        problem = highspy.Highs()
        wrapper = Model(
            problem=problem,
            variables=[],
            configuration=Interfacer.configuration(problem),
        )
        Interfacer.configure_model(wrapper, settings)
        variable_names = [str(name) for name in variable_names]
        if variable_names:
            inf = _highs_inf()
            lower = np.full(len(variable_names), -inf, dtype=np.float64)
            upper = np.full(len(variable_names), inf, dtype=np.float64)
            problem.addVars(int(len(variable_names)), lower, upper)
        wrapper.variables = [
            Variable(wrapper, index, name) for index, name in enumerate(variable_names)
        ]
        return wrapper

    @staticmethod
    def build_row_expressions(matrix, variables):
        matrix = sp.csr_matrix(matrix, dtype=np.float64)
        expressions = []
        for row_ind in range(matrix.shape[0]):
            start = matrix.indptr[row_ind]
            end = matrix.indptr[row_ind + 1]
            expressions.append(
                RowExpression(
                    index=matrix.indices[start:end].astype(np.int64, copy=True),
                    value=matrix.data[start:end].astype(np.float64, copy=True),
                )
            )
        return expressions

    @staticmethod
    def constraint_names(names):
        if names is None:
            return None
        return [str(name) for name in names]

    @staticmethod
    def add_constraint_system(model, matrix, rhs, names=None, equality=False):
        matrix = sp.csr_matrix(matrix, dtype=np.float64)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if matrix.shape[0] != rhs.size:
            raise ValueError("Constraint matrix and rhs dimensions do not match.")

        sense = "=" if equality else "<"
        constraint_names = Interfacer.constraint_names(names)
        if constraint_names is None:
            constraint_names = [
                f"constraint_{len(model.constraints) + i}" for i in range(rhs.size)
            ]
        elif len(constraint_names) != rhs.size:
            raise ValueError("Constraint name count does not match rhs length.")

        first_row = int(model.problem.getNumRow())
        constraints = ConstraintList()
        if rhs.size:
            inf = _highs_inf()
            lower = (
                rhs.copy() if equality else np.full(rhs.size, -inf, dtype=np.float64)
            )
            upper = rhs.copy()
            model.problem.addRows(
                int(matrix.shape[0]),
                lower,
                upper,
                int(matrix.nnz),
                matrix.indptr[:-1].astype(np.int32, copy=False),
                matrix.indices.astype(np.int32, copy=False),
                matrix.data.astype(np.float64, copy=False),
            )

        for row_ind in range(matrix.shape[0]):
            start = matrix.indptr[row_ind]
            end = matrix.indptr[row_ind + 1]
            coefficients = {
                int(col): float(value)
                for col, value in zip(
                    matrix.indices[start:end],
                    matrix.data[start:end],
                    strict=True,
                )
            }
            constraints.append(
                Constraint(
                    model,
                    first_row + row_ind,
                    constraint_names[row_ind],
                    coefficients,
                    rhs[row_ind],
                    sense,
                )
            )
        model.constraints.extend(constraints)
        model.update()
        return constraints

    @staticmethod
    def normalize_sense(sense):
        if sense in {"=", "=="}:
            return "="
        if sense in {">", ">="}:
            return ">"
        if sense in {"<", "<="}:
            return "<"
        raise ValueError("Unknown HiGHS constraint sense.")

    @staticmethod
    def set_objective(model, expression, maximize=False):
        if isinstance(expression, RowExpression):
            model.set_sparse_objective(
                expression.index,
                expression.value,
                maximize=maximize,
            )
            return
        coefficients = np.asarray(expression, dtype=np.float64).reshape(-1)
        model.set_dense_objective(coefficients, maximize=maximize)

    @staticmethod
    def linexpr(coefficients, variables):
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        nonzero = np.flatnonzero(coefficients)
        return RowExpression(nonzero, coefficients[nonzero])

    @staticmethod
    def solution(model, size=None):
        return model.solution_vector(size=size)

    @staticmethod
    def configure_model(m, settings):
        problem = m.problem if hasattr(m, "problem") else m
        feasibility_tol, optimality_tol = _tolerances(settings)

        options = {
            "output_flag": bool(getattr(settings, "sgp", False)),
            "log_to_console": bool(getattr(settings, "sgp", False)),
            "presolve": "on" if getattr(settings, "presolve", False) else "off",
            "solver": "simplex",
            "parallel": "off",
            "threads": 1,
            "primal_feasibility_tolerance": feasibility_tol,
            "dual_feasibility_tolerance": optimality_tol,
            "optimality_tolerance": optimality_tol,
        }

        legacy_options = getattr(settings, "hp_flags", {}) or {}
        Interfacer.apply_legacy_option_aliases(options, legacy_options)
        options["threads"] = 1

        verbose_print(settings, "highs", "options=" + str(options))
        for key, val in options.items():
            status = problem.setOptionValue(key, val)
            _check_highs_status(status, "set option " + key)

        handled_legacy_options = {
            "FeasibilityTol",
            "OptimalityTol",
            "TimeLimit",
            "Threads",
            "OutputFlag",
            "Presolve",
            "Method",
        }
        gurobi_only_options = {
            "NumericFocus",
            "MarkowitzTol",
        }
        for key in legacy_options:
            if key in gurobi_only_options:
                warnings.warn(
                    "legacy hp_flag " + key + " has no HiGHS equivalent and is ignored",
                    stacklevel=2,
                )
            elif key not in handled_legacy_options:
                warnings.warn(
                    "legacy hp_flag " + key + " not supported by backend HiGHS",
                    stacklevel=2,
                )

    @staticmethod
    def apply_legacy_option_aliases(options, legacy_options):
        if "TimeLimit" in legacy_options:
            options["time_limit"] = float(legacy_options["TimeLimit"])
        if "Threads" in legacy_options:
            options["threads"] = int(legacy_options["Threads"])
        if "OutputFlag" in legacy_options:
            output_flag = bool(legacy_options["OutputFlag"])
            options["output_flag"] = output_flag
            options["log_to_console"] = output_flag
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

    @staticmethod
    def solve(obj, A, b, settings, S=None, h=None):
        variable_names = [str(r) for r in range(A.shape[1])]
        model = Interfacer.make_model(variable_names, settings)
        Interfacer.set_objective(
            model,
            Interfacer.linexpr(obj, model.variables),
            maximize=False,
        )
        Interfacer.add_constraint_system(model, A, b, equality=False)
        if S is not None:
            assert h is not None
            Interfacer.add_constraint_system(model, S, h, equality=True)
        model.optimize()
        return Interfacer.solution(model, A.shape[1]), model

    @staticmethod
    def solve_model(obj, m):

        Interfacer.set_objective(
            m,
            Interfacer.linexpr(obj, m.variables),
            maximize=False,
        )
        m.optimize()
        return Interfacer.solution(m), m

    @staticmethod
    def regularize_chebyshev_center(obj_val, m):
        return Interfacer.minimum_norm_chebyshev_center(obj_val, m), m

    @staticmethod
    def minimum_norm_chebyshev_center(obj_val, m):
        chebyshev_radius = float(np.squeeze(obj_val))
        if not np.isfinite(chebyshev_radius):
            raise ValueError(
                "Cannot regularize Chebyshev center with non-finite radius."
            )
        # Match the Gurobi backend's regularization semantics: keep a radius of
        # at least half the Chebyshev optimum and select the minimum L2-norm
        # point on that face. HiGHS' native QP active-set solver can fail on
        # these degenerate LP faces, so solve the equivalent projection in the
        # equality null space with SciPy's dense SLSQP.
        lower_bound = chebyshev_radius * _CHEBYSHEV_REGULARIZATION_RADIUS_FRACTION

        inequality_matrix, inequality_rhs, equality_matrix, equality_rhs = (
            Interfacer.regularized_chebyshev_constraints(m, lower_bound)
        )
        feasibility_tol = (
            m.configuration.tolerances.feasibility * default_accepted_tol_violation
        )
        return Interfacer.minimum_norm_point(
            len(m.variables),
            inequality_matrix,
            inequality_rhs,
            equality_matrix,
            equality_rhs,
            feasibility_tol,
        )

    @staticmethod
    def regularized_chebyshev_constraints(m, lower_bound):
        dimension = len(m.variables)
        inequalities = []
        inequality_rhs = []
        equalities = []
        equality_rhs = []

        for constr in m.constraints:
            if not constr.active:
                continue

            row = np.zeros(dimension, dtype=np.float64)
            for index, value in constr.coefficients.items():
                if index < dimension:
                    row[index] = value

            if constr.Sense == "=":
                equalities.append(row)
                equality_rhs.append(constr.RHS)
            elif constr.Sense == "<":
                inequalities.append(row)
                inequality_rhs.append(constr.RHS)
            elif constr.Sense == ">":
                inequalities.append(-row)
                inequality_rhs.append(-constr.RHS)
            else:
                raise ValueError("Unknown HiGHS constraint sense.")

        radius_lower = np.zeros(dimension, dtype=np.float64)
        radius_lower[-1] = -1.0
        inequalities.append(radius_lower)
        inequality_rhs.append(-float(lower_bound))

        if inequalities:
            inequality_matrix = np.vstack(inequalities)
            inequality_rhs = np.asarray(inequality_rhs, dtype=np.float64)
        else:
            inequality_matrix = np.empty((0, dimension), dtype=np.float64)
            inequality_rhs = np.empty(0, dtype=np.float64)

        if equalities:
            equality_matrix = np.vstack(equalities)
            equality_rhs = np.asarray(equality_rhs, dtype=np.float64)
        else:
            equality_matrix = np.empty((0, dimension), dtype=np.float64)
            equality_rhs = np.empty(0, dtype=np.float64)

        return inequality_matrix, inequality_rhs, equality_matrix, equality_rhs

    @staticmethod
    def minimum_norm_point(
        dimension,
        inequality_matrix,
        inequality_rhs,
        equality_matrix,
        equality_rhs,
        feasibility_tol,
    ):
        if equality_matrix.shape[0] > 0:
            particular, *_ = np.linalg.lstsq(
                equality_matrix,
                equality_rhs,
                rcond=_CHEBYSHEV_REGULARIZATION_RCOND,
            )
            equality_residual = np.max(
                np.abs(equality_matrix @ particular - equality_rhs)
            )
            if equality_residual > feasibility_tol:
                raise ValueError("Regularized Chebyshev equalities are infeasible.")
            null_space = linalg.null_space(
                equality_matrix,
                rcond=_CHEBYSHEV_REGULARIZATION_RCOND,
            )
        else:
            particular = np.zeros(dimension, dtype=np.float64)
            null_space = np.eye(dimension, dtype=np.float64)

        if null_space.shape[1] == 0:
            Interfacer.check_minimum_norm_feasibility(
                particular,
                inequality_matrix,
                inequality_rhs,
                feasibility_tol,
            )
            return particular

        reduced_matrix = inequality_matrix @ null_space
        reduced_rhs = inequality_rhs - inequality_matrix @ particular
        if reduced_matrix.shape[0] == 0 or np.max(-reduced_rhs) <= feasibility_tol:
            return particular

        def objective(y):
            return 0.5 * float(np.dot(y, y))

        def gradient(y):
            return y

        constraints = {
            "type": "ineq",
            "fun": lambda y: reduced_rhs - reduced_matrix @ y,
            "jac": lambda y: -reduced_matrix,
        }
        result = optimize.minimize(
            objective,
            np.zeros(null_space.shape[1], dtype=np.float64),
            jac=gradient,
            constraints=[constraints],
            method="SLSQP",
            options={
                "ftol": _CHEBYSHEV_REGULARIZATION_FTOL,
                "maxiter": max(1000, 10 * reduced_matrix.shape[1]),
                "disp": False,
            },
        )
        point = particular + null_space @ result.x
        Interfacer.check_minimum_norm_feasibility(
            point,
            inequality_matrix,
            inequality_rhs,
            feasibility_tol,
        )
        return point

    @staticmethod
    def check_minimum_norm_feasibility(
        point,
        inequality_matrix,
        inequality_rhs,
        feasibility_tol,
    ):
        if inequality_matrix.shape[0] == 0:
            return
        violation = np.max(inequality_matrix @ point - inequality_rhs)
        if violation > feasibility_tol:
            raise ValueError(
                "HiGHS failed to solve regularized Chebyshev center within "
                f"feasibility tolerance: violation={violation}"
            )

    @staticmethod
    def polytope_to_model(polytope, settings):
        model = Interfacer.make_model(polytope.A.columns, settings)
        model.configuration.presolve = settings.presolve
        Interfacer.add_constraint_system(
            model,
            polytope.A.values,
            polytope.b.values,
            names=polytope.b.index,
            equality=False,
        )
        if polytope.S is not None:
            Interfacer.add_constraint_system(
                model,
                polytope.S.values,
                polytope.h.values,
                names=polytope.h.index,
                equality=True,
            )
        model._source_polytope = polytope.copy()
        return model

    @staticmethod
    def model_to_polytope(m):
        A, b = Interfacer.constraints_as_mat(m, sense="<")
        S, h = Interfacer.constraints_as_mat(m, sense="=")
        if S.size > 0:
            return Polytope(A, b, S, h)
        return Polytope(A, b)

    @staticmethod
    def constraint_record(m, constr):
        coefficients = {
            m.variables[index].VarName: float(value)
            for index, value in constr.coefficients.items()
        }
        rhs = float(constr.RHS)
        sense = constr.Sense
        if sense == ">":
            coefficients = {name: -value for name, value in coefficients.items()}
            rhs = -rhs
            sense = "<"
        return sense, coefficients, rhs

    @staticmethod
    def constraints_as_mat(m, sense="<"):
        r_names = [var.VarName for var in m.variables]

        if sense not in {"<", "="}:
            raise ValueError

        rows = []
        duplicate_counts = {}
        for constr in m.constraints:
            if not constr.active:
                continue
            live_sense, coefficients, rhs = Interfacer.constraint_record(m, constr)
            if live_sense != sense:
                continue

            raw_name = constr.ConstrName or "constraint"
            duplicate_counts.setdefault(raw_name, 0)
            count = duplicate_counts[raw_name]
            duplicate_counts[raw_name] += 1
            row_name = raw_name if count == 0 else f"{raw_name}__{count}"
            rows.append((row_name, coefficients, rhs))

        if not rows:
            return pd.DataFrame(dtype=np.float64), pd.Series(dtype=np.float64)

        b = pd.Series(
            [rhs for _, _, rhs in rows],
            index=[row_name for row_name, _, _ in rows],
            dtype=np.float64,
        )
        c_df = pd.DataFrame(0.0, index=b.index, columns=r_names, dtype=np.float64)
        for row_name, coefficients, _ in rows:
            for var_name, value in coefficients.items():
                c_df.loc[row_name, var_name] = value
        return c_df, b

    @staticmethod
    def check_tolerances(m):
        feasibility_threshold = (
            m.configuration.tolerances.feasibility * default_accepted_tol_violation
        )
        solution = m.solution_vector()
        worst_violation = 0.0
        for constr in m.constraints:
            if not constr.active:
                continue
            activity = 0.0
            for index, value in constr.coefficients.items():
                activity += value * solution[index]

            rhs = constr.RHS
            if constr.Sense == "<":
                violation = activity - rhs
            elif constr.Sense == ">":
                violation = rhs - activity
            elif constr.Sense == "=":
                violation = abs(activity - rhs)
            else:
                raise ValueError("Unknown HiGHS constraint sense.")

            worst_violation = max(worst_violation, violation)

        if worst_violation > feasibility_threshold:
            raise ValueError("Feasibility tolerance violated")

        r_costs = np.array(list(m.reduced_costs.values()))
        if r_costs.size == 0:
            return
        sense = float(m.objective.direction == "max") * 2 - 1
        opt_violation = np.max(r_costs * sense)

        if (
            opt_violation
            > m.configuration.tolerances.optimality * default_accepted_tol_violation
        ):
            raise ValueError("Optimality tolerance violated")

    @staticmethod
    def get_opt(m, settings):
        if m.status == "optimal":
            if settings.check_lps:
                Interfacer.check_tolerances(m)
            return m.objective.value
        if m.status in {"infeasible", "undefined", "unknown"}:
            # print("model in infeasible state, resetting lp")
            m.problem.clearSolver()
            m.optimize()
            if m.status == "optimal":
                return m.objective.value
            print("Solver status: " + str(m.status))
            raise ValueError("Optimization fails despite resetting")
        print("Solver status: " + str(m.status))
        raise ValueError
