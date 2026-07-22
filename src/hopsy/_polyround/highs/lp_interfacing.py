import numpy as np
import pandas as pd
from scipy import sparse as sp

try:
    import highspy
except Exception:
    highspy = None

from hopsy._polyround.default_settings import default_accepted_tol_violation
from hopsy._polyround.polytope import Polytope


def verbose_print(settings, backend, message):
    if getattr(settings, "verbose", bool(settings)):
        print(f"[{backend}] {message}")


class Configuration:
    def __init__(self, problem, tolerances=None, presolve=False, lp_method="primal"):
        self.problem = problem
        self.tolerances = tolerances or self
        self.lp_method = lp_method
        self.presolve = presolve

    @property
    def feasibility(self):
        status, value = self.problem.getOptionValue("primal_feasibility_tolerance")
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not read HiGHS feasibility tolerance.")
        return value

    @feasibility.setter
    def feasibility(self, value):
        status = self.problem.setOptionValue(
            "primal_feasibility_tolerance", float(value)
        )
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Invalid HiGHS feasibility tolerance.")

    @property
    def optimality(self):
        status, value = self.problem.getOptionValue("dual_feasibility_tolerance")
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not read HiGHS optimality tolerance.")
        return value

    @optimality.setter
    def optimality(self, value):
        value = float(value)
        statuses = (
            self.problem.setOptionValue("dual_feasibility_tolerance", value),
            self.problem.setOptionValue("optimality_tolerance", value),
        )
        if any(status != highspy.HighsStatus.kOk for status in statuses):
            raise ValueError("Invalid HiGHS optimality tolerance.")

    @property
    def lp_method(self):
        status, solver = self.problem.getOptionValue("solver")
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not read HiGHS LP method.")
        if solver == "ipm":
            return "barrier"
        if solver != "simplex":
            if getattr(self, "_lp_method", None) in {
                "concurrent",
                "deterministic_concurrent",
            }:
                return self._lp_method
            return "auto"

        status, strategy = self.problem.getOptionValue("simplex_strategy")
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not read HiGHS simplex strategy.")
        methods = {
            0: "auto",
            1: "dual",
            2: "dual",
            3: "dual",
            4: "primal",
        }
        return methods.get(strategy, "auto")

    @lp_method.setter
    def lp_method(self, value):
        methods = {
            "auto": ("choose", 0),
            "primal": ("simplex", 4),
            "dual": ("simplex", 1),
            "barrier": ("ipm", None),
            "concurrent": ("choose", None),
            "deterministic_concurrent": ("choose", None),
        }
        if value not in methods:
            raise ValueError("Invalid LP method.")

        solver, strategy = methods[value]
        status = self.problem.setOptionValue("solver", solver)
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Invalid LP method.")
        if strategy is not None:
            status = self.problem.setOptionValue("simplex_strategy", strategy)
            if status != highspy.HighsStatus.kOk:
                raise ValueError("Invalid LP method.")
        self._lp_method = value

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        if value is True:
            option = "on"
        elif value is False:
            option = "off"
        else:
            option = "choose"
        status = self.problem.setOptionValue("presolve", option)
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Invalid HiGHS presolve setting.")
        self._presolve = value


class Objective:
    def __init__(self, model):
        self._model = model

    @property
    def direction(self):
        status, sense = self._model.problem.getObjectiveSense()
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not read HiGHS objective sense.")
        if sense == highspy.ObjSense.kMinimize:
            return "min"
        return "max"

    @property
    def value(self):
        info = self._model.problem.getInfo()
        status = self._model.status
        no_solution_statuses = {
            "loaded",
            "infeasible",
            "infeasible_or_unbounded",
            "unbounded",
            "cutoff",
        }
        if status == "optimal" or (
            status not in no_solution_statuses
            and info.primal_solution_status
            == highspy.SolutionStatus.kSolutionStatusFeasible
        ):
            return self._model.problem.getObjectiveValue()
        return np.nan


class Model:
    def __init__(self, problem, variables, configuration, source_polytope=None):
        self.problem = problem
        self.variables = variables
        self.configuration = configuration
        self._source_polytope = source_polytope

    @property
    def status(self):
        return Interfacer.status(self.problem.getModelStatus())

    @property
    def primal_values(self):
        if not self.variables:
            return {}
        if (
            self.problem.getInfo().primal_solution_status
            != highspy.SolutionStatus.kSolutionStatusFeasible
        ):
            raise AttributeError("No HiGHS primal solution is available.")
        values = self.problem.variableValues(self.variables)
        return {
            variable.name: value
            for variable, value in zip(self.variables, values, strict=True)
        }

    @property
    def reduced_costs(self):
        if not self.variables:
            return {}
        if (
            self.problem.getInfo().dual_solution_status
            != highspy.SolutionStatus.kSolutionStatusFeasible
        ):
            raise AttributeError("No HiGHS dual solution is available.")
        values = self.problem.variableDuals(self.variables)
        return {
            variable.name: value
            for variable, value in zip(self.variables, values, strict=True)
        }

    @property
    def objective(self):
        return Objective(self)

    def optimize(self):
        self.update()
        # NOTE: Repeated primal-simplex re-solves can hit basis cycling during
        # cleanup and return unknown status (see https://github.com/ERGO-Code/HiGHS/issues/1785). 
        
        # Store the configured LP method across temporary recovery fallbacks
        requested_method = self.configuration.lp_method

        # Model states treated as "retryable" solver failures
        def failed():
            return self.status in {"loaded", "undefined", "numeric"}

        # A clean solver state for each retry with the selected LP method
        def cold_run(method):
            if self.problem.clearSolver() != highspy.HighsStatus.kOk:
                raise RuntimeError("Could not reset HiGHS.")
            self.configuration.lp_method = method
            return self.problem.optimize()

        run_status = self.problem.optimize()

        # Try recovery with the requested method
        if failed():
            run_status = cold_run(requested_method)

        # Try dual-simplex method fallback after primal failure
        if failed() and requested_method == "primal":
            try:
                run_status = cold_run("dual")
            finally:
                self.configuration.lp_method = requested_method

        if failed():
            raise RuntimeError(
                f"HiGHS numerical failure: run={run_status}, "
                f"model={self.problem.getModelStatus()}"
            )

        if self.status != "optimal" and self.configuration.presolve == "auto":
            status = self.problem.clearSolver()
            if status != highspy.HighsStatus.kOk:
                raise ValueError("Could not reset HiGHS solver.")
            self.configuration.presolve = True
            self.problem.optimize()
            self.configuration.presolve = "auto"

    def update(self):
        pass


class Interfacer:
    @staticmethod
    def require_package():
        if highspy is None:
            raise ImportError("This backend requires highspy.")

    @staticmethod
    def status(status):
        if highspy is None:
            return str(status)
        mapping = {
            highspy.HighsModelStatus.kOptimal: "optimal",
            highspy.HighsModelStatus.kNotset: "loaded",
            highspy.HighsModelStatus.kInfeasible: "infeasible",
            highspy.HighsModelStatus.kUnboundedOrInfeasible: (
                "infeasible_or_unbounded"
            ),
            highspy.HighsModelStatus.kUnbounded: "unbounded",
            highspy.HighsModelStatus.kObjectiveBound: "cutoff",
            highspy.HighsModelStatus.kIterationLimit: "iteration_limit",
            highspy.HighsModelStatus.kTimeLimit: "time_limit",
            highspy.HighsModelStatus.kSolutionLimit: "solution_limit",
            highspy.HighsModelStatus.kInterrupt: "interrupted",
            highspy.HighsModelStatus.kHighsInterrupt: "interrupted",
            highspy.HighsModelStatus.kLoadError: "numeric",
            highspy.HighsModelStatus.kModelError: "numeric",
            highspy.HighsModelStatus.kPresolveError: "numeric",
            highspy.HighsModelStatus.kSolveError: "numeric",
            highspy.HighsModelStatus.kPostsolveError: "numeric",
            highspy.HighsModelStatus.kMemoryLimit: "numeric",
            highspy.HighsModelStatus.kUnknown: "undefined",
            highspy.HighsModelStatus.kObjectiveTarget: "suboptimal",
            highspy.HighsModelStatus.kModelEmpty: "optimal",
        }
        return mapping.get(status, str(status))

    @staticmethod
    def configuration(problem):
        return Configuration(problem)

    @staticmethod
    def make_model(variable_names, settings):
        Interfacer.require_package()
        problem = highspy.Highs()
        status = problem.setOptionValue("output_flag", bool(settings.sgp))
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not configure HiGHS output.")
        wrapper = Model(
            problem=problem,
            variables=[],
            configuration=Interfacer.configuration(problem),
        )
        Interfacer.configure_model(wrapper, settings)
        wrapper.variables = [
            problem.addVariable(lb=-highspy.kHighsInf, name=str(name))
            for name in variable_names
        ]
        wrapper.update()
        return wrapper

    @staticmethod
    def build_row_expressions(matrix, variables):
        matrix = sp.csr_matrix(matrix, dtype=np.float64)
        expressions = []
        for row_ind in range(matrix.shape[0]):
            start = matrix.indptr[row_ind]
            end = matrix.indptr[row_ind + 1]
            coeffs = matrix.data[start:end].tolist()
            cols = matrix.indices[start:end].tolist()
            expressions.append(
                highspy.Highs.qsum(
                    coefficient * variables[column]
                    for coefficient, column in zip(coeffs, cols, strict=True)
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
        matrix = sp.csr_matrix(matrix, dtype=np.float64, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if rhs.size != matrix.shape[0]:
            raise ValueError("Constraint matrix and rhs dimensions do not match.")
        constraint_names = Interfacer.constraint_names(names)
        if constraint_names is not None and len(constraint_names) != rhs.size:
            raise ValueError("Constraint name count does not match rhs length.")
        lower = (
            rhs
            if equality
            else np.full(rhs.shape, -highspy.kHighsInf, dtype=np.float64)
        )
        first_row = model.problem.getNumRow()
        status = model.problem.addRows(
            matrix.shape[0],
            lower,
            rhs,
            matrix.nnz,
            matrix.indptr[:-1].astype(np.int32, copy=False),
            matrix.indices.astype(np.int32, copy=False),
            matrix.data.astype(np.float64, copy=False),
        )
        if status not in {
            highspy.HighsStatus.kOk,
            highspy.HighsStatus.kWarning,
        }:
            raise ValueError(f"Could not add HiGHS constraint system: {status}")

        if constraint_names is not None:
            row_indices = range(first_row, model.problem.getNumRow())
            for index, name in zip(row_indices, constraint_names, strict=True):
                status = model.problem.passRowName(index, name)
                if status != highspy.HighsStatus.kOk:
                    raise ValueError("Could not name HiGHS constraint.")
        model.update()
        return model.problem.getConstrs()[first_row:]

    @staticmethod
    def linexpr(coefficients, variables):
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        nonzero = np.flatnonzero(coefficients)
        return highspy.Highs.qsum(
            float(coefficients[index]) * variables[index] for index in nonzero.tolist()
        )

    @staticmethod
    def solution(model, size=None):
        if model.status == "optimal":
            return np.asarray(
                model.problem.variableValues(model.variables),
                dtype=np.float64,
            )
        if size is None:
            size = len(model.variables)
        x = np.zeros(size, dtype=np.float64)
        x[:] = np.nan
        return x

    @staticmethod
    def configure_model(m, settings):
        problem = m.problem if hasattr(m, "problem") else m
        status = problem.setOptionValue("output_flag", bool(settings.sgp))
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not configure HiGHS output.")
        verbose_print(settings, "highs", "hp flags=" + str(settings.hp_flags))
        aliases = {
            "FeasibilityTol": ("primal_feasibility_tolerance",),
            "OptimalityTol": (
                "dual_feasibility_tolerance",
                "optimality_tolerance",
            ),
        }
        for key, val in settings.hp_flags.items():
            for option in aliases.get(key, (key,)):
                status = problem.setOptionValue(option, val)
                if status != highspy.HighsStatus.kOk:
                    raise ValueError(f"Invalid HiGHS option: {key}")
        status = problem.setOptionValue("threads", 1)
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not configure HiGHS threads.")

    @staticmethod
    def solve(obj, A, b, settings, S=None, h=None):
        variable_names = [str(r) for r in range(A.shape[1])]
        model = Interfacer.make_model(variable_names, settings)
        model.problem.setObjective(
            Interfacer.linexpr(obj, model.variables),
            highspy.ObjSense.kMinimize,
        )
        Interfacer.add_constraint_system(model, A, b, equality=False)
        if S is not None:
            assert h is not None
            Interfacer.add_constraint_system(model, S, h, equality=True)
        model.optimize()
        return Interfacer.solution(model, A.shape[1]), model

    @staticmethod
    def solve_model(obj, m):

        if m.problem.getHessianNumNz() > 0:
            dimension = len(m.variables)
            status = m.problem.passHessian(
                dimension,
                0,
                highspy.HessianFormat.kTriangular.value,
                np.zeros(dimension + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            )
            if status != highspy.HighsStatus.kOk:
                raise ValueError("Could not clear HiGHS quadratic objective.")
        m.problem.setObjective(
            Interfacer.linexpr(obj, m.variables),
            highspy.ObjSense.kMinimize,
        )
        m.optimize()
        return Interfacer.solution(m), m

    @staticmethod
    def regularize_chebyshev_center(obj_val, m):

        lower_bound = float(np.squeeze(obj_val)) / 2.0
        last_var = m.variables[-1]
        m.problem.addConstr(last_var >= lower_bound)

        m.problem.setObjective(highspy.Highs.qsum([]), highspy.ObjSense.kMinimize)
        dimension = len(m.variables)
        status = m.problem.passHessian(
            dimension,
            dimension,
            highspy.HessianFormat.kTriangular.value,
            np.arange(dimension + 1, dtype=np.int32),
            np.arange(dimension, dtype=np.int32),
            np.full(dimension, 2.0, dtype=np.float64),
        )
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not set HiGHS quadratic objective.")

        status, solver = m.problem.getOptionValue("solver")
        if status != highspy.HighsStatus.kOk:
            raise ValueError("Could not read HiGHS solver.")
        if solver == "ipm":
            status = m.problem.setOptionValue("solver", "choose")
            if status != highspy.HighsStatus.kOk:
                raise ValueError("Could not configure HiGHS quadratic solver.")
        try:
            m.optimize()
        finally:
            if solver == "ipm":
                status = m.problem.setOptionValue("solver", solver)
                if status != highspy.HighsStatus.kOk:
                    raise ValueError("Could not restore HiGHS solver.")
        return Interfacer.solution(m), m

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
        expression = m.problem.getExpr(constr)
        coefficients = {}
        for index, value in zip(
            expression.idxs,
            expression.vals,
            strict=True,
        ):
            coefficients[m.variables[index].name] = float(value)

        lower, upper = expression.bounds
        if lower == upper:
            rhs = float(upper)
            sense = "="
        elif upper == highspy.kHighsInf:
            coefficients = {name: -value for name, value in coefficients.items()}
            rhs = -float(lower)
            sense = "<"
        else:
            rhs = float(upper)
            sense = "<"
        return sense, coefficients, rhs

    @staticmethod
    def constraints_as_mat(m, sense="<"):
        r_names = [variable.name for variable in m.variables]

        if sense == "<":
            target_sense = "<"
        elif sense == "=":
            target_sense = "="
        else:
            raise ValueError

        rows = []
        duplicate_counts = {}
        for constr in m.problem.getConstrs():
            live_sense, coefficients, rhs = Interfacer.constraint_record(m, constr)
            if live_sense != target_sense:
                continue

            status, raw_name = m.problem.getRowName(int(constr))
            if status != highspy.HighsStatus.kOk or not raw_name:
                raw_name = "constraint"
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
        values = m.problem.variableValues(m.variables)
        worst_violation = 0.0
        for constr in m.problem.getConstrs():
            expression = m.problem.getExpr(constr)
            activity = 0.0
            for index, coefficient in zip(
                expression.idxs,
                expression.vals,
                strict=True,
            ):
                activity += coefficient * values[index]

            lower, upper = expression.bounds
            if lower == upper:
                violation = abs(activity - upper)
            elif upper == highspy.kHighsInf:
                violation = lower - activity
            elif lower == -highspy.kHighsInf:
                violation = activity - upper
            else:
                violation = max(lower - activity, activity - upper)

            worst_violation = max(worst_violation, violation)

        if worst_violation > feasibility_threshold:
            raise ValueError("Feasibility tolerance violated")

        r_costs = np.array(list(m.reduced_costs.values()))
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
        if m.status in {"infeasible", "check_original_solver_status", "undefined"}:
            m.problem.clearSolver()
            m.optimize()
            if m.status == "optimal":
                return m.objective.value
            print("Solver status: " + str(m.status))
            raise ValueError("Optimization fails despite resetting")
        print("Solver status: " + str(m.status))
        raise ValueError
