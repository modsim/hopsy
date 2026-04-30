# ©2020-​2021 ETH Zurich, Axel Theorell
import numpy as np
import pandas as pd
from scipy import sparse as sp

try:
    import gurobipy as gp
except Exception:
    gp = None

from hopsy._polyround.default_settings import (
    default_accepted_tol_violation,
    default_hp_flags,
)
from hopsy._polyround.mutable_classes.polytope import Polytope


class NativeToleranceConfiguration:
    def __init__(self, feasibility, optimality):
        self.feasibility = feasibility
        self.optimality = optimality


class NativeConfiguration:
    def __init__(self, presolve, feasibility, optimality):
        self.presolve = presolve
        self.lp_method = "primal"
        self.tolerances = NativeToleranceConfiguration(feasibility, optimality)


class NativeObjective:
    def __init__(self, model):
        self._model = model

    @property
    def direction(self):
        if self._model.problem.ModelSense == gp.GRB.MINIMIZE:
            return "min"
        return "max"

    @property
    def value(self):
        if self._model.problem.SolCount > 0:
            return self._model.problem.ObjVal
        return np.nan


class NativeGurobiModel:
    def __init__(self, problem, variables, configuration, source_polytope=None):
        self.problem = problem
        self.variables = variables
        self.configuration = configuration
        self._source_polytope = source_polytope

    @property
    def status(self):
        return GurobiInterfacer.gurobi_status(self.problem.Status)

    @property
    def primal_values(self):
        return {var.VarName: var.X for var in self.variables}

    @property
    def reduced_costs(self):
        return {var.VarName: var.RC for var in self.variables}

    @property
    def objective(self):
        return NativeObjective(self)

    def optimize(self):
        self.problem.optimize()

    def update(self):
        self.problem.update()


class GurobiInterfacer:
    @staticmethod
    def is_native_gurobi_model(model):
        return isinstance(model, NativeGurobiModel)

    @staticmethod
    def require_native_gurobi():
        if gp is None:
            raise ImportError("hopsy's native PolyRound backend requires gurobipy.")

    @staticmethod
    def gurobi_status(status):
        if gp is None:
            return str(status)
        mapping = {
            gp.GRB.OPTIMAL: "optimal",
            gp.GRB.INFEASIBLE: "infeasible",
            gp.GRB.INF_OR_UNBD: "infeasible_or_unbounded",
            gp.GRB.UNBOUNDED: "unbounded",
            gp.GRB.NUMERIC: "numeric_error",
            gp.GRB.SUBOPTIMAL: "suboptimal",
            gp.GRB.TIME_LIMIT: "time_limit",
            gp.GRB.INTERRUPTED: "interrupted",
        }
        return mapping.get(status, str(status))

    @staticmethod
    def native_configuration(settings):
        return NativeConfiguration(
            presolve=settings.presolve,
            feasibility=settings.hp_flags.get(
                "FeasibilityTol", default_hp_flags["FeasibilityTol"]
            ),
            optimality=settings.hp_flags.get(
                "OptimalityTol", default_hp_flags["OptimalityTol"]
            ),
        )

    @staticmethod
    def make_native_model(variable_names, settings):
        GurobiInterfacer.require_native_gurobi()
        problem = gp.Model()
        wrapper = NativeGurobiModel(
            problem=problem,
            variables=[],
            configuration=GurobiInterfacer.native_configuration(settings),
        )
        GurobiInterfacer.configure_gurobi_model(wrapper, settings)
        wrapper.variables = [
            problem.addVar(lb=-gp.GRB.INFINITY, name=str(name))
            for name in variable_names
        ]
        problem.update()
        return wrapper

    @staticmethod
    def build_native_row_expressions(matrix, variables):
        matrix = sp.csr_matrix(matrix, dtype=np.float64)
        expressions = []
        for row_ind in range(matrix.shape[0]):
            start = matrix.indptr[row_ind]
            end = matrix.indptr[row_ind + 1]
            coeffs = matrix.data[start:end].tolist()
            cols = matrix.indices[start:end].tolist()
            expressions.append(gp.LinExpr(coeffs, [variables[col] for col in cols]))
        return expressions

    @staticmethod
    def native_constraint_names(names):
        if names is None:
            return None
        return [str(name) for name in names]

    @staticmethod
    def add_native_constraint_system(model, matrix, rhs, names=None, equality=False):
        matrix = sp.csr_matrix(matrix, dtype=np.float64)
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        sense = "=" if equality else "<"
        constraint_names = GurobiInterfacer.native_constraint_names(names)
        if constraint_names is None:
            constraints = model.problem.addMConstr(matrix, model.variables, sense, rhs)
        else:
            constraints = model.problem.addMConstr(
                matrix,
                model.variables,
                sense,
                rhs,
                name=constraint_names,
            )
        model.update()
        return constraints

    @staticmethod
    def add_native_constraints(problem, expressions, rhs, names=None, equality=False):
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        constraints = []
        for row_ind, bound in enumerate(rhs):
            name = "" if names is None else str(names[row_ind])
            if equality:
                constr = problem.addConstr(
                    expressions[row_ind] == float(bound), name=name
                )
            else:
                constr = problem.addConstr(
                    expressions[row_ind] <= float(bound), name=name
                )
            constraints.append(constr)
        problem.update()
        return constraints

    @staticmethod
    def native_linexpr(coefficients, variables):
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        nonzero = np.flatnonzero(coefficients)
        if nonzero.size == 0:
            return gp.LinExpr()
        return gp.LinExpr(
            coefficients[nonzero].tolist(),
            [variables[ind] for ind in nonzero.tolist()],
        )

    @staticmethod
    def native_solution(model, size=None):
        if model.status == "optimal":
            return np.asarray([var.X for var in model.variables], dtype=np.float64)
        if size is None:
            size = len(model.variables)
        x = np.zeros(size, dtype=np.float64)
        x[:] = np.nan
        return x

    @staticmethod
    def configure_gurobi_model(m, settings):
        problem = m.problem if hasattr(m, "problem") else m
        if not settings.sgp:
            problem.setParam("OutputFlag", 0)
        else:
            problem.setParam("OutputFlag", 1)
        problem.setParam("Method", 0)
        if settings.verbose:
            print("Using the hp flags: " + str(settings.hp_flags))
        for key, val in settings.hp_flags.items():
            problem.setParam(key, val)
        problem.setParam("Threads", 1)
        if settings.presolve:
            problem.setParam("Presolve", 2)
        else:
            problem.setParam("Presolve", 0)
        if GurobiInterfacer.is_native_gurobi_model(m):
            m.configuration = GurobiInterfacer.native_configuration(settings)

    @staticmethod
    def gurobi_solve(obj, A, b, settings, S=None, h=None):
        variable_names = [str(r) for r in range(A.shape[1])]
        model = GurobiInterfacer.make_native_model(variable_names, settings)
        problem = model.problem
        problem.setObjective(
            GurobiInterfacer.native_linexpr(obj, model.variables),
            gp.GRB.MINIMIZE,
        )
        GurobiInterfacer.add_native_constraint_system(model, A, b, equality=False)
        if S is not None:
            assert h is not None
            GurobiInterfacer.add_native_constraint_system(model, S, h, equality=True)
        model.optimize()
        return GurobiInterfacer.native_solution(model, A.shape[1]), model

    @staticmethod
    def gurobi_solve_model(obj, m):
        if not GurobiInterfacer.is_native_gurobi_model(m):
            raise ValueError("The gurobi branch only supports native gurobi models.")

        m.problem.setObjective(
            GurobiInterfacer.native_linexpr(obj, m.variables),
            gp.GRB.MINIMIZE,
        )
        m.optimize()
        return GurobiInterfacer.native_solution(m), m

    @staticmethod
    def gurobi_regularize_chebyshev_center(obj_val, m):
        if not GurobiInterfacer.is_native_gurobi_model(m):
            raise ValueError("The gurobi branch only supports native gurobi models.")

        lower_bound = float(np.squeeze(obj_val)) / 2.0
        last_var = m.variables[-1]
        m.problem.addConstr(last_var >= lower_bound)
        quadratic_objective = gp.QuadExpr()
        for var in m.variables:
            quadratic_objective.add(var * var)
        m.problem.setObjective(quadratic_objective, gp.GRB.MINIMIZE)
        m.optimize()
        return GurobiInterfacer.native_solution(m), m

    @staticmethod
    def polytope_to_model(polytope, settings):
        model = GurobiInterfacer.make_native_model(polytope.A.columns, settings)
        GurobiInterfacer.add_native_constraint_system(
            model,
            polytope.A.values,
            polytope.b.values,
            names=polytope.b.index,
            equality=False,
        )
        if polytope.S is not None:
            GurobiInterfacer.add_native_constraint_system(
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
        A, b = GurobiInterfacer.constraints_as_mat(m, sense="<")
        S, h = GurobiInterfacer.constraints_as_mat(m, sense="=")
        if S.size > 0:
            return Polytope(A, b, S, h)
        return Polytope(A, b)

    @staticmethod
    def native_constraint_record(m, constr):
        row = m.problem.getRow(constr)
        coefficients = {}
        for index in range(row.size()):
            coefficients[row.getVar(index).VarName] = float(row.getCoeff(index))

        rhs = float(constr.RHS)
        sense = constr.Sense
        if sense == gp.GRB.GREATER_EQUAL:
            coefficients = {name: -value for name, value in coefficients.items()}
            rhs = -rhs
            sense = gp.GRB.LESS_EQUAL
        return sense, coefficients, rhs

    @staticmethod
    def constraints_as_mat(m, sense="<"):
        if not GurobiInterfacer.is_native_gurobi_model(m):
            raise ValueError("The gurobi branch only supports native gurobi models.")
        r_names = [var.VarName for var in m.variables]

        if sense == "<":
            target_sense = gp.GRB.LESS_EQUAL
        elif sense == "=":
            target_sense = gp.GRB.EQUAL
        else:
            raise ValueError

        rows = []
        duplicate_counts = {}
        for constr in m.problem.getConstrs():
            live_sense, coefficients, rhs = GurobiInterfacer.native_constraint_record(
                m, constr
            )
            if live_sense != target_sense:
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
        if not GurobiInterfacer.is_native_gurobi_model(m):
            raise ValueError("The gurobi branch only supports native gurobi models.")

        feasibility_threshold = (
            m.configuration.tolerances.feasibility * default_accepted_tol_violation
        )
        worst_violation = 0.0
        for constr in m.problem.getConstrs():
            row = m.problem.getRow(constr)
            activity = 0.0
            for ind in range(row.size()):
                activity += row.getCoeff(ind) * row.getVar(ind).X

            rhs = constr.RHS
            if constr.Sense == gp.GRB.LESS_EQUAL:
                violation = activity - rhs
            elif constr.Sense == gp.GRB.GREATER_EQUAL:
                violation = rhs - activity
            elif constr.Sense == gp.GRB.EQUAL:
                violation = abs(activity - rhs)
            else:
                raise ValueError("Unknown Gurobi constraint sense.")

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
                GurobiInterfacer.check_tolerances(m)
            return m.objective.value
        if (
            m.status == "infeasible"
            or m.status == "infeasible_or_unbounded"
            or m.status == "unbounded"
            or m.status == "numeric_error"
        ):
            # print("model in infeasible state, resetting lp")
            m.problem.reset()
            m.optimize()
            if m.status == "optimal":
                return m.objective.value
            print("Solver status: " + str(m.status))
            raise ValueError("Optimization fails despite resetting")
        print("Solver status: " + str(m.status))
        raise ValueError
