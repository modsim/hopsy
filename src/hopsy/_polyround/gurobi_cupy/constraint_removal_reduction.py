import numpy as np
import pandas as pd
import scipy.sparse as sp

from hopsy._polyround.default_settings import default_solver_timeout

from .lp_interfacing import Interfacer, gp


def verbose_print(settings, backend, message):
    if getattr(settings, "verbose", bool(settings)):
        print(f"[{backend}] {message}")


def constraint_removal(polytope, settings):
    """
    Removes redundant constraints and removes narrow directions by turning them into equality constraints
    :param polytope: Polytope object to round
    :param hp_flags: Dictionary of gurobi flags for high precision solution
    :param thresh: Float determining how narrow a direction has to be to declare an equality constraint
    :param verbose: Bool regulating output level
    :return: Polytope object with non-empty interior and no redundant constraints, number of removed constraints,
    number of inequality constraints turned to equality constraints.
    """
    if gp is None:
        raise ImportError(
            "hopsy's gurobi-cupy PolyRound backend requires gurobipy for constraint reduction."
        )

    model = Interfacer.make_model(polytope.A.columns, settings)
    model.configuration.presolve = settings.presolve
    problem = model.problem
    problem.setParam("TimeLimit", default_solver_timeout)

    inequality_expressions = Interfacer.build_row_expressions(
        polytope.A.values, model.variables
    )
    inequality_constraints = Interfacer.add_constraint_system(
        model,
        polytope.A.values,
        polytope.b.values,
        names=polytope.b.index,
        equality=False,
    ).tolist()

    if polytope.S is not None:
        Interfacer.add_constraint_system(
            model,
            polytope.S.values,
            polytope.h.values,
            names=polytope.h.index,
            equality=True,
        )

    (
        _active_mask,
        _equality_mask,
        removed,
        refunctioned,
    ) = constraint_removal_loop(
        model,
        inequality_constraints,
        inequality_expressions,
        polytope.b.values,
        settings,
    )

    model.update()
    reduced_polytope = Interfacer.model_to_polytope(model)
    verbose_print(settings, "gurobi-cupy", f"removed constraints={removed}")
    verbose_print(settings, "gurobi-cupy", f"refunctioned constraints={refunctioned}")
    return reduced_polytope, removed, refunctioned


def constraint_removal_loop(
    model,
    inequality_constraints,
    inequality_expressions,
    rhs,
    settings,
):
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    active_mask = np.ones(rhs.shape[0], dtype=bool)
    equality_mask = np.zeros(rhs.shape[0], dtype=bool)
    removed = 0
    refunctioned = 0

    for index, constr in enumerate(inequality_constraints):
        if not active_mask[index]:
            continue

        if index % 50 == 0:
            verbose_print(settings, "gurobi-cupy", f"investigating constraint={index}")

        model.problem.setObjective(inequality_expressions[index], gp.GRB.MAXIMIZE)
        model.optimize()
        max_val = Interfacer.get_opt(model, settings)

        if settings.reduce:
            original_rhs = rhs[index]
            constr.RHS = float(original_rhs + 1.0)
            model.optimize()
            perturbed_val = Interfacer.get_opt(model, settings)
            constr.RHS = float(original_rhs)
            if np.abs(max_val - perturbed_val) < settings.thresh:
                removed += 1
                active_mask[index] = False
                model.problem.remove(constr)
                continue
        elif rhs[index] - max_val >= settings.thresh:
            continue

        if not settings.simplify_only:
            model.problem.setObjective(inequality_expressions[index], gp.GRB.MINIMIZE)
            model.optimize()
            min_val = Interfacer.get_opt(model, settings)
            if np.abs(max_val - min_val) < settings.thresh:
                constr.Sense = gp.GRB.EQUAL
                equality_mask[index] = True
                refunctioned += 1

    return active_mask, equality_mask, removed, refunctioned


def null_space(S, eps=1e-10):
    """
    Returns the null space of a matrix
    :param S: Numpy array
    :param eps: Threshold for declaring 0 singular values
    :return: Numpy array of null space
    """
    u, s, vh = np.linalg.svd(S)
    s = np.array(s.tolist())
    vh = np.array(vh.tolist())
    null_mask = s <= eps
    null_mask = np.append(null_mask, True)
    null_ind = np.argmax(null_mask)
    null = vh[null_ind:, :]
    return np.transpose(null)
