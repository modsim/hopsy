import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from hopsy._rounding.polytope import Polytope

from .lp_interfacing import Interfacer

try:
    import cobra
except Exception:
    cobra = None


def verbose_print(settings, backend, message):
    if getattr(settings, "verbose", bool(settings)):
        print(f"[{backend}] {message}", flush=True)


def solution_summary(x):
    finite = bool(np.isfinite(x).all())
    radius = np.asarray(x[-1]).reshape(-1)[0] if x.size else np.nan
    return finite, radius


def assert_finite_solution(x, model, stage):
    if np.isfinite(x).all():
        return
    status = getattr(model, "status", "unknown")
    raise ValueError(
        f"Chebyshev center {stage} returned non-finite solution "
        f"(solver status={status})."
    )


def chebyshev_center(polytope, settings):
    # get norm col
    a_norm = np.linalg.norm(polytope.A.values, axis=1).reshape((polytope.A.shape[0], 1))
    A_ext = np.concatenate((polytope.A.values, a_norm), axis=1)
    obj = np.zeros(A_ext.shape[1])
    obj[-1] = -1
    if polytope.inequality_only:
        x, m = Interfacer.solve(obj, A_ext, polytope.b.values, settings)
    else:
        s_0_col = np.zeros(shape=(polytope.S.shape[0], 1))
        S_ext = np.concatenate((polytope.S.values, s_0_col), axis=1)
        x, m = Interfacer.solve(
            obj,
            A_ext,
            polytope.b.values,
            settings,
            S=S_ext,
            h=polytope.h.values,
        )
    finite, radius = solution_summary(x)
    verbose_print(
        settings,
        "gurobi-cupy",
        "chebyshev LP "
        + f"status={m.status}, solcount={m.problem.SolCount}, "
        + f"finite={finite}, radius={radius}",
    )
    assert_finite_solution(x, m, "LP")
    if settings.regularize:
        x_reg, m = Interfacer.regularize_chebyshev_center(x[-1], m)
        x = x_reg
        finite, radius = solution_summary(x)
        verbose_print(
            settings,
            "gurobi-cupy",
            "chebyshev regularization "
            + f"status={m.status}, solcount={m.problem.SolCount}, "
            + f"finite={finite}, radius={radius}",
        )
        assert_finite_solution(x, m, "regularization")
    x = x.reshape((x.shape[0], 1))
    return x[:-1], x[-1]


def fva(polytope, settings):
    n_reac = polytope.A.shape[1]
    output = pd.DataFrame(index=polytope.A.columns)
    # make the first run
    obj = np.ones(n_reac)

    if polytope.inequality_only:
        x, m = Interfacer.solve(obj, polytope.A.values, polytope.b.values, settings)
    else:
        x, m = Interfacer.solve(
            obj,
            polytope.A.values,
            polytope.b.values,
            settings,
            S=polytope.S.values,
            h=polytope.h.values,
        )

    obj = np.zeros(n_reac)
    # Now run all the remaining directions
    for i in range(0, n_reac * 2):
        ind = i // 2
        if i % 2 == 0:
            obj[ind] = 1
        else:
            obj[ind] = -1
        x, m = Interfacer.solve_model(obj, m)
        obj[ind] = 0
        output.loc[:, i] = x
    return output


def polytope_to_csv(polytope, dirname):
    Path(dirname).mkdir(parents=True, exist_ok=True)
    name = dirname.rstrip("/").split("/")[-1]
    for attribute in dir(polytope):
        tentative_df = getattr(polytope, attribute)
        if isinstance(tentative_df, pd.DataFrame) or isinstance(
            tentative_df, pd.Series
        ):
            if attribute == "transformation":
                zero_solution_df = pd.Series(0, index=tentative_df.columns)
                zero_solution_df.to_csv(
                    os.path.join(dirname, "start_" + name + "_rounded.csv"),
                    header=False,
                    index=False,
                )
                tentative_df.to_csv(
                    os.path.join(dirname, "N_" + name + "_rounded.csv"),
                    header=False,
                    index=False,
                )
            elif attribute == "shift":
                tentative_df.to_csv(
                    os.path.join(dirname, "p_shift_" + name + "_rounded.csv"),
                    header=False,
                    index=False,
                )
                name_series = pd.Series(tentative_df.index)
                name_series.to_csv(
                    os.path.join(dirname, "reaction_names_" + name + "_rounded.csv"),
                    header=False,
                    index=False,
                )
            else:
                tentative_df.to_csv(
                    os.path.join(dirname, attribute + "_" + name + "_rounded.csv"),
                    header=False,
                    index=False,
                )


def parse_sbml_cobrapy(file, inf_bound=1e5, prescale=False):
    if cobra is None:
        raise NotImplementedError(
            "Missing optional cobrapy dependency required for parsing SBML. "
            "Use pip install 'hopsy[sbml]'."
        )
    model = read_sbml_model(file)

    if prescale:
        # prefix reactions
        reactions = list(model.reactions)
        for r in reactions:
            model.remove_reactions([r])
            r.id = "R_" + r.id
        model.add_reactions(reactions)
        model.repair()
        fva = cobra.flux_analysis.flux_variability_analysis(
            model, fraction_of_optimum=0
        )
        ranges = fva.maximum - fva.minimum
        threshold = 1e-9
        ranges[ranges > 1] = 1
        ranges[ranges < threshold] = 1
        transformation = pd.DataFrame(np.eye(ranges.size), columns=ranges.index)
        transformation = transformation * ranges

    p = extract_polytope(model, inf_bound=inf_bound)
    if prescale:
        p.apply_transformation(transformation.values)

    return p


def read_sbml_model(file):
    if cobra is None:
        raise NotImplementedError(
            "Missing optional cobrapy dependency required for parsing SBML. "
            "Use pip install 'hopsy[sbml]'."
        )
    model = cobra.io.read_sbml_model(file)
    return model


def extract_polytope(model, inf_bound=1e5):
    if cobra is None:
        raise NotImplementedError(
            "Missing optional cobrapy dependency required for parsing SBML. "
            "Use pip install 'hopsy[sbml]'."
        )
    S = cobra.util.array.create_stoichiometric_matrix(model, array_type="DataFrame")
    # make bounds matrix
    n_react = len(model.reactions)
    uids = [uuid.uuid4().hex for i in range(n_react * 2)]
    A = pd.DataFrame(0.0, index=uids, columns=S.columns)
    b = pd.Series(0.0, index=uids)
    for ind, r in enumerate(list(model.reactions)):
        if r.bounds[1] == float("inf"):
            b[uids[ind]] = inf_bound
        else:
            b[uids[ind]] = r.bounds[1]
        if r.bounds[0] == float("-inf"):
            b[uids[ind + n_react]] = inf_bound
        else:
            b[uids[ind + n_react]] = -r.bounds[0]
        A.loc[uids[ind], r.id] += 1
        A.loc[uids[ind + n_react], r.id] -= 1
    p = Polytope(A, b, S=S)
    return p


def make_precision_truncated_integer_polytope(polytope, max_decimals):
    # this is only used for the sparse transform, which requires a homogeneous system
    assert all(polytope.h == 0)
    truncated_p = polytope.copy()
    precision = 10 ** (max_decimals - 1)
    tempS = polytope.S.abs()
    tempS[tempS == 0] = np.nan
    row_norm = np.nanmin(tempS, axis=1)
    row_norm[row_norm == 0] = 1
    # potency version
    row_norm_potency = np.power(
        10.0, np.floor(np.log10(np.abs(row_norm))).astype(np.int64) * -1
    )
    for attribute in dir(truncated_p):
        tentative_df = getattr(truncated_p, attribute)
        # do not truncate transformation or shift
        if attribute == "transformation":
            assert np.all(tentative_df.values == np.eye(tentative_df.shape[0]))
        elif attribute == "shift":
            assert np.all(tentative_df.values == np.zeros(tentative_df.shape[0]))
        elif attribute == "S":
            temp_df = (
                ((tentative_df.T * row_norm_potency).T * precision).round().astype(int)
            )
            setattr(truncated_p, attribute, temp_df)

    return truncated_p
