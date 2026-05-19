# ©2020-​2021 ETH Zurich, Axel Theorell

import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cupy as cp
except ImportError as error:
    raise ImportError("hopsy's exp2 PolyRound backend requires CuPy.") from error

from hopsy._polyround.polytope import Polytope

from . import lp_interfacing as lp

try:
    import cobra
except Exception:
    cobra = None


def chebyshev_center(polytope, settings):
    # get norm col
    A = cp.asarray(polytope.A.values, dtype=cp.float64)
    a_norm = cp.linalg.norm(A, axis=1).reshape((polytope.A.shape[0], 1))
    A_ext = cp.concatenate((A, a_norm), axis=1)
    obj = cp.zeros(A_ext.shape[1], dtype=A.dtype)
    obj[-1] = -1

    obj = cp.asnumpy(obj)
    A_ext = cp.asnumpy(A_ext)
    if polytope.inequality_only:
        x, m = lp.solve(obj, A_ext, polytope.b.values, settings)
    else:
        S = cp.asarray(polytope.S.values, dtype=cp.float64)
        s_0_col = cp.zeros(shape=(polytope.S.shape[0], 1), dtype=S.dtype)
        S_ext = cp.concatenate((S, s_0_col), axis=1)
        x, m = lp.solve(
            obj,
            A_ext,
            polytope.b.values,
            settings,
            S=cp.asnumpy(S_ext),
            h=polytope.h.values,
        )
    if settings.regularize:
        x_reg, m = lp.regularize_chebyshev_center(x[-1], m)
        x = x_reg
    x = x.reshape((x.shape[0], 1))
    return x[:-1], x[-1]


def fva(polytope, settings):
    n_reac = polytope.A.shape[1]
    output = pd.DataFrame(index=polytope.A.columns)
    # make the first run
    obj = cp.ones(n_reac, dtype=cp.float64)
    A = cp.asarray(polytope.A.values, dtype=cp.float64)

    if polytope.inequality_only:
        x, m = lp.solve(cp.asnumpy(obj), cp.asnumpy(A), polytope.b.values, settings)
    else:
        S = cp.asarray(polytope.S.values, dtype=cp.float64)
        x, m = lp.solve(
            cp.asnumpy(obj),
            cp.asnumpy(A),
            polytope.b.values,
            settings,
            S=cp.asnumpy(S),
            h=polytope.h.values,
        )

    obj = cp.zeros(n_reac, dtype=cp.float64)
    # Now run all the remaining directions
    for i in range(0, n_reac * 2):
        ind = i // 2
        if i % 2 == 0:
            obj[ind] = 1
        else:
            obj[ind] = -1
        x, m = lp.solve_model(cp.asnumpy(obj), m)
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
            "missing optional cobrapy dependency required for parsing sbml. Use pip install 'PolyRound[extras]'"
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
            "missing optional cobrapy dependency required for parsing sbml. Use pip install 'PolyRound[extras]'"
        )
    model = cobra.io.read_sbml_model(file)
    return model


def extract_polytope(model, inf_bound=1e5):
    if cobra is None:
        raise NotImplementedError(
            "missing optional cobrapy dependency required for parsing sbml. Use pip install 'PolyRound[extras]'"
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
    # polytope.S = ((polytope.S*precision).astype(int).astype(float))/precision
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
