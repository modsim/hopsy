# ©2020-​2021 ETH Zurich, Axel Theorell
import pandas as pd

from hopsy._polyround.mutable_classes.polytope import Polytope

try:
    import cobra
except:
    cobra = None
import uuid

import numpy as np

from hopsy._polyround.static_classes.lp_utils import ChebyshevFinder


class StoichiometryParser:
    @staticmethod
    def parse_sbml_cobrapy(file, inf_bound=1e5, prescale=False):
        if cobra is None:
            raise NotImplementedError(
                "missing optional cobra dependency required for parsing sbml."
            )
        model = StoichiometryParser.read_sbml_model(file)

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
            # fva = ChebyshevFinder.fva(p, 'gurobi')
            ranges = fva.maximum - fva.minimum
            threshold = 1e-9
            ranges[ranges > 1] = 1
            ranges[ranges < threshold] = 1
            transformation = pd.DataFrame(np.eye(ranges.size), columns=ranges.index)
            transformation = transformation * ranges

        p = StoichiometryParser.extract_polytope(model, inf_bound=inf_bound)
        if prescale:
            p.apply_transformation(transformation.values)

        return p

    @staticmethod
    def read_sbml_model(file):
        if cobra is None:
            raise NotImplementedError(
                "missing optional cobra dependency required for parsing sbml."
            )
        model = cobra.io.read_sbml_model(file)
        return model

    @staticmethod
    def extract_polytope(model, inf_bound=1e5):
        if cobra is None:
            raise NotImplementedError(
                "missing optional cobra dependency required for parsing sbml."
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

    @staticmethod
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
                    ((tentative_df.T * row_norm_potency).T * precision)
                    .round()
                    .astype(int)
                )
                setattr(truncated_p, attribute, temp_df)

        return truncated_p
