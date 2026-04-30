# ©2020-​2021 ETH Zurich, Axel Theorell

import numpy as np
import pandas as pd

from hopsy._polyround.static_classes.lp_interfacing import GurobiInterfacer


class ChebyshevFinder:
    @staticmethod
    def chebyshev_center(polytope, settings):
        # get norm col
        a_norm = np.linalg.norm(polytope.A.values, axis=1).reshape(
            (polytope.A.shape[0], 1)
        )
        A_ext = np.concatenate((polytope.A.values, a_norm), axis=1)
        obj = np.zeros(A_ext.shape[1])
        obj[-1] = -1
        if polytope.inequality_only:
            x, m = GurobiInterfacer.gurobi_solve(
                obj, A_ext, polytope.b.values, settings
            )
        else:
            s_0_col = np.zeros(shape=(polytope.S.shape[0], 1))
            S_ext = np.concatenate((polytope.S.values, s_0_col), axis=1)
            x, m = GurobiInterfacer.gurobi_solve(
                obj,
                A_ext,
                polytope.b.values,
                settings,
                S=S_ext,
                h=polytope.h.values,
            )
        if settings.regularize:
            x_reg, m = GurobiInterfacer.gurobi_regularize_chebyshev_center(x[-1], m)
            x = x_reg
        x = x.reshape((x.shape[0], 1))
        return x[:-1], x[-1]

    @staticmethod
    def fva(polytope, settings):
        n_reac = polytope.A.shape[1]
        output = pd.DataFrame(index=polytope.A.columns)
        # make the first run
        obj = np.ones(n_reac)

        if polytope.inequality_only:
            x, m = GurobiInterfacer.gurobi_solve(
                obj, polytope.A.values, polytope.b.values, settings
            )
        else:
            x, m = GurobiInterfacer.gurobi_solve(
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
            x, m = GurobiInterfacer.gurobi_solve_model(obj, m)
            obj[ind] = 0
            output.loc[:, i] = x
        return output
