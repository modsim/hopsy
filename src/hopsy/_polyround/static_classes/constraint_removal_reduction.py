# ©2020-​2021 ETH Zurich, Axel Theorell
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hopsy._polyround.default_settings import default_solver_timeout
from hopsy._polyround.static_classes.lp_interfacing import GurobiInterfacer, gp


class PolytopeReducer:
    @staticmethod
    def constraint_removal(
        polytope,
        settings,
    ):
        """
        Removes redundant constraints and removes narrow directions by turning them into equality constraints
        :param polytope: Polytope object to round
        :param hp_flags: Dictionary of gurobi flags for high precision solution
        :param thresh: Float determining how narrow a direction has to be to declare an equality constraint
        :param verbose: Bool regulating output level
        :return: Polytope object with non-empty interior and no redundant constraints, number of removed constraints,
        number of inequality constraints turned to equality constraints.
        """
        reduced_polytope, removed, refunctioned = (
            PolytopeReducer.native_constraint_removal(polytope, settings)
        )

        if settings.verbose:
            print("Number of removed constraints: " + str(removed))
            print("Number of refunctioned constraints: " + str(refunctioned))

        return reduced_polytope, removed, refunctioned

    @staticmethod
    def native_constraint_removal(polytope, settings):
        if gp is None:
            raise ImportError(
                "hopsy's native PolyRound backend requires gurobipy for constraint reduction."
            )

        model = GurobiInterfacer.make_native_model(polytope.A.columns, settings)
        problem = model.problem
        problem.setParam("TimeLimit", default_solver_timeout)

        inequality_expressions = GurobiInterfacer.build_native_row_expressions(
            polytope.A.values, model.variables
        )
        inequality_constraints = GurobiInterfacer.add_native_constraint_system(
            model,
            polytope.A.values,
            polytope.b.values,
            names=polytope.b.index,
            equality=False,
        ).tolist()

        if polytope.S is not None:
            GurobiInterfacer.add_native_constraint_system(
                model,
                polytope.S.values,
                polytope.h.values,
                names=polytope.h.index,
                equality=True,
            )

        (
            active_mask,
            equality_mask,
            removed,
            refunctioned,
        ) = PolytopeReducer.native_constraint_removal_loop(
            model,
            inequality_constraints,
            inequality_expressions,
            polytope.b.values,
            settings,
        )

        inequality_mask = active_mask & ~equality_mask
        reduced_A = polytope.A.iloc[inequality_mask].copy()
        reduced_b = polytope.b.iloc[inequality_mask].copy()

        if polytope.S is None:
            if np.any(equality_mask):
                reduced_S = polytope.A.iloc[equality_mask].copy()
                reduced_h = polytope.b.iloc[equality_mask].copy()
                reduced_polytope = type(polytope)(
                    reduced_A, reduced_b, reduced_S, reduced_h
                )
            else:
                reduced_polytope = type(polytope)(reduced_A, reduced_b)
        else:
            if np.any(equality_mask):
                reduced_S = pd.concat(
                    [polytope.A.iloc[equality_mask].copy(), polytope.S.copy()], axis=0
                )
                reduced_h = pd.concat(
                    [polytope.b.iloc[equality_mask].copy(), polytope.h.copy()], axis=0
                )
            else:
                reduced_S = polytope.S.copy()
                reduced_h = polytope.h.copy()
            reduced_polytope = type(polytope)(
                reduced_A, reduced_b, reduced_S, reduced_h
            )

        return reduced_polytope, removed, refunctioned

    @staticmethod
    def native_constraint_removal_loop(
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

            if settings.verbose and index % 50 == 0:
                print("\n Investigating constraint number: " + str(index) + "\n")

            model.problem.setObjective(inequality_expressions[index], gp.GRB.MAXIMIZE)
            model.optimize()
            max_val = GurobiInterfacer.get_opt(model, settings)

            if settings.reduce:
                original_rhs = rhs[index]
                constr.RHS = float(original_rhs + 1.0)
                model.optimize()
                perturbed_val = GurobiInterfacer.get_opt(model, settings)
                constr.RHS = float(original_rhs)
                if np.abs(max_val - perturbed_val) < settings.thresh:
                    removed += 1
                    active_mask[index] = False
                    model.problem.remove(constr)
                    continue
            elif rhs[index] - max_val >= settings.thresh:
                continue

            if not settings.simplify_only:
                model.problem.setObjective(
                    inequality_expressions[index], gp.GRB.MINIMIZE
                )
                model.optimize()
                min_val = GurobiInterfacer.get_opt(model, settings)
                if np.abs(max_val - min_val) < settings.thresh:
                    constr.Sense = gp.GRB.EQUAL
                    equality_mask[index] = True
                    refunctioned += 1

        return active_mask, equality_mask, removed, refunctioned

    @staticmethod
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

    @staticmethod
    def sparse_null_space(S_df):
        from sympy import SparseMatrix

        if not (S_df.dtypes == int).all():
            raise TypeError(
                "Polytope has to be formulated in integers for sparse null space."
            )

        react_sum = np.sum(S_df != 0, axis=0)
        order = np.argsort(react_sum)
        S = S_df.values[:, order]

        csr = sp.csr_matrix(S)
        dokform = csr.todok()
        in_matrix = SparseMatrix(dokform.shape[0], dokform.shape[1], dict(dokform))

        rref_m, rref_pivot = in_matrix.rref()
        rank = len(rref_pivot)
        rref_inverted = list()
        for i in range(rref_m.cols):
            if i not in rref_pivot:
                rref_inverted.append(i)
        rref_inverted = tuple(rref_inverted)
        trans = SparseMatrix.zeros(in_matrix.cols, in_matrix.cols - rank)
        eye_ind = 0
        rref_ind = 0
        for ind in range(in_matrix.cols):
            if ind in rref_pivot:
                trans[ind, :] = -rref_m[rref_ind, rref_inverted]
                rref_ind += 1
            else:
                trans[ind, eye_ind] = 1
                eye_ind += 1
        back_order = list(range(len(order)))
        for ind, num in enumerate(order):
            back_order[num] = ind
        trans = trans[back_order, :]
        trans_df = pd.DataFrame(np.array(trans), index=S_df.columns, dtype=np.float64)
        return trans_df
