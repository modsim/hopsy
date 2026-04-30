# ©2020-​2021 ETH Zurich, Axel Theorell
from typing import Dict

from hopsy._polyround.default_settings import (
    default_0_width,
    default_accepted_tol_violation,
    default_hp_flags,
    default_numerics_threshold,
)

try:
    import gurobipy
except Exception:
    gurobipy = None


class PolyRoundSettings:
    """
    Settings object specifying numerical tolerance and other program options
    """

    def __init__(
        self,
        hp_flags: Dict = default_hp_flags,
        thresh: float = default_0_width,
        verbose: bool = False,
        sgp: bool = False,
        reduce: bool = True,
        regularize: bool = False,
        check_lps: bool = False,
        simplify_only: bool = False,
        presolve: bool = False,
        numerics_threshold: float = default_numerics_threshold,
        accepted_tol_violation: float = default_accepted_tol_violation,
    ):
        """
        @param hp_flags: dictionary with gurobi flags as keys and related numbers as values. Passed directly to the
        gurobi model.
        @param thresh: numerical threshold determining the smallest facette width in absolute values (default 1e-7).
        @param verbose: allow runtime information to be printed to terminal.
        @param sgp: specifically control the terminal print level of gurobi.
        @param reduce: remove redundant constraints (True by default). Setting it False is only for testing purposes.
        @param regularize: impose quadratic penalty term to control position of Chebyshec center. Currently not
        functional.
        @param check_lps: perform extra checks on whether solutions to linear programs really fulfill the imposed
        tolerances. Degrades performance significantly.
        @param simplify_only: only remove redundant constraints (and not zero facettes). Does not yield roundable
        polytope.
        @param presolve: use linear programming solver presolve option.
        """
        if gurobipy is None:
            raise ImportError("hopsy's native PolyRound backend requires gurobipy.")

        self.hp_flags = dict(hp_flags)
        self.thresh = thresh
        self.verbose = verbose
        self.sgp = sgp
        self.reduce = reduce
        self.regularize = regularize
        self.check_lps = check_lps
        self.simplify_only = simplify_only
        self.presolve = presolve
        self.numerics_threshold = numerics_threshold
        self.accepted_tol_violation = accepted_tol_violation
