import numpy as np

from hopsy._polyround.polytope import Polytope
from hopsy._polyround.settings import PolyRoundSettings

from .constraint_removal_reduction import constraint_removal, null_space
from .lp_interfacing import Interfacer
from .lp_utils import (
    chebyshev_center,
    extract_polytope,
    parse_sbml_cobrapy,
    polytope_to_csv,
)
from .maximum_volume_ellipsoid import iterative_solve

try:
    from cobra.core.model import Model
except Exception:
    Model = None


class Exp1Backend:
    name = "exp1"

    def simplify_polytope(
        self,
        polytope: Polytope,
        settings: PolyRoundSettings,
        normalize: bool = True,
    ) -> Polytope:
        """
        Remove redundant constraints and refunction inequality constraints to equality constraints in case of dimension
        width less than thresh
        @param polytope:
        @param settings:
        @return:
        """
        polytope = polytope.copy()
        if normalize:
            polytope.normalize()
        removed, refunctioned = 1, 1
        while (removed != 0 or refunctioned != 0) and polytope.A.size > 0:
            polytope, removed, refunctioned = constraint_removal(
                polytope,
                settings,
            )
        if polytope.A.shape[0] == 0:
            raise ValueError(
                "All inequality constraints are redundant, implying that the polytope is a single point."
            )
        return polytope

    def transform_polytope(
        self,
        polytope: Polytope,
        settings: PolyRoundSettings,
    ) -> Polytope:
        """
        Express polytope in a (shifted) orthogonal basis in the null space of the equality constraints to remove all
        equality constraints
        @param polytope:
        @param settings:
        @return:
        """
        if polytope.inequality_only:
            raise ValueError(
                "Polytope already transformed (only contains inequality constraints)"
            )
        polytope = polytope.copy()
        x, dist = chebyshev_center(polytope, settings)
        if polytope.border_distance(x) <= 0:
            raise ValueError("Chebyshev center outside polytope before transforming")
        if settings.verbose:
            print("chebyshev distance is : " + str(dist))
            pre_b_dist = polytope.border_distance(x)
            print("border distance pre-transformation is: " + str(pre_b_dist))
        # put x at zero!
        polytope.apply_shift(x)
        if settings.verbose:
            x_0 = np.zeros(x.shape)
            b_dist_at_zero = polytope.border_distance(x_0)
            print("border distance zero-transformation is: " + str(b_dist_at_zero))
        stoichiometry = polytope.S.values
        transformation = null_space(stoichiometry, eps=settings.numerics_threshold)
        polytope.apply_transformation(transformation)
        if settings.verbose:
            u = np.zeros((transformation.shape[1], 1))
            norm_check = np.linalg.norm(np.matmul(stoichiometry, transformation))
            print("norm of the null space is: " + str(norm_check))
            b_dist = polytope.border_distance(u)
            print("border distance after transformation is: " + str(b_dist))
            # test if we can reproduce the original x
            trans_x = polytope.back_transform(u)
            x_rec_diff = np.max(trans_x - np.squeeze(x))
            print("the deviation of the back transform is: " + str(x_rec_diff))
        return polytope

    def round_polytope(
        self,
        polytope: Polytope,
        settings: PolyRoundSettings,
    ) -> Polytope:
        """
        Round polytope using the maximum volume ellipsoid approach
        @param polytope:
        @param settings:
        @return:
        """
        # check if there are Nans
        bool = False
        bool += np.isinf(polytope.A.values).any()
        bool += np.isinf(polytope.b.values).any()
        if bool:
            raise ValueError("Polytope assigned for rounding contains inf")

        # create a blank polytope so that we can make isolated checks on the rounding transform
        blank_polytope = Polytope(polytope.A, polytope.b)
        iterative_solve(blank_polytope, settings)
        # iterative_solve(
        #     o_polytope, backend, hp_flags=hp_flags, verbose=verbose, sgp=sgp
        # )
        # check if the transformation is full dimensional
        _, s, _ = np.linalg.svd(blank_polytope.transformation)
        if not np.min(s) > settings.thresh / settings.accepted_tol_violation:
            raise ValueError("Rounding transformation not full dimensional")
        # check if 0 is a solution
        if not blank_polytope.b.min() > 0:
            raise ValueError("Zero point not inside rounded polytope")
        polytope.apply_shift(blank_polytope.shift.values)
        polytope.apply_transformation(blank_polytope.transformation.values)

        # assert polytope == o_polytope
        return polytope

    def simplify_transform_and_round(
        self,
        polytope: Polytope,
        settings: PolyRoundSettings,
    ) -> Polytope:
        """
        Conveniently execute simplify_polytope, transform_polytope and round polytope in sequence
        @param polytope:
        @param settings:
        @return:
        """
        polytope = self.simplify_polytope(
            polytope,
            settings=settings,
        )
        if not polytope.inequality_only:
            polytope = self.transform_polytope(
                polytope,
                settings=settings,
            )
        polytope = self.round_polytope(
            polytope,
            settings=settings,
        )
        return polytope

    def cobra_model_to_polytope(self, model):
        """
        Turn cobrapy model into polytope
        @param model: cobrapy model
        @return:
        """
        if Model is None:
            raise NotImplementedError(
                "Cobra not currently supported. Install Polyround with extras to support cobra "
                "(pip install 'PolyRound[extras]')"
            )
        return extract_polytope(model)

    def polytope_to_csvs(self, polytope: Polytope, dirname: str):
        polytope_to_csv(polytope, dirname)

    def sbml_to_polytope(
        self,
        file_name: str,
        settings: PolyRoundSettings,
        inf_bound=1e5,
        prescale=False,
    ) -> Polytope:
        if Model is None:
            raise NotImplementedError(
                "Cobra not currently supported. Install Polyround with extras to support cobra "
                "(pip install 'PolyRound[extras]')"
            )
        polytope = parse_sbml_cobrapy(
            file_name,
            inf_bound=inf_bound,
            prescale=prescale,
        )
        return polytope

    def chebyshev_center(
        self,
        polytope: Polytope,
        settings: PolyRoundSettings,
    ):
        return chebyshev_center(polytope, settings)

    def iterative_solve(
        self,
        polytope: Polytope,
        settings: PolyRoundSettings,
    ):
        return iterative_solve(polytope, settings)

    def polytope_to_model(
        self,
        polytope: Polytope,
        settings: PolyRoundSettings,
    ):
        return Interfacer.polytope_to_model(polytope, settings)
