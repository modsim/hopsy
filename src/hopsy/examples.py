"""

"""


class _core:
    from .core import Gaussian, Mixture, Problem


_c = _core


class _submodules:
    import typing

    import numpy as np
    import numpy.typing
    import scipy


from .misc import add_box_constraints, sample
from .setup import setup

_s = _submodules


def _to_angle(gamma):
    r"""
    turn radian to degree

    Parameters
    ----------
    gamma : float


    Returns
    -------
    float
        angle in degree
    """
    return gamma * 90


def _to_rad(gamma):
    r"""
    generate radians from a given gamma

    Parameters
    ----------
    gamma : float


    Returns
    -------
    float
        angle in radian
    """
    return (0.25 + 0.25 * gamma) * _s.np.pi


def generate_unit_hypercube(
    dimension: int,
) -> _s.typing.Tuple[_s.numpy.ndarray, _s.numpy.ndarray]:
    r"""
    Generate matrix A and vector b of the unit N-dimensional hypercube.

    Parameters
    ----------
    dimension : int
        Dimension N of the cube

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        First value is the matrix A of shape (2*N, N), second the vector b of shape (2*N,)
    """
    assert 1 <= dimension

    A = _s.np.vstack((_s.np.identity(dimension), -_s.np.identity(dimension)))
    b = _s.np.concatenate((_s.np.repeat(1, dimension), _s.np.repeat(0, dimension)))

    return A, b


def generate_unit_simplex(
    dimension: int,
) -> _s.typing.Tuple[_s.numpy.ndarray, _s.numpy.ndarray]:
    r"""
    Generate matrix A and vector b of the unit N-dimensional simplex.

    Parameters
    ----------
    dimension : int
        Dimension N of the cube

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        First value is the matrix A of shape (2*N, N), second the vector b of shape (2*N,)
    """
    assert 1 <= dimension

    A = _s.np.vstack((_s.np.ones(dimension), -_s.np.identity(dimension)))
    b = _s.np.concatenate(([1], _s.np.repeat(0, dimension)))

    return A, b


def generate_gaussian_mixture(
    dim: _s.typing.Optional[int] = None,
    n_mix: _s.typing.Optional[int] = None,
    n_nonident: _s.typing.Optional[int] = 0,
    means: _s.typing.Optional[_s.typing.List[_s.numpy.ndarray]] = None,
    covs: _s.typing.Optional[_s.typing.List[_s.numpy.ndarray]] = None,
    polytope_type: _s.typing.Optional[str] = None,
    angle: _s.typing.Optional[float] = None,
    A: _s.typing.Optional[_s.numpy.ndarray] = None,
    b: _s.typing.Optional[_s.numpy.ndarray] = None,
    seed: _s.typing.Optional[int] = None,
):
    r"""
    Generate Gaussian mixture distribution on a certain polytope type. The Gaussian mixtures model can be fully
    specified by setting means and covariances of the individual Gaussians. However, it can also be generated partially
    or fully by specifying the following characteristics:
    - Dimension of the problem
    - Number of mixture components
    - Number of non-identifiable parameters (very large variance)
    - Polytope type (spike, cone, diamond) and its angle

    Parameters
    ----------
    dim : int, optional
        The dimensionality of the Gaussian modes. Default is None.
    n_mix : int, optional
        Number of Gaussian modes in the mixture. Default is None.
    n_nonident : int, optional
        Number of non-identifiable parameters in the problem. Default is 0.
    means : list, optional
        A list of means for each Gaussian mode in the mixture. Default is an empty list.
    covs : list, optional
        A list of covariance matrices for each Gaussian mode in the mixture. Default is an empty list.
    polytope_type : str, optional
        Can be either of "spike", "cone" or "diamond" polytope. Default is None (equivalent with unit hypercube).
    angle : float, optional
        Parameter that must be supplied if polytope type is not the unit hypercube. Default is None.
    A : array-like, optional
        A transformation matrix applied to the Gaussian mixture. Default is None.
    b : array-like, optional
        A translation vector applied to the Gaussian mixture. Default is None.
    seed : int, optional
        A seed for random number generation to ensure reproducibility. Default is None.

    Returns
    -------
    hopsy.Problem
        Gaussian mixture model on a polytope
    """
    generator = GaussianMixtureGenerator(
        dim=dim,
        n_mix=n_mix,
        n_nonident=n_nonident,
        means=means,
        covs=covs,
        polytope_type=polytope_type,
        angle=angle,
        A=A,
        b=b,
        seed=seed,
    )

    return generator.get_problem()


class BirkhoffPolytope:
    r"""
    Birkhoff polytope helper that manages transformation of states and constraints
    """

    def __init__(self, size: int):
        r"""
        Construct Birkhoff polytope class.

        :param size : int
            Number of rows/columns of the matrices that the Birkhof polytope is constructed for
        """

        self.size = size
        self.size_squared = size * size

        self.ineq_matrix_full = _s.numpy.vstack(
            (_s.numpy.eye(self.size_squared), -_s.numpy.eye(self.size_squared))
        )
        self.ineq_bounds_full = _s.numpy.concatenate(
            (_s.numpy.ones(self.size_squared), _s.numpy.zeros(self.size_squared))
        )

        row_sums_mat, row_sums_rhs = _s.numpy.zeros(
            (self.size, self.size_squared)
        ), _s.numpy.ones(self.size)
        for i in range(size):
            row_sums_mat[i, i * self.size : (i + 1) * self.size] = 1.0
        col_sums_mat, col_sums_rhs = _s.numpy.zeros(
            (self.size, self.size_squared)
        ), _s.numpy.ones(self.size)
        for i in range(size):
            col_sums_mat[i, [i + j * self.size for j in range(self.size)]] = 1.0

        self.eq_matrix = _s.numpy.vstack((row_sums_mat, col_sums_mat))
        self.eq_rhs = _s.numpy.concatenate((row_sums_rhs, col_sums_rhs))

        self.kernel_basis = _s.scipy.linalg.null_space(self.eq_matrix, rcond=None)
        self.particular_sol = _s.numpy.linalg.lstsq(
            self.eq_matrix, self.eq_rhs, rcond=None
        )[0]

        self.ineq_matrix_reduced = self.ineq_matrix_full @ self.kernel_basis
        self.ineq_bounds_reduced = self.ineq_bounds_full - self.ineq_matrix_full.dot(
            self.particular_sol
        )

    @property
    def A(self):
        return self.ineq_matrix_reduced

    @property
    def b(self):
        return self.ineq_bounds_reduced

    def convert_to_full_space(self, samples: _s.numpy.ndarray) -> _s.numpy.ndarray:
        r"""
        Convert reduced space samples to full space samples.

        :param samples : numpy.ndarray
            (N, D) array where N is the number of samples and D the number of dimensions
        """

        return self.kernel_basis @ samples.T + self.particular_sol.reshape((-1, 1))


class ConePolytope:
    r"""
    Conic polytope with specified angle and dimension
    """

    def __init__(self, dim: int, gamma: float) -> None:
        r"""
        Construct Conic polytope class.

        :param dim : int
            Dimensionality of the problem
        :param gamma : float
            Control problem complexity by changing the angle of the cone: 0 <= gamma <= 1
        """
        angle = _to_rad(gamma)
        x, y = _s.np.cos(angle), _s.np.sin(angle)
        A = _s.np.zeros((int(dim * (dim - 1)) + 1, dim))
        b = _s.np.zeros(int(dim * (dim - 1)) + 1)

        l = _s.np.sqrt(x**2 + y**2)

        # ax - by = 0
        # y = ax/b
        #

        k = 0
        for i in range(dim):
            for j in range(1, dim):
                A[k, i] = x / l
                A[k, (i + j) % dim] = -y / l

                k += 1

        A[k] = _s.np.ones(dim)
        b[k] = x / y + 1

        self.angle = angle
        self.ineq_matrix = A
        self.ineq_bounds = b

    @property
    def A(self):
        return self.ineq_matrix

    @property
    def b(self):
        return self.ineq_bounds


class SpikePolytope:
    r"""
    Spike polytope with specified angle and dimension
    """

    def __init__(self, dim: int, gamma: float) -> None:
        r"""
        Construct Conic polytope class.

        :param dim : int
            Dimensionality of the problem
        :param gamma : float
            Control problem complexity by changing the angle of the cone: 0 <= gamma <= 1
        """
        angle = _to_rad(gamma)
        x, y = _s.np.cos(angle), _s.np.sin(angle)
        A = _s.np.zeros((int(dim * (dim - 1)), dim))
        b = _s.np.zeros(int(dim * (dim - 1)))

        l = _s.np.sqrt(x**2 + y**2)

        k = 0
        for i in range(dim):
            for j in range(1, dim):
                A[k, i] = x / l
                A[k, (i + j) % dim] = -y / l

                k += 1

        prob = _c.Problem(A, b)
        prob = add_box_constraints(prob, 0, 1)

        self.angle = angle
        self.ineq_matrix = prob.A
        self.ineq_bounds = prob.b

    @property
    def A(self):
        return self.ineq_matrix

    @property
    def b(self):
        return self.ineq_bounds


class DiamondPolytope:
    r"""
    Diamond-shaped polytope with specified angle and dimension
    """

    def __init__(self, dim: int, gamma: float) -> None:
        r"""
        Construct Conic polytope class.

        :param dim : int
            Dimensionality of the problem
        :param gamma : float
            Control problem complexity by changing the angle of the cone: 0 <= gamma <= 1
        """

        angle = _to_rad(gamma)
        x, y = _s.np.cos(angle), _s.np.sin(angle)
        A = _s.np.zeros((2 * int(dim * (dim - 1)), dim))
        b = _s.np.zeros(2 * int(dim * (dim - 1)))

        l = _s.np.sqrt(x**2 + y**2)

        k = 0
        for i in range(dim):
            for j in range(1, dim):
                A[k, i] = x / l
                A[k, (i + j) % dim] = -y / l

                k += 1

        for i in range(dim):
            for j in range(1, dim):
                A[k, i] = -x / l
                A[k, (i + j) % dim] = y / l
                b[k] = y - x

                k += 1

        prob = _c.Problem(A, b)

        self.angle = angle
        self.ineq_matrix = prob.A
        self.ineq_bounds = prob.b

    @property
    def A(self):
        return self.ineq_matrix

    @property
    def b(self):
        return self.ineq_bounds


class GaussianMixtureGenerator:
    r"""Generator class for Gaussian mixtures"""

    def __init__(
        self,
        dim: _s.typing.Optional[int] = None,
        n_mix: _s.typing.Optional[int] = None,
        n_nonident: _s.typing.Optional[int] = 0,
        means: _s.typing.Optional[_s.typing.List[_s.numpy.ndarray]] = None,
        covs: _s.typing.Optional[_s.typing.List[_s.numpy.ndarray]] = None,
        polytope_type: _s.typing.Optional[str] = None,
        angle: _s.typing.Optional[float] = None,
        A: _s.typing.Optional[_s.numpy.ndarray] = None,
        b: _s.typing.Optional[_s.numpy.ndarray] = None,
        seed: _s.typing.Optional[int] = None,
    ):
        r"""
        Creates Gaussian mixture model.

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the Gaussian modes. Default is None.
        n_mix : int, optional
            Number of Gaussian in the mixture. Default is None.
        n_nonident : int, optional
            Number of non-identifiable parameters in the problem. Default is 0.
        means : list, optional
            A list of locations for each Gaussian mode in the mixture. Default is an empty list.
        covs : list, optional
            A list of covariance matrices for each Gaussian mode in the mixture. Default is an empty list.
        polytope_type : str, optional
            Can be either of "spike", "cone" or "diamond" polytope. Default is None (equivalent with unit hypercube).
        angle : float, optional
            Parameter that must be supplied if polytope type is not the unit hypercube. Default is None.
        A : array-like, optional
            A transformation matrix applied to the Gaussian mixture. Default is None.
        b : array-like, optional
            A translation vector applied to the Gaussian mixture. Default is None.
        seed : int, optional
            A seed for random number generation to ensure reproducibility. Default is None.
        """
        self.dim = dim
        self.n_mix = n_mix
        self.n_nonident = n_nonident
        self.means = means if means is not None else []
        self.covs = covs if covs is not None else []
        self.A = A
        self.b = b
        self.seed = seed

        if self.seed is None:
            self.seed = _s.np.random.randint(0, 2_147_483_647)

        if self.seed is not None:
            _s.np.random.seed(self.seed)

        if self.n_mix is None and len(self.covs) == 0:
            raise ValueError("Either n_mix or covs must be provided")

        if self.n_mix is not None and len(self.covs) > 0:
            raise ValueError("Only one of n_mix or covs can be provided")

        if self.dim is None:
            if self.A is None:
                raise ValueError("A must be provided if dim is not provided")
            if self.b is None:
                raise ValueError("b must be provided if dim is not provided")
            if self.A.shape[0] != self.b.shape[0]:
                raise ValueError("A and b must have the same number of rows")
        else:
            if self.A is not None or self.b is not None:
                raise ValueError("If dim is provided, A and b must not be provided")

        if self.A is None and self.b is not None:
            raise ValueError("A and b must be provided together")

        if (self.A is not None) and (self.b is None):
            raise ValueError("A and b must be provided together")

        if self.A is not None and self.polytope_type is not None:
            raise ValueError("A and polytope_type must not be provided together")

        self.polytope_type = polytope_type

        if self.polytope_type is not None:
            if self.polytope_type == "spike":
                if angle is None:
                    raise ValueError("angle must be provided for spike polytope")
                polytope = SpikePolytope(self.dim, angle)
                self.A, self.b = polytope.A, polytope.b
            elif self.polytope_type == "cone":
                if angle is None:
                    raise ValueError("angle must be provided for spike polytope")
                polytope = ConePolytope(self.dim, angle)
                self.A, self.b = polytope.A, polytope.b
            elif self.polytope_type == "diamond":
                if angle is None:
                    raise ValueError("angle must be provided for spike polytope")
                polytope = DiamondPolytope(self.dim, angle)
                self.A, self.b = polytope.A, polytope.b
            else:
                raise ValueError(f"Unknown polytope type")

        if self.A is None and self.b is None:
            self.A, self.b = generate_unit_hypercube(self.dim)

        if self.dim is None:
            self.dim = self.A.shape[1]

        if self.covs is not None:
            covariances = [self.generate_covariance_mat() for i in range(self.n_mix)]
            self.covs, self.scales = map(list, zip(*covariances))

        if len(self.means) == 0:
            # sample modes
            problem = _c.Problem(A=self.A, b=self.b)
            num_samples = 10_000
            chains, seeds = setup(problem=problem, random_seed=self.seed)

            _, samples = sample(
                chains[0], seeds[1], n_samples=num_samples, n_procs=1, thinning=1_000
            )

            self.means = samples[0, : self.n_mix, :]

    def generate_covariance_mat(
        self,
        scales_range: _s.typing.Tuple[float, float] = (-3, -0.5),
        nonident_scale: float = 1e6,
    ) -> _s.np.ndarray:
        r"""
        Generate a random covariance matrix with non-identifiable components.

        Parameters
        ----------
        scales_range: tuple(float, float)
            Range from which the identifiable component's standard deviations are sampled.
        nonident_scale: float
            Standard deviation of non-identifiable components.

        Returns
        -------
            Covariance matrix and list of standard deviations
        """

        # Generate random correlation matrix
        w = _s.np.random.rand(self.dim, self.dim)
        s = w @ w.T + _s.np.diag(_s.np.random.randint(0, self.dim, self.dim))
        r = (
            _s.np.diag(1 / _s.np.sqrt(_s.np.diag(s)))
            @ s
            @ _s.np.diag(1 / _s.np.sqrt(_s.np.diag(s)))
        )

        log_scales = _s.np.random.uniform(scales_range[0], scales_range[1], self.dim)
        scales = 10**log_scales

        if self.n_nonident > 0:
            idx_nonident = _s.np.random.choice(self.dim, self.n_nonident, replace=False)
            scales[idx_nonident] = nonident_scale

        cov = _s.np.diag(scales) @ r @ _s.np.diag(scales)

        return cov, scales

    def get_problem(self) -> _c.Problem:
        r"""
        Get hopsy.Problem generated by this object.

        Returns
        -------
        hopsy.Problem
            Gaussian mixture model on a polytope
        """

        models = [
            _c.Gaussian(mean, covariance)
            for mean, covariance in zip(self.means, self.covs)
        ]

        mixture = _c.Mixture(models)

        return _c.Problem(A=self.A, b=self.b, model=mixture)
