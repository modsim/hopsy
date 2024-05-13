"""

"""


class _core:
    from .core import Problem


_c = _core


class _submodules:
    import typing

    import numpy as np
    import numpy.typing
    import scipy


_s = _submodules


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

    A = _s.np.row_stack((_s.np.identity(dimension), -_s.np.identity(dimension)))
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

    A = _s.np.row_stack((_s.np.ones(dimension), -_s.np.identity(dimension)))
    b = _s.np.concatenate(([1], _s.np.repeat(0, dimension)))

    return A, b


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

        self.ineq_matrix_full = _s.numpy.row_stack(
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

        self.eq_matrix = _s.numpy.row_stack((row_sums_mat, col_sums_mat))
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





def generate_covariance_mat(
        dim : int, 
        n_nonident : int, 
        scales_range : _s.typing.Tuple[float, float] = (1, 1)
) -> _s.numpy.ndarray:
    
    a = _s.np.random.rand(dim, dim)
    q,_ = _s.np.linalg.qr(a)
    # q is othonormal
    idx_nonident = _s.np.random.choice(dim, n_nonident, replace=False)

    eig = _s.np.identity(dim)
    scales = _s.np.array([scales_range[0]]*dim)
    scales[idx_nonident] = scales_range[1]

    eig = _s.np.diag(scales) @ eig
    cov = q @ eig @ q.T

    return cov



def generate_gaussian_mixture_toy_problem(
        polytope : _s.typing.Tuple[_s.np.ndarray, _s.np.ndarray],
        modes : _s.typing.List[_s.typing.Tuple[_c.Gaussian, float]] = None,
        n_modes : int = None,
        non_ident : int = 0,
        vol : float = 1
    ) -> _s.hopsy.Problem:

    """
    Generate a toy problem with a Gaussian mixture model as the objective function.
    The means and covariances are randomly generated.
    """


    assert (modes is not None and len(modes) > 0) or n_modes > 0, "Either modes or n_modes must be provided"

    A, b = polytope

    if modes is not None and len(modes) > 0:
        models = modes

    else:
        dim = A.shape[1]

        mean_sampler = _c.setup(
            problem = _c.Problem(
                A=A,b=b
            ), random_seed=42
        )

        means = _s.hopsy.sample(mean_sampler[0][0], mean_sampler[1][0],n_modes)
        means = means[1][0]
        means = [means[i] for i in range(n_modes)]


        covs = [ 
            generate_covariance_mat(
                dim=dim, 
                n_nonident=non_ident,
                scales_range=(1e-2*vol, 1e6*vol),
            ) for i in range(n_modes)
        ]


        models = [
            _c.Gaussian(mean, covariance)
            for mean, covariance in zip(means, covs)
        ]

    mixture = _c.Mixture(models)

    return _c.Problem(A=A,b=b, model=mixture)
