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
