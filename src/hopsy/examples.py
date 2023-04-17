"""

"""


class _core:
    from .core import Problem


_c = _core


class _submodules:
    import typing

    import numpy
    import numpy.typing
    import scipy


_s = _submodules


def generate_birkhoff_polytope(
    n: int,
) -> _s.typing.Tuple[_s.numpy.ndarray, _s.numpy.ndarray]:
    r"""
    Generate Birkhoff polytope

    :param n : int
        Number of rows/columns of squared matrices (n x n).
    """
    N = n * n

    A = _s.numpy.column_stack((_s.numpy.eye(N), -_s.numpy.eye(N)))
    b = _s.numpy.column_stack((_s.numpy.ones(N), _s.numpy.zeros(N)))

    row_sums_mat, row_sums_rhs = _s.numpy.zeros((n, N)), _s.numpy.ones(n)
    for i in range(n):
        row_sums_mat[i, i * n : (i + 1) * n] = 1.0
    col_sums_mat, col_sums_rhs = _s.numpy.zeros((n, N)), _s.numpy.ones(n)
    for i in range(n):
        col_sums_mat[i, [i + j * n for j in range(n)]] = 1.0

    C = _s.numpy.column_stack((row_sums_mat, col_sums_mat))
    d = _s.numpy.column_stack((row_sums_rhs, col_sums_rhs))

    kernel_basis = _s.scipy.null_space(C)
    particular_sol = _s.numpy.linalg.lstsq(C, d)

    return A @ kernel_basis, b - A.dot(particular_sol)
