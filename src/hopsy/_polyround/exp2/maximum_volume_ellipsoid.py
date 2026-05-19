import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
except ImportError as error:
    raise ImportError("hopsy's exp1 PolyRound backend requires CuPy.") from error

from hopsy._polyround.default_settings import (
    default_0_width,
    default_max_ratio_bmAx0,
    default_numerics_threshold,
)

from .geometric_mean_scaling import geometric_mean_scaling
from .lp_utils import chebyshev_center
from .nearest_symmetric_positive_definite import get_NSPD


def as_float(value):
    return float(cp.asnumpy(value))


def as_floats(*values):
    stacked = cp.stack([cp.asarray(value) for value in values])
    return tuple(float(value) for value in cp.asnumpy(stacked))


def cholesky_solve(factor, rhs):
    y = solve_triangular(factor, rhs, lower=True, check_finite=False)
    return solve_triangular(factor, y, lower=True, trans="T", check_finite=False)


def positive_spacing(x, dtype):
    return cp.finfo(dtype).eps * cp.maximum(1.0, cp.abs(x))


def correct_numerics(A_hat):
    k = 0
    while True:
        mineig = cp.min(cp.linalg.eigvalsh(A_hat))
        if as_float(mineig) > 0:
            return A_hat
        k += 1
        diag_min = cp.min(cp.diag(A_hat))
        jitter = positive_spacing(diag_min, A_hat.dtype)
        A_hat = A_hat + (-mineig * k**2 + jitter) * cp.eye(
            A_hat.shape[0], dtype=A_hat.dtype
        )


def iterative_solve(
    polytope,
    settings,
):
    # only round reduced polytopes
    assert polytope.inequality_only
    nDimensions = polytope.A.shape[1]
    columnScale, _ = geometric_mean_scaling(polytope.A.values, 0, 0.99)
    # just this once, do the wasteful expansion
    c_scale_transform = np.diag(1 / columnScale)
    polytope.apply_transformation(c_scale_transform)

    # Normalize
    polytope.normalize()

    # Perform iterative rounding
    maxIterations = 20
    iteration = 0
    reg = 1e-3
    iterationTransform = np.eye(nDimensions)
    converged = 0
    temp_eig = np.linalg.eig(iterationTransform)[0]
    delta_b = delta_s = np.array([1])
    while (
        (np.max(temp_eig) > 6 * np.min(temp_eig) and converged != 1)
        or reg > 1e-8
        or converged == 2
        or (
            np.max(np.abs(delta_b)) > default_numerics_threshold
            or np.max(np.abs(delta_s)) > default_numerics_threshold
        )
    ):
        del temp_eig
        iteration = iteration + 1
        center, _ = chebyshev_center(polytope, settings)
        reg = np.maximum(reg / 10, 1e-10)

        # Calculate and apply transform
        (
            iterationShift,
            iterationTransform,
            converged,
            delta_b,
            delta_s,
        ) = run_mve(
            polytope.A.values.copy(),
            polytope.b.values.copy()[:, None],
            center,
            reg,
        )
        polytope.apply_shift(iterationShift)
        polytope.apply_transformation(iterationTransform)
        polytope.normalize()
        if iteration == maxIterations:
            break
        temp_eig_all = np.linalg.eig(iterationTransform)
        temp_eig = temp_eig_all[0]

    if iteration == maxIterations:
        if not (
            np.max(np.abs(delta_b)) < default_numerics_threshold
            and np.max(np.abs(delta_s)) < default_numerics_threshold
        ):
            raise ValueError(
                "Polytope distortions delta_b and delta_s non-zero after reaching max iterations."
            )

    if np.min(polytope.b.values) <= 0:
        center, _ = chebyshev_center(polytope, settings)
        polytope.apply_shift(center)


def run_mve(A, b, x0, reg):
    maxiter = 150
    tol2 = 1.0e-6
    (
        x,
        E2,
        msg,
        _y,
        _z,
        _iter,
        delta_b,
        delta_s,
    ) = solve_mve(A, b, x0, reg, maxiter=maxiter, tol=tol2)
    transform = cp.linalg.cholesky(get_NSPD(E2))
    return (
        cp.asnumpy(x),
        cp.asnumpy(transform),
        msg,
        cp.asnumpy(delta_b),
        cp.asnumpy(delta_s),
    )


def check_convergence(
    E2,
    r1,
    r2,
    last_r1,
    last_r2,
    res,
    tol,
    bnrm,
    rmu,
    minmu,
    prev_obj,
    x,
    x0,
    reg,
    iter,
):
    msg = 0
    E2_eig = cp.linalg.eigvalsh(E2)
    min_eig_gpu = cp.min(E2_eig)
    safe_eig = cp.where(E2_eig > 0, E2_eig, 1.0)
    min_eig, max_eig, sum_det = as_floats(
        min_eig_gpu,
        cp.max(E2_eig),
        cp.sum(cp.log(safe_eig)),
    )
    if min_eig > 0:
        objval = sum_det / 2
        # objval = np.log(np.linalg.det(E2)) / 2
    else:
        objval = -np.inf

    if (
        np.abs((last_r1 - r1) / np.minimum(np.abs(last_r1), np.abs(r1))) < 1e-2
        and np.abs((last_r2 - r2) / np.minimum(np.abs(last_r2), np.abs(r2))) < 1e-2
        and max_eig / min_eig > 100
        and reg > 1e-10
    ):
        x = x + x0.squeeze()
        msg = 2

    if (res < tol * (1 + bnrm) and rmu <= minmu) or (
        iter > 100
        and prev_obj != -np.inf
        and (prev_obj >= (1 - tol) * objval or prev_obj <= (1 - tol) * objval)
    ):
        x = x + x0.squeeze()
        msg = 1

    return objval, msg, x


def solve_mve(A, b, x0, reg, maxiter=50, tol=1e-4):
    # check that x and b are 2 dimensional
    assert len(b.shape) == 2 and len(x0.shape) == 2
    A = cp.asarray(A, dtype=cp.float64)
    b = cp.asarray(b, dtype=cp.float64)
    x0 = cp.asarray(x0, dtype=cp.float64)
    m, n = A.shape
    bnrm = as_float(cp.linalg.norm(b))

    minmu = 1.0e-8
    tau0 = 0.75

    smallest_scaling = default_0_width
    last_r1 = -np.inf
    last_r2 = -np.inf

    bmAx0 = b - A @ x0
    positive = cp.maximum(bmAx0, smallest_scaling)
    delta_b = positive - bmAx0
    bmAx0 = positive
    min_el = cp.min(bmAx0)
    limited = cp.minimum(bmAx0, min_el * default_max_ratio_bmAx0)
    delta_s = bmAx0 - limited
    bmAx0 = limited
    A = cp.divide(A, bmAx0)
    b = cp.ones((m,), dtype=A.dtype)
    x = cp.zeros((n,), dtype=A.dtype)
    y = cp.ones((m,), dtype=A.dtype)
    bmAx = b

    prev_obj = -np.inf
    astep = None
    Adx = None
    msg = 0
    for iter in range(1, maxiter + 1):
        if astep is not None and Adx is not None:
            bmAx = bmAx - astep * Adx

        Aty = cp.multiply(cp.transpose(A), cp.squeeze(y))
        assert A.shape[0] == Aty.shape[1]
        assert A.shape[1] == Aty.shape[0]
        prod = Aty @ A
        E2 = cp.linalg.inv(prod)
        Q = A @ E2 @ cp.transpose(A)
        h = cp.sqrt(cp.diag(Q))
        if iter == 1:
            t = cp.min(bmAx / h)
            y = y / cp.power(t, 2)
            h = t * h
            z = cp.maximum(1.0e-1, cp.squeeze(bmAx) - h)
            Q = cp.power(t, 2) * Q

        yz = cp.squeeze(y) * z
        yh = cp.squeeze(y) * h

        gap = cp.sum(yz) / m
        rmu = cp.minimum(0.5, gap) * gap
        rmu = cp.maximum(rmu, minmu)

        R1 = -cp.transpose(A) @ yh
        R2 = cp.squeeze(bmAx) - h - z
        R3 = rmu - yz

        residuals = (
            cp.max(cp.abs(R1)),
            cp.max(cp.abs(R2)),
            cp.max(cp.abs(R3)),
        )
        if iter % 10 == 0:
            r1, r2, r3, rmu_value = as_floats(*residuals, rmu)
        else:
            r1, r2, r3 = as_floats(*residuals)
        res = max(r1, r2, r3)

        if iter % 10 == 0:
            objval, msg, x = check_convergence(
                E2,
                r1,
                r2,
                last_r1,
                last_r2,
                res,
                tol,
                bnrm,
                rmu_value,
                minmu,
                prev_obj,
                x,
                x0,
                reg,
                iter,
            )
            if msg == 1 or msg == 2:
                break
            last_r2 = r2
            last_r1 = r1
            prev_obj = objval
        #
        YQ = cp.multiply(Q, cp.squeeze(y))
        YQQY = YQ * cp.transpose(YQ)
        y2h = 2 * yh
        YA = cp.transpose(cp.multiply(cp.transpose(A), cp.squeeze(y)))
        G = YQQY
        elementwise_max = cp.maximum(reg, y2h * z)
        diagonal = cp.arange(G.shape[0])
        G[diagonal, diagonal] += elementwise_max
        temp_rhs = cp.transpose(cp.multiply(cp.transpose(YA), h + z))
        G_c = cp.linalg.cholesky(G)
        T = cholesky_solve(G_c, temp_rhs)
        temp = cp.transpose(cp.multiply(cp.transpose(T), y2h))
        ATP = cp.transpose(temp - YA)
        R3Dy = R3 / cp.squeeze(y)
        R23 = R2 - R3Dy
        ATP_A = ATP @ A
        diagonal = cp.arange(ATP_A.shape[0])
        ATP_A[diagonal, diagonal] += reg
        dx = cp.linalg.solve(ATP_A, R1 + ATP @ R23)

        Adx = A @ dx
        dyDy = cholesky_solve(G_c, y2h * (Adx - R23))

        dy = y * dyDy
        dz = R3Dy - z * dyDy

        min_ax, min_ay, min_az = as_floats(
            cp.min(-Adx / bmAx),
            cp.min(dyDy),
            cp.min(dz / z),
        )
        ax = -1 / min(min_ax, -0.5)
        ay = -1 / min(min_ay, -0.5)
        az = -1 / min(min_az, -0.5)
        tau = max(tau0, 1 - res)
        astep = tau * min(1, ax, ay, az)

        x = x + astep * dx
        y = y + astep * dy
        z = z + astep * dz

        if reg > 1e-6 and iter >= 10:
            break

    return x, E2, msg, y, z, iter, delta_b, delta_s
