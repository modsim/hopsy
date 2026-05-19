from contextlib import contextmanager
from time import perf_counter

import numpy as np

from hopsy._polyround.default_settings import (
    default_0_width,
    default_max_ratio_bmAx0,
    default_numerics_threshold,
)

from .geometric_mean_scaling import geometric_mean_scaling
from .lp_utils import chebyshev_center
from .nearest_symmetric_positive_definite import get_NSPD


def verbose_print(settings, backend, message):
    if getattr(settings, "verbose", bool(settings)):
        print(f"[{backend}] {message}", flush=True)


try:
    # CPU version: NumPy + scipy.linalg
    # GPU version: CuPy arrays + CuPyX triangular solves for the Cholesky systems
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
except ImportError as error:
    raise ImportError("hopsy's exp1 PolyRound backend requires CuPy.") from error


def as_float(value):
    # Branching and logging happen on the host,
    # scalar reductions from CuPy are materialized as Python floats
    return float(cp.asnumpy(value))


def as_floats(*values):
    stacked = cp.stack([cp.asarray(value) for value in values])
    return tuple(float(value) for value in cp.asnumpy(stacked))


def scalar_float(value):
    return float(np.asarray(value).reshape(-1)[0])


def format_bytes(nbytes):
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(nbytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0


def matrix_size(shape, dtype=np.float64):
    return format_bytes(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)


def sync_if_verbose(settings):
    if getattr(settings, "verbose", bool(settings)):
        cp.cuda.Stream.null.synchronize()


@contextmanager
def verbose_timer(settings, backend, label):
    if not getattr(settings, "verbose", bool(settings)):
        yield
        return

    verbose_print(settings, backend, f"start {label}")
    start = perf_counter()
    try:
        yield
    finally:
        sync_if_verbose(settings)
        elapsed = perf_counter() - start
        verbose_print(settings, backend, f"done  {label} ({elapsed:.3f}s)")


def cholesky_solve(factor, rhs):
    # scipy.linalg.cho_solve in the CPU backend accepts the packed factor tuple.
    # On GPU, keep the lower Cholesky factor and perform the two triangular
    # solves explicitly.
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


def iterative_solve(polytope, settings):
    # only round reduced polytopes
    assert polytope.inequality_only
    verbose = getattr(settings, "verbose", bool(settings))
    nDimensions = polytope.A.shape[1]
    verbose_print(
        settings,
        "exp1",
        "iterative_solve start "
        + f"A={polytope.A.shape}, b={polytope.b.shape}, "
        + f"nonzeros={np.count_nonzero(polytope.A.values)}",
    )
    with verbose_timer(
        settings,
        "exp1",
        f"geometric_mean_scaling A={polytope.A.shape}",
    ):
        # exp1/geometric_mean_scaling uses CuPy internally, but returns NumPy
        # scales so the host-side Polytope transformation code can stay shared
        # with the CPU backend.
        columnScale, _ = geometric_mean_scaling(polytope.A.values, 0, 0.99)
    # just this once, do the wasteful expansion
    c_scale_transform = np.diag(1 / columnScale)
    polytope.apply_transformation(c_scale_transform)

    # Normalize
    polytope.normalize()

    # Iterative rounding
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
        verbose_print(
            settings,
            "exp1",
            f"outer iteration={iteration} begin "
            + f"reg={reg:.3e}, A={polytope.A.shape}, b_min={polytope.b.min():.3e}",
        )
        with verbose_timer(settings, "exp1", f"chebyshev_center outer={iteration}"):
            [center, distance] = chebyshev_center(polytope, settings)
        verbose_print(
            settings,
            "exp1",
            f"outer iteration={iteration} chebyshev distance={scalar_float(distance):.6e}",
        )
        reg = np.maximum(reg / 10, 1e-10)

        # Calculate and apply transform
        with verbose_timer(
            settings,
            "exp1",
            f"run_mve outer={iteration} reg={reg:.3e}",
        ):
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
                verbose=verbose,
            )
        verbose_print(
            settings,
            "exp1",
            f"outer iteration={iteration} run_mve finished "
            + f"msg={converged}, max_delta_b={np.max(np.abs(delta_b)):.3e}, "
            + f"max_delta_s={np.max(np.abs(delta_s)):.3e}",
        )
        polytope.apply_shift(iterationShift)
        polytope.apply_transformation(iterationTransform)
        polytope.normalize()
        if iteration == maxIterations:
            break
        temp_eig_all = np.linalg.eig(iterationTransform)
        temp_eig = temp_eig_all[0]
        verbose_print(
            settings,
            "exp1",
            "iteration="
            + str(iteration)
            + ", reg="
            + str(reg)
            + ", log ellipsoid vol="
            + str(np.sum(np.log(temp_eig)))
            + ", longest axis="
            + str(np.max(temp_eig))
            + ", shortest axis="
            + str(np.min(temp_eig))
            + ", border distance="
            + str(distance)
            + ", max_delta_b="
            + str(np.max(np.abs(delta_b)))
            + ", max_delta_s="
            + str(np.max(np.abs(delta_s))),
        )

    if iteration == maxIterations:
        verbose_print(
            settings,
            "exp1",
            "maximum number of iterations reached; rounding may not be ideal",
        )
        if not (
            np.max(np.abs(delta_b)) < default_numerics_threshold
            and np.max(np.abs(delta_s)) < default_numerics_threshold
        ):
            raise ValueError(
                "Polytope distortions delta_b and delta_s non-zero after reaching max iterations."
            )

    verbose_print(settings, "exp1", "maximum volume ellipsoid found")
    if np.min(polytope.b.values) <= 0:
        center, _ = chebyshev_center(polytope, settings)
        polytope.apply_shift(center)
        verbose_print(
            settings,
            "exp1",
            "shifting so that the origin is inside the polytope",
        )


def run_mve(A, b, x0, reg, verbose=False):
    verbose_output = getattr(verbose, "verbose", bool(verbose))
    maxiter = 150
    tol2 = 1.0e-6
    m, n = A.shape
    verbose_print(
        verbose,
        "exp1",
        "run_mve start "
        + f"A={A.shape}, b={b.shape}, x0={x0.shape}, reg={reg:.3e}, "
        + f"Q/G={matrix_size((m, m))}, E2={matrix_size((n, n))}",
    )
    with verbose_timer(
        verbose_output,
        "exp1",
        f"solve_mve A={A.shape} maxiter={maxiter}",
    ):
        (
            x,
            E2,
            msg,
            _y,
            _z,
            _iter,
            delta_b,
            delta_s,
        ) = solve_mve(A, b, x0, reg, maxiter=maxiter, tol=tol2, verbose=verbose)
    verbose_print(
        verbose,
        "exp1",
        f"solve_mve returned msg={msg}, inner_iterations={_iter}",
    )
    with verbose_timer(verbose_output, "exp1", f"get_NSPD E2={E2.shape}"):
        E2 = get_NSPD(E2)
    with verbose_timer(
        verbose_output,
        "exp1",
        f"cholesky transform E2={E2.shape}",
    ):
        # Keep the SPD correction and Cholesky on the device.  The resulting
        # transform is copied back below because Polytope.apply_transformation
        # wants a NumPy array.
        transform = cp.linalg.cholesky(E2)
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
    verbose,
):
    msg = 0
    # E2 is symmetric by construction
    # The CPU backend uses np.linalg.eig
    # GPU backend uses eigvalsh, the symmetric/Hermitian equivalent
    # Avoids complex/general-eigenvalue work
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

    verbose_print(verbose, "exp1", "inner MVE iteration=" + str(iter))
    if (
        np.abs((last_r1 - r1) / np.minimum(np.abs(last_r1), np.abs(r1))) < 1e-2
        and np.abs((last_r2 - r2) / np.minimum(np.abs(last_r2), np.abs(r2))) < 1e-2
        and max_eig / min_eig > 100
        and reg > 1e-10
    ):
        verbose_print(verbose, "exp1", "stopped making progress; restarting")
        x = x + x0.squeeze()
        msg = 2

    if (res < tol * (1 + bnrm) and rmu <= minmu) or (
        iter > 100
        and prev_obj != -np.inf
        and (prev_obj >= (1 - tol) * objval or prev_obj <= (1 - tol) * objval)
    ):
        verbose_print(verbose, "exp1", "inner MVE converged")
        x = x + x0.squeeze()
        msg = 1

    return objval, msg, x


def solve_mve(A, b, x0, reg, maxiter=50, tol=1e-4, verbose=False):
    # check that x and b are 2 dimensional
    assert len(b.shape) == 2 and len(x0.shape) == 2
    verbose_output = getattr(verbose, "verbose", bool(verbose))
    verbose_print(
        verbose,
        "exp1",
        "solve_mve upload "
        + f"A={A.shape} ({matrix_size(A.shape)}), b={b.shape}, x0={x0.shape}",
    )
    # Main CPU-to-GPU boundary
    # After this, the dense MVE iteration keeps A, b, x0, residuals, and Newton systems on the device
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
    if verbose_output:
        min_slack, max_slack, max_delta_b, max_delta_s = as_floats(
            cp.min(bmAx0),
            cp.max(bmAx0),
            cp.max(cp.abs(delta_b)),
            cp.max(cp.abs(delta_s)),
        )
        verbose_print(
            verbose,
            "exp1",
            "solve_mve slack scaling "
            + f"min={min_slack:.3e}, max={max_slack:.3e}, "
            + f"max_delta_b={max_delta_b:.3e}, max_delta_s={max_delta_s:.3e}",
        )
    # if np.any(bmAx0 <= 0):
    #     if verbose:
    #         print("x0 not interior, use absolute value")
    #     bmAx0 = np.abs(bmAx0)
    #     # raise ValueError

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
        if verbose_output:
            verbose_print(
                verbose,
                "exp1",
                f"solve_mve iter={iter} begin reg={reg:.3e}",
            )
        if astep is not None and Adx is not None:
            bmAx = bmAx - astep * Adx

        # Same as CPU backend, but all operands are CuPy arrays
        # Matrix products below are CUDA BLAS/LAPACK operations
        Aty = cp.multiply(cp.transpose(A), cp.squeeze(y))
        assert A.shape[0] == Aty.shape[1]
        assert A.shape[1] == Aty.shape[0]
        with verbose_timer(
            verbose_output,
            "exp1",
            f"iter={iter} form prod/E2 {n}x{n}",
        ):
            prod = Aty @ A
            E2 = cp.linalg.inv(prod)
        with verbose_timer(
            verbose_output,
            "exp1",
            f"iter={iter} form Q {m}x{m} ({matrix_size((m, m))})",
        ):
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
        # The convergence test is Python, large vectors/matrices remain on device
        if iter % 10 == 0:
            r1, r2, r3, rmu_value = as_floats(*residuals, rmu)
        else:
            r1, r2, r3 = as_floats(*residuals)
        res = max(r1, r2, r3)
        if verbose_output:
            gap_value, rmu_log_value = as_floats(gap, rmu)
            verbose_print(
                verbose,
                "exp1",
                f"solve_mve iter={iter} residuals "
                + f"r1={r1:.3e}, r2={r2:.3e}, r3={r3:.3e}, "
                + f"res={res:.3e}, gap={gap_value:.3e}, rmu={rmu_log_value:.3e}",
            )

        if iter % 10 == 0:
            # print(r2, r1, r3, objval);
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
                verbose,
            )
            if msg == 1 or msg == 2:
                break
            last_r2 = r2
            last_r1 = r1
            prev_obj = objval
        #
        with verbose_timer(
            verbose_output,
            "exp1",
            f"iter={iter} form G {m}x{m} ({matrix_size((m, m))})",
        ):
            YQ = cp.multiply(Q, cp.squeeze(y))
            YQQY = YQ * cp.transpose(YQ)
            y2h = 2 * yh
            YA = cp.transpose(cp.multiply(cp.transpose(A), cp.squeeze(y)))
            G = YQQY
            elementwise_max = cp.maximum(reg, y2h * z)
            # NumPy's flat diagonal update is written with explicit GPU indices
            diagonal = cp.arange(G.shape[0])
            G[diagonal, diagonal] += elementwise_max
            temp_rhs = cp.transpose(cp.multiply(cp.transpose(YA), h + z))
        with verbose_timer(
            verbose_output,
            "exp1",
            f"iter={iter} cholesky G {m}x{m}",
        ):
            # Explicit lower Cholesky factor, then cholesky_solve() above
            G_c = cp.linalg.cholesky(G)
        with verbose_timer(
            verbose_output,
            "exp1",
            f"iter={iter} triangular solve T {m}x{n}",
        ):
            T = cholesky_solve(G_c, temp_rhs)
            temp = cp.transpose(cp.multiply(cp.transpose(T), y2h))
            ATP = cp.transpose(temp - YA)
        R3Dy = R3 / cp.squeeze(y)
        R23 = R2 - R3Dy
        with verbose_timer(
            verbose_output,
            "exp1",
            f"iter={iter} solve ATP_A {n}x{n}",
        ):
            ATP_A = ATP @ A
            diagonal = cp.arange(ATP_A.shape[0])
            ATP_A[diagonal, diagonal] += reg
            dx = cp.linalg.solve(ATP_A, R1 + ATP @ R23)

        Adx = A @ dx
        with verbose_timer(
            verbose_output,
            "exp1",
            f"iter={iter} triangular solve dyDy {m}",
        ):
            dyDy = cholesky_solve(G_c, y2h * (Adx - R23))

        dy = y * dyDy
        dz = R3Dy - z * dyDy

        min_ax, min_ay, min_az = as_floats(
            cp.min(-Adx / bmAx),
            cp.min(dyDy),
            cp.min(dz / z),
        )
        # Step-size selection: host scalar logic
        # The three minima: device
        ax = -1 / min(min_ax, -0.5)
        ay = -1 / min(min_ay, -0.5)
        az = -1 / min(min_az, -0.5)
        tau = max(tau0, 1 - res)
        astep = tau * min(1, ax, ay, az)
        if verbose_output:
            verbose_print(
                verbose,
                "exp1",
                f"solve_mve iter={iter} step "
                + f"astep={astep:.3e}, ax={ax:.3e}, ay={ay:.3e}, az={az:.3e}",
            )

        x = x + astep * dx
        y = y + astep * dy
        z = z + astep * dz

        if reg > 1e-6 and iter >= 10:
            break

    return x, E2, msg, y, z, iter, delta_b, delta_s
