import numpy as np
from scipy.linalg import cho_factor, cho_solve

from hopsy._polyround.default_settings import (
    default_0_width,
    default_max_ratio_bmAx0,
    default_numerics_threshold,
)
from hopsy._polyround.static_classes.lp_utils import ChebyshevFinder
from hopsy._polyround.static_classes.rounding.geometric_mean_scaling import (
    geometric_mean_scaling,
)
from hopsy._polyround.static_classes.rounding.nearest_symmetric_positive_definite import (
    NSPD,
)


class MaximumVolumeEllipsoidFinder:
    @staticmethod
    def iterative_solve(
        polytope,
        settings,
    ):
        # only round reduced polytopes
        assert polytope.inequality_only
        nDimensions = polytope.A.shape[1]
        columnScale, rowScale = geometric_mean_scaling(polytope.A.values, 0, 0.99)
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
            [center, distance] = ChebyshevFinder.chebyshev_center(polytope, settings)
            reg = np.maximum(reg / 10, 1e-10)

            # Calculate and apply transform
            (
                iterationShift,
                iterationTransform,
                converged,
                delta_b,
                delta_s,
            ) = MaximumVolumeEllipsoidFinder.run_mve(
                polytope.A.values.copy(),
                polytope.b.values.copy()[:, None],
                center,
                reg,
                verbose=settings.verbose,
            )
            polytope.apply_shift(iterationShift)
            polytope.apply_transformation(iterationTransform)
            polytope.normalize()
            if iteration == maxIterations:
                break
            temp_eig_all = np.linalg.eig(iterationTransform)
            temp_eig = temp_eig_all[0]
            if settings.verbose:
                print(
                    "Iteration: "
                    + str(iteration)
                    + ", reg: "
                    + str(reg)
                    + ", log ellipsoid vol: "
                    + str(np.sum(np.log(temp_eig)))
                    + ", longest axis: "
                    + str(np.max(temp_eig))
                    + ", shortest axis: "
                    + str(np.min(temp_eig))
                    + ", border distance: "
                    + str(distance)
                    + ", max_delta_b: "
                    + str(np.max(np.abs(delta_b)))
                    + ", max_delta_s: "
                    + str(np.max(np.abs(delta_s)))
                )

        if iteration == maxIterations:
            if settings.verbose:
                print(
                    "Maximum number of iterations reached, rounding may not be ideal."
                )
            if not (
                np.max(np.abs(delta_b)) < default_numerics_threshold
                and np.max(np.abs(delta_s)) < default_numerics_threshold
            ):
                raise ValueError(
                    "Polytope distortions delta_b and delta_s non-zero after reaching max iterations."
                )

        if settings.verbose:
            print("Maximum volume ellipsoid found.")
        if np.min(polytope.b.values) <= 0:
            center, distance = ChebyshevFinder.chebyshev_center(polytope, settings)
            polytope.apply_shift(center)
            if settings.verbose:
                print("Shifting so that the origin is inside the polytope.")

    @staticmethod
    def run_mve(A, b, x0, reg, verbose=False):
        maxiter = 150
        tol2 = 1.0e-6
        (
            x,
            E2,
            msg,
            y,
            z,
            iter,
            delta_b,
            delta_s,
        ) = MaximumVolumeEllipsoidFinder.solve_mve(
            A, b, x0, reg, maxiter=maxiter, tol=tol2, verbose=verbose
        )
        return x, np.linalg.cholesky(NSPD.get_NSPD(E2)), msg, delta_b, delta_s

    @staticmethod
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
        E2_eig_all = np.linalg.eig(E2)
        E2_eig = E2_eig_all[0]
        if np.min(E2_eig) > 0:
            sum_det = np.sum(np.log(E2_eig))
            objval = sum_det / 2
            # objval = np.log(np.linalg.det(E2)) / 2
        else:
            objval = -np.inf

        if verbose:
            print("Iteration number in inner loop of MVE optimization: " + str(iter))
        if (
            np.abs((last_r1 - r1) / np.minimum(np.abs(last_r1), np.abs(r1))) < 1e-2
            and np.abs((last_r2 - r2) / np.minimum(np.abs(last_r2), np.abs(r2))) < 1e-2
            and np.max(E2_eig) / np.min(E2_eig) > 100
            and reg > 1e-10
        ):
            if verbose:
                print("Stopped making progress, stopping and restarting.")
            x = x + x0.squeeze()
            msg = 2

        if (res < tol * (1 + bnrm) and rmu <= minmu) or (
            iter > 100
            and prev_obj != -np.inf
            and (prev_obj >= (1 - tol) * objval or prev_obj <= (1 - tol) * objval)
        ):
            if verbose:
                print("Converged")
            x = x + x0.squeeze()
            msg = 1

        return objval, msg, x

    @staticmethod
    def solve_mve(A, b, x0, reg, maxiter=50, tol=1e-4, verbose=False):
        # check that x and b are 2 dimensional
        assert len(b.shape) == 2 and len(x0.shape) == 2
        m, n = A.shape
        bnrm = np.linalg.norm(b)

        minmu = 1.0e-8
        tau0 = 0.75

        smallest_scaling = default_0_width
        last_r1 = -np.inf
        last_r2 = -np.inf

        bmAx0 = b - np.matmul(A, x0)
        positive = np.maximum(bmAx0, smallest_scaling)
        delta_b = positive - bmAx0
        bmAx0 = positive
        min_el = np.min(bmAx0)
        limited = np.minimum(bmAx0, min_el * default_max_ratio_bmAx0)
        delta_s = bmAx0 - limited
        bmAx0 = limited
        # if np.any(bmAx0 <= 0):
        #     if verbose:
        #         print("x0 not interior, use absolute value")
        #     bmAx0 = np.abs(bmAx0)
        #     # raise ValueError

        A = np.divide(A, bmAx0)
        b = np.ones((m,))
        x = np.zeros((n,))
        y = np.ones((m,))
        bmAx = b

        prev_obj = -np.inf
        astep = None
        Adx = None
        msg = 0
        for iter in range(1, maxiter + 1):

            if astep is not None and Adx is not None:
                bmAx = bmAx - astep * Adx

            Aty = np.multiply(np.transpose(A), np.squeeze(y))
            assert A.shape[0] == Aty.shape[1]
            assert A.shape[1] == Aty.shape[0]
            prod = np.matmul(Aty, A)
            E2 = np.linalg.inv(prod)
            Q = np.matmul(np.matmul(A, E2), np.transpose(A))
            h = np.sqrt(np.diag(Q))
            if iter == 1:
                t = np.min(bmAx / h)
                y = y / np.power(t, 2)
                h = t * h
                z = np.maximum(1.0e-1, np.squeeze(bmAx) - h)
                Q = np.power(t, 2) * Q

            yz = np.squeeze(y) * z
            yh = np.squeeze(y) * h

            gap = np.sum(yz) / m
            rmu = np.minimum(0.5, gap) * gap
            rmu = np.maximum(rmu, minmu)

            R1 = -np.matmul(np.transpose(A), yh)
            R2 = np.squeeze(bmAx) - h - z
            R3 = rmu - yz

            r1 = np.max(np.abs(R1))
            r2 = np.max(np.abs(R2))
            r3 = np.max(np.abs(R3))
            res = np.max([r1, r2, r3])

            if iter % 10 == 0:
                # print(r2, r1, r3, objval);
                objval, msg, x = MaximumVolumeEllipsoidFinder.check_convergence(
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
                )
                if msg == 1 or msg == 2:
                    break
                last_r2 = r2
                last_r1 = r1
                prev_obj = objval
            #
            YQ = np.multiply(Q, np.squeeze(y))
            YQQY = YQ * np.transpose(YQ)
            y2h = 2 * yh
            YA = np.transpose(np.multiply(np.transpose(A), np.squeeze(y)))
            G = YQQY
            elementwise_max = np.maximum(reg, y2h * z)
            G.flat[:: G.shape[1] + 1] += elementwise_max
            temp_rhs = np.transpose(np.multiply(np.transpose(YA), h + z))
            G_c, low = cho_factor(G)
            T = cho_solve((G_c, low), temp_rhs)
            temp = np.transpose(np.multiply(np.transpose(T), y2h))
            ATP = np.transpose(temp - YA)
            R3Dy = R3 / np.squeeze(y)
            R23 = R2 - R3Dy
            ATP_A = np.matmul(ATP, A)
            ATP_A.flat[:: ATP_A.shape[1] + 1] += reg
            dx = np.linalg.solve(ATP_A, R1 + np.matmul(ATP, R23))

            Adx = np.matmul(A, dx)
            dyDy = cho_solve((G_c, low), y2h * (Adx - R23))

            dy = y * dyDy
            dz = R3Dy - z * dyDy

            ax = -1 / np.minimum(np.min(-Adx / bmAx), -0.5)
            ay = -1 / np.minimum(np.min(dyDy), -0.5)
            az = -1 / np.minimum(np.min(dz / z), -0.5)
            tau = np.maximum(tau0, 1 - res)
            astep = tau * np.min([1, ax, ay, az])

            x = x + astep * dx
            y = y + astep * dy
            z = z + astep * dz

            if reg > 1e-6 and iter >= 10:
                break

        return x, E2, msg, y, z, iter, delta_b, delta_s
