#ifndef DRIFT_H
#define DRIFT_H

#include <cmath>

namespace hopsy {
    struct ZeroDrift {
        static constexpr bool scales_quadratically = true;
        // The output is a displacement, including any step-size scaling.
        // Geometry workspace is read-only and remains owned by the geometry.
        template <typename State, typename Target, typename Metric, typename Geometry,
                  typename Workspace>
        void compute(const State &, typename State::Scalar, const Target &, const Metric &,
                     const Geometry &, const Workspace &, State &displacement) const noexcept {
            displacement.setZero();
        }
    };

    struct ClippedGradientDrift {
        // compute(..., sigma=1) is the cacheable coefficient of sigma^2.
        // Policies with scale-dependent clipping must leave this opt-in false.
        static constexpr bool scales_quadratically = true;
        // Clip the log-density gradient at norm 1 before preconditioning.
        // Small gradients remain unchanged, so the displacement is continuous
        // at zero. sigma is a standard deviation, hence the sigma^2/2 factor.
        template <typename State, typename Target, typename Metric, typename Geometry,
                  typename Workspace>
        void compute(const State &state, typename State::Scalar sigma, const Target &target,
                     const Metric &metric, const Geometry &, const Workspace &,
                     State &displacement) const {
            displacement = target.log_gradient(state);
            const auto norm = displacement.stableNorm();
            if (norm > 1.)
                displacement /= norm;
            metric.solve_in_place(displacement);
            displacement *= sigma;
            displacement *= sigma / 2;
        }
    };
} // namespace hopsy

#endif // DRIFT_H
