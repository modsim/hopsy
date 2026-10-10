#ifndef DRAW_H
#define DRAW_H

#include <Eigen/Core>
#include <stdexcept>

#include <hopsy/sampling/step.h>

namespace hopsy {
    template<typename Kernel, typename State, typename Target,
        typename Geometry, typename Workspace, typename Derived>
    void draw_into(
        Kernel &kernel,
        State &state,
        RandomNumberGenerator &rng,
        const Target &target,
        const Geometry &geometry,
        Workspace &workspace,
        Eigen::MatrixBase<Derived> &samples,
        Eigen::Index thinning = 1
    ) {
        if (thinning < 1) {
            throw std::invalid_argument("draw_into: thinning must be positive.");
        }
        if (samples.rows() != state.size()) {
            throw std::invalid_argument(
                "draw_into: sample rows must match the state dimension."
            );
        }

        for (Eigen::Index sample = 0; sample < samples.cols(); ++sample) {
            for (Eigen::Index iteration = 0; iteration < thinning; ++iteration) {
                step(kernel, state, rng, target, geometry, workspace);
            }

            samples.col(sample) = state;
        }
    }

    template<typename Kernel, typename State, typename Target,
        typename Geometry, typename Workspace>
    [[nodiscard]] auto draw(
        Kernel &kernel,
        State &state,
        RandomNumberGenerator &rng,
        const Target &target,
        const Geometry &geometry,
        Workspace &workspace,
        Eigen::Index num_samples,
        Eigen::Index thinning = 1
    ) {
        if (num_samples < 0) {
            throw std::invalid_argument("draw: num_samples must be nonnegative.");
        }
        if (thinning < 1) {
            throw std::invalid_argument("draw: thinning must be positive.");
        }

        using Scalar = typename State::Scalar;
        using SampleMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

        SampleMatrix samples(state.size(), num_samples);

        draw_into(
            kernel, state, rng, target, geometry, workspace,
            samples, thinning
        );

        return samples;
    }
}

#endif //DRAW_H
