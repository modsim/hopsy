#ifndef DRAW_H
#define DRAW_H

#include <Eigen/Core>
#include <stdexcept>

#include <hopsy/sampling/step.h>

namespace hopsy {
    namespace detail {
        inline constexpr auto ignore_sample = [](Eigen::Index, const auto &...) {};

        template <typename Step, typename Record>
        void collect_draws(Eigen::Index count, Eigen::Index thinning, const Step &step,
                           const Record &record) {
            for (Eigen::Index sample = 0; sample < count; ++sample) {
                for (Eigen::Index iteration = 0; iteration < thinning; ++iteration)
                    step();
                record(sample);
            }
        }
    } // namespace detail

    template <typename Kernel, typename State, typename Target, typename Geometry,
              typename Workspace, typename Derived>
    void draw_into(Kernel &kernel, State &state, RandomNumberGenerator &rng, const Target &target,
                   const Geometry &geometry, Workspace &workspace,
                   Eigen::MatrixBase<Derived> &samples, Eigen::Index thinning = 1) {
        if (thinning < 1) {
            throw std::invalid_argument("draw_into: thinning must be positive.");
        }
        if (samples.rows() != state.size()) {
            throw std::invalid_argument("draw_into: sample rows must match the state dimension.");
        }

        // A batch binds one fixed target. Start fresh so changing its parameters
        // between batches cannot reuse stale target data or gradients.
        if constexpr (requires { kernel.invalidate_target_cache(); }) {
            if (samples.cols() > 0)
                kernel.invalidate_target_cache();
        }
        const auto record = [&](Eigen::Index sample) { samples.col(sample) = state; };
        if constexpr (detail::CachesLogDensity<Kernel> && !target_is_uniform_v<Target> &&
                      !Kernel::template always_accept<Target> &&
                      !requires { target.log_density_increment(state, state); }) {
            detail::LogDensityCache cache;
            const detail::CachedTarget cached(target, cache);
            detail::collect_draws(
                samples.cols(), thinning,
                [&] {
                    detail::step_with_density_cache(kernel, state, rng, cached, geometry, workspace,
                                                    cache);
                },
                record);
        } else {
            detail::collect_draws(
                samples.cols(), thinning,
                [&] { step(kernel, state, rng, target, geometry, workspace); }, record);
        }
    }

    template <typename Kernel, typename State, typename Target, typename Geometry,
              typename Workspace>
    [[nodiscard]] auto draw(Kernel &kernel, State &state, RandomNumberGenerator &rng,
                            const Target &target, const Geometry &geometry, Workspace &workspace,
                            Eigen::Index num_samples, Eigen::Index thinning = 1) {
        if (num_samples < 0) {
            throw std::invalid_argument("draw: num_samples must be nonnegative.");
        }
        if (thinning < 1) {
            throw std::invalid_argument("draw: thinning must be positive.");
        }

        using Scalar = typename State::Scalar;
        using SampleMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

        SampleMatrix samples(state.size(), num_samples);

        draw_into(kernel, state, rng, target, geometry, workspace, samples, thinning);

        return samples;
    }
} // namespace hopsy

#endif // DRAW_H
