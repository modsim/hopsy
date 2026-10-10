#ifndef STEP_H
#define STEP_H

#include <cmath>
#include <limits>
#include <random>
#include <type_traits>

#include <hopsy/random_number_generator.h>

namespace hopsy {
    template<typename Kernel, typename State, typename Target,
        typename Geometry, typename Workspace>
    bool step(
        Kernel &kernel,
        State &state,
        RandomNumberGenerator &rng,
        const Target &target,
        const Geometry &geometry,
        Workspace &workspace
    ) {
        kernel.propose(state, rng, target, geometry, workspace);

        if constexpr (Kernel::template always_accept<Target>) {
            kernel.accept(state, target, geometry, workspace);
            return true;
        } else {
            const auto log_ratio = kernel.log_acceptance_ratio(
                state, target, geometry, workspace
            );

            using Scalar = std::remove_cvref_t<decltype(log_ratio)>;

            if (std::isnan(log_ratio)) {
                kernel.reject(state, target, geometry, workspace);
                return false;
            }

            bool accepted = log_ratio >= Scalar{0};

            if (!accepted && log_ratio != -std::numeric_limits<Scalar>::infinity()) {
                std::uniform_real_distribution<Scalar> uniform(
                    Scalar{0}, Scalar{1}
                );
                accepted = std::log(uniform(rng)) < log_ratio;
            }

            if (accepted) {
                kernel.accept(state, target, geometry, workspace);
            } else {
                kernel.reject(state, target, geometry, workspace);
            }

            return accepted;
        }
    }
}

#endif //STEP_H
