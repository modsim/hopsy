#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <ranges>
#include <stdexcept>

#include <hopsy/random_number_generator.h>

namespace hopsy {
    // Linear overrelaxation on ONE chord, with the current point at t=0.
    // In normalized coordinates: p(v|u) = 1 - (2u-1)(2v-1).
    // This transition is symmetric and preserves the uniform line distribution.
    class OverRelaxedLineKernel {
    public:
        template<typename LineDomain>
            requires std::ranges::forward_range<const LineDomain>
        [[nodiscard]] auto propose(RandomNumberGenerator &rng, const LineDomain &domain) const {
            using Interval = std::ranges::range_value_t<const LineDomain>;
            using Scalar = typename Interval::scalar_type;
            auto it = std::ranges::begin(domain);
            const auto end = std::ranges::end(domain);
            if (it == end) throw std::invalid_argument("OverRelaxedLineKernel: empty domain.");
            const auto interval = *it;
            if (++it != end) {
                throw std::invalid_argument("OverRelaxedLineKernel: exactly one interval is required.");
            }
            const Scalar a = interval.lower;
            const Scalar b = interval.upper;
            const Scalar length = b - a;
            if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(length) ||
                !(length > Scalar{0}) || a > Scalar{0} || b < Scalar{0}) {
                throw std::invalid_argument("OverRelaxedLineKernel: finite positive chord containing zero required.");
            }
            const Scalar u = -a / length;
            const Scalar correlation = Scalar{2} * u - Scalar{1};
            std::uniform_real_distribution<Scalar> uniform(Scalar{0}, Scalar{1});
            const Scalar r = uniform(rng);
            if (r == Scalar{0}) return a;
            // Rationalized inverse of F(v)=(1+A)v-A*v*v. Unlike the
            // direct quadratic formula, this is stable at A=0 and small r.
            const Scalar discriminant = std::max(Scalar{0},
                                                 (Scalar{1} + correlation) * (Scalar{1} + correlation) -
                                                 Scalar{4} * correlation * r);
            const Scalar v = Scalar{2} * r /
                             (Scalar{1} + correlation + std::sqrt(discriminant));
            return std::lerp(a, b, std::clamp(v, Scalar{0}, Scalar{1}));
        }
    };

    template<>
    struct line_kernel_requires_mh<OverRelaxedLineKernel> : std::false_type {
    };
} // namespace hopsy
