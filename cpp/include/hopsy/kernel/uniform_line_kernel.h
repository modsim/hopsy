#ifndef UNIFORM_LINE_H
#define UNIFORM_LINE_H

#include <cmath>
#include <random>
#include <ranges>
#include <stdexcept>

#include <hopsy/kernel/traits.h>
#include <hopsy/random_number_generator.h>

namespace hopsy {
    class UniformLineKernel {
      public:
        // log q(reverse) - log q(forward) for a translated chord domain.
        template <typename LineDomain, typename Scalar>
        [[nodiscard]] constexpr double log_proposal_ratio(const LineDomain &,
                                                          Scalar) const noexcept {
            return 0.;
        }

        template <typename LineDomain>
            requires std::ranges::forward_range<const LineDomain>
        [[nodiscard]] auto propose(RandomNumberGenerator &rng,
                                   const LineDomain &line_domain) const {
            using Interval = std::ranges::range_value_t<const LineDomain>;
            using Scalar = typename Interval::scalar_type;

            // A single chord needs only one scan, whether its container is
            // fixed-size or a dynamically sized interval union.
            if constexpr (std::ranges::sized_range<const LineDomain>) {
                if (std::ranges::size(line_domain) == 1) {
                    const auto &interval = *std::ranges::begin(line_domain);
                    const Scalar length = interval.upper - interval.lower;
                    if (!std::isfinite(interval.lower) || !std::isfinite(interval.upper) ||
                        length < Scalar{0})
                        throw std::invalid_argument(
                            "UniformLineKernel: invalid or unbounded interval.");
                    if (!(length > Scalar{0}) || !std::isfinite(length))
                        throw std::invalid_argument(
                            "UniformLineKernel: total length must be positive and finite.");
                    std::uniform_real_distribution<Scalar> uniform(Scalar{0}, length);
                    Scalar offset = uniform(rng);
                    if (offset < length) {
                        Scalar point = interval.lower + offset;
                        return point < interval.upper ? point : interval.upper;
                    }
                    return interval.upper;
                }
            }

            Scalar total_length{0};

            for (const auto &interval : line_domain) {
                const Scalar length = interval.upper - interval.lower;

                if (!std::isfinite(interval.lower) || !std::isfinite(interval.upper) ||
                    length < Scalar{0}) {
                    throw std::invalid_argument(
                        "UniformLineKernel: invalid or unbounded interval.");
                }

                total_length += length;
            }

            if (!(total_length > Scalar{0}) || !std::isfinite(total_length)) {
                throw std::invalid_argument(
                    "UniformLineKernel: total length must be positive and finite.");
            }

            std::uniform_real_distribution<Scalar> uniform(Scalar{0}, total_length);
            Scalar offset = uniform(rng);
            Scalar last_upper{0};

            for (const auto &interval : line_domain) {
                const Scalar length = interval.upper - interval.lower;

                if (length == Scalar{0}) {
                    continue;
                }

                last_upper = interval.upper;

                if (offset < length) {
                    // Clamp possible rounding overshoot at the upper endpoint.
                    const Scalar point = interval.lower + offset;
                    return point < interval.upper ? point : interval.upper;
                }

                offset -= length;
            }

            // Floating-point rounding can leave offset at the total-length edge.
            return last_upper;
        }
    };

    template <> struct line_kernel_requires_mh<UniformLineKernel> : std::false_type {};

} // namespace hopsy

#endif // UNIFORM_LINE_H
