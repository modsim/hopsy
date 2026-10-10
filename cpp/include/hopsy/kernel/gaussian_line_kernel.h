#pragma once

#include <boost/math/distributions/normal.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <ranges>
#include <stdexcept>
#include <type_traits>

#include <hopsy/random_number_generator.h>

namespace hopsy {
    // Gaussian random-walk displacement, centered at zero, conditioned on the
    // supplied union of disjoint intervals. This is NOT a Gaussian target's
    // conditional sampler. The scale must remain fixed until acceptance/rejection.
    class GaussianLineKernel {
    public:
        explicit GaussianLineKernel(double step_size = 1.0) {
            set_standard_deviation(step_size);
        }

        [[nodiscard]] double standard_deviation() const noexcept { return sigma_; }

        void set_standard_deviation(double value) {
            if (!(value > 0) || !std::isfinite(value)) {
                throw std::invalid_argument("GaussianLineKernel: scale must be positive and finite.");
            }
            sigma_ = value;
        }

        template<typename LineDomain>
            requires std::ranges::forward_range<const LineDomain>
        [[nodiscard]] auto propose(RandomNumberGenerator &rng, const LineDomain &domain) const {
            using Interval = std::ranges::range_value_t<const LineDomain>;
            using Scalar = typename Interval::scalar_type;
            static_assert(std::is_floating_point_v<Scalar>);

            const auto total = normalization(domain, 0);
            std::uniform_real_distribution<long double> uniform(0, total);
            const long double draw = uniform(rng);
            long double remaining = draw;
            long double last_a = 0, last_b = 0, last_mass = 0;

            for (const auto &interval: domain) {
                const long double a = static_cast<long double>(interval.lower) / sigma_;
                const long double b = static_cast<long double>(interval.upper) / sigma_;
                const auto mass = interval_mass(a, b);
                if (mass == 0) continue;
                last_a = a;
                last_b = b;
                last_mass = mass;
                if (remaining < mass) {
                    return static_cast<Scalar>(sigma_ * inverse_within(a, b, remaining / mass));
                }
                remaining -= mass;
            }
            // Accumulated rounding can place the draw at the final mass boundary.
            if (last_mass > 0) {
                return static_cast<Scalar>(sigma_ * inverse_within(last_a, last_b,
                                                                   std::nextafter(1.0L, 0.0L)));
            }
            throw std::runtime_error("GaussianLineKernel: no representable interval mass.");
        }

        // log q(reverse) - log q(forward); deliberately NOT clipped at zero.
        template<typename LineDomain, typename Scalar>
            requires std::ranges::forward_range<const LineDomain>
        [[nodiscard]] Scalar log_proposal_ratio(const LineDomain &domain, Scalar step) const {
            if (!std::isfinite(step)) {
                throw std::invalid_argument("GaussianLineKernel: step must be finite.");
            }
            return static_cast<Scalar>(std::log(normalization(domain, 0)) -
                                       std::log(normalization(domain, step)));
        }

    private:
        using Normal = boost::math::normal_distribution<long double>;
        double sigma_ = 1;

        static long double interval_mass(long double a, long double b) {
            if (std::isnan(a) || std::isnan(b) || a > b) {
                throw std::invalid_argument("GaussianLineKernel: invalid interval.");
            }
            if (a == b) return 0;
            const Normal normal;
            // Subtract survival probabilities on the positive side to avoid
            // cancellation between two CDF values close to one.
            const auto mass = a >= 0
                                  ? boost::math::cdf(boost::math::complement(normal, a)) -
                                    boost::math::cdf(boost::math::complement(normal, b))
                                  : boost::math::cdf(normal, b) - boost::math::cdf(normal, a);
            if (!(mass > 0) || !std::isfinite(mass)) {
                throw std::runtime_error("GaussianLineKernel: interval mass lost numerically.");
            }
            return mass;
        }

        template<typename LineDomain>
        long double normalization(const LineDomain &domain, long double step) const {
            long double total = 0;
            for (const auto &interval: domain) {
                total += interval_mass((static_cast<long double>(interval.lower) - step) / sigma_,
                                       (static_cast<long double>(interval.upper) - step) / sigma_);
            }
            if (!(total > 0) || !std::isfinite(total)) {
                throw std::invalid_argument("GaussianLineKernel: domain must have positive Gaussian mass.");
            }
            return total;
        }

        static long double inverse_within(long double a, long double b, long double u) {
            const Normal normal;
            // Use an interior probability to avoid infinite results at unbounded
            // endpoints. This only adjusts floating-point endpoint draws.
            u = std::clamp(u, std::nextafter(0.0L, 1.0L), std::nextafter(1.0L, 0.0L));
            long double value;
            if (a >= 0) {
                const auto lo = boost::math::cdf(boost::math::complement(normal, b));
                const auto hi = boost::math::cdf(boost::math::complement(normal, a));
                auto p = std::lerp(hi, lo, u);
                p = std::clamp(p, std::nextafter(0.0L, 1.0L), std::nextafter(1.0L, 0.0L));
                value = boost::math::quantile(boost::math::complement(normal, p));
            } else {
                const auto lo = boost::math::cdf(normal, a);
                const auto hi = boost::math::cdf(normal, b);
                auto p = std::lerp(lo, hi, u);
                p = std::clamp(p, std::nextafter(0.0L, 1.0L), std::nextafter(1.0L, 0.0L));
                value = boost::math::quantile(normal, p);
            }
            // Quantile roundoff can cross a finite endpoint by a few ulps.
            return std::clamp(value, a, b);
        }
    };
} // namespace hopsy
