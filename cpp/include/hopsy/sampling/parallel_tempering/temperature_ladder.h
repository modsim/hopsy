#ifndef HOPSY_TEMPERATURE_LADDER_H
#define HOPSY_TEMPERATURE_LADDER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <boost/math/interpolators/cubic_hermite.hpp>

namespace hopsy {
    // A strictly increasing inverse-temperature schedule from reference to target.
    class TemperatureLadder {
      public:
        explicit TemperatureLadder(std::vector<double> inverse_temperatures)
            : betas_(std::move(inverse_temperatures)) {
            if (betas_.size() < 2 || betas_.front() != 0. || betas_.back() != 1.)
                throw std::invalid_argument("Temperature ladder requires endpoints 0 and 1");
            for (std::size_t i = 1; i < betas_.size(); ++i) {
                if (!(betas_[i - 1] < betas_[i]))
                    throw std::invalid_argument("Inverse temperatures must strictly increase");
            }
        }

        static TemperatureLadder linear(std::size_t count) {
            if (count < 2)
                throw std::invalid_argument("Temperature ladder requires at least two slots");
            std::vector<double> betas(count);
            for (std::size_t i = 1; i < count; ++i)
                betas[i] = static_cast<double>(i) / static_cast<double>(count - 1);
            return TemperatureLadder(std::move(betas));
        }

        std::span<const double> inverse_temperatures() const noexcept { return betas_; }
        std::size_t size() const noexcept { return betas_.size(); }
        double operator[](std::size_t slot) const noexcept { return betas_[slot]; }

        // Interpolate the inverse cumulative rejection curve (PCHIP) and space
        // the new betas evenly along it. One rejection estimate per adjacent pair.
        TemperatureLadder adapted(std::span<const double> mean_rejections) const {
            if (size() < 2 || mean_rejections.size() != size() - 1)
                throw std::invalid_argument("Expected one mean rejection per temperature gap");
            bool has_rejection = false;
            for (double rate : mean_rejections) {
                if (!(0. <= rate && rate <= 1.))
                    throw std::invalid_argument("Mean rejections must be in [0, 1]");
                has_rejection |= rate > 0.;
            }
            if (!has_rejection || size() == 2)
                return *this;

            std::vector<double> cumulative(size());
            std::vector<double> widths(size() - 1), slopes(size() - 1), derivatives(size());
            for (std::size_t i = 0; i < widths.size(); ++i) {
                cumulative[i + 1] = cumulative[i] + std::max(mean_rejections[i], 1e-10);
                widths[i] = cumulative[i + 1] - cumulative[i];
                slopes[i] = (betas_[i + 1] - betas_[i]) / widths[i];
            }
            derivatives.front() = endpoint_slope(widths[0], widths[1], slopes[0], slopes[1]);
            const auto gap_count = widths.size();
            const auto last = gap_count - 1;
            derivatives.back() =
                endpoint_slope(widths[last], widths[last - 1], slopes[last], slopes[last - 1]);
            for (std::size_t i = 1; i + 1 < size(); ++i) {
                // Weighted harmonic mean of two positive secant slopes.
                const auto w1 = 2 * widths[i] + widths[i - 1];
                const auto w2 = widths[i] + 2 * widths[i - 1];
                const auto smaller = std::min(slopes[i - 1], slopes[i]);
                derivatives[i] =
                    smaller *
                    ((w1 + w2) / (w1 * (smaller / slopes[i - 1]) + w2 * (smaller / slopes[i])));
            }
            const auto barrier = cumulative.back();
            const boost::math::interpolators::cubic_hermite inverse_barrier(
                std::move(cumulative), std::vector<double>(betas_), std::move(derivatives));
            std::vector<double> next(size());
            next.back() = 1.;
            for (std::size_t i = 1; i + 1 < next.size(); ++i) {
                next[i] = inverse_barrier(
                    barrier * (static_cast<double>(i) / static_cast<double>(gap_count)));
                // Near-coincident knots can lose strict ordering in double precision.
                if (!(next[i - 1] < next[i] && next[i] < 1.))
                    return *this;
            }
            return TemperatureLadder(std::move(next));
        }

      private:
        static double endpoint_slope(double h0, double h1, double m0, double m1) noexcept {
            return std::max(0., ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1));
        }

        std::vector<double> betas_;
    };
} // namespace hopsy
#endif
