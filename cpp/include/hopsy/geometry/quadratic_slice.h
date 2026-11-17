#ifndef HOPSY_QUADRATIC_SLICE_H
#define HOPSY_QUADRATIC_SLICE_H
#include <algorithm>
#include <array>
#include <cmath>
#include <hopsy/geometry/line_interval.h>
#include <limits>
#include <optional>
#include <stdexcept>

namespace hopsy::detail {
    struct QuadraticSlice {
        std::array<LineInterval<long double>, 2> intervals{};
        unsigned size = 0;
    };
    // Error-free two-product (Dekker splitting). Inputs are range checked/scaled
    // below; long-double fma is a slow software routine on the measured x86 host.
    inline std::pair<long double, long double> slice_product(long double a, long double b) {
        constexpr int half = (std::numeric_limits<long double>::digits + 1) / 2;
        const long double splitter = std::ldexp(1.L, half) + 1;
        long double ca = splitter * a, cb = splitter * b;
        long double ah = ca - (ca - a), bh = cb - (cb - b), al = a - ah, bl = b - bh;
        long double product = a * b;
        long double e1 = product - ah * bh, e2 = e1 - al * bh, e3 = e2 - ah * bl;
        return {product, al * bl - e3};
    }
    // Closed solution of a*t^2+b*t+c <= 0. Optional discriminant is supplied
    // only after exact coefficient reconstruction in an ill-conditioned geometry.
    inline QuadraticSlice quadratic_slice(long double a, long double b, long double c,
                                          std::optional<long double> exact_discriminant = {}) {
        const long double inf = std::numeric_limits<long double>::infinity();
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
            throw std::overflow_error("quadratic slice: nonfinite coefficients");
        if (a == 0) {
            if (b == 0)
                return c <= 0 ? QuadraticSlice{{{{-inf, inf}, {0, 0}}}, 1} : QuadraticSlice{};
            long double root = -c / b;
            if (!std::isfinite(root))
                throw std::overflow_error("quadratic slice: linear root overflow");
            return b > 0 ? QuadraticSlice{{{{-inf, root}, {0, 0}}}, 1}
                         : QuadraticSlice{{{{root, inf}, {0, 0}}}, 1};
        }
        long double aa = a, bb = b, cc = c;
        const long double scale = std::max({std::abs(a), std::abs(b), std::abs(c)});
        const long double safe_max =
            std::ldexp(1.L, std::numeric_limits<long double>::max_exponent / 2 -
                                std::numeric_limits<long double>::digits - 8);
        const long double safe_min =
            std::ldexp(1.L, std::numeric_limits<long double>::min_exponent / 2 +
                                std::numeric_limits<long double>::digits + 8);
        const long double smallest =
            std::min({std::abs(a), b == 0 ? scale : std::abs(b), c == 0 ? scale : std::abs(c)});
        if (scale > safe_max || smallest < safe_min) {
            int exponent = 0;
            std::frexp(scale, &exponent);
            aa = std::scalbn(a, -exponent);
            bb = std::scalbn(b, -exponent);
            cc = std::scalbn(c, -exponent);
            if (aa == 0 || (b != 0 && bb == 0) || (c != 0 && cc == 0))
                throw std::overflow_error(
                    "quadratic slice: coefficient range cannot be represented");
            if (exact_discriminant) {
                const long double scaled = std::scalbn(*exact_discriminant, -2 * exponent);
                if (*exact_discriminant != 0 && scaled == 0)
                    throw std::overflow_error("quadratic slice: discriminant underflow");
                exact_discriminant = scaled;
            }
        }
        const auto b2 = slice_product(bb, bb), ac = slice_product(4 * aa, cc);
        if (!exact_discriminant &&
            ((bb != 0 && b2.first == 0) || (aa != 0 && cc != 0 && ac.first == 0)))
            throw std::underflow_error("quadratic slice: discriminant products underflow");
        const long double discriminant =
            exact_discriminant.value_or((b2.first - ac.first) + (b2.second - ac.second));
        if (!std::isfinite(discriminant))
            throw std::overflow_error("quadratic slice: discriminant overflow");
        if (discriminant < 0)
            return aa < 0 ? QuadraticSlice{{{{-inf, inf}, {0, 0}}}, 1} : QuadraticSlice{};
        if (discriminant == 0) {
            if (aa < 0)
                return QuadraticSlice{{{{-inf, inf}, {0, 0}}}, 1};
            const long double root = -bb / (2 * aa);
            return QuadraticSlice{{{{root, root}, {0, 0}}}, 1};
        }
        const long double q = -0.5L * (bb + std::copysign(std::sqrt(discriminant), bb));
        long double first = q / aa, second = cc / q;
        if (!std::isfinite(first) || !std::isfinite(second))
            throw std::overflow_error("quadratic slice: root overflow");
        if (first == second)
            throw std::range_error(
                "quadratic slice: distinct roots cannot be resolved at working precision");
        if (first > second)
            std::swap(first, second);
        if (aa > 0)
            return QuadraticSlice{{{{first, second}, {0, 0}}}, 1};
        return QuadraticSlice{{{{-inf, first}, {second, inf}}}, 2};
    }
} // namespace hopsy::detail
#endif
