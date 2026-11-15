#ifndef BILINEAR_POLYTOPE_H
#define BILINEAR_POLYTOPE_H

#include <vector>
#include <cmath>
#include <type_traits>
#include <concepts>
#include <new>
#include <boost/multiprecision/cpp_int.hpp>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/geometry/quadratic_slice.h>

namespace hopsy {
    namespace detail {
        // Reused tiny interval buffers are written by independent chains. Give
        // them separate cache lines just like the existing kernel/RNG storage.
        template <typename T> struct SliceAllocator {
            using value_type = T;
            SliceAllocator() = default;
            template <typename U> SliceAllocator(const SliceAllocator<U> &) noexcept {}
            T *allocate(std::size_t n) {
                static_assert(alignof(T) <= 64);
                if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
                    throw std::bad_array_new_length();
                return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t(64)));
            }
            void deallocate(T *p, std::size_t) noexcept {
                ::operator delete(p, std::align_val_t(64));
            }
            template <typename U> bool operator==(const SliceAllocator<U> &) const noexcept {
                return true;
            }
        };
    } // namespace detail
    // z=[x,y], split=dim(x). A*z<=b and x^T B[k] y+affine.row(k)*z<=rhs[k].
    // Cross terms never contain x_i*x_j or y_i*y_j. All boundaries are closed.
    template <typename Scalar = double> class BilinearPolytope {
      public:
        static constexpr bool hit_and_run_only = true;
        using scalar_type = Scalar;
        using vector_type = Vector<Scalar>;
        using matrix_type = Matrix<Scalar>;
        using line_domain_type =
            std::vector<LineInterval<Scalar>, detail::SliceAllocator<LineInterval<Scalar>>>;
        using Wide = long double;
        struct Workspace {
            typename LinearPolytope<Scalar>::workspace_type linear;
            line_domain_type domain, scratch;
            unsigned accepted_since_refresh = 0;
        };
        using workspace_type = Workspace;
        BilinearPolytope(matrix_type A, vector_type b, Index split, std::vector<matrix_type> cross,
                         vector_type rhs, matrix_type affine = {})
            : linear_(std::move(A), std::move(b)), split_(split), cross_(std::move(cross)),
              affine_(std::move(affine)), rhs_(std::move(rhs)) {
            static_assert(std::is_floating_point_v<Scalar>);
            static_assert(std::numeric_limits<Scalar>::digits <=
                              std::numeric_limits<double>::digits,
                          "BilinearPolytope currently validates float/double input precision");
            const Index n = num_dimension(), k = static_cast<Index>(cross_.size());
            if (split_ <= 0 || split_ >= n)
                throw std::invalid_argument(
                    "BilinearPolytope: split must partition z into two nonempty blocks");
            if (!linear_.A().allFinite() || !linear_.b().allFinite() || rhs_.size() != k ||
                !rhs_.allFinite())
                throw std::invalid_argument(
                    "BilinearPolytope: finite compatible linear/bilinear bounds required");
            if (affine_.rows() == 0 && affine_.cols() == 0)
                affine_ = matrix_type::Zero(k, n);
            if (affine_.rows() != k || affine_.cols() != n || !affine_.allFinite())
                throw std::invalid_argument("BilinearPolytope: affine must have one row per cross "
                                            "term and one column per parameter");
            terms_.resize(k);
            affine_terms_.resize(k);
            for (Index j = 0; j < k; ++j) {
                for (Index i = 0; i < n; ++i)
                    if (affine_(j, i) != 0)
                        affine_terms_[j].push_back({i, affine_(j, i)});
                if (cross_[j].rows() != split_ || cross_[j].cols() != n - split_ ||
                    !cross_[j].allFinite())
                    throw std::invalid_argument("BilinearPolytope: every B must have shape (split, "
                                                "dimension-split) and be finite");
                for (Index col = 0; col < n - split_; ++col)
                    for (Index row = 0; row < split_; ++row)
                        if (cross_[j](row, col) != 0)
                            terms_[j].push_back({row, col, cross_[j](row, col)});
            }
        }
        Index num_dimension() const noexcept { return linear_.num_dimension(); }
        Index num_constraints() const noexcept { return linear_.num_constraints() + rhs_.size(); }
        Index split() const noexcept { return split_; }
        const matrix_type &A() const noexcept { return linear_.A(); }
        const vector_type &b() const noexcept { return linear_.b(); }
        const std::vector<matrix_type> &cross_terms() const noexcept { return cross_; }
        const matrix_type &affine() const noexcept { return affine_; }
        const vector_type &bilinear_bounds() const noexcept { return rhs_; }
        bool is_feasible(const vector_type &z, Scalar tolerance = 0) const {
            validate_point(z);
            if (!std::isfinite(tolerance))
                throw std::invalid_argument("feasibility tolerance must be finite");
            for (Index row = 0; row < linear_.num_constraints(); ++row) {
                Wide residual = Wide(linear_.b()(row)) - Wide(tolerance),
                     magnitude = std::abs(Wide(linear_.b()(row))) + std::abs(Wide(tolerance));
                for (Index col = 0; col < num_dimension(); ++col) {
                    Wide term = multiply(Wide(linear_.A()(row, col)), Wide(z(col)));
                    residual -= term;
                    magnitude += std::abs(term);
                }
                if (!std::isfinite(residual))
                    throw std::overflow_error("BilinearPolytope: linear evaluation overflow");
                if (std::abs(residual) <=
                    16 * std::numeric_limits<Wide>::epsilon() * (num_dimension() + 1) * magnitude) {
                    Rational precise = exact(linear_.b()(row)) - exact(tolerance);
                    for (Index col = 0; col < num_dimension(); ++col)
                        precise -= exact(linear_.A()(row, col)) * exact(z(col));
                    if (precise < 0)
                        return false;
                } else if (residual < 0)
                    return false;
            }
            for (Index k = 0; k < rhs_.size(); ++k)
                if (!constraint_feasible(k, z, tolerance))
                    return false;
            return true;
        }
        workspace_type create_workspace(const vector_type &z) const {
            validate_point(z);
            workspace_type result{linear_.create_workspace(z), {}, {}, 0};
            if (!result.linear.slacks.allFinite())
                throw std::overflow_error("BilinearPolytope: linear workspace overflow");
            result.domain.reserve(rhs_.size() + 1);
            result.scratch.reserve(rhs_.size() + 1);
            return result;
        }
        void reset_workspace(const vector_type &z, workspace_type &w) const {
            validate_point(z);
            linear_.reset_workspace(z, w.linear);
            w.accepted_since_refresh = 0;
            if (!w.linear.slacks.allFinite())
                throw std::overflow_error("BilinearPolytope: linear workspace overflow");
        }
        const line_domain_type &line_domain(const vector_type &z, const vector_type &d,
                                            workspace_type &w) const {
            // As for LinearPolytope, origin and workspace must remain paired.
            // Creation/reset validates the state; the owning chain maintains it.
            if (z.size() != num_dimension())
                throw std::invalid_argument("origin dimension mismatch");
            validate_point(d);
            return domain(z, d, w);
        }
        const line_domain_type &line_domain(const vector_type &z, const CoordinateDirection &d,
                                            workspace_type &w) const {
            if (z.size() != num_dimension())
                throw std::invalid_argument("origin dimension mismatch");
            if (d.coordinate < 0 || d.coordinate >= num_dimension())
                throw std::invalid_argument("coordinate direction out of range");
            return domain(z, d, w);
        }
        template <typename Direction>
        void advance_workspace(Scalar step, const Direction &d, workspace_type &w) const {
            linear_.advance_workspace(step, d, w.linear);
            ++w.accepted_since_refresh;
            if (!w.linear.slacks.allFinite()) {
                w.accepted_since_refresh = 256;
                throw std::overflow_error("BilinearPolytope: linear workspace update overflow");
            }
        }

      private:
        struct Term {
            Index row, col;
            Scalar coefficient;
        };
        static Wide multiply(Wide a, Wide b) {
            Wide result = a * b;
            if constexpr (std::numeric_limits<Wide>::max_exponent <=
                          3 * std::numeric_limits<Scalar>::max_exponent) {
                if (a != 0 && b != 0 && std::abs(result) < std::numeric_limits<Wide>::min())
                    throw std::underflow_error(
                        "BilinearPolytope: product underflow at working precision");
            }
            return result;
        }
        void validate_point(const vector_type &z) const {
            if (z.size() != num_dimension() || !z.allFinite())
                throw std::invalid_argument(
                    "BilinearPolytope: finite vector matching geometry dimension required");
        }
        bool constraint_feasible(Index k, const vector_type &z, Scalar tolerance) const {
            Wide result = Wide(tolerance) - Wide(rhs_(k)),
                 magnitude = std::abs(Wide(tolerance)) + std::abs(Wide(rhs_(k)));
            for (const auto &[i, a] : affine_terms_[k]) {
                Wide term = multiply(Wide(a), Wide(z(i)));
                result += term;
                magnitude += std::abs(term);
            }
            for (const auto &entry : terms_[k]) {
                Wide term = multiply(multiply(Wide(z(entry.row)), Wide(entry.coefficient)),
                                     Wide(z(split_ + entry.col)));
                result += term;
                magnitude += std::abs(term);
            }
            if (!std::isfinite(result))
                throw std::overflow_error("BilinearPolytope: constraint evaluation overflow");
            if (std::abs(result) > 32 * std::numeric_limits<Wide>::epsilon() *
                                       (1 + terms_[k].size() + affine_terms_[k].size()) * magnitude)
                return result < 0;
            Rational precise = exact(tolerance) - exact(rhs_(k));
            for (const auto &[i, a] : affine_terms_[k])
                precise += exact(a) * exact(z(i));
            for (const auto &entry : terms_[k])
                precise +=
                    exact(z(entry.row)) * exact(entry.coefficient) * exact(z(split_ + entry.col));
            return precise <= 0;
        }
        static Wide component(const vector_type &d, Index i) { return d(i); }
        static Wide component(const CoordinateDirection &d, Index i) {
            return d.coordinate == i ? 1 : 0;
        }
        using Rational = boost::multiprecision::cpp_rational;
        static Rational exact(Wide value) {
            if (value == 0)
                return 0;
            int exponent = 0;
            Wide fraction = std::frexp(value, &exponent);
            constexpr int digits = std::numeric_limits<Wide>::digits;
            Rational result = boost::multiprecision::cpp_int(std::ldexp(fraction, digits));
            if (exponent >= digits)
                result *= boost::multiprecision::cpp_int(1) << (exponent - digits);
            else
                result /= boost::multiprecision::cpp_int(1) << (digits - exponent);
            return result;
        }
        template <typename Direction>
        std::array<Rational, 3> exact_coefficients(Index k, const vector_type &z,
                                                   const Direction &d) const {
            std::array<Rational, 3> p{0, 0, -exact(rhs_(k))};
            for (const auto &[i, a] : affine_terms_[k]) {
                p[1] += exact(a) * exact(component(d, i));
                p[2] += exact(a) * exact(z(i));
            }
            for (const auto &term : terms_[k]) {
                Rational dx = exact(component(d, term.row)),
                         dy = exact(component(d, split_ + term.col)), B = exact(term.coefficient);
                p[0] += dx * B * dy;
                p[1] += dx * B * exact(z(split_ + term.col)) + exact(z(term.row)) * B * dy;
                p[2] += exact(z(term.row)) * B * exact(z(split_ + term.col));
            }
            return p;
        }
        template <typename Direction>
        auto coefficients(Index k, const vector_type &z, const Direction &d) const {
            // Coefficients and absolute term sums for the roundoff filter.
            std::array<Wide, 6> p{0, 0, -Wide(rhs_(k)), 0, 0, std::abs(Wide(rhs_(k)))};
            for (const auto &[i, a] : affine_terms_[k]) {
                Wide b = multiply(Wide(a), component(d, i)), c = multiply(Wide(a), Wide(z(i)));
                p[1] += b;
                p[2] += c;
                p[4] += std::abs(b);
                p[5] += std::abs(c);
            }
            for (const auto &term : terms_[k]) {
                const Wide B = term.coefficient;
                const Wide c = multiply(multiply(Wide(z(term.row)), B), Wide(z(split_ + term.col)));
                p[2] += c;
                p[5] += std::abs(c);
                if constexpr (std::same_as<Direction, CoordinateDirection>) {
                    // A single-coordinate line is exactly affine for this
                    // partitioned model; do not compute zero quadratic terms.
                    Wide derivative = 0;
                    if (d.coordinate == term.row)
                        derivative = multiply(B, Wide(z(split_ + term.col)));
                    else if (d.coordinate == split_ + term.col)
                        derivative = multiply(Wide(z(term.row)), B);
                    p[1] += derivative;
                    p[4] += std::abs(derivative);
                } else {
                    const Wide dx = component(d, term.row), dy = component(d, split_ + term.col);
                    const Wide a = multiply(multiply(dx, B), dy),
                               b1 = multiply(multiply(dx, B), Wide(z(split_ + term.col))),
                               b2 = multiply(multiply(Wide(z(term.row)), B), dy);
                    p[0] += a;
                    p[1] += b1 + b2;
                    p[3] += std::abs(a);
                    p[4] += std::abs(b1) + std::abs(b2);
                }
            }
            for (auto v : p)
                if (!std::isfinite(v))
                    throw std::overflow_error("BilinearPolytope: polynomial coefficient overflow");
            return p;
        }
        template <typename Direction>
        detail::QuadraticSlice factored_slice(Index k, const vector_type &z,
                                              const Direction &d) const {
            const auto &term = terms_[k][0];
            Wide x = z(term.row), y = z(split_ + term.col), dx = component(d, term.row),
                 dy = component(d, split_ + term.col);
            const Wide inf = std::numeric_limits<Wide>::infinity();
            auto all = [&] { return detail::QuadraticSlice{{{{-inf, inf}, {0, 0}}}, 1}; };
            auto negative = [](Wide a, Wide b, Wide c) {
                return std::signbit(a) ^ std::signbit(b) ^ std::signbit(c);
            };
            if (dx == 0 && dy == 0)
                return x == 0 || y == 0 || negative(x, y, term.coefficient)
                           ? all()
                           : detail::QuadraticSlice{};
            if ((dx == 0 && x == 0) || (dy == 0 && y == 0))
                return all();
            if (dx == 0 || dy == 0) {
                Wide root = dx == 0 ? -y / dy : -x / dx;
                if (!std::isfinite(root))
                    throw std::overflow_error("BilinearPolytope: factored linear root overflow");
                bool falling =
                    dx == 0 ? negative(x, dy, term.coefficient) : negative(dx, y, term.coefficient);
                return falling ? detail::QuadraticSlice{{{{root, inf}, {0, 0}}}, 1}
                               : detail::QuadraticSlice{{{{-inf, root}, {0, 0}}}, 1};
            }
            Wide first = -x / dx, second = -y / dy;
            if (!std::isfinite(first) || !std::isfinite(second))
                throw std::overflow_error("BilinearPolytope: factored root overflow");
            bool falling = negative(dx, dy, term.coefficient);
            if (first == second) {
                if (exact(x) * exact(dy) != exact(y) * exact(dx))
                    throw std::range_error(
                        "BilinearPolytope: distinct factored roots cannot be resolved");
                return falling ? all() : detail::QuadraticSlice{{{{first, first}, {0, 0}}}, 1};
            }
            if (first > second)
                std::swap(first, second);
            return falling ? detail::QuadraticSlice{{{{-inf, first}, {second, inf}}}, 2}
                           : detail::QuadraticSlice{{{{first, second}, {0, 0}}}, 1};
        }
        template <typename Direction>
        detail::QuadraticSlice solve(Index k, const vector_type &z, const Direction &d, Wide lo,
                                     Wide hi) const {
            if (terms_[k].size() == 1 && affine_terms_[k].empty() && rhs_(k) == 0)
                return factored_slice(k, z, d);
            auto p = coefficients(k, z, d);
            const Wide gamma = 32 * std::numeric_limits<Wide>::epsilon() *
                               (1 + 3 * terms_[k].size() + affine_terms_[k].size());
            if (std::isfinite(lo) && std::isfinite(hi)) {
                auto eval = [&](Wide t) { return (p[0] * t + p[1]) * t + p[2]; };
                Wide maximum = std::max(eval(lo), eval(hi));
                if (p[0] < 0) {
                    Wide vertex = -p[1] / (2 * p[0]);
                    if (vertex > lo && vertex < hi)
                        maximum = std::max(maximum, eval(vertex));
                }
                Wide extent = std::max(std::abs(lo), std::abs(hi));
                Wide error = gamma * ((p[3] * extent + p[4]) * extent + p[5]);
                if (std::isfinite(maximum) && std::isfinite(error) && maximum < -error)
                    return detail::QuadraticSlice{{{{-std::numeric_limits<Wide>::infinity(),
                                                     std::numeric_limits<Wide>::infinity()},
                                                    {0, 0}}},
                                                  1};
            }
            bool uncertain = false;
            for (unsigned j = 0; j < 3; ++j)
                if (p[j + 3] > 0 && std::abs(p[j]) <= gamma * p[j + 3])
                    uncertain = true;
            Wide discriminant = p[1] * p[1] - 4 * p[0] * p[2];
            if (p[0] != 0 && std::abs(discriminant) <=
                                 gamma * (2 * std::abs(p[1]) * p[4] +
                                          4 * (std::abs(p[2]) * p[3] + std::abs(p[0]) * p[5])))
                uncertain = true;
            if (uncertain) {
                const std::array<Rational, 3> exact_p = exact_coefficients(k, z, d);
                const Rational exact_d = exact_p[1] * exact_p[1] - 4 * exact_p[0] * exact_p[2];
                Wide a = exact_p[0].convert_to<Wide>(), b = exact_p[1].convert_to<Wide>(),
                     c = exact_p[2].convert_to<Wide>(), delta = exact_d.convert_to<Wide>();
                if ((exact_d != 0 && delta == 0) || !std::isfinite(delta))
                    throw std::overflow_error(
                        "BilinearPolytope: exact discriminant outside working range");
                return detail::quadratic_slice(a, b, c, delta);
            }
            return detail::quadratic_slice(p[0], p[1], p[2]);
        }
        static Scalar inward(Wide bound, bool lower) {
            if (!std::isfinite(bound))
                return static_cast<Scalar>(bound);
            if (std::abs(bound) > Wide(std::numeric_limits<Scalar>::max()))
                throw std::overflow_error(
                    "BilinearPolytope: finite slice endpoint exceeds scalar range");
            Scalar result = static_cast<Scalar>(bound);
            if ((lower && Wide(result) < bound) || (!lower && Wide(result) > bound))
                result = std::nextafter(result, lower ? std::numeric_limits<Scalar>::infinity()
                                                      : -std::numeric_limits<Scalar>::infinity());
            return result;
        }
        static void append(line_domain_type &out, Wide lo, Wide hi) {
            if (lo > hi)
                return;
            Scalar lower = inward(lo, true), upper = inward(hi, false);
            if (lo < hi && !(lower < upper))
                throw std::range_error(
                    "BilinearPolytope: positive-width slice component cannot be represented");
            if (lower > upper)
                return;
            if (!out.empty() && lower <= out.back().upper)
                out.back().upper = std::max(out.back().upper, upper);
            else
                out.push_back({lower, upper});
        }
        template <typename Direction>
        const line_domain_type &domain(const vector_type &z, const Direction &d,
                                       workspace_type &w) const {
            if (w.accepted_since_refresh >= 256)
                reset_workspace(z, w);
            const auto linear_domain = linear_.line_domain(z, d, w.linear);
            if constexpr (std::same_as<Direction, vector_type>)
                if (!w.linear.direction_product.allFinite())
                    throw std::overflow_error(
                        "BilinearPolytope: linear direction product overflow");
            if (std::isnan(linear_domain[0].lower) || std::isnan(linear_domain[0].upper))
                throw std::overflow_error("BilinearPolytope: invalid linear slice");
            w.domain.clear();
            if (linear_domain[0].lower > linear_domain[0].upper)
                return w.domain;
            w.domain.push_back(linear_domain[0]);
            for (Index k = 0; k < rhs_.size() && !w.domain.empty(); ++k) {
                const auto solution =
                    solve(k, z, d, Wide(w.domain.front().lower), Wide(w.domain.back().upper));
                if (solution.size == 1 && std::isinf(solution.intervals[0].lower) &&
                    solution.intervals[0].lower < 0 && std::isinf(solution.intervals[0].upper) &&
                    solution.intervals[0].upper > 0)
                    continue;
                w.scratch.clear();
                for (const auto &interval : w.domain)
                    for (unsigned j = 0; j < solution.size; ++j)
                        append(w.scratch,
                               std::max(Wide(interval.lower), solution.intervals[j].lower),
                               std::min(Wide(interval.upper), solution.intervals[j].upper));
                w.domain.swap(w.scratch);
            }
            return w.domain;
        }
        LinearPolytope<Scalar> linear_;
        Index split_;
        std::vector<matrix_type> cross_;
        matrix_type affine_;
        vector_type rhs_;
        std::vector<std::vector<Term>> terms_;
        std::vector<std::vector<std::pair<Index, Scalar>>> affine_terms_;
    };
} // namespace hopsy
#endif
