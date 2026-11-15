#ifndef HOPSY_PROBLEM_H
#define HOPSY_PROBLEM_H

#include <concepts>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <hopsy/problem/coordinates.h>
#include <hopsy/target/traits.h>

namespace hopsy {
    namespace detail {
        template <typename T> T &unwrap(T &value) noexcept { return value; }
        template <typename T> const T &unwrap(const T &value) noexcept { return value; }
        template <typename T> T &unwrap(const std::reference_wrapper<T> &value) noexcept {
            return value.get();
        }
        template <typename T> T &unwrap(std::reference_wrapper<T> &value) noexcept {
            return value.get();
        }
        template <typename T>
        using unwrapped_type = std::remove_cvref_t<decltype(unwrap(std::declval<const T &>()))>;
        template <typename Coordinates>
        inline constexpr bool constant_coordinate_volume = [] {
            if constexpr (requires { Coordinates::constant_volume; })
                return Coordinates::constant_volume;
            else
                return false;
        }();
    } // namespace detail

    // Own values by default. std::cref explicitly borrows immutable data: it must
    // outlive the problem and remain mathematically unchanged while bound.
    // The supplied geometry is ALREADY in sampling coordinates. No constraints
    // are silently transformed; this works for non-linear/disconnected geometry.
    template <typename Geometry, typename Target, typename Coordinates = IdentityCoordinates>
    class Problem {
      public:
        using geometry_type = detail::unwrapped_type<Geometry>;
        using target_type = detail::unwrapped_type<Target>;
        using coordinates_type = detail::unwrapped_type<Coordinates>;
        static constexpr bool identity_coordinates =
            std::same_as<coordinates_type, IdentityCoordinates>;

        Problem(Geometry geometry, Target target, Coordinates coordinates = {})
            : geometry_(std::move(geometry)), target_(std::move(target)),
              coordinates_(std::move(coordinates)) {
            if constexpr (requires { this->geometry().num_dimension(); }) {
                const auto n = this->geometry().num_dimension();
                if constexpr (requires { this->coordinates().sampling_dimension(); }) {
                    if (this->coordinates().sampling_dimension() != n)
                        throw std::invalid_argument(
                            "Coordinate map and sampling geometry dimensions differ");
                }
                if constexpr (requires { this->target().dimension(); }) {
                    if (this->target().dimension() != this->coordinates().target_dimension(n))
                        throw std::invalid_argument("Coordinate map and target dimensions differ");
                }
            }
        }
        const geometry_type &geometry() const noexcept { return detail::unwrap(geometry_); }
        const target_type &target() const noexcept { return detail::unwrap(target_); }
        const coordinates_type &coordinates() const noexcept {
            return detail::unwrap(coordinates_);
        }

      private:
        Geometry geometry_;
        Target target_;
        [[no_unique_address]] Coordinates coordinates_;
    };

    namespace detail {
        template <typename Problem, typename State, bool Identity = Problem::identity_coordinates>
        class TargetEvaluation;

        // Identity evaluation is empty and returns the original target type and
        // reference. Uniform traits, allocations, and derivative capabilities
        // are exactly those of that target; no adapter reaches the hot loop.
        template <typename Problem, typename State> class TargetEvaluation<Problem, State, true> {
          public:
            TargetEvaluation(const Problem &, const State &) noexcept {}
            const typename Problem::target_type &target(const Problem &problem) const noexcept {
                return problem.target();
            }
        };

        template <typename Problem, typename State> class TargetEvaluation<Problem, State, false> {
          public:
            TargetEvaluation(const Problem &problem, const State &initial)
                : problem_(problem),
                  mapped_(problem.coordinates().target_dimension(initial.size())),
                  gradient_(initial), volume_gradient_(initial) {}
            TargetEvaluation &target(const Problem &) noexcept { return *this; }
            double log_density(const State &state) const {
                problem_.coordinates().to_target(state, mapped_);
                return static_cast<double>(problem_.target().log_density(mapped_)) +
                       problem_.coordinates().log_volume(state);
            }
            const State &log_gradient(const State &state) const
                requires requires(const typename Problem::target_type &t,
                                  const Vector<typename State::Scalar> &x) { t.log_gradient(x); }
            {
                problem_.coordinates().to_target(state, mapped_);
                const auto &gradient = problem_.target().log_gradient(mapped_);
                problem_.coordinates().pullback_gradient(state, gradient, gradient_);
                if constexpr (!constant_coordinate_volume<typename Problem::coordinates_type>) {
                    problem_.coordinates().log_volume_gradient(state, volume_gradient_);
                    gradient_ += volume_gradient_;
                }
                return gradient_;
            }

          private:
            const Problem &problem_;
            // Per-chain evaluation buffers; shared problem/target stay immutable.
            mutable Vector<typename State::Scalar> mapped_;
            mutable State gradient_, volume_gradient_;
        };
    } // namespace detail

    template <typename P, typename State>
    struct target_is_uniform<detail::TargetEvaluation<P, State, false>>
        : std::bool_constant<target_is_uniform_v<typename P::target_type> &&
                             detail::constant_coordinate_volume<typename P::coordinates_type>> {};
} // namespace hopsy
#endif
