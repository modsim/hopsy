#ifndef UNIFORM_H
#define UNIFORM_H

#include <Eigen/Dense>

namespace hopsy {
    class UniformTarget {
    public:
        static constexpr bool thread_safe = true;

        template <typename State>
        double log_density(const State&) const noexcept {
            return 0.;
        }

        template <typename Derived>
            requires (!std::integral<typename Derived::Scalar>)
        auto log_gradient(
            const Eigen::MatrixBase<Derived>& state
        ) const {
            using Scalar = typename Derived::Scalar;

            return Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(
                state.size()
            );
        }

        template <typename Derived>
            requires (!std::integral<typename Derived::Scalar>)
        auto log_hessian(
            const Eigen::MatrixBase<Derived>& state
        ) const {
            using Scalar = typename Derived::Scalar;

            return Eigen::Matrix<
                Scalar,
                Eigen::Dynamic,
                Eigen::Dynamic
            >::Zero(state.size(), state.size());
        }
    };
}

#endif //UNIFORM_H
