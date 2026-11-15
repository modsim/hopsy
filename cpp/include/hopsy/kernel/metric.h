#ifndef METRIC_H
#define METRIC_H

#include <cmath>
#include <random>
#include <Eigen/Cholesky>
#include <hopsy/linalg.h>

namespace hopsy {
    struct IdentityMetric {
        template <typename State, typename Geometry, typename Workspace>
        bool prepare(const State &, const Geometry &, const Workspace &) const noexcept {
            return true;
        }

        template <typename RNG, typename State> void sample(RNG &rng, State &noise) const {
            std::normal_distribution<typename State::Scalar> normal;
            for (decltype(noise.size()) i = 0; i < noise.size(); ++i) {
                noise(i) = normal(rng);
            }
        }
        template <typename State> void solve_in_place(State &) const noexcept {}

        template <typename State> double squared_norm(const State &delta) const {
            return delta.squaredNorm();
        }

        double log_sqrt_determinant() const noexcept { return 0.; }
    };

    // Precision metric G = A^T diag(slacks^-2) A. Scale is kept outside
    // the factorization: Gaussian covariance is sigma^2 G^-1.
    template <typename Scalar = double> class DikinMetric {
      public:
        template <typename State, typename Geometry, typename Workspace>
        bool prepare(const State &state, const Geometry &geometry, const Workspace &)
            requires requires {
                geometry.A();
                geometry.b();
            }
        {
            // Also works at a candidate, whose slacks are not the accepted cache.
            slacks_.noalias() = geometry.A() * state;
            slacks_ = geometry.b() - slacks_;
            if (!slacks_.allFinite() || !(slacks_.array() > Scalar{0}).all())
                return false;
            norm_buffer_.resize(state.size());
            // Reuse the slack buffer for reciprocals; scale contiguous columns
            // with multiplication rather than dividing every matrix element.
            slacks_ = slacks_.cwiseInverse();
            weighted_A_.noalias() = slacks_.asDiagonal() * geometry.A();
            precision_.noalias() = weighted_A_.transpose() * weighted_A_;
            if (!precision_.allFinite())
                return false;
            solver_.compute(precision_);
            if (solver_.info() != Eigen::Success)
                return false;
            log_sqrt_determinant_ = solver_.matrixLLT().diagonal().array().log().sum();
            return std::isfinite(log_sqrt_determinant_);
        }

        template <typename RNG, typename State> void sample(RNG &rng, State &noise) const {
            IdentityMetric{}.sample(rng, noise);
            // If G = L L^T, L^-T z has covariance G^-1 (not L^-1 z).
            solver_.matrixU().solveInPlace(noise);
        }

        template <typename State> void solve_in_place(State &vector) const {
            solver_.matrixL().solveInPlace(vector);
            solver_.matrixU().solveInPlace(vector);
        }

        template <typename State> double squared_norm(const State &delta) {
            norm_buffer_.noalias() = solver_.matrixU() * delta;
            return norm_buffer_.squaredNorm();
        }

        double log_sqrt_determinant() const noexcept { return log_sqrt_determinant_; }
        const Matrix<Scalar> &precision() const noexcept { return precision_; }

      private:
        Vector<Scalar> slacks_, norm_buffer_;
        Matrix<Scalar> weighted_A_, precision_;
        Eigen::LLT<Matrix<Scalar>> solver_;
        double log_sqrt_determinant_{};
    };
} // namespace hopsy

#endif // METRIC_H
