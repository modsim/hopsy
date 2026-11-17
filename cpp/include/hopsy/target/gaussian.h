#ifndef HOPSY_GAUSSIAN_TARGET_H
#define HOPSY_GAUSSIAN_TARGET_H

#include <cmath>
#include <stdexcept>
#include <utility>
#include <Eigen/Cholesky>
#include <hopsy/linalg.h>

namespace hopsy {
    // Immutable native density, safe to share between chains without the GIL.
    class GaussianTarget {
      public:
        GaussianTarget(Vector<double> mean, const Matrix<double> &covariance)
            : mean_(std::move(mean)) {
            if (!mean_.size() || covariance.rows() != mean_.size() ||
                covariance.cols() != mean_.size() || !mean_.allFinite() ||
                !covariance.allFinite() || !covariance.isApprox(covariance.transpose()))
                throw std::invalid_argument("Gaussian mean/covariance must be finite with "
                                            "compatible dimensions and symmetric covariance");
            Eigen::LLT<Matrix<double>> solver(covariance);
            if (solver.info() != Eigen::Success)
                throw std::invalid_argument("Gaussian covariance must be positive definite");
            precision_ = solver.solve(Matrix<double>::Identity(mean_.size(), mean_.size()));
            if (!precision_.allFinite())
                throw std::invalid_argument("Gaussian precision is not finite");
            log_normalizer_ = -0.5 * mean_.size() * std::log(2. * std::acos(-1.)) -
                              solver.matrixLLT().diagonal().array().log().sum();
        }
        double log_density(const Vector<double> &x) const {
            validate(x);
            const Vector<double> delta = x - mean_;
            return log_normalizer_ - 0.5 * delta.dot(precision_ * delta);
        }
        Vector<double> log_gradient(const Vector<double> &x) const {
            validate(x);
            return -precision_ * (x - mean_);
        }
        Index dimension() const noexcept { return mean_.size(); }

      private:
        void validate(const Vector<double> &x) const {
            if (x.size() != mean_.size())
                throw std::invalid_argument("Gaussian state dimension mismatch");
        }
        Vector<double> mean_;
        Matrix<double> precision_;
        double log_normalizer_{};
    };
} // namespace hopsy
#endif
