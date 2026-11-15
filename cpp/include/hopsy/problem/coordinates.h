#ifndef HOPSY_COORDINATES_H
#define HOPSY_COORDINATES_H

#include <cmath>
#include <stdexcept>
#include <utility>
#include <Eigen/QR>
#include <hopsy/linalg.h>

namespace hopsy {
    struct IdentityCoordinates {
        static constexpr bool constant_volume = true;
        Index target_dimension(Index sampling_dimension) const noexcept {
            return sampling_dimension;
        }
        template <typename State, typename Output>
        void to_target(const State &state, Output &output) const {
            output = state;
        }
    };

    // x = T z + shift. Geometry and kernel state use z; target evaluations use x.
    // Full-column-rank rectangular T represents an affine subspace, with volume
    // measured in its induced Euclidean measure, not an ambient marginalization.
    template <typename Scalar = double> class AffineCoordinates {
      public:
        static constexpr bool constant_volume = true;
        AffineCoordinates(Matrix<Scalar> matrix, Vector<Scalar> shift)
            : matrix_(std::move(matrix)), shift_(std::move(shift)) {
            if (matrix_.cols() == 0 || matrix_.rows() < matrix_.cols() ||
                shift_.size() != matrix_.rows() || !matrix_.allFinite() || !shift_.allFinite())
                throw std::invalid_argument("AffineCoordinates requires finite compatible "
                                            "dimensions and rows >= columns > 0");
            Eigen::ColPivHouseholderQR<Matrix<Scalar>> qr(matrix_);
            if (qr.rank() != matrix_.cols())
                throw std::invalid_argument("AffineCoordinates requires full column rank");
            // sum(log(abs(diag(R)))) = log sqrt(det(T^T T)); QR avoids forming
            // the Gram matrix and squaring its condition number.
            log_volume_ = qr.matrixR()
                              .topLeftCorner(matrix_.cols(), matrix_.cols())
                              .diagonal()
                              .array()
                              .abs()
                              .log()
                              .sum();
            if (!std::isfinite(log_volume_))
                throw std::invalid_argument("AffineCoordinates volume must be finite");
        }
        Index sampling_dimension() const noexcept { return matrix_.cols(); }
        Index target_dimension(Index = 0) const noexcept { return matrix_.rows(); }
        const Matrix<Scalar> &matrix() const noexcept { return matrix_; }
        const Vector<Scalar> &shift() const noexcept { return shift_; }
        template <typename State, typename Output>
        void to_target(const State &state, Output &output) const {
            output.noalias() = matrix_ * state;
            output += shift_;
        }
        template <typename State> double log_volume(const State &) const noexcept {
            return log_volume_;
        }
        template <typename State, typename Gradient, typename Output>
        void pullback_gradient(const State &, const Gradient &gradient, Output &output) const {
            output.noalias() = matrix_.transpose() * gradient;
        }

      private:
        Matrix<Scalar> matrix_;
        Vector<Scalar> shift_;
        double log_volume_{};
    };
    template <typename MatrixDerived, typename ShiftDerived>
    AffineCoordinates(const Eigen::MatrixBase<MatrixDerived> &,
                      const Eigen::MatrixBase<ShiftDerived> &)
        -> AffineCoordinates<typename MatrixDerived::Scalar>;
} // namespace hopsy
#endif
