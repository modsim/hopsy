#ifndef HOPSY_BASIS_REFERENCE_H
#define HOPSY_BASIS_REFERENCE_H

#include <stdexcept>
#include <utility>
#include <Eigen/QR>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/step.h>
#include <hopsy/target/uniform.h>

namespace hopsy {
    // Random-coordinate Hit-and-Run in an affine chart x = T z + s. Each
    // uniform chord maps to a uniform chord along a column of T, so we can
    // execute it directly in x and retain the original constraints exactly.
    template <typename Scalar> class BasisReference {
        using State = Vector<Scalar>;
        using Geometry = LinearPolytope<Scalar>;

        class Directions {
          public:
            explicit Directions(const Matrix<Scalar> &basis)
                : basis_(&basis), coordinate_(basis.cols()), direction_(basis.rows()) {}

            const State &generate(RandomNumberGenerator &rng) {
                direction_ = basis_->col(coordinate_.generate(rng).coordinate);
                return direction_;
            }
            const State &direction() const noexcept { return direction_; }

          private:
            const Matrix<Scalar> *basis_;
            RandomCoordinateDirectionGenerator coordinate_;
            State direction_;
        };

      public:
        BasisReference(Matrix<Scalar> basis, double max_beta)
            : basis_(std::move(basis)), max_beta_(max_beta) {
            if (basis_.rows() == 0 || basis_.rows() != basis_.cols() || !basis_.allFinite() ||
                Eigen::ColPivHouseholderQR<Matrix<Scalar>>(basis_).rank() != basis_.cols())
                throw std::invalid_argument("Reference basis must be finite, square and full rank");
            if (!(0. <= max_beta_ && max_beta_ <= 1.))
                throw std::invalid_argument("Reference maximum beta must be in [0, 1]");
        }

        bool uses(double beta) const noexcept { return beta <= max_beta_; }

        // The configuration owns a shared, immutable basis. Each replica
        // owns its separate mutable kernel and workspace.
        struct Local {
            HitAndRunKernel<State, Directions, UniformLineKernel> kernel;
            typename Geometry::workspace_type workspace;
            bool active = false;

            Local(const BasisReference &reference, const State &initial, const Geometry &geometry)
                : kernel(initial, Directions(reference.basis_for(initial.size())),
                         UniformLineKernel{}),
                  workspace(geometry.create_workspace(initial)) {}
        };

      private:
        const Matrix<Scalar> &basis_for(Index dimension) const {
            if (basis_.rows() != dimension)
                throw std::invalid_argument("Reference basis dimension mismatch");
            return basis_;
        }

        Matrix<Scalar> basis_;
        double max_beta_;
    };
} // namespace hopsy
#endif
