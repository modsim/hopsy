#ifndef LINEAR_POLYTOPE_H
#define LINEAR_POLYTOPE_H

#include <array>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <utility>

#include <hopsy/geometry/line_interval.h>
#include <hopsy/geometry/workspace.h>
#include <hopsy/kernel/direction.h>
#include <hopsy/linalg.h>

namespace hopsy {
template <typename Scalar = double> class LinearPolytope {
public:
  using scalar_type = Scalar;
  using vector_type = Vector<Scalar>;
  using matrix_type = Matrix<Scalar>;
  using line_domain_type = std::array<LineInterval<Scalar>, 1>;

  struct Workspace {
    vector_type slacks;
    vector_type direction_product;
  };

  using workspace_type = Workspace;

  LinearPolytope(matrix_type A, vector_type b)
      : A_(std::move(A)), b_(std::move(b)) {
    if (A_.rows() != b_.size()) {
      throw std::invalid_argument(
          "LinearPolytope: number of rows in A must match the size of b.");
    }

    if (A_.cols() == 0) {
      throw std::invalid_argument(
          "LinearPolytope: A must have at least one column.");
    }
  }

  [[nodiscard]] Index num_dimension() const noexcept { return A_.cols(); };

  [[nodiscard]] Index num_constraints() const noexcept { return A_.rows(); };

  [[nodiscard]] bool contains(const vector_type &point,
                              Scalar tolerance = Scalar{0}) const {
    return ((b_ - A_ * point).array() >= tolerance).all();
  }

  //
  // General direction, no cache
  //
  [[nodiscard]] line_domain_type line_domain(const vector_type &origin,
                                             const vector_type &direction,
                                             EmptyWorkspace &) const {
    const vector_type slacks = b_ - A_ * origin;

    return compute_line_domain(slacks, A_ * direction);
  }

  //
  // General direction, cached slacks
  //
  [[nodiscard]] line_domain_type line_domain(const vector_type &origin,
                                             const vector_type &direction,
                                             workspace_type &workspace) const {
    workspace.direction_product.noalias() = A_ * direction;
    return compute_line_domain(workspace.slacks, workspace.direction_product);
  }

  //
  // Coordinate direction, no cache
  //
  [[nodiscard]] line_domain_type
  line_domain(const vector_type &origin, const CoordinateDirection &direction,
              EmptyWorkspace &) const {
    const vector_type slacks = b_ - A_ * origin;

    return compute_line_domain(slacks, A_.col(direction.coordinate));
  }

  //
  // Coordinate direction, cached slacks
  //
  [[nodiscard]] line_domain_type
  line_domain(const vector_type &origin, const CoordinateDirection &direction,
              workspace_type &workspace) const {
    return compute_line_domain(workspace.slacks, A_.col(direction.coordinate));
  }

  [[nodiscard]] const matrix_type &A() const noexcept { return A_; }

  [[nodiscard]] const vector_type &b() const noexcept { return b_; }

  [[nodiscard]] workspace_type
  create_workspace(const vector_type &origin) const {
    return workspace_type{b_ - A_ * origin, vector_type(num_constraints())};
  }

  void reset_workspace(const vector_type &origin,
                       workspace_type &workspace) const {
    workspace.slacks = b_ - A_ * origin;
  }

  void reset_workspace(const vector_type &, EmptyWorkspace &) const {}

  void advance_workspace(Scalar step, const vector_type &,
                         workspace_type &workspace) const {
    workspace.slacks -= step * workspace.direction_product;
  }

  void advance_workspace(Scalar step, const CoordinateDirection &direction,
                         workspace_type &workspace) const {
    workspace.slacks -= step * A_.col(direction.coordinate);
  }

  void advance_workspace(Scalar, const vector_type &, EmptyWorkspace &) const {}

  void advance_workspace(Scalar, const CoordinateDirection &,
                         EmptyWorkspace &) const {}

private:
  template <typename Slacks, typename DirectionProduct>
  [[nodiscard]] line_domain_type compute_line_domain(
      const Eigen::MatrixBase<Slacks> &slacks,
      const Eigen::MatrixBase<DirectionProduct> &direction_product) const {
    Scalar lower = -std::numeric_limits<Scalar>::infinity();
    Scalar upper = std::numeric_limits<Scalar>::infinity();

    for (Index i = 0; i < A_.rows(); ++i) {
      const Scalar denominator = direction_product(i);

      if (denominator > Scalar{0}) {
        upper = std::min(upper, slacks(i) / denominator);
      } else if (denominator < Scalar{0}) {
        lower = std::max(lower, slacks(i) / denominator);
      } else if (slacks(i) < Scalar{0}) {
        // Parallel to a violated constraint:
        // the line has no feasible intersection.
        return {LineInterval<Scalar>{Scalar{1}, Scalar{0}}};
      }
    }

    return {Lint seInterval<Scalar>{lower, upper}};
  }

  matrix_type A_;
  vector_type b_;
};
} // namespace hopsy

#endif // LINEAR_POLYTOPE_H
