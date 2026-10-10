#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include <hopsy/geometry/linear_polytope.h>

#include <stdexcept>

namespace {

using Scalar = double;
using Polytope = hopsy::LinearPolytope<Scalar>;
using Vector = Polytope::vector_type;
using Matrix = Polytope::matrix_type;
using Catch::Approx;

Polytope make_box() {
  // -1 <= x <= 1
  // -1 <= y <= 1
  Matrix A(4, 2);
  A << 1, 0, -1, 0, 0, 1, 0, -1;

  Vector b = Vector::Ones(4);
  return Polytope(std::move(A), std::move(b));
}

Vector point(Scalar x, Scalar y) {
  Vector result(2);
  result << x, y;
  return result;
}

void check_interval(const Polytope::line_domain_type &domain, Scalar lower,
                    Scalar upper) {
  REQUIRE(domain.size() == 1);
  CHECK(domain[0].lower == Approx(lower));
  CHECK(domain[0].upper == Approx(upper));
}

} // namespace

TEST_CASE("LinearPolytope validates its dimensions", "[geometry][polytope]") {
  SECTION("Valid dimensions") {
    auto polytope = make_box();

    CHECK(polytope.num_dimension() == 2);
    CHECK(polytope.num_constraints() == 4);
  }

  SECTION("Mismatching A and b") {
    Matrix A = Matrix::Zero(3, 2);
    Vector b = Vector::Zero(4);

    CHECK_THROWS_AS(Polytope(A, b), std::invalid_argument);
  }

  SECTION("Zero-dimensional polytope") {
    Matrix A(2, 0);
    Vector b = Vector::Ones(2);

    CHECK_THROWS_AS(Polytope(A, b), std::invalid_argument);
  }
}

TEST_CASE("LinearPolytope checks containment", "[geometry][polytope]") {
  auto polytope = make_box();

  CHECK(polytope.contains(point(0, 0)));
  CHECK(polytope.contains(point(0.5, -0.5)));
  CHECK(polytope.contains(point(1, 1)));

  CHECK_FALSE(polytope.contains(point(1.1, 0)));
  CHECK_FALSE(polytope.contains(point(0, -1.1)));

  SECTION("Positive tolerance shrinks the feasible set") {
    constexpr Scalar tolerance = 0.25;

    CHECK(polytope.contains(point(0.5, 0), tolerance));
    CHECK(polytope.contains(point(0.75, 0), tolerance));
    CHECK_FALSE(polytope.contains(point(0.875, 0), tolerance));
  }
}

TEST_CASE("LinearPolytope computes general line domains",
          "[geometry][polytope]") {
  auto polytope = make_box();
  hopsy::EmptyWorkspace empty;

  const auto origin = point(0.25, -0.5);

  SECTION("Horizontal direction") {
    check_interval(polytope.line_domain(origin, point(1, 0), empty), -1.25,
                   0.75);
  }

  SECTION("Vertical direction") {
    check_interval(polytope.line_domain(origin, point(0, 1), empty), -0.5, 1.5);
  }

  SECTION("Diagonal direction") {
    check_interval(polytope.line_domain(origin, point(1, 1), empty), -0.5,
                   0.75);
  }

  SECTION("Negative direction") {
    check_interval(polytope.line_domain(origin, point(-1, 0), empty), -0.75,
                   1.25);
  }

  SECTION("Non-unit direction") {
    check_interval(polytope.line_domain(origin, point(2, 0), empty), -0.625,
                   0.375);
  }

  SECTION("Zero direction") {
    check_interval(polytope.line_domain(origin, point(0, 0), empty),
                   -std::numeric_limits<Scalar>::infinity(),
                   std::numeric_limits<Scalar>::infinity());
  }
}

TEST_CASE("LinearPolytope computes coordinate line domains",
          "[geometry][polytope]") {
  auto polytope = make_box();
  hopsy::EmptyWorkspace empty;

  const auto origin = point(0.25, -0.5);

  check_interval(
      polytope.line_domain(origin, hopsy::CoordinateDirection{0}, empty), -1.25,
      0.75);

  check_interval(
      polytope.line_domain(origin, hopsy::CoordinateDirection{1}, empty), -0.5,
      1.5);
}

TEST_CASE("Cached and uncached line domains agree",
          "[geometry][polytope][workspace]") {
  auto polytope = make_box();

  const auto origin = point(0.25, -0.5);
  auto workspace = polytope.create_workspace(origin);
  hopsy::EmptyWorkspace empty;

  // SECTION("General direction") {
  //     const auto direction = point(1, 2);
  //     const Scalar step = 0.25;
  //
  //     // Proposal phase: compute and cache A*d.
  //     [[maybe_unused]] const auto domain =
  //         polytope.line_domain(origin, direction, workspace);
  //
  //     // Advance phase: reuse the cached product.
  //     polytope.advance_workspace(step, direction, workspace);
  //
  //     origin += step * direction;
  //
  //     const auto expected = polytope.create_workspace(origin);
  //
  //     CHECK(
  //         workspace.slacks.isApprox(
  //             expected.slacks, 1e-12
  //         )
  //     );
  // }

  SECTION("Coordinate direction") {
    const hopsy::CoordinateDirection direction{1};

    const auto cached = polytope.line_domain(origin, direction, workspace);

    const auto uncached = polytope.line_domain(origin, direction, empty);

    check_interval(cached, uncached[0].lower, uncached[0].upper);
  }
}

TEST_CASE("Workspace initialization computes slacks",
          "[geometry][polytope][workspace]") {
  auto polytope = make_box();
  auto workspace = polytope.create_workspace(point(0.25, -0.5));

  REQUIRE(workspace.slacks.size() == 4);

  CHECK(workspace.slacks(0) == Approx(0.75));
  CHECK(workspace.slacks(1) == Approx(1.25));
  CHECK(workspace.slacks(2) == Approx(1.5));
  CHECK(workspace.slacks(3) == Approx(0.5));
}

TEST_CASE("Workspace updates match recomputation",
          "[geometry][polytope][workspace]") {
  auto polytope = make_box();

  auto origin = point(0.25, -0.5);
  auto workspace = polytope.create_workspace(origin);

  SECTION("General direction") {
    const auto direction = point(1, 2);
    workspace.direction_product = polytope.A() * direction;
    const Scalar step = 0.25;

    polytope.advance_workspace(step, direction, workspace);

    origin += step * direction;

    const auto expected = polytope.create_workspace(origin);

    CHECK(workspace.slacks.isApprox(expected.slacks, 1e-12));
  }

  SECTION("Coordinate direction") {
    const Scalar step = 0.5;

    polytope.advance_workspace(step, hopsy::CoordinateDirection{0}, workspace);

    origin(0) += step;

    const auto expected = polytope.create_workspace(origin);

    CHECK(workspace.slacks.isApprox(expected.slacks, 1e-12));
  }

  SECTION("Reset workspace") {
    const auto new_origin = point(-0.5, 0.75);

    polytope.reset_workspace(new_origin, workspace);

    const auto expected = polytope.create_workspace(new_origin);

    CHECK(workspace.slacks.isApprox(expected.slacks, 1e-12));
  }

  SECTION("Empty workspace updates are valid no-ops") {
    hopsy::EmptyWorkspace empty;

    polytope.advance_workspace(0.25, point(1, 0), empty);

    polytope.advance_workspace(0.25, hopsy::CoordinateDirection{1}, empty);
  }
}

TEST_CASE("Parallel violated constraints produce empty lines",
          "[geometry][polytope]") {
  auto polytope = make_box();
  hopsy::EmptyWorkspace empty;

  // x > 1, while moving exclusively in y.
  const auto origin = point(1.2, 0.5);
  const auto direction = point(0, 1);

  const auto domain = polytope.line_domain(origin, direction, empty);

  REQUIRE(domain.size() == 1);

  CHECK(domain[0].lower > domain[0].upper);
}
