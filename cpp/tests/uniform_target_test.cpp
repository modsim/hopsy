#include <catch2/catch_test_macros.hpp>

#include <hopsy/target/concepts.h>
#include <hopsy/target/uniform.h>
#include <hopsy/types.h>

TEST_CASE("UniformTarget satisfies target density concepts", "[target][uniform]")
{
    using State = hopsy::Vector<double>;

    static_assert(hopsy::TargetDensity<hopsy::UniformTarget, State>);
    static_assert(hopsy::DifferentiableTargetDensity<hopsy::UniformTarget, State>);
    static_assert(hopsy::TwiceDifferentiableTargetDensity<hopsy::UniformTarget, State>);
}

TEST_CASE("UniformTarget has constant zero log density", "[target][uniform]")
{
    hopsy::UniformTarget target;

    hopsy::Vector<double> x(3);
    x << 1.0, -2.0, 42.0;

    REQUIRE(target.log_density(x) == 0.0);
}

TEST_CASE("UniformTarget has zero gradient", "[target][uniform]")
{
    hopsy::UniformTarget target;

    hopsy::Vector<double> x(3);
    x << 1.0, -2.0, 42.0;

    const auto gradient = target.log_gradient(x);

    REQUIRE(gradient.size() == x.size());
    REQUIRE(gradient.isZero());
}

TEST_CASE("UniformTarget has zero Hessian", "[target][uniform]")
{
    hopsy::UniformTarget target;

    hopsy::Vector<double> x(3);
    x << 1.0, -2.0, 42.0;

    const auto hessian = target.log_hessian(x);

    REQUIRE(hessian.rows() == x.size());
    REQUIRE(hessian.cols() == x.size());
    REQUIRE(hessian.isZero());
}
