#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <hops/MarkovChain/Proposal/CSmMALAProposal.hpp>

namespace {
    struct ConstantGradient : hops::Model {
        double gradient;
        explicit ConstantGradient(double g) : gradient(g) {}
        double computeNegativeLogLikelihood(const hops::VectorType &x) override {
            return -gradient * x(0);
        }
        std::optional<hops::VectorType>
        computeLogLikelihoodGradient(const hops::VectorType &) override {
            return hops::VectorType::Constant(1, gradient);
        }
        std::vector<std::string> getDimensionNames() const override { return {}; }
        std::unique_ptr<hops::Model> copyModel() const override {
            return std::make_unique<ConstantGradient>(*this);
        }
    };
} // namespace
TEST_CASE("Legacy CSmMALA clips at one continuously instead of normalizing small gradients",
          "[legacy][csmala]") {
    hops::MatrixType A(2, 1);
    A << 1., -1.;
    auto b = hops::VectorType::Ones(2).eval();
    auto x = hops::VectorType::Zero(1).eval();
    for (double g : {0., 1e-12, 0.25, 1., 1.000001, 2., -0.25, -2.}) {
        hops::CSmMALAProposal<ConstantGradient, hops::MatrixType> proposal(
            A, b, x, ConstantGradient(g), 0., 0.4);
        hops::RandomNumberGenerator rng(81), reference(81);
        const double z = std::normal_distribution<double>{}(reference);
        const double expected = 0.4 / std::sqrt(2.) * z + 0.04 * std::clamp(g, -1., 1.);
        REQUIRE(proposal.propose(rng)(0) == Catch::Approx(expected).margin(1e-14));
    }
}
