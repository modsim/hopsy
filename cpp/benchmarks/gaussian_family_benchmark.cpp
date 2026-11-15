// Native-core comparison: identical target arithmetic, storage, warm-up and MH
// decisions for both backends. Legacy CSmMALA now clips gradients at norm 1.
#include <hopsy/kernel/gaussian_kernel.h>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/sampling/step.h>
#include <hops/MarkovChain/Proposal/GaussianProposal.hpp>
#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/MarkovChain/Proposal/CSmMALAProposal.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using State = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
#ifdef HOPSY_COUNT_TARGET_CALLS
static std::size_t density_calls = 0, gradient_calls = 0;
void count_density() { ++density_calls; }
void count_gradient() { ++gradient_calls; }
#else
void count_density() {}
void count_gradient() {}
#endif
struct NormalTarget {
    double log_density(const State &x) const {
        count_density();
        return -0.5 * x.squaredNorm();
    }
    State log_gradient(const State &x) const {
        count_gradient();
        return -x;
    }
};
struct LegacyNormal : hops::Model {
    double computeNegativeLogLikelihood(const State &x) override {
        count_density();
        return 0.5 * x.squaredNorm();
    }
    std::optional<State> computeLogLikelihoodGradient(const State &x) override {
        count_gradient();
        return -x;
    }
    std::optional<Matrix> computeExpectedFisherInformation(const State &x) override {
        return Matrix::Identity(x.size(), x.size());
    }
    std::vector<std::string> getDimensionNames() const override { return {}; }
    std::unique_ptr<hops::Model> copyModel() const override {
        return std::make_unique<LegacyNormal>(*this);
    }
};

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: benchmark mode seed samples sigma output.bin\n";
        return 1;
    }
    const std::string mode = argv[1];
    const int seed = std::stoi(argv[2]), count = std::stoi(argv[3]);
    const double sigma = std::stod(argv[4]);
    if (count <= 0 || !std::isfinite(sigma) || sigma <= 0)
        return 1;
    constexpr int n = 10, warmup = 20000;
    Matrix A(2 * n, n);
    A.topRows(n).setIdentity();
    A.bottomRows(n) = -A.topRows(n);
    const State b = State::Constant(2 * n, 3.), initial = State::Zero(n);
    Matrix samples(n, count);
    int accepted = 0;
    double elapsed = 0.;
    NormalTarget target;
    auto measure = [&](auto transition, auto state) {
        for (int i = 0; i < warmup; ++i)
            transition();
#ifdef HOPSY_COUNT_TARGET_CALLS
        density_calls = gradient_calls = 0;
#endif
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i) {
            accepted += transition();
            samples.col(i) = state();
        }
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    if (mode.starts_with("old_")) {
        hops::RandomNumberGenerator rng(seed);
        const State *state = &initial;
        double log_density = target.log_density(initial);
        auto run = [&]<class Proposal>(Proposal &proposal) {
            measure(
                [&] {
                    const auto &y = proposal.propose(rng);
                    double ratio = proposal.computeLogAcceptanceProbability();
                    double candidate_density = log_density;
                    if (std::isfinite(ratio)) {
                        if constexpr (!std::is_same_v<
                                          Proposal, hops::CSmMALAProposal<LegacyNormal, Matrix>>) {
                            candidate_density = target.log_density(y);
                            ratio += candidate_density - log_density;
                        }
                    }
                    bool accept =
                        !std::isnan(ratio) &&
                        (ratio >= 0. ||
                         (ratio != -std::numeric_limits<double>::infinity() &&
                          std::log(std::uniform_real_distribution<double>{}(rng)) < ratio));
                    if (accept) {
                        state = &proposal.acceptProposal();
                        log_density = candidate_density;
                    }
                    return accept;
                },
                [&]() -> const State & { return *state; });
        };
        if (mode == "old_grw") {
            hops::GaussianProposal<Matrix, State> proposal(A, b, initial, sigma);
            run(proposal);
        } else if (mode == "old_dikin") {
            // Legacy covariance = legacy_step/n * G^-1.
            hops::DikinProposal<Matrix, State> proposal(A, b, initial, n * sigma * sigma);
            run(proposal);
        } else if (mode == "old_csmala") {
            // Legacy covariance = legacy_step^2/n * G^-1, fisher_weight=0.
            hops::CSmMALAProposal<LegacyNormal, Matrix> proposal(A, b, initial, LegacyNormal{}, 0.,
                                                                 std::sqrt(n) * sigma);
            run(proposal);
        } else
            return 1;
    } else {
        State x = initial;
        hopsy::LinearPolytope<double> geometry(A, b);
        // These kernels need no accepted-slack cache. Legacy proposals likewise
        // recompute candidate slacks rather than maintaining a shared HR cache.
        hopsy::EmptyWorkspace workspace;
        hopsy::RandomNumberGenerator rng(seed);
        auto run = [&](auto &kernel) {
            measure([&] { return hopsy::step(kernel, x, rng, target, geometry, workspace); },
                    [&]() -> const State & { return x; });
        };
        if (mode == "new_grw") {
            hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::IdentityMetric> kernel(x, {}, {},
                                                                                         sigma);
            run(kernel);
#ifndef HOPSY_BENCHMARK_BASELINE
        } else if (mode == "new_dikin") {
            hopsy::GaussianKernel<State, hopsy::ZeroDrift, hopsy::DikinMetric<double>> kernel(
                x, {}, {}, sigma);
            run(kernel);
        } else if (mode == "new_csmala") {
            hopsy::GaussianKernel<State, hopsy::ClippedGradientDrift, hopsy::DikinMetric<double>>
                kernel(x, {}, {}, sigma);
            run(kernel);
#endif
        } else
            return 1;
    }
#ifdef HOPSY_COUNT_TARGET_CALLS
    std::cerr << mode << ",steps=" << count << ",density_calls=" << density_calls
              << ",gradient_calls=" << gradient_calls << '\n';
#endif
    std::ofstream file(argv[5], std::ios::binary);
    file.write(reinterpret_cast<const char *>(samples.data()), samples.size() * sizeof(double));
    if (!file)
        return 2;
    std::cout << std::setprecision(12) << mode << ',' << seed << ',' << count << ',' << sigma << ','
              << count / elapsed << ',' << double(accepted) / count << '\n';
}
