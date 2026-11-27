#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#include <hopsy/geometry/linear_polytope.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/parallel_tempering/non_reversible_parallel_tempering.h>

struct StandardNormal {
    double log_density(const hopsy::Vector<double> &x) const { return -0.5 * x.squaredNorm(); }
};

int main() {
    using namespace hopsy;
    Matrix<double> A(4, 2);
    A << 1, 0, -1, 0, 0, 1, 0, -1;
    auto problem =
        Problem(LinearPolytope<double>(A, Vector<double>::Constant(4, 4.)), StandardNormal{});
    const auto ladder = TemperatureLadder::linear(6);
    const Vector<double> initial = Vector<double>::Zero(2);
    auto make_kernel = [&] {
        return HitAndRunKernel(initial, RandomCoordinateDirectionGenerator(initial.size()),
                               UniformLineKernel{});
    };
    std::vector<TemperingReplica<decltype(make_kernel())>> replicas;
    for (std::size_t i = 0; i < ladder.size(); ++i)
        replicas.push_back({initial, make_kernel(), RandomNumberGenerator(42, i)});

    auto nrpt = NonReversibleParallelTempering(std::move(problem), ladder, std::move(replicas),
                                               RandomNumberGenerator(43, ladder.size()));
    for (int i = 0; i < 500; ++i)
        nrpt.scan();
    nrpt.reset_statistics();
    const auto samples = nrpt.draw(5000);
    std::cout << "Cold sample mean " << samples.rowwise().mean().transpose() << '\n';

    const auto rejections = nrpt.mean_exchange_rejections();
    const auto stats = nrpt.exchange_statistics();
    for (std::size_t edge = 0; edge < stats.size(); ++edge)
        std::cout << "Edge " << edge << " accepted " << stats[edge].accepts << '/'
                  << stats[edge].attempts << ", mean rejection " << rejections[edge] << '\n';
    const auto trips = nrpt.round_trips();
    std::cout << "Round trips " << std::accumulate(trips.begin(), trips.end(), std::size_t{0})
              << ", cold replica " << nrpt.replica_indices().back() << '\n';
}
