// Standalone ownership/allocation smoke check; registered with CTest.
#include <hopsy/geometry/bilinear_polytope.h>
#include <hopsy/kernel/hit_and_run.h>
#include <hopsy/kernel/direction_generator.h>
#include <hopsy/kernel/uniform_line_kernel.h>
#include <hopsy/sampling/owned_chain.h>
#include <hopsy/target/uniform.h>
#include <thread>
#include <vector>
#include <iostream>
int main() {
    using V = hopsy::Vector<double>;
    using M = hopsy::Matrix<double>;
    M A(4, 2);
    A << 1, 0, 0, 1, -1, 0, 0, -1;
    hopsy::BilinearPolytope<> geometry(A, V::Ones(4), 1, {M::Ones(1, 1)}, V::Zero(1));
    const auto problem = hopsy::Problem(std::cref(geometry), hopsy::UniformTarget{});
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 4; ++worker)
        workers.emplace_back([&, worker] {
            V state(2);
            state << -.5, .5;
            auto kernel = hopsy::HitAndRunKernel(state, hopsy::RandomDirectionGenerator<double>(2),
                                                 hopsy::UniformLineKernel{});
            auto chain = hopsy::bind_chain(problem, kernel, V(state));
            hopsy::RandomNumberGenerator rng(42 + worker);
            for (int batch = 0; batch < 10; ++batch) {
                auto samples = chain->draw(rng, 1000, 2);
                for (hopsy::Index i = 0; i < samples.cols(); ++i)
                    if (!geometry.is_feasible(samples.col(i), -1e-11))
                        std::terminate();
                auto workspace = geometry.create_workspace(chain->sampling_state());
                auto copy =
                    geometry.line_domain(chain->sampling_state(), V::Ones(2).eval(), workspace);
                if (copy.empty())
                    std::terminate();
            }
        });
    for (auto &worker : workers)
        worker.join();
    std::cout << "bilinear sanitizer smoke passed\n";
}
