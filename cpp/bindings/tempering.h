#ifndef HOPSY_BINDINGS_TEMPERING_H
#define HOPSY_BINDINGS_TEMPERING_H

#include "common.h"
#include <atomic>
#include <tuple>
#include <hopsy/sampling/basis_reference.h>
#include <hopsy/sampling/non_reversible_parallel_tempering.h>

namespace hopsy::bindings {
    std::vector<std::shared_ptr<TargetMethods>>
    private_tempering_targets(const std::vector<nb::object> &targets, std::size_t replicas);

    // Dispatch once per Python call. Local transitions use the typed native sampler.
    class TemperingBinding {
        using ReplicaTrace = Matrix<Index>;

        struct Trace {
            ReplicaTrace ids;
            DenseMatrix log_densities;
        };

        class Operation {
          public:
            explicit Operation(std::atomic_flag &busy) : busy_(busy) {
                if (busy_.test_and_set(std::memory_order_acquire))
                    throw std::runtime_error("parallel tempering sampler is already in use");
            }
            ~Operation() { busy_.clear(std::memory_order_release); }
            Operation(const Operation &) = delete;
            Operation &operator=(const Operation &) = delete;

          private:
            std::atomic_flag &busy_;
        };

      public:
        TemperingBinding(nb::handle problem, bool parallel_ready)
            : problem_owner_(nb::borrow(problem)), parallel_ready_(parallel_ready) {}
        virtual ~TemperingBinding() = default;

        auto lock() const { return Operation(busy_); }
        virtual DenseMatrix draw(Index count, Index local_steps, Index reference_steps) = 0;
        DenseMatrix draw_parallel(Index count, Index local_steps, Index reference_steps,
                                  std::size_t workers,
                                  std::optional<std::vector<Index>> steps_per_slot);
        std::tuple<DenseMatrix, ReplicaTrace, DenseMatrix>
        draw_traced(Index count, Index local_steps, Index reference_steps, bool parallel,
                    std::size_t workers, std::optional<std::vector<Index>> steps_per_slot);
        virtual void scan(Index local_steps, Index reference_steps) = 0;
        virtual std::size_t explore(std::size_t slot, Index steps) = 0;
        virtual void communicate() = 0;
        virtual State sampling_state(std::size_t slot) const = 0;
        virtual State target_state(std::size_t slot) const = 0;
        virtual double state_log_density(std::size_t slot) const = 0;
        virtual std::optional<double> proposal_log_density(std::size_t slot) const = 0;
        virtual TemperatureLadder ladder() const = 0;
        virtual void set_ladder(TemperatureLadder ladder) = 0;
        virtual std::vector<std::size_t> replica_indices() const = 0;
        virtual std::vector<std::size_t> round_trips() const = 0;
        virtual std::vector<ReplicaExchangeStatistics> exchange_statistics() const = 0;
        virtual std::vector<double> mean_exchange_rejections() const = 0;
        virtual void reset_statistics() = 0;

      private:
        virtual void record_snapshot(Index *ids, double *log_densities) const = 0;
        void record_scan(DenseMatrix &samples, Index scan, std::size_t cold, Trace *trace);
        DenseMatrix run_parallel_draw(Index count, const std::vector<Index> &steps_per_slot,
                                      Trace *trace, std::size_t workers);
        nb::object problem_owner_; // Retains the geometry, target and coordinate map.
        const bool parallel_ready_;
        mutable std::atomic_flag busy_{};
    };

    template <class Problem, class Kernel, class Reference = detail::DefaultReference>
    class TypedTemperingBinding final : public TemperingBinding {
      public:
        TypedTemperingBinding(nb::handle owner, Problem problem, TemperatureLadder ladder,
                              std::vector<TemperingReplica<Kernel>> replicas,
                              RandomNumberGenerator exchange_rng,
                              std::vector<Problem> replica_problems, Reference reference = {})
            : TemperingBinding(owner, !runtime_target<typename Problem::target_type> ||
                                          !replica_problems.empty()),
              sampler_(std::move(problem), std::move(ladder), std::move(replicas),
                       std::move(exchange_rng), std::move(replica_problems), std::move(reference)) {}

        DenseMatrix draw(Index count, Index local, Index reference) override {
            return sampler_.draw(count, local, reference);
        }
        void scan(Index local, Index reference) override { sampler_.scan(local, reference); }
        std::size_t explore(std::size_t slot, Index steps) override {
            return sampler_.explore(slot, steps);
        }
        void communicate() override { sampler_.communicate(); }
        State sampling_state(std::size_t slot) const override {
            return sampler_.sampling_state(slot);
        }
        State target_state(std::size_t slot) const override { return sampler_.target_state(slot); }
        double state_log_density(std::size_t slot) const override {
            return sampler_.state_log_density(slot);
        }
        std::optional<double> proposal_log_density(std::size_t slot) const override {
            return sampler_.proposal_log_density(slot);
        }
        TemperatureLadder ladder() const override { return sampler_.ladder(); }
        void set_ladder(TemperatureLadder ladder) override {
            sampler_.set_ladder(std::move(ladder));
        }
        std::vector<std::size_t> replica_indices() const override {
            const auto ids = sampler_.replica_indices();
            return {ids.begin(), ids.end()};
        }
        std::vector<std::size_t> round_trips() const override {
            const auto trips = sampler_.round_trips();
            return {trips.begin(), trips.end()};
        }
        std::vector<ReplicaExchangeStatistics> exchange_statistics() const override {
            const auto stats = sampler_.exchange_statistics();
            return {stats.begin(), stats.end()};
        }
        std::vector<double> mean_exchange_rejections() const override {
            return sampler_.mean_exchange_rejections();
        }
        void reset_statistics() override { sampler_.reset_statistics(); }

      private:
        void record_snapshot(Index *ids, double *log_densities) const override {
            const auto current = sampler_.replica_indices();
            for (std::size_t slot = 0; slot < current.size(); ++slot) {
                ids[slot] = static_cast<Index>(current[slot]);
                log_densities[slot] = sampler_.state_log_density(slot);
            }
        }

        NonReversibleParallelTempering<Problem, Kernel, Reference> sampler_;
    };

    template <class Kernel> struct TemperingSetup {
        nb::handle owner;
        TemperatureLadder ladder;
        std::vector<TemperingReplica<Kernel>> replicas;
        RandomNumberGenerator exchange_rng;
        std::vector<std::shared_ptr<TargetMethods>> replica_targets;
        std::optional<DenseMatrix> reference_basis;
        double reference_max_beta;

        template <typename Problem>
        std::unique_ptr<TemperingBinding> make(Problem problem,
                                               std::vector<Problem> replica_problems = {}) {
            if constexpr (!runtime_target<typename Problem::target_type>) {
                if (!replica_targets.empty())
                    throw nb::type_error("replica_targets requires a native capsule problem");
            }
            if (reference_basis) {
                if constexpr (std::same_as<Kernel, CSmMALAKernel<State>> &&
                              Problem::identity_coordinates &&
                              std::same_as<typename Problem::geometry_type, Geometry>) {
                    BasisReference<double> reference(std::move(*reference_basis), reference_max_beta);
                    return std::make_unique<TypedTemperingBinding<Problem, Kernel,
                                                                  BasisReference<double>>>(
                        owner, std::move(problem), std::move(ladder), std::move(replicas),
                        std::move(exchange_rng), std::move(replica_problems), std::move(reference));
                } else {
                    throw nb::type_error("reference_basis requires CSmMALA on a linear problem "
                                         "in its original coordinates");
                }
            }
            return std::make_unique<TypedTemperingBinding<Problem, Kernel>>(
                owner, std::move(problem), std::move(ladder), std::move(replicas),
                std::move(exchange_rng), std::move(replica_problems));
        }

        template <class C, class G, int Order>
        std::unique_ptr<TemperingBinding> make_runtime(const P<RuntimeTarget<Order>, C, G> &p) {
            constexpr int required_order = std::same_as<Kernel, CSmMALAKernel<State>> ? 1 : 0;
            if constexpr (Order < required_order) {
                throw nb::type_error("kernel requires log_gradient");
            } else {
                using Target = RuntimeTarget<required_order>;
                using NormalizedProblem = P<Target, C, G>;
                // Kernels need at most the gradient, regardless of the original
                // problem's derivative order.
                const auto build = [&](const std::shared_ptr<TargetMethods> &methods) {
                    if constexpr (std::same_as<C, IdentityCoordinates>)
                        return NormalizedProblem(std::cref(p.geometry()), Target{methods});
                    else
                        return NormalizedProblem(std::cref(p.geometry()), Target{methods},
                                                 std::cref(p.coordinates()));
                };
                if (!replica_targets.empty() && !p.target().methods->native)
                    throw nb::type_error("replica_targets requires a native capsule problem");
                std::vector<NormalizedProblem> replica_problems;
                replica_problems.reserve(replica_targets.size());
                for (const auto &methods : replica_targets) {
                    if (methods->order < required_order)
                        throw nb::type_error("kernel requires log_gradient");
                    if (methods->native->dimension != p.target().methods->native->dimension)
                        throw nb::value_error("replica target dimension mismatch");
                    replica_problems.push_back(build(methods));
                }
                return make(build(p.target().methods), std::move(replica_problems));
            }
        }

        template <class C, class G> std::unique_ptr<TemperingBinding> try_bind() {
            if (nb::isinstance<P<Uniform, C, G>>(owner))
                return make(nb::cast<const P<Uniform, C, G> &>(owner));
            if (nb::isinstance<P<Gaussian, C, G>>(owner))
                return make(nb::cast<const P<Gaussian, C, G> &>(owner));
            if (nb::isinstance<P<RuntimeTarget<0>, C, G>>(owner))
                return make_runtime<C, G, 0>(nb::cast<const P<RuntimeTarget<0>, C, G> &>(owner));
            if (nb::isinstance<P<RuntimeTarget<1>, C, G>>(owner))
                return make_runtime<C, G, 1>(nb::cast<const P<RuntimeTarget<1>, C, G> &>(owner));
            if (nb::isinstance<P<RuntimeTarget<2>, C, G>>(owner))
                return make_runtime<C, G, 2>(nb::cast<const P<RuntimeTarget<2>, C, G> &>(owner));
            return {};
        }

        std::unique_ptr<TemperingBinding> bind() {
            if (auto result = try_bind<IdentityCoordinates, Geometry>())
                return result;
            if (auto result = try_bind<Coordinates, Geometry>())
                return result;
            if constexpr (kernel_supports_geometry_v<Kernel, Bilinear>) {
                if (auto result = try_bind<IdentityCoordinates, Bilinear>())
                    return result;
                if (auto result = try_bind<Coordinates, Bilinear>())
                    return result;
            }
            throw nb::type_error("invalid problem or kernel does not support this geometry");
        }
    };

    template <class Kernel> void bind_tempering_kernel(nb::module_ &m) {
        m.def(
            "bind_parallel_tempering",
            [](nb::handle problem, TemperatureLadder ladder, const std::vector<Kernel *> &kernels,
               std::vector<State> initial_states, const std::vector<RandomNumberGenerator *> &rngs,
               RandomNumberGenerator &exchange_rng,
               const std::vector<nb::object> &replica_targets,
               std::optional<DenseMatrix> reference_basis, double reference_max_beta) {
                const auto size = ladder.size();
                if (size < 2 || kernels.size() != size || initial_states.size() != size ||
                    rngs.size() != size)
                    throw nb::value_error(
                        "expected one kernel, initial state and RNG per temperature");
                if (!reference_basis && reference_max_beta != 0.)
                    throw nb::value_error("reference_max_beta requires reference_basis");
                auto targets = private_tempering_targets(replica_targets, size);
                std::vector<TemperingReplica<Kernel>> replicas;
                replicas.reserve(size);
                for (std::size_t i = 0; i < size; ++i) {
                    if (!kernels[i] || !rngs[i])
                        throw nb::type_error("replica kernel and RNG must not be None");
                    ExclusiveLease<Kernel> kernel_lease(*kernels[i]);
                    ExclusiveLease<RandomNumberGenerator> rng_lease(*rngs[i]);
                    replicas.push_back({std::move(initial_states[i]), *kernels[i], *rngs[i]});
                }
                auto exchange = [&] {
                    ExclusiveLease<RandomNumberGenerator> lease(exchange_rng);
                    return exchange_rng;
                }();
                return TemperingSetup<Kernel>{problem, std::move(ladder), std::move(replicas),
                                              std::move(exchange), std::move(targets),
                                              std::move(reference_basis), reference_max_beta}
                    .bind();
            },
            "problem"_a, "ladder"_a, "kernels"_a, "initial_states"_a, "rngs"_a, "exchange_rng"_a,
            "replica_targets"_a = nb::tuple(), "reference_basis"_a = nb::none(),
            "reference_max_beta"_a = 0.);
    }

    template <class Direction, class Line>
    void bind_hit_and_run_with_tempering(nb::module_ &m, const char *name) {
        bind_hit_and_run<Direction, Line>(m, name);
        bind_tempering_kernel<HitAndRunKernel<State, Direction, Line>>(m);
    }

    template <class Kernel> void bind_gaussian_with_tempering(nb::module_ &m, const char *name) {
        bind_gaussian<Kernel>(m, name);
        bind_tempering_kernel<Kernel>(m);
    }
} // namespace hopsy::bindings
#endif
