#include "tempering.h"

namespace hopsy::bindings {
    namespace {
        std::vector<Index> draw_steps(std::size_t slots, Index local_steps, Index reference_steps,
                                     std::optional<std::vector<Index>> steps_per_slot) {
            if (steps_per_slot)
                return std::move(*steps_per_slot);
            return detail::tempering_steps(slots, local_steps, reference_steps);
        }

        template <auto Method, bool ReleaseGIL = false> auto exclusive_call() {
            return [](TemperingBinding &sampler, auto &&...args) -> decltype(auto) {
                auto guard = sampler.lock();
                if constexpr (ReleaseGIL) {
                    nb::gil_scoped_release release;
                    return (sampler.*Method)(std::forward<decltype(args)>(args)...);
                } else {
                    return (sampler.*Method)(std::forward<decltype(args)>(args)...);
                }
            };
        }
    } // namespace

    std::vector<std::shared_ptr<TargetMethods>>
    private_tempering_targets(const std::vector<nb::object> &targets, std::size_t replicas) {
        if (!targets.empty() && targets.size() != replicas)
            throw nb::value_error("expected one private capsule target per replica");
        std::vector<std::shared_ptr<TargetMethods>> methods;
        methods.reserve(targets.size());
        for (const auto &target : targets) {
            if (!nb::isinstance<CapsuleTarget>(target))
                throw nb::type_error("each replica target must be a CapsuleTarget");
            auto adapter = std::make_shared<TargetMethods>(target);
            for (const auto &previous : methods) {
                if (adapter->native == previous->native ||
                    (adapter->native->context &&
                     adapter->native->context == previous->native->context))
                    throw nb::value_error("replica targets must have distinct native contexts");
            }
            methods.push_back(std::move(adapter));
        }
        return methods;
    }

    void TemperingBinding::require_parallel_target() const {
        if (!parallel_ready_)
            throw std::invalid_argument(
                "parallel sampling requires a built-in target or one native capsule per replica");
    }

    DenseMatrix TemperingBinding::draw(Index count, Index local_steps, Index reference_steps) {
        return draw_samples(count, detail::tempering_steps(size(), local_steps, reference_steps),
                            std::nullopt);
    }

    std::tuple<DenseMatrix, Matrix<Index>, DenseMatrix>
    TemperingBinding::draw_traced(Index count, Index local_steps, Index reference_steps,
                                  bool parallel, std::size_t workers,
                                  std::optional<std::vector<Index>> steps_per_slot) {
        auto steps = draw_steps(size(), local_steps, reference_steps, std::move(steps_per_slot));
        if (parallel)
            require_parallel_target();
        else if (workers != 0)
            throw std::invalid_argument("A worker count requires parallel=True");
        auto result = draw_trace(count, steps, parallel ? std::optional{workers} : std::nullopt);
        return {std::move(result.samples), std::move(result.replica_ids),
                std::move(result.log_densities)};
    }

    DenseMatrix TemperingBinding::draw_parallel(Index count, Index local_steps,
                                                Index reference_steps, std::size_t workers,
                                                std::optional<std::vector<Index>> steps_per_slot) {
        auto steps = draw_steps(size(), local_steps, reference_steps, std::move(steps_per_slot));
        require_parallel_target();
        return draw_samples(count, steps, workers);
    }

    void bind_tempering(nb::module_ &m) {
        nb::class_<TemperatureLadder>(m, "TemperatureLadder")
            .def(nb::init<std::vector<double>>(), "inverse_temperatures"_a)
            .def_static("linear", &TemperatureLadder::linear, "num_replicas"_a)
            .def("__len__", &TemperatureLadder::size)
            .def_prop_ro("inverse_temperatures",
                         [](const TemperatureLadder &ladder) {
                             const auto betas = ladder.inverse_temperatures();
                             return std::vector<double>(betas.begin(), betas.end());
                         })
            .def(
                "adapted",
                [](const TemperatureLadder &ladder, const std::vector<double> &rates) {
                    return ladder.adapted(rates);
                },
                "mean_rejections"_a);
        nb::class_<ReplicaExchangeStatistics>(m, "ReplicaExchangeStatistics")
            .def_ro("attempts", &ReplicaExchangeStatistics::attempts)
            .def_ro("accepts", &ReplicaExchangeStatistics::accepts);
        nb::class_<TemperingBinding>(m, "NonReversibleParallelTempering")
            .def(
                "draw",
                [](TemperingBinding &sampler, Index count, Index local, Index reference) {
                    return exclusive_call<&TemperingBinding::draw, true>()(sampler, count, local,
                                                                           reference);
                },
                "num_samples"_a, "local_steps"_a = 1, "reference_steps"_a = 10)
            .def(
                "draw_parallel",
                [](TemperingBinding &sampler, Index count, Index local, Index reference,
                   std::size_t workers, std::optional<std::vector<Index>> steps_per_slot) {
                    return exclusive_call<&TemperingBinding::draw_parallel, true>()(
                        sampler, count, local, reference, workers, std::move(steps_per_slot));
                },
                "num_samples"_a, "local_steps"_a = 1, "reference_steps"_a = 10, "workers"_a = 0,
                "steps_per_slot"_a = nb::none())
            .def(
                "draw_traced",
                [](TemperingBinding &sampler, Index count, Index local, Index reference,
                   bool parallel, std::size_t workers,
                   std::optional<std::vector<Index>> steps_per_slot) {
                    return exclusive_call<&TemperingBinding::draw_traced, true>()(
                        sampler, count, local, reference, parallel, workers,
                        std::move(steps_per_slot));
                },
                "num_samples"_a, "local_steps"_a = 1, "reference_steps"_a = 10,
                "parallel"_a = false, "workers"_a = 0, "steps_per_slot"_a = nb::none())
            .def(
                "scan",
                [](TemperingBinding &sampler, Index local, Index reference) {
                    exclusive_call<&TemperingBinding::scan, true>()(sampler, local, reference);
                },
                "local_steps"_a = 1, "reference_steps"_a = 10)
            .def(
                "explore",
                [](TemperingBinding &sampler, std::size_t slot, Index steps) {
                    return exclusive_call<&TemperingBinding::explore, true>()(sampler, slot, steps);
                },
                "slot"_a, "steps"_a = 1)
            .def("communicate",
                 [](TemperingBinding &sampler) {
                     exclusive_call<&TemperingBinding::communicate, true>()(sampler);
                 })
            .def(
                "sampling_state",
                [](TemperingBinding &sampler, std::size_t slot) {
                    return exclusive_call<&TemperingBinding::sampling_state>()(sampler, slot);
                },
                "slot"_a)
            .def(
                "target_state",
                [](TemperingBinding &sampler, std::size_t slot) {
                    return exclusive_call<&TemperingBinding::target_state>()(sampler, slot);
                },
                "slot"_a)
            .def(
                "state_log_density",
                [](TemperingBinding &sampler, std::size_t slot) {
                    return exclusive_call<&TemperingBinding::state_log_density>()(sampler, slot);
                },
                "slot"_a)
            .def(
                "proposal_log_density",
                [](TemperingBinding &sampler, std::size_t slot) {
                    return exclusive_call<&TemperingBinding::proposal_log_density>()(sampler, slot);
                },
                "slot"_a)
            .def_prop_ro("ladder",
                         [](TemperingBinding &sampler) {
                             return exclusive_call<&TemperingBinding::ladder>()(sampler);
                         })
            .def(
                "set_ladder",
                [](TemperingBinding &sampler, TemperatureLadder ladder) {
                    exclusive_call<&TemperingBinding::set_ladder>()(sampler, std::move(ladder));
                },
                "ladder"_a)
            .def_prop_ro("replica_indices",
                         [](TemperingBinding &sampler) {
                             return exclusive_call<&TemperingBinding::replica_indices>()(sampler);
                         })
            .def_prop_ro("round_trips",
                         [](TemperingBinding &sampler) {
                             return exclusive_call<&TemperingBinding::round_trips>()(sampler);
                         })
            .def_prop_ro("exchange_statistics",
                         [](TemperingBinding &sampler) {
                             return exclusive_call<&TemperingBinding::exchange_statistics>()(
                                 sampler);
                         })
            .def("mean_exchange_rejections",
                 [](TemperingBinding &sampler) {
                     return exclusive_call<&TemperingBinding::mean_exchange_rejections>()(sampler);
                 })
            .def("reset_statistics", [](TemperingBinding &sampler) {
                exclusive_call<&TemperingBinding::reset_statistics>()(sampler);
            });
    }
} // namespace hopsy::bindings
