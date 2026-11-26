#include "tempering.h"
#include <algorithm>
#include <barrier>
#include <exception>
#include <latch>
#include <limits>
#include <thread>
#include <nanobind/stl/tuple.h>

namespace hopsy::bindings {
    namespace {
        std::vector<Index> draw_steps(Index count, std::size_t slots, Index local_steps,
                                      Index reference_steps,
                                      std::optional<std::vector<Index>> steps_per_slot) {
            if (count < 0)
                throw std::invalid_argument("Invalid sample count");
            if (steps_per_slot) {
                if (steps_per_slot->size() != slots ||
                    std::any_of(steps_per_slot->begin(), steps_per_slot->end(),
                                [](Index steps) { return steps < 1; }))
                    throw std::invalid_argument("Expected one positive step count per slot");
                return std::move(*steps_per_slot);
            }
            if (local_steps < 1 || reference_steps < 1)
                throw std::invalid_argument("Invalid sample count or local step count");
            std::vector<Index> steps(slots, local_steps);
            steps.front() = reference_steps;
            return steps;
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

    void TemperingBinding::record_scan(DenseMatrix &samples, Index scan, std::size_t cold,
                                       Trace *trace) {
        samples.col(scan) = target_state(cold);
        communicate();
        if (trace)
            record_snapshot(trace->ids.col(scan + 1).data(),
                            trace->log_densities.col(scan + 1).data());
    }

    std::tuple<DenseMatrix, TemperingBinding::ReplicaTrace, DenseMatrix>
    TemperingBinding::draw_traced(Index count, Index local_steps, Index reference_steps,
                                  bool parallel, std::size_t workers,
                                  std::optional<std::vector<Index>> steps_per_slot) {
        const auto slots = ladder().size();
        const auto steps = draw_steps(count, slots, local_steps, reference_steps,
                                      std::move(steps_per_slot));
        if (count == std::numeric_limits<Index>::max())
            throw std::invalid_argument("Trace requires room for the initial snapshot");
        Trace trace{ReplicaTrace(slots, count + 1), DenseMatrix(slots, count + 1)};
        record_snapshot(trace.ids.col(0).data(), trace.log_densities.col(0).data());
        DenseMatrix samples;
        if (parallel) {
            samples = run_parallel_draw(count, steps, &trace, workers);
        } else {
            if (workers != 0)
                throw std::invalid_argument("A worker count requires parallel=True");
            samples.resize(target_state(slots - 1).size(), count);
            for (Index scan = 0; scan < count; ++scan) {
                for (std::size_t slot = 0; slot < slots; ++slot)
                    explore(slot, steps[slot]);
                record_scan(samples, scan, slots - 1, &trace);
            }
        }
        return {std::move(samples), std::move(trace.ids), std::move(trace.log_densities)};
    }

    DenseMatrix TemperingBinding::draw_parallel(Index count, Index local_steps,
                                                Index reference_steps, std::size_t workers,
                                                std::optional<std::vector<Index>> steps_per_slot) {
        auto steps = draw_steps(count, ladder().size(), local_steps, reference_steps,
                                std::move(steps_per_slot));
        return run_parallel_draw(count, steps, nullptr, workers);
    }

    DenseMatrix TemperingBinding::run_parallel_draw(Index count, const std::vector<Index> &steps,
                                                    Trace *trace,
                                                    std::size_t worker_count) {
        if (!parallel_ready_)
            throw std::invalid_argument(
                "parallel sampling requires a built-in target or one native capsule per replica");
        const auto slots = steps.size();
        if (worker_count == 0)
            worker_count = slots;
        if (worker_count > slots)
            throw std::invalid_argument("Workers must not exceed the number of temperature slots");
        const auto cold = slots - 1;
        DenseMatrix samples(target_state(cold).size(), count);
        if (count == 0)
            return samples;

        Index scan = 0;
        std::vector<std::exception_ptr> exploration_errors(slots);
        std::exception_ptr failure;
        std::barrier finish_scan(static_cast<std::ptrdiff_t>(worker_count), [&]() noexcept {
            for (const auto &error : exploration_errors) {
                if (error) {
                    failure = error;
                    return;
                }
            }
            try {
                record_scan(samples, scan, cold, trace);
                ++scan;
            } catch (...) {
                failure = std::current_exception();
            }
        });

        // A failed thread launch must release workers without entering a
        // barrier whose participant count can no longer be reached.
        std::latch start(1);
        std::vector<std::jthread> workers;
        workers.reserve(worker_count);
        try {
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                workers.emplace_back([&, worker](std::stop_token stop) {
                    start.wait();
                    if (stop.stop_requested())
                        return;
                    for (Index i = 0; i < count; ++i) {
                        for (std::size_t slot = worker; slot < slots; slot += worker_count) {
                            try {
                                explore(slot, steps[slot]);
                            } catch (...) {
                                exploration_errors[slot] = std::current_exception();
                            }
                        }
                        finish_scan.arrive_and_wait();
                        if (failure)
                            return;
                    }
                });
            }
        } catch (...) {
            for (auto &worker : workers)
                worker.request_stop();
            start.count_down();
            throw;
        }
        start.count_down();
        for (auto &worker : workers)
            worker.join();
        if (failure)
            std::rethrow_exception(failure);
        return samples;
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
