#ifndef HOPSY_SAMPLING_PARALLEL_H
#define HOPSY_SAMPLING_PARALLEL_H

#include <barrier>
#include <cstddef>
#include <exception>
#include <functional>
#include <latch>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hopsy::detail {
    // Work must handle its exceptions if it participates in a barrier.
    template <typename Work> void run_workers(std::size_t count, const Work &work) {
        std::latch start(1);
        std::vector<std::jthread> workers;
        workers.reserve(count);
        try {
            for (std::size_t worker = 0; worker < count; ++worker) {
                workers.emplace_back([&, worker](std::stop_token stop) {
                    start.wait();
                    if (!stop.stop_requested())
                        work(worker);
                });
            }
        } catch (...) {
            // A partial launch cannot satisfy a barrier. Cancel before releasing it.
            for (auto &worker : workers)
                worker.request_stop();
            start.count_down();
            throw;
        }
        start.count_down();
        for (auto &worker : workers)
            worker.join();
    }

    // Keep one worker loop for all kernel/target combinations. Callbacks
    // dispatch once per task or barrier, outside individual MCMC transitions.
    inline void parallel_rounds(std::size_t tasks, std::size_t rounds, std::size_t workers,
                                const std::function<void(std::size_t)> &task,
                                const std::function<void(std::size_t)> &finish) {
        const auto count = workers == 0 ? tasks : workers;
        if (count > tasks)
            throw std::invalid_argument("Workers must not exceed the number of tasks");
        if (rounds == 0 || tasks == 0)
            return;

        std::vector<std::exception_ptr> errors(tasks);
        std::exception_ptr failure;
        std::size_t completed = 0;
        std::barrier barrier(static_cast<std::ptrdiff_t>(count), [&]() noexcept {
            for (const auto &error : errors) {
                if (error) {
                    failure = error;
                    return;
                }
            }
            try {
                finish(completed++);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        run_workers(count, [&](std::size_t worker) {
            for (std::size_t round = 0; round < rounds; ++round) {
                for (std::size_t i = worker; i < tasks; i += count) {
                    try {
                        task(i);
                    } catch (...) {
                        errors[i] = std::current_exception();
                    }
                }
                barrier.arrive_and_wait();
                if (failure)
                    return;
            }
        });
        if (failure)
            std::rethrow_exception(failure);
    }

    // Finish runs once all tasks in a round have returned, before the next round.
    // A task failure skips Finish and is propagated after all workers have joined.
    template <typename Task, typename Finish>
    void for_each_round(std::size_t tasks, std::size_t rounds, std::optional<std::size_t> workers,
                        const Task &task, const Finish &finish) {
        if (workers) {
            parallel_rounds(tasks, rounds, *workers, std::cref(task), std::cref(finish));
            return;
        }
        for (std::size_t round = 0; round < rounds; ++round) {
            for (std::size_t i = 0; i < tasks; ++i)
                task(i);
            finish(round);
        }
    }
} // namespace hopsy::detail

namespace hopsy {
    // Run independent tasks to completion. Each task owns its mutable resources.
    // A worker count of zero uses one worker per task.
    template <typename Task>
    void parallel_for(std::size_t tasks, std::size_t workers, const Task &task) {
        detail::parallel_rounds(tasks, 1, workers, std::cref(task), [](std::size_t) {});
    }
} // namespace hopsy

#endif
