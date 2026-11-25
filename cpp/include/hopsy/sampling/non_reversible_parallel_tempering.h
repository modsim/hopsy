#ifndef HOPSY_NON_REVERSIBLE_PARALLEL_TEMPERING_H
#define HOPSY_NON_REVERSIBLE_PARALLEL_TEMPERING_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <hopsy/sampling/chain.h>
#include <hopsy/sampling/temperature_ladder.h>
#include <hopsy/target/uniform.h>

namespace hopsy {
    namespace detail {
        inline double checked_tempering_log_density(double value) {
            if (std::isnan(value) || value == std::numeric_limits<double>::infinity())
                throw std::domain_error(
                    "Tempering requires finite or negative-infinite log density");
            return value;
        }

        template <typename Target> class PowerTarget {
          public:
            PowerTarget(const Target &target, double beta) : target_(target), beta_(beta) {}

            template <typename State> double log_density(const State &state) const {
                return beta_ * checked_tempering_log_density(target_.log_density(state));
            }

            template <typename State>
                requires requires(const Target &t, const State &s) { t.log_gradient(s); }
            State log_gradient(const State &state) const {
                return beta_ * target_.log_gradient(state);
            }

            template <typename State, typename Output>
                requires(
                    requires(const Target &t, const State &s, Output &out) {
                        t.log_gradient(s, out);
                    } || requires(const Target &t, const State &s) { t.log_gradient(s); })
            void log_gradient(const State &state, Output &out) const {
                if constexpr (requires { target_.log_gradient(state, out); })
                    target_.log_gradient(state, out);
                else
                    out = target_.log_gradient(state);
                out *= beta_;
            }

          private:
            const Target &target_;
            double beta_;
        };
    } // namespace detail

    template <typename Target>
    struct target_is_uniform<detail::PowerTarget<Target>> : target_is_uniform<Target> {};

    // Log Metropolis acceptance for adjacent states with untempered log densities.
    // Both states must have positive mass at their current beta. At beta zero
    // even a state with zero target density has positive reference density.
    inline double replica_exchange_log_acceptance(double lower_beta, double upper_beta,
                                                  double lower_log_density,
                                                  double upper_log_density) {
        if (!(0. <= lower_beta && lower_beta < upper_beta && upper_beta <= 1.))
            throw std::invalid_argument(
                "Exchange requires increasing inverse temperatures in [0, 1]");
        detail::checked_tempering_log_density(lower_log_density);
        if (!std::isfinite(upper_log_density) ||
            (lower_beta > 0. && !std::isfinite(lower_log_density)))
            throw std::domain_error("Exchange state has invalid mass at its current temperature");
        if (lower_log_density >= upper_log_density)
            return 0.;
        const double gap = upper_beta - lower_beta;
        // Opposite-sign finite inputs can overflow before scaling.
        if (upper_log_density > 0. &&
            lower_log_density <= upper_log_density - std::numeric_limits<double>::max())
            return gap * lower_log_density - gap * upper_log_density;
        return gap * (lower_log_density - upper_log_density);
    }

    template <typename Kernel> struct TemperingReplica {
        typename Kernel::state_type state;
        Kernel kernel;
        RandomNumberGenerator rng;
    };

    struct ReplicaExchangeStatistics {
        std::size_t attempts = 0;
        std::size_t accepts = 0;
    };

    // States, kernels, workspaces and local RNGs belong to physical replicas.
    // A slot contains only a beta and a physical replica index.
    template <typename Problem, typename Kernel> class NonReversibleParallelTempering {
      public:
        using state_type = typename Kernel::state_type;
        using scalar_type = typename Kernel::scalar_type;

        NonReversibleParallelTempering(Problem problem, TemperatureLadder ladder,
                                       std::vector<TemperingReplica<Kernel>> replicas,
                                       RandomNumberGenerator exchange_rng)
            : problem_(std::move(problem)), ladder_(std::move(ladder)),
              exchange_rng_(std::move(exchange_rng)) {
            static_assert(detail::constant_coordinate_volume<typename Problem::coordinates_type>,
                          "Tempering with a uniform reference requires a constant-volume map");
            if (ladder_.size() < 2 || replicas.size() != ladder_.size())
                throw std::invalid_argument("Expected one replica per inverse temperature");
            validate_streams(replicas);
            replicas_.reserve(size());
            for (std::size_t i = 0; i < size(); ++i)
                replicas_.emplace_back(problem_, std::move(replicas[i]), ladder_[i]);
            replica_at_slot_.resize(size());
            for (std::size_t i = 0; i < size(); ++i)
                replica_at_slot_[i] = i;
            edges_.resize(size() - 1);
            log_acceptance_.resize(size() - 1);
            rejection_sums_.resize(size() - 1);
            trip_phases_.resize(size());
            round_trips_.resize(size());
            reset_statistics();
        }

        NonReversibleParallelTempering(const NonReversibleParallelTempering &) = delete;
        NonReversibleParallelTempering &operator=(const NonReversibleParallelTempering &) = delete;
        NonReversibleParallelTempering(NonReversibleParallelTempering &&) = delete;
        NonReversibleParallelTempering &operator=(NonReversibleParallelTempering &&) = delete;

        std::size_t size() const noexcept { return ladder_.size(); }
        const TemperatureLadder &ladder() const noexcept { return ladder_; }
        const state_type &sampling_state(std::size_t slot) const {
            return replicas_[replica_at_slot_.at(slot)].state;
        }
        Vector<scalar_type> target_state(std::size_t slot) const {
            const auto &state = sampling_state(slot);
            Vector<scalar_type> result(problem_.coordinates().target_dimension(state.size()));
            problem_.coordinates().to_target(state, result);
            return result;
        }
        std::span<const std::size_t> replica_indices() const noexcept { return replica_at_slot_; }
        std::span<const std::size_t> round_trips() const noexcept { return round_trips_; }
        std::span<const ReplicaExchangeStatistics> exchange_statistics() const noexcept {
            return edges_;
        }
        std::vector<double> mean_exchange_rejections() const {
            if (scans_ == 0)
                throw std::logic_error(
                    "Mean exchange rejection requires at least one communication");
            auto means = rejection_sums_;
            for (double &rate : means)
                rate /= static_cast<double>(scans_);
            return means;
        }

        void reset_statistics() noexcept {
            scans_ = 0;
            std::fill(edges_.begin(), edges_.end(), ReplicaExchangeStatistics{});
            std::fill(rejection_sums_.begin(), rejection_sums_.end(), 0.);
            std::fill(trip_phases_.begin(), trip_phases_.end(), TripPhase::awaiting_hot);
            std::fill(round_trips_.begin(), round_trips_.end(), 0);
            trip_phases_[replica_at_slot_.front()] = TripPhase::seeking_cold;
        }

        // Change the ladder between scans. Keep the communication parity and
        // restart diagnostics for the new schedule.
        void set_ladder(TemperatureLadder ladder) {
            if (ladder.size() != size())
                throw std::invalid_argument("New ladder must have the same number of slots");
            ladder_ = std::move(ladder);
            reset_statistics();
        }

        // A caller may explore distinct slots concurrently. Join all such work
        // before accessing the ensemble or starting communication.
        std::size_t explore(std::size_t slot, Index steps = 1) {
            if (steps < 1)
                throw std::invalid_argument("Local exploration requires positive steps");
            return replicas_[replica_at_slot_.at(slot)].explore(problem_, ladder_[slot], steps);
        }

        // Attempt disjoint neighboring exchanges. The first scan attempts odd
        // lower indices, matching annealing.py's one-based scan numbering.
        void communicate() {
            // Compute the mean-rejection observations from the pre-swap states
            // on all edges, including the edges this scan does not attempt.
            for (std::size_t i = 0; i < edges_.size(); ++i) {
                log_acceptance_[i] = replica_exchange_log_acceptance(
                    ladder_[i], ladder_[i + 1], replicas_[replica_at_slot_[i]].log_density,
                    replicas_[replica_at_slot_[i + 1]].log_density);
            }
            std::uniform_real_distribution<double> uniform(0., 1.);
            for (std::size_t i = 0; i < edges_.size(); ++i) {
                const double log_alpha = log_acceptance_[i];
                rejection_sums_[i] -= std::expm1(log_alpha);
                if (i % 2 != next_edge_)
                    continue;
                ++edges_[i].attempts;
                const double u = uniform(exchange_rng_);
                if (log_alpha == 0. || (log_alpha != -std::numeric_limits<double>::infinity() &&
                                        std::log(u) < log_alpha)) {
                    std::swap(replica_at_slot_[i], replica_at_slot_[i + 1]);
                    ++edges_[i].accepts;
                }
            }
            ++scans_;
            next_edge_ ^= 1;
            record_endpoints();
        }

        void scan(Index local_steps = 1, Index reference_steps = 10) {
            explore_all(local_steps, reference_steps);
            communicate();
        }

        // One column per scan at beta one, in target coordinates, after
        // exploration and before communication, as in annealing.py.
        Matrix<scalar_type> draw(Index count, Index local_steps = 1, Index reference_steps = 10) {
            if (count < 0)
                throw std::invalid_argument("Sample count must be nonnegative");
            validate_steps(local_steps, reference_steps);
            Matrix<scalar_type> samples(
                problem_.coordinates().target_dimension(sampling_state(0).size()), count);
            for (Index i = 0; i < count; ++i) {
                explore_all(local_steps, reference_steps);
                auto column = samples.col(i);
                problem_.coordinates().to_target(sampling_state(size() - 1), column);
                communicate();
            }
            return samples;
        }

      private:
        using Workspace = decltype(Kernel::make_workspace(
            std::declval<const Problem &>().geometry(), std::declval<const state_type &>()));

        struct Replica {
            state_type state;
            Kernel kernel;
            RandomNumberGenerator rng;
            Workspace workspace;
            [[no_unique_address]] detail::TargetEvaluation<Problem, state_type> evaluation;
            double log_density;
            double beta;

            Replica(const Problem &problem, TemperingReplica<Kernel> &&input, double initial_beta)
                : state(validated(problem, input)), kernel(std::move(input.kernel)),
                  rng(std::move(input.rng)),
                  workspace(Kernel::make_workspace(problem.geometry(), state)),
                  evaluation(problem, state), log_density(evaluate(problem)), beta(initial_beta) {
                if (beta > 0. && !std::isfinite(log_density))
                    throw std::invalid_argument(
                        "Initial state must have positive mass at its beta");
                if constexpr (requires { kernel.invalidate_state_cache(); })
                    kernel.invalidate_state_cache();
                else if constexpr (requires { kernel.invalidate_target_cache(); })
                    kernel.invalidate_target_cache();
            }

            std::size_t explore(const Problem &problem, double next_beta, Index steps) {
                if (beta != next_beta) {
                    if constexpr (requires { kernel.invalidate_target_cache(); })
                        kernel.invalidate_target_cache();
                    beta = next_beta;
                }
                const auto run = [&](const auto &target) {
                    try {
                        std::size_t accepts = 0;
                        for (Index i = 0; i < steps; ++i) {
                            if (hopsy::step(kernel, state, rng, target, problem.geometry(),
                                            workspace)) {
                                log_density = std::numeric_limits<double>::quiet_NaN();
                                ++accepts;
                            }
                        }
                        return accepts;
                    } catch (...) {
                        kernel.reject(state);
                        throw;
                    }
                };
                const auto accepts =
                    beta == 0. ? run(UniformTarget{})
                               : run(detail::PowerTarget(evaluation.target(problem), beta));
                if (std::isnan(log_density))
                    log_density = evaluate(problem);
                return accepts;
            }

          private:
            static state_type validated(const Problem &problem, TemperingReplica<Kernel> &input) {
                detail::validate_binding(problem, input.kernel, input.state);
                if (input.state != input.kernel.proposal().state)
                    throw std::invalid_argument(
                        "Initial state must match the kernel proposal state");
                return std::move(input.state);
            }
            double evaluate(const Problem &problem) {
                return detail::checked_tempering_log_density(
                    evaluation.target(problem).log_density(state));
            }
        };

        enum class TripPhase { awaiting_hot, seeking_cold, returning_hot };

        void record_endpoints() noexcept {
            const auto cold = replica_at_slot_.back();
            if (trip_phases_[cold] == TripPhase::seeking_cold)
                trip_phases_[cold] = TripPhase::returning_hot;
            const auto hot = replica_at_slot_.front();
            if (trip_phases_[hot] == TripPhase::returning_hot)
                ++round_trips_[hot];
            trip_phases_[hot] = TripPhase::seeking_cold;
        }

        static void validate_steps(Index local_steps, Index reference_steps) {
            if (local_steps < 1 || reference_steps < 1)
                throw std::invalid_argument(
                    "Local and reference exploration require positive steps");
        }
        void explore_all(Index local_steps, Index reference_steps) {
            validate_steps(local_steps, reference_steps);
            for (std::size_t i = 0; i < size(); ++i)
                explore(i, i == 0 ? reference_steps : local_steps);
        }
        void validate_streams(std::vector<TemperingReplica<Kernel>> &replicas) {
            std::vector<decltype(exchange_rng_.stream())> streams;
            streams.reserve(replicas.size() + 1);
            streams.push_back(exchange_rng_.stream());
            for (auto &replica : replicas)
                streams.push_back(replica.rng.stream());
            std::sort(streams.begin(), streams.end());
            if (std::adjacent_find(streams.begin(), streams.end()) != streams.end())
                throw std::invalid_argument("Tempering requires distinct RNG streams");
        }

        const Problem problem_;
        TemperatureLadder ladder_;
        RandomNumberGenerator exchange_rng_;
        std::vector<Replica> replicas_;
        std::vector<std::size_t> replica_at_slot_;
        std::vector<ReplicaExchangeStatistics> edges_;
        std::vector<double> log_acceptance_, rejection_sums_;
        std::vector<TripPhase> trip_phases_;
        std::vector<std::size_t> round_trips_;
        std::size_t scans_ = 0;
        std::size_t next_edge_ = 1;
    };
} // namespace hopsy
#endif
