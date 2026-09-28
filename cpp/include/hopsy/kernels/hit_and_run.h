#ifndef HIT_AND_RUN_H
#define HIT_AND_RUN_H

namespace hopsy {
    template <typename DirectionSampler, typename ChordSampler>
    class HitAndRunKernel {
        public:

        template <typename Geometry, typename TargetDensity>
        auto propose(const state_type& current_state, RandomNumberGenerator& rng, Geometry geometry, TargetDensity target_density) {
            auto direction = direction_sampler_(rng, current_state, geometry, target_density);
            return chord_sampler_(rng, direction, current_state, geometry, target_density);
        }

        private:
            [[no_unique_address]] DirectionSampler direction_sampler_;
            [[no_unique_address]] ChordSampler chord_sampler_;
    }
}

#endif //HIT_AND_RUN_H
