#ifndef DIRECTION_H
#define DIRECTION_H

#include <hopsy/linalg.h>
#include <hopsy/random_number_generator.h>

namespace hopsy {
    struct CoordinateDirection {
        Index coordinate;
    };

    class RandomCoordinateDirectectionGenerator {
    public:
        [[nodiscard] CoordinateDirection generate(RandomNumberGenerator& rng) const {
        }
    };

    class CyclicCoordinateDirectectionGenerator {
    public:
        [[nodiscard] CoordinateDirection generate(RandomNumberGenerator& rng) const {
        }

    private:
        Index next_coordinate_{0};
    };

    class RandomDirectectionGenerator {
    public:
    };

}

#endif //DIRECTION_H
