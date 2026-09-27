#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Core>
#include <pcg/random.hpp>

namespace hopsy {

    template <typename Scalar>
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    template <typename Scalar>
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    using RandomNumberGenerator = pcg64;

} // namespace hopsy

#endif //TYPES_H
