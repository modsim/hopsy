#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Core>

namespace hopsy {

    template <typename Scalar>
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    template <typename Scalar>
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    using Index = Eigen::Index;

} // namespace hopsy

#endif //TYPES_H
