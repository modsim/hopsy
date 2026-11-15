#ifndef BILINEAR_POLYTOPE_H
#define BILINEAR_POLYTOPE_H

#include <vector>


#include <hopsy/geometry/line_interval.h>
#include <hopsy/linalg.h>

namespace hopsy {
    template <typename Scalar=double>
    class BilinearPolytope {
        using scalar_type = Scalar;
        using vector_type = Vector<Scalar>;
        using matrix_type = Matrix<Scalar>;
        using line_domain_type = std::vector<LineInterval<Scalar>>;
    };
}

#endif //BILINEAR_POLYTOPE_H
