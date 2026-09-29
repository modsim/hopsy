#ifndef LINEAR_POLYTOPE_H
#define LINEAR_POLYTOPE_H

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <utility>

#include <hopsy/linalg.h>

namespace hopsy {

    template <typename Scalar = double>
    class LinearPolytope {
    public:
        using scalar_type = Scalar;
        using vector_type = Vector<Scalar>;
        using matrix_type = MatrixX<Scalar>;
        using state_type  = vector_type;
        struct Workspace;
        using workspace_type = Workspace;
        using line_domain_type = std::array<LineInterval<Scalar>, 1>;

        LinearPolytope(matrix_type A, vector_type b)
            : A_(std::move(A)),
              b_(std::move(b))
        {
            if (A_.rows() != b_.size()) {
                throw std::invalid_argument(
                    "LinearPolytope: number of rows in A must match the size of b."
                );
            }

            if (A_.cols() == 0) {
                throw std::invalid_argument(
                    "LinearPolytope: A must have at least one column."
                );
            }
        }

        [[nodiscard]] Index num_dimension() const noexcept {
            return A_.cols()
        };

        [[nodiscard]] Index num_constraints() const noexcept {
            return A_.rows();
        };

        [[nodiscard]] bool contains(
            const state_type& x,
            Scalar tolerance = Scalar{0}
        ) const {
            return ((b-A*x).array() < tolerance).all();
        }

        [[nodiscard]] LineBounds<Scalar> line_bounds(
            const state_type& x,
            const state_type& direction
        ) const {

        }

        [[nodiscard]] const matrix_type& A() const noexcept {
            return A_;
        }

        [[nodiscard]] const vector_type& b() const noexcept {
            return b_;
        }

        struct Workspace {
            vector_type slacks;
        };


    private:
        matrix_type A_;
        vector_type b_;
    };

    struct Workspace {
        vector_type slacks;
    };
}

#endif //LINEAR_POLYTOPE_H
