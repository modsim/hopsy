#ifndef LINE_INTERVAL_H
#define LINE_INTERVAL_H

namespace hopsy {
    template<typename Scalar>
    struct LineInterval {
        using scalar_type = Scalar;
        Scalar lower;
        Scalar upper;
    };
}

#endif //LINE_INTERVAL_H
