#ifndef LINE_INTERVAL_H
#define LINE_INTERVAL_H

namespace hopsy {

    template <typename Scalar>
    struct LineInterval {
        Scalar lower;
        Scalar upper;
    };

}

#endif //LINE_INTERVAL_H
