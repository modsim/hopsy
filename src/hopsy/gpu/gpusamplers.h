#pragma once

// Declarations to GPU samplers that will be exposed to Python
// Actual implementation in 

#include "dvector.h"
#include "dmatrix.h"

namespace hopsy {
    namespace GPU {
        DMatrix<double> WarumUp(DMatrix<double>& A_d,
                                DVector<double>& b_d,
                                DVector<double>& x0_d,
                                int nwarmup,
                                int nchains,
                                int tpb_rd = -1,
                                int tpb_ss = -1);

        DMatrix<double> MatrixHitAndRun     (   DMatrix<double>& A_d,
                                                DVector<double>& b_d,
                                                DMatrix<double>& X_d,
                                                int nspc,
                                                int thinning,
                                                int nchains,
                                                int tpb_rd = -1,
                                                int tpb_ss = -1);

        DMatrix<double> CoordinateHitAndRun(    DMatrix<double>& A_d,
                                                DVector<double>& b_d,
                                                DMatrix<double>& X_d,
                                                int nspc,
                                                int thinning,
                                                int nchains,
                                                int tpb_ss = -1);

        DVector<double> rhat(const DMatrix<double>& samples_d,
                             int nspc,
                             int nchains,
                             int dim);
    }

}