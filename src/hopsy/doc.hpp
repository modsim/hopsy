#ifndef HOPSY_DOC_HPP
#define HOPSY_DOC_HPP

#include <string>

namespace hopsy {
    namespace doc {
        extern const char* numberOfThreads;

        namespace Model {
            extern const char* base;
            extern const char* __init__;
            extern const char* computeNegativeLogLikelihood;
            extern const char* computeLogLikelihoodGradient;
            extern const char* computeExpectedFisherInformation;
        }

        namespace DegenerateGaussian {
            extern const char* base;
            extern const char* __init__;
            extern const char* computeNegativeLogLikelihood;
            extern const char* computeLogLikelihoodGradient;
            extern const char* computeExpectedFisherInformation;
        }

        namespace GaussianMixtureModel {
            extern const char* base;
            extern const char* __init__;
            extern const char* computeNegativeLogLikelihood;
            extern const char* computeLogLikelihoodGradient;
            extern const char* computeExpectedFisherInformation;
        }

        namespace Mixture {
            extern const char* base;
            extern const char* __init__;
            extern const char* computeNegativeLogLikelihood;
            extern const char* computeLogLikelihoodGradient;
            extern const char* computeExpectedFisherInformation;
        }

        namespace MultivariateGaussianModel {
            extern const char* base;
            extern const char* __init__;
            extern const char* computeNegativeLogLikelihood;
            extern const char* computeLogLikelihoodGradient;
            extern const char* computeExpectedFisherInformation;
        }

        namespace PyModel {
            extern const char* base;
            extern const char* __init__;
            extern const char* computeNegativeLogLikelihood;
            extern const char* computeLogLikelihoodGradient;
            extern const char* computeExpectedFisherInformation;
        }

        namespace Rosenbrock {
            extern const char* base;
            extern const char* __init__;
            extern const char* computeNegativeLogLikelihood;
            extern const char* computeLogLikelihoodGradient;
            extern const char* computeExpectedFisherInformation;
        }

        namespace UniformModel {
            extern const char* base;
            extern const char* __init__;
            extern const char* computeNegativeLogLikelihood;
            extern const char* computeLogLikelihoodGradient;
            extern const char* computeExpectedFisherInformation;
        }

        namespace Problem {
            extern const char* base;
            extern const char* __init__;
            extern const char* A;
            extern const char* b;
            extern const char* model;
            extern const char* startingPoint;
            extern const char* unroundingTransformation;
            extern const char* unroundingShift;
        }

        extern const char* addBoxConstraintsToProblem;
        extern const char* computeChebyshevCenter;
        extern const char* round;

        namespace Run {
            extern const char* base;
            extern const char* __init__;
            extern const char* init;
            extern const char* sample;
            extern const char* data;
            extern const char* problem;
            extern const char* startingPoints;
            extern const char* numberOfChains;
            extern const char* numberOfSamples;
            extern const char* thinning;
            extern const char* stepSize;
            extern const char* fisherWeight;
            extern const char* randomSeed;
            extern const char* samplingUntilConvergence;
            extern const char* convergenceThreshold;
            extern const char* maxRepetitions;
        }

        namespace AcceptanceRateTarget {
            extern const char* base;
            extern const char* __init__;
            extern const char* acceptanceRate;
        }

        namespace ExpectedSquaredJumpDistanceTarget {
            extern const char* base;
            extern const char* __init__;
            extern const char* lags;
            extern const char* considerTimeCost;
        }

        extern const char* tune;

        namespace Data {
            extern const char* base;
            extern const char* __init__;
            extern const char* flatten;
            extern const char* reset;
            extern const char* subsample;
            extern const char* thin;
            extern const char* write;
            extern const char* acceptanceRates;
            extern const char* negativeLogLikelihood;
            extern const char* parameterNames;
            extern const char* states;
            extern const char* timestamps;
            extern const char* numberOfChains;
            extern const char* numberOfSamples;
            extern const char* dims;
            extern const char* shape;
            extern const char* __getitem__;
        }

        extern const char* computeAcceptanceRate;
        extern const char* computeEffectiveSampleSize;
        extern const char* computeExpectedSquaredJumpDistance;
        extern const char* computePotentialScaleReductionFactor;
        extern const char* computeTotalTimeTaken;
    }
}

#endif // HOPSY_DOC_HPP