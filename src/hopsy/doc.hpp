#ifndef HOPSY_DOC_HPP
#define HOPSY_DOC_HPP

#include <string>

namespace hopsy {
    namespace doc {
        extern const char* base;

        struct RandomNumberGenerator {
            static const char* base;
            static const char* __init__;
            static const char* __call__;
        };

        struct Uniform {
            static const char* base;
            static const char* __init__;
            static const char* __call__;
        };

        struct Normal {
            static const char* base;
            static const char* __init__;
            static const char* __call__;
        };

        struct Model {
            static const char* base;
            static const char* __init__;
            static const char* logDensity;
            static const char* logGradient;
            static const char* logCurvature;
            static const char* computeNegativeLogLikelihood;
            static const char* computeLogLikelihoodGradient;
            static const char* computeExpectedFisherInformation;
        };

        struct Gaussian {
            static const char* base;
            static const char* __init__;
            static const char* mean;
            static const char* covariance;
            static const char* inactives;
            static const char* logDensity;
            static const char* logGradient;
            static const char* logCurvature;
            static const char* computeNegativeLogLikelihood;
            static const char* computeLogLikelihoodGradient;
            static const char* computeExpectedFisherInformation;
        };

        struct GaussianMixtureModel {
            static const char* base;
            static const char* __init__;
            static const char* logDensity;
            static const char* logGradient;
            static const char* logCurvature;
            static const char* computeNegativeLogLikelihood;
            static const char* computeLogLikelihoodGradient;
            static const char* computeExpectedFisherInformation;
        };

        struct Mixture {
            static const char* base;
            static const char* __init__;
            static const char* components;
            static const char* weights;
            static const char* logDensity;
            static const char* logGradient;
            static const char* logCurvature;
            static const char* computeNegativeLogLikelihood;
            static const char* computeLogLikelihoodGradient;
            static const char* computeExpectedFisherInformation;
        };

        struct MultivariateGaussianModel {
            static const char* base;
            static const char* __init__;
            static const char* logDensity;
            static const char* logGradient;
            static const char* logCurvature;
            static const char* computeNegativeLogLikelihood;
            static const char* computeLogLikelihoodGradient;
            static const char* computeExpectedFisherInformation;
        };

        struct PyModel {
            static const char* base;
            static const char* __init__;
            static const char* model;
            static const char* logDensity;
            static const char* logGradient;
            static const char* logCurvature;
            static const char* computeNegativeLogLikelihood;
            static const char* computeLogLikelihoodGradient;
            static const char* computeExpectedFisherInformation;
        };

        struct Rosenbrock {
            static const char* base;
            static const char* __init__;
            static const char* scale;
            static const char* shift;
            static const char* logDensity;
            static const char* logGradient;
            static const char* logCurvature;
            static const char* computeNegativeLogLikelihood;
            static const char* computeLogLikelihoodGradient;
            static const char* computeExpectedFisherInformation;
        };

        struct UniformModel {
            static const char* base;
            static const char* __init__;
            static const char* logDensity;
            static const char* logGradient;
            static const char* logCurvature;
            static const char* computeNegativeLogLikelihood;
            static const char* computeLogLikelihoodGradient;
            static const char* computeExpectedFisherInformation;
        };

        struct Problem {
            static const char* base;
            static const char* __init__;
            static const char* A;
            static const char* b;
            static const char* model;
            static const char* startingPoint;
            static const char* transformation;
            static const char* shift;
            static const char* slacks;
        };

        struct Proposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* boundaryCushion;
            static const char* epsilon;
            static const char* stepSize;
            static const char* warmUp;
        };

        struct AdaptiveMetropolisProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* boundaryCushion;
            static const char* epsilon;
            static const char* warmUp;
        };

        struct BallWalkProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* stepSize;
        };

        struct BilliardAdaptiveMetropolisProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* maxReflections;
            static const char* boundaryCushion;
            static const char* epsilon;
            static const char* stepSize;
            static const char* warmUp;
        };

        struct BilliardMALAProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* maxReflections;
            static const char* stepSize;
        };

        struct BilliardWalkProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* maxReflections;
            static const char* stepSize;
        };

        struct CSmMALAProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* fisherWeight;
            static const char* stepSize;
        };

        struct DikinWalkProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* boundaryCushion;
            static const char* stepSize;
        };

        struct GaussianCoordinateHitAndRunProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* stepSize;
        };

        struct GaussianHitAndRunProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* stepSize;
        };

        struct GaussianProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* stepSize;
        };

        struct ReversibleJumpProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getStepSize;
            static const char* getParameter;
            static const char* setParameter;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;

            static const char* modelJumpProbability;
            static const char* activationProbability;
            static const char* deactivationProbability;
        };

        struct PyProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;
        };

        struct TruncatedGaussianProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;
        };

        struct UniformCoordinateHitAndRunProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;
        };

        struct UniformHitAndRunProposal {
            static const char* base;
            static const char* __init__;

            static const char* propose;
            static const char* acceptProposal;
            static const char* logAcceptanceProbability;
            static const char* proposal;
            static const char* state;
            static const char* getParameter;
            static const char* setParameter;
            static const char* getStepSize;
            static const char* name;
            static const char* stateNegativeLogLikelihood;
            static const char* proposalNegativeLogLikelihood;
            static const char* hasNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* proposalLogDensity;
            static const char* hasLogDensity;
            static const char* copyProposal;
        };

        struct MarkovChain {
            static const char* base;
            static const char* __init__;
            static const char* draw;
            static const char* state;
            static const char* model;
            static const char* proposal;
            static const char* stateNegativeLogLikelihood;
            static const char* stateLogDensity;
            static const char* exchangeAttemptProbability;
        };

        struct TuningTarget {
            static const char* base;
        };

        struct AcceptanceRateTarget {
            static const char* base;
            static const char* __init__;
            static const char* markovChains;
            static const char* numberOfTestSamples;
            static const char* acceptanceRate;
            static const char* __call__;
        };

        struct ExpectedSquaredJumpDistanceTarget {
            static const char* base;
            static const char* __init__;
            static const char* markovChains;
            static const char* numberOfTestSamples;
            static const char* lags;
            static const char* considerTimeCost;
            static const char* __call__;
        };

        struct PyTuningTarget {
            static const char* base;
            static const char* __init__;
            static const char* __call__;
        };

        struct ThompsonSampling {
            static const char* base;
            static const char* __init__;
            static const char* posteriorUpdateIterations;
            static const char* pureSamplingIterations;
            static const char* iterationsForConvergence;
            static const char* posteriorUpdateIterationsNeeded;
            static const char* stepSizeGridSize;
            static const char* stepSizeLowerBound;
            static const char* stepSizeUpperBound;
            static const char* smoothingLength;
            static const char* randomSeed;
            static const char* recordData;
        };

        extern const char* tune;

        extern const char* computeAcceptanceRate;
        extern const char* computeEffectiveSampleSize;
        extern const char* computeExpectedSquaredJumpDistance;
        extern const char* computePotentialScaleReductionFactor;
        extern const char* computeTotalTimeTaken;
    } // namespace doc
} // namespace hopsy

#endif // HOPSY_DOC_HPP
