#ifndef HOPSY_HPP
#define HOPSY_HPP

#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

#include <Eigen/Core>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "../../extern/hops/src/hops/hops.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hopsy {
    using VectorType = hops::VectorType;
    using MatrixType = hops::MatrixType;

    using Model = hops::Model;
	using Proposal = hops::Proposal;
} // namespace hopsy

    //using RandomNumberGenerator = hops::RandomNumberGenerator;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Model);

namespace hopsy {
    struct RandomNumberGenerator {
        std::optional<unsigned int> seed;
        std::optional<unsigned int> stream;
        hops::RandomNumberGenerator rng;

        RandomNumberGenerator() {
            // 
        }

        RandomNumberGenerator(unsigned int seed) :
            seed(seed),
            rng(seed) {
            // 
        }

        RandomNumberGenerator(unsigned int seed, unsigned int stream) :
            seed(seed),
            stream(stream),
            rng(seed, stream) {
            // 
        }

        unsigned int operator()() {
            return rng();
        }

        std::string __repr__() const {
            std::string repr = "hopsy.RandomNumberGenerator("; 
            repr += (seed ? "seed=" + std::to_string(*seed) : "");
            repr += (stream ? ", stream=" + std::to_string(*stream) : "");
            repr += ")";
            return repr;
        }
    };

    template<typename ModelBase = Model>
    class ModelTrampoline : public ModelBase, public py::trampoline_self_life_support {
	public:
		/* Inherit the constructors */
		using ModelBase::ModelBase;

		double computeNegativeLogLikelihood(const VectorType& x) const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				double,     /* Return type */
				ModelBase,       /* Parent class */
                "compute_negative_log_likelihood",
				computeNegativeLogLikelihood,  /* Name of function in C++ (must match Python name) */
                x
			);
        }

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType& x) const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::optional<MatrixType>,     /* Return type */
				ModelBase,       /* Parent class */
                "compute_expected_fisher_information",
				computeExpectedFisherInformation,  /* Name of function in C++ (must match Python name) */
                x
			);
		}

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType& x) const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::optional<VectorType>,     /* Return type */
				ModelBase,       /* Parent class */
                "compute_log_likelihood_gradient",
				computeLogLikelihoodGradient,  /* Name of function in C++ (must match Python name) */
                x
			);
		}

        std::unique_ptr<Model> deepCopy() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::unique_ptr<Model>,     // Return type 
				ModelBase,                  // Parent class
                "deepcopy",                 // Python function name
				deepCopy                    // C++ function name
            );
		}
	};

    class PyModel : public Model {
	public:
        using MatrixType = MatrixType;
        using VectorType = VectorType;

		PyModel(py::object pyObj) : pyObj(std::move(pyObj)) {};

		double computeNegativeLogLikelihood(const VectorType& x) const override {
			return pyObj.attr("compute_negative_log_likelihood")(x).cast<double>();
		}

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType& x) const override {
			return pyObj.attr("compute_expected_fisher_information")(x).cast<std::optional<MatrixType>>();
		}

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType& x) const override {
			return pyObj.attr("compute_log_likelihood_gradient")(x).cast<std::optional<VectorType>>();
		}

        std::unique_ptr<Model> deepCopy() const override {
			return std::make_unique<PyModel>(PyModel(pyObj));
		}

        std::string __repr__() const {
            std::string repr{""};
            repr += "PyModel(";
            repr += "udm=" + py::cast<std::string>(pyObj.attr("__repr__")());
            repr += ")";
            return repr;
        }

		py::object pyObj;
	};

    template<typename ProposalBase = Proposal>
	class ProposalTrampoline : public ProposalBase {
	public:
		/* Inherit the constructors */
		using ProposalBase::ProposalBase;

        std::pair<double, VectorType> propose(hops::RandomNumberGenerator& rng) override {
			PYBIND11_OVERRIDE_PURE(
				PYBIND11_TYPE(std::pair<double, VectorType>),
				ProposalBase,
				propose,
                rng
			);
        }

        VectorType acceptProposal() override {
			PYBIND11_OVERRIDE_PURE_NAME(
				VectorType,     /* Return type */
				ProposalBase,       /* Parent class */
                "accept_proposal",
				acceptProposal  /* Name of function in C++ (must match Python name) */
			);
        }

        VectorType getProposal() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				VectorType,
				ProposalBase,
                "get_proposal",
				getProposal
			);
        }

        VectorType getState() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				VectorType,
				ProposalBase,
                "get_state",
				getState
			);
        }

        void setState(VectorType state) override {
			PYBIND11_OVERRIDE_PURE_NAME(
				void,
				ProposalBase,
                "set_state",
				setState,
                state
			);
        }

        std::optional<double> getStepSize() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::optional<double>,
				ProposalBase,
                "get_stepsize",
				getStepSize
			);
        }

        void setStepSize(double stepSize) override {
			PYBIND11_OVERRIDE_PURE_NAME(
				void,
				ProposalBase,
                "set_stepsize",
				setStepSize,
                stepSize
			);
        }

        bool hasStepSize() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				bool,
				ProposalBase,
                "has_stepsize",
				hasStepSize
			);
        }

        std::string getProposalName() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::string,
				ProposalBase,
                "get_proposal_name",
				getProposalName,
			);
        }

        double getNegativeLogLikelihood() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				double,
				ProposalBase,
                "get_negative_log_likelihood",
                getNegativeLogLikelihood
			);
        }

        std::unique_ptr<Proposal> deepCopy() const override {
            return nullptr;
        }
	};

    class PyProposal : public Proposal {
    public:
        using StateType = VectorType;

        PyProposal() = default;

		PyProposal(py::object pyObj) : pyObj(std::move(pyObj)) {};

        std::pair<double, VectorType> propose(hops::RandomNumberGenerator& rng) override {
            return pyObj.attr("propose")(rng).cast<std::pair<double, VectorType>>();
        }

        VectorType acceptProposal() override {
            return pyObj.attr("accept_proposal")().cast<VectorType>();
        }

        VectorType getProposal() const override {
            return pyObj.attr("get_proposal")().cast<VectorType>();
        }

        VectorType getState() const override {
            return pyObj.attr("get_state")().cast<VectorType>();
        }

        void setState(VectorType newState) override {
            pyObj.attr("set_state")(newState);
        }

        std::optional<double> getStepSize() const override {
            return pyObj.attr("get_stepsize")().cast<std::optional<double>>();
        }

        void setStepSize(double newStepSize) override {
            pyObj.attr("set_stepsize")(newStepSize);
        }

        bool hasStepSize() const override {
            return pyObj.attr("has_stepsize")().cast<bool>();
        }

        std::string getProposalName() const override {
            return pyObj.attr("get_name")().cast<std::string>();
        }

        double getNegativeLogLikelihood() const override {
            return pyObj.attr("get_negative_log_likelihood")().cast<double>();
        }
	private:
		py::object pyObj;
    };


    class PyTuningTarget {
    public:
        PyTuningTarget() = default;

		PyTuningTarget(py::object pyObj) : pyObj(std::move(pyObj)) {};

        std::tuple<double, double> operator() (const VectorType x) {
            return pyObj.attr("__call__")(x).cast<std::tuple<double, double>>();
        }

        std::string getName() {
            try {
                return pyObj.attr("get_name")().cast<std::string>();
            } catch (...) {
                return "";
            }
        }

        std::vector<std::shared_ptr<hops::MarkovChain>> markovChain;
        std::vector<hops::RandomNumberGenerator>* randomNumberGenerator;
        unsigned long numberOfTestSamples;

	private:
		py::object pyObj;
    };

    using Normal = std::normal_distribution<double>;
    using Uniform = std::uniform_real_distribution<double>;

    using DegenerateGaussian = hops::DegenerateGaussian;
    using Mixture = hops::Mixture;
    //typedef hops::MultivariateGaussianModel<MatrixType, VectorType> MultivariateGaussianModel;
    using Rosenbrock =  hops::Rosenbrock;

    using AdaptiveMetropolis = hops::AdaptiveMetropolisProposal<MatrixType, VectorType>;
    //using BallWalk = hops::BallWalk;
    using CoordinateHitAndRun = hops::CoordinateHitAndRunProposal<MatrixType, VectorType>;
    using CSmMALA = hops::CSmMALAProposal<MatrixType, VectorType>;
    using DikinWalk = hops::DikinProposal<MatrixType, VectorType>;
    using Gaussian = hops::GaussianProposal<MatrixType, VectorType>;
    using HitAndRun = hops::HitAndRunProposal<MatrixType, VectorType>;

    using MarkovChain = hops::MarkovChainPrototypeImpl;

	//typedef hops::Run<DegenerateMultivariateGaussianModel> DegenerateMultivariateGaussianRun;
    //typedef hops::Run<GaussianMixtureModel> GaussianMixtureRun;
    //typedef hops::Run<MixtureModel> MixtureRun;
    ////typedef hops::Run<MultivariateGaussianModel> MultivariateGaussianRun;
    //typedef hops::Run<PyModel> PyRun;
    //typedef hops::Run<RosenbrockModel> RosenbrockRun;
    //typedef hops::Run<UniformModel> UniformRun;

	//typedef hops::RunBase<DegenerateMultivariateGaussianModel, PyProposal> DegenerateMultivariateGaussianPyProposalRun;
    //typedef hops::RunBase<GaussianMixtureModel, PyProposal> GaussianMixturePyProposalRun;
    //typedef hops::RunBase<MixtureModel, PyProposal> MixturePyProposalRun;
    ////typedef hops::RunBase<MultivariateGaussianModel, PyProposal> MultivariateGaussianPyProposalRun;
    //typedef hops::RunBase<PyModel, PyProposal> PyPyProposalRun;
    //typedef hops::RunBase<RosenbrockModel, PyProposal> RosenbrockPyProposalRun;
    //typedef hops::RunBase<UniformModel, PyProposal> UniformPyProposalRun;


	//template<typename T>
	//hops::Run<T> createRun(const hops::Problem<T>& t, 
    //                       std::string chainTypeString = "HitAndRun", 
    //                       unsigned long numberOfSamples = 1000, 
    //                       unsigned long numberOfChains = 1,
    //                       std::vector<VectorType> startingPoints = std::vector<VectorType>(),
    //                       unsigned long thinning = 1,
    //                       double stepSize = 1,
    //                       double fisherWeight = 0.5,
    //                       unsigned long randomSeed = 0,
    //                       bool sampleUntilConvergence = false,
    //                       double diagnosticsThreshold = 1.05,
    //                       unsigned long maxRepetitions = 10) {
	//	hops::MarkovChainType chainType;
	//	if (chainTypeString == "AdaptiveMetropolis" || chainTypeString == "AM") {
	//		chainType = hops::MarkovChainType::BallWalk;
    //    } else if (chainTypeString == "BallWalk" || chainTypeString == "BW") {
	//		chainType = hops::MarkovChainType::BallWalk;
	//	} else if (chainTypeString == "CoordinateHitAndRun" || chainTypeString == "CHR") {
	//		chainType = hops::MarkovChainType::CoordinateHitAndRun;
	//	} else if (chainTypeString == "DikinWalk" || chainTypeString == "DW") {
	//		chainType = hops::MarkovChainType::DikinWalk;
	//	} else if (chainTypeString == "Gaussian" || chainTypeString == "GRW") {
	//		chainType = hops::MarkovChainType::Gaussian;
	//	} else if (chainTypeString == "HitAndRun" || chainTypeString == "HR") {
	//		chainType = hops::MarkovChainType::HitAndRun;
	//	} else {
    //        throw std::invalid_argument("Proposal algorithm not known.");
    //    }

    //    // initialize missing starting points with chebyshev center
    //    if (startingPoints.size() < numberOfChains) {
    //        VectorType chebyshevCenter = computeChebyshevCenter(t);
    //        for (size_t i = startingPoints.size(); i < numberOfChains; ++i) {
    //            startingPoints.push_back(chebyshevCenter);
    //        }
    //    }

    //    auto run = hops::Run<T>(t, chainType, numberOfSamples, numberOfChains);
    //    run.setStartingPoints(startingPoints);
    //    run.setStartingPoints(startingPoints);
    //    run.setThinning(thinning);
    //    run.setStepSize(stepSize);
    //    run.setFisherWeight(fisherWeight);
    //    run.setRandomSeed(randomSeed);
    //    run.setSamplingUntilConvergence(sampleUntilConvergence);
    //    run.setConvergenceThreshold(diagnosticsThreshold);
    //    run.setMaxRepetitions(maxRepetitions);
    //    return run;
	//}

	//template<typename T>
	//hops::RunBase<T, PyProposal> createRunFromPyProposal(const hops::Problem<T>& t, 
    //                                                     PyProposal proposal, 
    //                                                     unsigned long numberOfSamples = 1000, 
    //                                                     unsigned long numberOfChains = 1,
    //                                                     std::vector<VectorType> startingPoints = std::vector<VectorType>(),
    //                                                     unsigned long thinning = 1,
    //                                                     double stepSize = 1,
    //                                                     double fisherWeight = 0.5,
    //                                                     unsigned long randomSeed = 0,
    //                                                     bool sampleUntilConvergence = false,
    //                                                     double diagnosticsThreshold = 1.05,
    //                                                     unsigned long maxRepetitions = 10) {
    //    // initialize missing starting points with chebyshev center
    //    if (startingPoints.size() < numberOfChains) {
    //        VectorType chebyshevCenter = computeChebyshevCenter(t);
    //        for (size_t i = startingPoints.size(); i < numberOfChains; ++i) {
    //            startingPoints.push_back(chebyshevCenter);
    //        }
    //    }

    //    auto run = hops::RunBase<T, PyProposal>(t, proposal, numberOfSamples, numberOfChains);
    //    run.setStartingPoints(startingPoints);
    //    run.setThinning(thinning);
    //    run.setStepSize(stepSize);
    //    run.setFisherWeight(fisherWeight);
    //    run.setRandomSeed(randomSeed);
    //    run.setSamplingUntilConvergence(sampleUntilConvergence);
    //    run.setConvergenceThreshold(diagnosticsThreshold);
    //    run.setMaxRepetitions(maxRepetitions);
    //    return run;
	//}


	//template<typename T>
	//hops::RunBase<T, PyProposal> createRunFromPyObject(const hops::Problem<T>& t, 
    //                                                   py::object pyObj, 
    //                                                   unsigned long numberOfSamples = 1000, 
    //                                                   unsigned long numberOfChains = 1,
    //                                                   std::vector<VectorType> startingPoints = std::vector<VectorType>(),
    //                                                   unsigned long thinning = 1,
    //                                                   double stepSize = 1,
    //                                                   double fisherWeight = 0.5,
    //                                                   unsigned long randomSeed = 0,
    //                                                   bool sampleUntilConvergence = false,
    //                                                   double diagnosticsThreshold = 1.05,
    //                                                   unsigned long maxRepetitions = 10) {
    //    return createRunFromPyProposal(t, 
    //                                   PyProposal(pyObj), 
    //                                   numberOfSamples, 
    //                                   numberOfChains, 
    //                                   startingPoints, 
    //                                   thinning, 
    //                                   stepSize, 
    //                                   fisherWeight, 
    //                                   randomSeed, 
    //                                   sampleUntilConvergence, 
    //                                   diagnosticsThreshold, 
    //                                   maxRepetitions);
	//}

    //std::pair<MatrixType, VectorType> addBoxConstraintsToMatrixVector(const MatrixType& A, const VectorType& b, const VectorType& lowerBounds, const VectorType& upperBounds) {
    //    MatrixType newA(A.rows() + 2*A.cols(), A.cols());
    //    VectorType newB(newA.rows());

    //    newA << A, MatrixType::Identity(A.cols(), A.cols()), -MatrixType::Identity(A.cols(), A.cols());
    //    newB << b, upperBounds, -lowerBounds;

    //    return {newA, newB};
    //}

    //std::pair<MatrixType, VectorType> addBoxConstraintsToMatrixVector(const MatrixType& A, const VectorType& b, double lowerBound, double upperBound) {
    //    VectorType lowerBounds = lowerBound * VectorType::Ones(A.cols());
    //    VectorType upperBounds = upperBound * VectorType::Ones(A.cols());
    //    return addBoxConstraintsToMatrixVector(A, b, lowerBounds, upperBounds);
    //}

    //template<typename Problem>
    //Problem addBoxConstraintsToProblem(const Problem& problem, const VectorType& lowerBounds, const VectorType& upperBounds) {
    //    auto[A, b] = addBoxConstraintsToMatrixVector(problem.getA(), problem.getB(), lowerBounds, upperBounds);
    //    Problem newProblem(problem);
    //    newProblem.setA(A);
    //    newProblem.setB(b);
    //    return newProblem;
    //}

    //template<typename Problem>
    //Problem addBoxConstraintsToProblem(const Problem& problem, double lowerBound, double upperBound) {
    //    VectorType lowerBounds = lowerBound * VectorType::Ones(problem.getA().cols());
    //    VectorType upperBounds = upperBound * VectorType::Ones(problem.getA().cols());
    //    return addBoxConstraintsToProblem(problem, lowerBounds, upperBounds);
    //}


    //py::dict tuningDataToDict(const hops::TuningData& tuningData) {
    //    py::dict dict;
    //    dict["method"] = tuningData.method;
    //    dict["target"] = tuningData.target;
    //    dict["n_samples"] = tuningData.totalNumberOfSamples;
    //    dict["n_iterations"] = tuningData.totalNumberOfIterations;
    //    dict["tuned_f"] = tuningData.tunedObjectiveValue;
    //    dict["time_taken"] = tuningData.totalTimeTaken;
    //    dict["data"] = tuningData.data;
    //    dict["posterior"] = tuningData.posterior;
    //    return dict;
    //}

    //template<typename Run, typename Target>
    //std::tuple<VectorType, py::dict> tune(const Run& run, 
    //                                           Target& target,
    //                                           std::string method,
    //                                           size_t numberOfTestSamples,
    //                                           size_t posteriorUpdateIterations,
    //                                           size_t pureSamplingIterations,
    //                                           size_t iterationsForConvergence,
    //                                           size_t stepSizeGridSize,
    //                                           double stepSizeLowerBound,
    //                                           double stepSizeUpperBound,
    //                                           double smoothingLength,
    //                                           size_t randomSeed,
    //                                           bool recordData) {
    //    if (method == "ThompsonSampling" || method == "TS") {
    //        hops::ThompsonSamplingTuner::param_type parameters{
    //                  numberOfTestSamples,
    //                  posteriorUpdateIterations,
    //                  pureSamplingIterations,
    //                  iterationsForConvergence,
    //                  stepSizeGridSize,
    //                  stepSizeLowerBound,
    //                  stepSizeUpperBound,
    //                  smoothingLength,
    //                  randomSeed,
    //                  recordData};
    //        auto[tunedParameters, tuningData] = tune(run, parameters, target);
    //        return std::make_tuple(tunedParameters, tuningDataToDict(tuningData));
    //    } else {
    //        throw std::invalid_argument(method + std::string(" does not name a known tuning method."));
    //    }
    //}

    //using AcceptanceRateTarget = hops::AcceptanceRateTarget<VectorType>;
    //using ExpectedSquaredJumpDistanceTarget = hops::ExpectedSquaredJumpDistanceTarget<VectorType, MatrixType>;

    ///*
    // *
    // *  Data __getitem__
    // *
    // */

    ////using SimpleData = std::vector<         // chains
    ////    std::tuple<
    ////        std::vector<double>,            // acceptance rates
    ////        std::vector<double>,            // negative log likelihood
    ////        std::vector<VectorType>,   // states
    ////        std::vector<long>               // timestamps
    ////    >>;

    //hops::Data getDataItem (const hops::Data& data, const py::slice& slice) {
    //    py::ssize_t numberOfChains = static_cast<py::ssize_t>(data.chains.size());
    //    py::ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
    //    if (!slice.compute(numberOfChains, &start, &stop, &step, &slicelength))
    //        throw py::error_already_set();
    //    size_t istart = static_cast<size_t>(start);
    //    size_t istop  = static_cast<size_t>(stop);
    //    size_t istep  = static_cast<size_t>(step);

    //    hops::Data newData{};
    //    for (size_t i = istart; i < istop; i += istep) {
    //        newData.chains.push_back(
    //                hops::ChainData(
    //                    data.chains[i].getAcceptanceRates(), 
    //                    data.chains[i].getNegativeLogLikelihood(), 
    //                    data.chains[i].getStates(), 
    //                    data.chains[i].getTimestamps()
    //                ));
    //    }

    //    return newData;
    //}

    //hops::Data getDataItem (const hops::Data& data, const std::tuple<py::slice, py::slice>& slices) {
    //    py::ssize_t numberOfChains = static_cast<py::ssize_t>(data.chains.size());

    //    py::slice chainSlice = std::get<0>(slices);
    //    py::slice stateSlice = std::get<1>(slices);

    //    py::ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
    //    if (!chainSlice.compute(numberOfChains, &start, &stop, &step, &slicelength))
    //        throw py::error_already_set();
    //    size_t istart = static_cast<size_t>(start);
    //    size_t istop  = static_cast<size_t>(stop);
    //    size_t istep  = static_cast<size_t>(step);

    //    py::ssize_t maxNumberOfStates = static_cast<py::ssize_t>(data.chains[0].getStates().size());
    //    for (const auto& chain : data.chains) {
    //        maxNumberOfStates = std::max(maxNumberOfStates, static_cast<py::ssize_t>(chain.getStates().size()));
    //    }
    //    start = 0, stop = 0, step = 0, slicelength = 0;
    //    if (!stateSlice.compute(maxNumberOfStates, &start, &stop, &step, &slicelength))
    //        throw py::error_already_set();
    //    size_t jstart = static_cast<size_t>(start);
    //    size_t jstop  = static_cast<size_t>(stop);
    //    size_t jstep  = static_cast<size_t>(step);

    //    hops::Data newData{};
    //    for (size_t i = istart; i < istop; i += istep) {
    //        newData.chains.push_back(hops::ChainData());
    //        std::vector<double>& acceptanceRates = *newData.chains.back().acceptanceRates;
    //        std::vector<double>& negativeLogLikelihood = *newData.chains.back().negativeLogLikelihood;
    //        std::vector<VectorType>& states = *newData.chains.back().states;
    //        std::vector<long>& timestamps = *newData.chains.back().timestamps;

    //        for (size_t j = jstart; j < jstop; j += jstep) {
    //            if (j < data.chains[i].getAcceptanceRates().size()) {
    //                acceptanceRates.push_back(data.chains[i].getAcceptanceRates()[j]);
    //            }
    //            if (j < data.chains[i].getNegativeLogLikelihood().size()) {
    //                negativeLogLikelihood.push_back(data.chains[i].getNegativeLogLikelihood()[j]);
    //            }
    //            if (j < data.chains[i].getStates().size()) {
    //                states.push_back(data.chains[i].getStates()[j]);
    //            }
    //            if (j < data.chains[i].getTimestamps().size()) {
    //                timestamps.push_back(data.chains[i].getTimestamps()[j]);
    //            }
    //        }
    //    }

    //    return newData;
    //}

    //hops::Data constructDataFromSimpleData(const SimpleData& simpleData) {
    //    hops::Data data;
    //    for (const auto& simpleChain : simpleData) {
    //        data.chains.push_back(
    //                hops::ChainData(
    //                    std::get<0>(simpleChain),
    //                    std::get<1>(simpleChain),
    //                    std::get<2>(simpleChain),
    //                    std::get<3>(simpleChain)
    //                )
    //            );
    //    }
    //    return data;
    //}
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::DegenerateGaussian);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Mixture);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::PyModel);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Rosenbrock);


#endif // HOPSY_HPP
