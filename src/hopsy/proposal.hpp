#ifndef HOPSY_PROPOSAL_HPP
#define HOPSY_PROPOSAL_HPP

#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "../../extern/hops/src/hops/hops.hpp"

#include "doc.hpp"
#include "misc.hpp"
#include "model.hpp"
#include "problem.hpp"
#include "random.hpp"

namespace py = pybind11;

namespace hopsy {
	using Proposal = hops::Proposal;
    using ProposalParameter = hops::ProposalParameter;
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Proposal);

namespace hopsy {
    template<typename ProposalBase = Proposal>
	class ProposalTrampoline : public ProposalBase, public py::trampoline_self_life_support {
	public:
		/* Inherit the constructors */
		using ProposalBase::ProposalBase;

        virtual ~ProposalTrampoline() = default;

        virtual VectorType& propose(hops::RandomNumberGenerator& rng) override {
			PYBIND11_OVERRIDE_PURE(
				VectorType&,
				ProposalBase,
				propose,
                rng
			);
        }

        VectorType& acceptProposal() override {
			PYBIND11_OVERRIDE_PURE_NAME(
				VectorType&,     /* Return type */
				ProposalBase,       /* Parent class */
                "accept_proposal",
				acceptProposal  /* Name of function in C++ (must match Python name) */
			);
        }

        VectorType getProposal() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				VectorType,
				ProposalBase,
                "proposal",
				getProposal
			);
        }

        double computeLogAcceptanceProbability() override {
			PYBIND11_OVERRIDE_PURE_NAME(
				double,
				ProposalBase,
                "compute_log_acceptance_probability",
                computeLogAcceptanceProbability
			);
        }

        VectorType getState() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				VectorType,
				ProposalBase,
                "state",
				getState
			);
        }

        void setState(const VectorType& state) override {
			PYBIND11_OVERRIDE_PURE_NAME(
				void,
				ProposalBase,
                "state",
				setState,
                state
			);
        }

        std::optional<std::vector<std::string>> getDimensionNames() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::optional<std::vector<std::string>>,
				ProposalBase,
                "get_dimension_names",
				getDimension
			);
        }

        std::vector<std::string> getParameterNames() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::vector<std::string>,
				ProposalBase,
                "get_parameter_names",
				getParameter
			);
        }

        std::string getParameterType(const ProposalParameter &parameter) const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::string,
				ProposalBase,
                "get_parameter_type",
				getParameter,
                parameter
			);
        }

        //std::any getParameter(const ProposalParameter &parameter) const override {
		//	PYBIND11_OVERRIDE_PURE_NAME(
		//		std::any,
		//		ProposalBase,
        //        "_get_parameter",
		//		getParameter,
        //        parameter
		//	);
        //}

        //void setParameter(const ProposalParameter &parameter, const std::any &value) override {
		//	PYBIND11_OVERRIDE_PURE_NAME(
		//		void,
		//		ProposalBase,
        //        "_set_parameter",
		//		setParameter,
        //        parameter,
        //        value
		//	);
        //}

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

        double getStateNegativeLogLikelihood() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				double,
				ProposalBase,
                "get_state_negative_log_likelihood",
                getStateNegativeLogLikelihood
			);
        }

        double getProposalNegativeLogLikelihood() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				double,
				ProposalBase,
                "get_proposal_negative_log_likelihood",
                getProposalNegativeLogLikelihood
			);
        }

        bool hasNegativeLogLikelihood() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				bool,
				ProposalBase,
                "has_negative_log_likelihood",
                hasNegativeLogLikelihood
			);
        }

        std::unique_ptr<Proposal> copyProposal() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::unique_ptr<Proposal>,  // Return type 
				ProposalBase,               // Parent class
                "deepcopy",                 // Python function name
				copyProposal                    // C++ function name
            );
        }
	};

    class PyProposal : public Proposal {
    public:
        PyProposal() = default;

		PyProposal(py::object pyObj) : pyObj(std::move(pyObj)) {};

        VectorType& propose(hops::RandomNumberGenerator& rng) override {
            proposal = pyObj.attr("propose")(rng).cast<VectorType>();
            return proposal;
        }

        VectorType& acceptProposal() override {
            state = pyObj.attr("accept_proposal")().cast<VectorType>();
            return state;
        }

        double computeLogAcceptanceProbability() override {
            return pyObj.attr("compute_log_acceptance_probability")().cast<double>();
        }

        VectorType getProposal() const override {
            return pyObj.attr("get_proposal")().cast<VectorType>();
        }

        VectorType getState() const override {
            return pyObj.attr("get_state")().cast<VectorType>();
        }

        void setState(const VectorType& newState) override {
            pyObj.attr("set_state")(newState);
        }

        std::optional<std::vector<std::string>> getDimensionNames() const override {
            return pyObj.attr("get_dimension_names")().cast<std::optional<std::vector<std::string>>>();
        }

        std::vector<std::string> getParameterNames() const override {
            return pyObj.attr("get_parameter_names")().cast<std::vector<std::string>>();
        }

        std::any getParameter(const ProposalParameter& parameter) const override {
            return pyObj.attr("_get_parameter")(hops::ProposalParameterName[static_cast<int>(parameter)]).cast<std::any>();
        }

        std::string getParameterType(const ProposalParameter& parameter) const override {
            return pyObj.attr("get_parameter_type")(hops::ProposalParameterName[static_cast<int>(parameter)]).cast<std::string>();
        }

        void setParameter(const ProposalParameter& parameter, const std::any &value) override {
            pyObj.attr("_set_parameter")(hops::ProposalParameterName[static_cast<int>(parameter)], value);
        }

        bool hasStepSize() const override {
            return pyObj.attr("has_stepsize")().cast<bool>();
        }

        std::string getProposalName() const override {
            return pyObj.attr("name").cast<std::string>();
        }

        double getStateNegativeLogLikelihood() const override {
            return pyObj.attr("get_state_negative_log_likelihood")().cast<double>();
        }

        double getProposalNegativeLogLikelihood() const override {
            return pyObj.attr("get_proposal_negative_log_likelihood")().cast<double>();
        }

        bool hasNegativeLogLikelihood() const override {
            return pyObj.attr("has_negative_log_likelihood")().cast<double>();
        }

        std::unique_ptr<Proposal> copyProposal() const override {
            return pyObj.attr("deepcopy")().cast<std::unique_ptr<Proposal>>();
        }
	private:
		py::object pyObj;

        VectorType proposal;
        VectorType state;
    };


    class ProposalWrapper : public Proposal {
    public:
		ProposalWrapper(const std::shared_ptr<Proposal> proposal) {
            proposalPtr = std::move(proposal->copyProposal());
        }

        VectorType& propose(hops::RandomNumberGenerator& rng) override {
            return proposalPtr->propose(rng);
        }

        VectorType& acceptProposal() override {
            return proposalPtr->acceptProposal();
        }

        double computeLogAcceptanceProbability() override {
            return proposalPtr->computeLogAcceptanceProbability();
        }

        VectorType getProposal() const override {
            return proposalPtr->getProposal();
        }

        VectorType getState() const override {
            return proposalPtr->getState();
        }

        void setState(const VectorType& newState) override {
            proposalPtr->setState(newState);
        }

        std::optional<std::vector<std::string>> getDimensionNames() const override {
            return proposalPtr->getDimensionNames();
        }

        std::vector<std::string> getParameterNames() const override {
            return proposalPtr->getParameterNames();
        }

        std::any getParameter(const ProposalParameter& parameter) const override {
            return proposalPtr->getParameter(parameter);
        }

        std::string getParameterType(const ProposalParameter& parameter) const override {
            return proposalPtr->getParameterType(parameter);
        }

        void setParameter(const ProposalParameter& parameter, const std::any &value) override {
            proposalPtr->setParameter(parameter, value);
        }

        bool hasStepSize() const override {
            return proposalPtr->hasStepSize();
        }

        std::string getProposalName() const override {
            return proposalPtr->getProposalName();
        }

        double getStateNegativeLogLikelihood() const override {
            return proposalPtr->getStateNegativeLogLikelihood();
        }

        double getProposalNegativeLogLikelihood() const override {
            return proposalPtr->getProposalNegativeLogLikelihood();
        }

        bool hasNegativeLogLikelihood() const override {
            return proposalPtr->hasNegativeLogLikelihood();;
        }

        std::unique_ptr<Proposal> copyProposal() const override {
            return proposalPtr->copyProposal();
        }

        std::shared_ptr<Proposal> getProposalPtr() {
            return proposalPtr;
        }

    private:
        std::shared_ptr<Proposal> proposalPtr;
    };

    template<typename ProposalImpl, typename... Args>
    class UninitializedProposalWrapper : public Proposal {
    public:
        UninitializedProposalWrapper() : 
                isInitialized(false) { }

        UninitializedProposalWrapper(const MatrixType& A, 
                                     const VectorType& b, 
                                     const VectorType& startingPoint,
                                     const Args&... args) : 
                proposal(ProposalImpl(A, b, startingPoint, args...)), 
                isInitialized(true) { }
                
        static UninitializedProposalWrapper<ProposalImpl, Args...> createFromProblem(const Problem* problem, 
                                                                                     const Args&... args) {
            if (problem) {
                if (!problem->startingPoint) {
                    throw std::runtime_error("Cannot setup a proposal without starting point.");
                }

                return UninitializedProposalWrapper<ProposalImpl, Args...>(problem->A, problem->b, *problem->startingPoint, args...);
            } else {
                throw std::runtime_error(std::string("Internal error in ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
            }
        }

        static UninitializedProposalWrapper<ProposalImpl, Args...> create(const Problem* problem,
                                                                          const VectorType* startingPoint,
                                                                          const Args&... args) {
            if (problem) {
                if (!(problem->startingPoint || startingPoint)) {
                    throw std::runtime_error("Cannot setup a proposal without starting point.");
                }

                VectorType _startingPoint = ( startingPoint ? *startingPoint : *problem->startingPoint );
                return UninitializedProposalWrapper<ProposalImpl, Args...>(problem->A, problem->b, _startingPoint, args...);
            } else {
                throw std::runtime_error(std::string("Internal error in ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
            }
        }

        virtual ~UninitializedProposalWrapper() = default;

        VectorType& propose(hops::RandomNumberGenerator& rng) override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("propose"));
            return proposal->propose(rng);
        }

        VectorType& acceptProposal() override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("accept_proposal"));
            return proposal->acceptProposal();
        }

        double computeLogAcceptanceProbability() override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("log_acceptance_probability"));
            return proposal->computeLogAcceptanceProbability();
        }

        VectorType getProposal() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("proposal"));
            return proposal->getProposal();
        }

        VectorType getState() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("state"));
            return proposal->getState();
        }

        void setState(const VectorType& newState) override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("state"));
            proposal->setState(newState);
        }

        std::optional<std::vector<std::string>> getDimensionNames() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("dimension_names"));
            return proposal->getDimensionNames();
        }

        std::vector<std::string> getParameterNames() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("parameter_names"));
            return proposal->getParameterNames();
        }

        std::any getParameter(const ProposalParameter& parameter) const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("_get_parameter"));
            return proposal->getParameter(parameter);
        }

        std::string getParameterType(const ProposalParameter& parameter) const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("get_parameter_type"));
            return proposal->getParameterType(parameter);
        }

        void setParameter(const ProposalParameter& parameter, const std::any &value) override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("_set_parameter"));
            proposal->setParameter(parameter, value);
        }

        bool hasStepSize() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("has_stepsize"));
            return proposal->hasStepSize();
        }

        std::string getProposalName() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("name"));
            return proposal->getProposalName();
        }

        double getStateNegativeLogLikelihood() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("state_negative_log_likelihood"));
            return proposal->getStateNegativeLogLikelihood();
        }

        double getProposalNegativeLogLikelihood() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("proposal_negative_log_likelihood"));
            return proposal->getProposalNegativeLogLikelihood();
        }

        bool hasNegativeLogLikelihood() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("has_negative_log_likelihood"));
            return proposal->hasNegativeLogLikelihood();;
        }

        std::unique_ptr<Proposal> copyProposal() const override {
            return std::make_unique<UninitializedProposalWrapper<ProposalImpl, Args...>>(*this);
        }

    private:
        std::string uninitializedMethod(const std::string& name) const {
            return "Tried to access " + name + " on an uninitialized proposal.";
        }

        std::optional<ProposalImpl> proposal;
        bool isInitialized;
    };

    using AdaptiveMetropolisProposal = UninitializedProposalWrapper<
            hops::AdaptiveMetropolisProposal<MatrixType, VectorType>, MatrixType, double, double, unsigned long>;
    using BallWalkProposal = UninitializedProposalWrapper<
            hops::BallWalkProposal<MatrixType, VectorType>, double>;
    using CSmMALAProposal = UninitializedProposalWrapper<
            hops::CSmMALAProposal<ModelWrapper, MatrixType>, ModelWrapper, double, double>;
    using DikinWalkProposal = UninitializedProposalWrapper<
			hops::DikinProposal<MatrixType, VectorType>, double>;
    using GaussianCoordinateHitAndRunProposal = UninitializedProposalWrapper<
			hops::CoordinateHitAndRunProposal<MatrixType, VectorType, hops::GaussianStepDistribution<double>>, double>;
    using GaussianHitAndRunProposal = UninitializedProposalWrapper<
			hops::HitAndRunProposal<MatrixType, VectorType, hops::GaussianStepDistribution<double>>, double>;
    using GaussianProposal = UninitializedProposalWrapper<
			hops::GaussianProposal<MatrixType, VectorType>, double>;
    using UniformCoordinateHitAndRunProposal = UninitializedProposalWrapper<
			hops::CoordinateHitAndRunProposal<MatrixType, VectorType>>;
    using UniformHitAndRunProposal = UninitializedProposalWrapper<
			hops::HitAndRunProposal<MatrixType, VectorType>>;
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::AdaptiveMetropolisProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::BallWalkProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::CSmMALAProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::DikinWalkProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::GaussianCoordinateHitAndRunProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::GaussianHitAndRunProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::GaussianProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::PyProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::UniformCoordinateHitAndRunProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::UniformHitAndRunProposal);

namespace hopsy {

    void addProposalParameters(py::module& m) {
        py::enum_<ProposalParameter>(m, "ProposalParameter", py::arithmetic())
            .value("BoundaryCushion", ProposalParameter::BOUNDARY_CUSHION)
            .value("Epsilon", ProposalParameter::EPSILON)
            .value("FisherWeight", ProposalParameter::FISHER_WEIGHT)
            .value("StepSize", ProposalParameter::STEP_SIZE)
            .value("WarmUp", ProposalParameter::WARM_UP)
            .def("__repr__", [] (const ProposalParameter& parameter) {
                        return "hopsy." + py::cast(parameter).attr("__str__")().cast<std::string>();
                    }, py::prepend())
        ;

    }

    namespace proposal {
        template<typename ProposalType, typename ClassType>
        void addCommon(ClassType& prop) {
            //py::classh<ProposalType, hopsy::Proposal, hopsy::ProposalTrampoline<ProposalType>> proposal(m, proposalName);
            prop.def("propose", [](ProposalType& self, hopsy::RandomNumberGenerator& rng) -> hops::VectorType& { 
                    return self.propose(rng.rng); 
                })
                .def("accept_proposal", &ProposalType::acceptProposal)
                .def_property_readonly("log_acceptance_probability", &ProposalType::computeLogAcceptanceProbability)
                .def_property_readonly("proposal", &ProposalType::getProposal)
                .def_property("state", &ProposalType::getState, &ProposalType::setState)
                .def("_get_parameter", [] (const ProposalType& self, 
                                           const ProposalParameter& parameter) {
                            return std::any_cast<double>(self.getParameter(parameter));
                        }, py::arg("param"))
                .def("_set_parameter", [] (ProposalType& self, 
                                           const ProposalParameter& parameter,
                                           double value) {
                            return self.setParameter(parameter, std::any(value));
                        }, py::arg("param"), py::arg("value"))
                .def_property_readonly("has_stepsize", &ProposalType::hasStepSize)
                .def_property_readonly("name", &ProposalType::getProposalName)
                .def_property_readonly("state_negative_log_likelihood", &ProposalType::getStateNegativeLogLikelihood)
                .def_property_readonly("proposal_negative_log_likelihood", &ProposalType::getProposalNegativeLogLikelihood)
                .def_property_readonly("has_negative_log_likelihood", &ProposalType::hasNegativeLogLikelihood)
                .def("deepcopy", &ProposalType::copyProposal)
            ;
        }

        template<typename ProposalType, typename ClassType, typename ParameterType = double>
        void addParameter(ClassType& prop, const hops::ProposalParameter &parameter, const char* parameterName) {
            prop.def_property(parameterName, 
                    [=] (const ProposalType& self) { return std::any_cast<ParameterType>(self.getParameter(parameter)); },
                    [=] (ProposalType& self, ParameterType value) { self.setParameter(parameter, value); }
                )
            ;
        }
    }

    void addProposals(py::module &m) {
        // register abstract Proposal
        py::classh<Proposal, ProposalTrampoline<>> proposal(m, "Proposal");
        // common
        proposal::addCommon<Proposal>(proposal);


        // register AdaptiveMetropolisProposal
        py::classh<AdaptiveMetropolisProposal, Proposal, ProposalTrampoline<AdaptiveMetropolisProposal>> adaptiveMetropolisProposal(
                m, "AdaptiveMetropolisProposal");
        // constructor
        adaptiveMetropolisProposal
            .def(py::init<>()) 
            .def(py::init([] (const Problem* problem,
                              double stepSize,
                              double boundaryCushion,
                              double eps,
                              unsigned long warmUp) {
                        MatrixType sqrtMve = computeSqrtMaximumVolumeEllipsoid(*problem);
                        auto proposal = AdaptiveMetropolisProposal::createFromProblem(problem, sqrtMve, stepSize, eps, warmUp);
                        proposal.setParameter(ProposalParameter::BOUNDARY_CUSHION, boundaryCushion);
                        return proposal;
                }),
                py::arg("problem"),
                py::arg("stepsize") = 1, 
                py::arg("boundary_cushion") = 0,
                py::arg("eps") = 1.e-3, 
                py::arg("warm_up") = 100)
            .def(py::init([] (const Problem* problem,
                              const VectorType* startingPoint,
                              double stepSize,
                              double boundaryCushion,
                              double eps,
                              unsigned long warmUp) {
                        MatrixType sqrtMve = computeSqrtMaximumVolumeEllipsoid(*problem);
                        auto proposal = AdaptiveMetropolisProposal::create(problem, startingPoint, sqrtMve, stepSize, eps, warmUp);
                        proposal.setParameter(ProposalParameter::BOUNDARY_CUSHION, boundaryCushion);
                        return proposal;
                }),
                py::arg("problem"),
                py::arg("starting_point") = py::none(), 
                py::arg("stepsize") = 1, 
                py::arg("boundary_cushion") = 0,
                py::arg("eps") = 1.e-3, 
                py::arg("warm_up") = 100)
            ;
        // common
        proposal::addCommon<AdaptiveMetropolisProposal>(adaptiveMetropolisProposal);
        // parameters
        proposal::addParameter<AdaptiveMetropolisProposal>(
                adaptiveMetropolisProposal, ProposalParameter::BOUNDARY_CUSHION, "boundary_cushion");
        proposal::addParameter<AdaptiveMetropolisProposal>(
                adaptiveMetropolisProposal, ProposalParameter::EPSILON, "eps");
        proposal::addParameter<AdaptiveMetropolisProposal>(
                adaptiveMetropolisProposal, ProposalParameter::STEP_SIZE, "stepsize");
        proposal::addParameter<AdaptiveMetropolisProposal, decltype(adaptiveMetropolisProposal), unsigned long>(
                adaptiveMetropolisProposal, ProposalParameter::WARM_UP, "warm_up");
        
        
        // register BallWalkProposal
        py::classh<BallWalkProposal, Proposal, ProposalTrampoline<BallWalkProposal>> ballWalkProposal(
                m, "BallWalkProposal");
        // constructor
        ballWalkProposal
            .def(py::init<>()) 
            .def(py::init(&BallWalkProposal::createFromProblem), 
                py::arg("problem"),
                py::arg("stepsize") = 1)
            .def(py::init(&BallWalkProposal::create), 
                py::arg("problem"),
                py::arg("starting_point"), 
                py::arg("stepsize") = 1)
            ;
        // common
        proposal::addCommon<BallWalkProposal>(ballWalkProposal);
        // parameters
        proposal::addParameter<BallWalkProposal>(
                ballWalkProposal, ProposalParameter::STEP_SIZE, "stepsize");


        // register CSmMALAProposal
        py::classh<CSmMALAProposal, Proposal, ProposalTrampoline<CSmMALAProposal>> csmmalaProposal(
                m, "CSmMALAProposal");
        // constructor
        csmmalaProposal    
            .def(py::init<>()) 
            .def(py::init([] (const Problem* problem, 
                              double stepSize,
                              double fisherWeight) -> CSmMALAProposal {
                        if (problem) {
                            if (!problem->model) {
                                throw std::runtime_error("Cannot initialize hopsy.CSmMALAProposal for uniform problem (problem.model == None).");
                            }

                            return CSmMALAProposal::createFromProblem(problem, ModelWrapper(problem->model), fisherWeight, stepSize);
                        } else {
                            throw std::runtime_error(std::string("Internal error in ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
                        }
                }), 
                py::arg("problem"), 
                py::arg("stepsize") = 1, 
                py::arg("fisher_weight") = 1)
            .def(py::init([] (const Problem* problem, 
                              const VectorType* startingPoint, 
                              double stepSize,
                              double fisherWeight) -> CSmMALAProposal {
                        if (problem) {
                            if (!problem->model) {
                                throw std::runtime_error("Cannot initialize hopsy.CSmMALAProposal for uniform problem (problem.model == None).");
                            }

                            return CSmMALAProposal::create(problem, startingPoint, ModelWrapper(problem->model), fisherWeight, stepSize);
                        } else {
                            throw std::runtime_error(std::string("Internal error in ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
                        }
                }), 
                py::arg("problem"), 
				py::arg("starting_point"), 
                py::arg("stepsize") = 1, 
                py::arg("fisher_weight") = 1)
            ;
        // common
        proposal::addCommon<CSmMALAProposal>(csmmalaProposal);
        // parameters
        proposal::addParameter<CSmMALAProposal>(
                csmmalaProposal, ProposalParameter::FISHER_WEIGHT, "fisher_weight");
        proposal::addParameter<CSmMALAProposal>(
                csmmalaProposal, ProposalParameter::STEP_SIZE, "stepsize");
        
        
        // register DikinWalkProposal
        py::classh<DikinWalkProposal, Proposal, ProposalTrampoline<DikinWalkProposal>> dikinWalkProposal(
                m, "DikinWalkProposal");
        // constructor
        dikinWalkProposal
            .def(py::init<>()) 
            .def(py::init([] (const Problem* problem,
                              double stepSize,
                              double boundaryCushion) {
                        auto proposal = DikinWalkProposal::createFromProblem(problem, stepSize);
                        proposal.setParameter(ProposalParameter::BOUNDARY_CUSHION, boundaryCushion);
                        return proposal;
                }),
                py::arg("problem"),
                py::arg("stepsize") = 1, 
                py::arg("boundary_cushion") = 0)
            .def(py::init([] (const Problem* problem,
                              const VectorType* startingPoint,
                              double stepSize,
                              double boundaryCushion) {
                        auto proposal = DikinWalkProposal::create(problem, startingPoint, stepSize);
                        proposal.setParameter(ProposalParameter::BOUNDARY_CUSHION, boundaryCushion);
                        return proposal;
                }),
                py::arg("problem"),
                py::arg("starting_point") = py::none(), 
                py::arg("stepsize") = 1, 
                py::arg("boundary_cushion") = 0)
            ;
        // common
        proposal::addCommon<DikinWalkProposal>(dikinWalkProposal);
        // parameters
        proposal::addParameter<DikinWalkProposal>(
                dikinWalkProposal, ProposalParameter::BOUNDARY_CUSHION, "boundary_cushion");
        proposal::addParameter<DikinWalkProposal>(
                dikinWalkProposal, ProposalParameter::STEP_SIZE, "stepsize");
        
        
        // register GaussianCoordinateHitAndRun
        py::classh<GaussianCoordinateHitAndRunProposal, Proposal, ProposalTrampoline<GaussianCoordinateHitAndRunProposal>> gaussianCoordinateHitAndRunProposal(
                m, "GaussianCoordinateHitAndRunProposal");
        // constructor
        gaussianCoordinateHitAndRunProposal
            .def(py::init<>()) 
            .def(py::init(&GaussianCoordinateHitAndRunProposal::createFromProblem),
                py::arg("problem"),
                py::arg("stepsize") = 1)
            .def(py::init(&GaussianCoordinateHitAndRunProposal::create),
                py::arg("problem"),
                py::arg("starting_point"), 
                py::arg("stepsize") = 1)
            ;        
		// common
        proposal::addCommon<GaussianCoordinateHitAndRunProposal>(gaussianCoordinateHitAndRunProposal);
        // parameters
        proposal::addParameter<GaussianCoordinateHitAndRunProposal>(
                gaussianCoordinateHitAndRunProposal, ProposalParameter::STEP_SIZE, "stepsize");


        // register GaussianHitAndRun
        py::classh<GaussianHitAndRunProposal, Proposal, ProposalTrampoline<GaussianHitAndRunProposal>> gaussianHitAndRunProposal(
                m, "GaussianHitAndRunProposal");
        // constructor
        gaussianHitAndRunProposal
            .def(py::init<>()) 
            .def(py::init(&GaussianHitAndRunProposal::createFromProblem),
                py::arg("problem"),
                py::arg("stepsize") = 1)
            .def(py::init(&GaussianHitAndRunProposal::create),
                py::arg("problem"),
                py::arg("starting_point"), 
                py::arg("stepsize") = 1)
            ;
        // common
        proposal::addCommon<GaussianHitAndRunProposal>(gaussianHitAndRunProposal);
        // parameters
        proposal::addParameter<GaussianHitAndRunProposal>(
                gaussianHitAndRunProposal, ProposalParameter::STEP_SIZE, "stepsize");


        // register GaussianProposal
        py::classh<GaussianProposal, Proposal, ProposalTrampoline<GaussianProposal>> gaussianProposal(
                m, "GaussianProposal");
        // constructor
        gaussianProposal
            .def(py::init<>()) 
            .def(py::init(&GaussianProposal::createFromProblem),
                py::arg("problem"),
                py::arg("stepsize") = 1)
            .def(py::init(&GaussianProposal::create),
                py::arg("problem"),
                py::arg("starting_point"),
                py::arg("stepsize") = 1)
            ;
        // common
        proposal::addCommon<GaussianProposal>(gaussianProposal);
        // parameters
        proposal::addParameter<GaussianProposal>(
                gaussianProposal, ProposalParameter::STEP_SIZE, "stepsize");


        // register PyProposal
        py::classh<PyProposal, Proposal, ProposalTrampoline<PyProposal>> 
            pyProposal(m, "PyProposal");
        // constructor
        pyProposal.def(py::init<py::object>(), py::arg("proposal"));
        // common
        proposal::addCommon<PyProposal>(pyProposal);


        // register UniformCoordinateHitAndRun
        py::classh<UniformCoordinateHitAndRunProposal, Proposal, ProposalTrampoline<UniformCoordinateHitAndRunProposal>> uniformCoordinateHitAndRunProposal(
                m, "UniformCoordinateHitAndRunProposal");
        // constructor
        uniformCoordinateHitAndRunProposal
            .def(py::init<>()) 
            .def(py::init(&UniformCoordinateHitAndRunProposal::createFromProblem), 
                py::arg("problem"))
            .def(py::init(&UniformCoordinateHitAndRunProposal::create), 
                py::arg("problem"),
                py::arg("starting_point"))
            ;
        // common
        proposal::addCommon<UniformCoordinateHitAndRunProposal>(uniformCoordinateHitAndRunProposal);


        // register UniformHitAndRun
        py::classh<UniformHitAndRunProposal, Proposal, ProposalTrampoline<UniformHitAndRunProposal>> uniformHitAndRunProposal(
                m, "UniformHitAndRunProposal");
        // constructor
        uniformHitAndRunProposal
            .def(py::init<>()) 
            .def(py::init(&UniformHitAndRunProposal::createFromProblem), 
                py::arg("problem")) 
            .def(py::init(&UniformHitAndRunProposal::create), 
                py::arg("problem"),
                py::arg("starting_point")) 
            ;
        // common
        proposal::addCommon<UniformHitAndRunProposal>(uniformHitAndRunProposal);


    }
} // namespace hopsy

#endif // HOPSY_PROPOSAL_HPP
