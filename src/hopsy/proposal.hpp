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
                "log_acceptance_probability",
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

        std::vector<std::string> getDimensionNames() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::vector<std::string>,
				ProposalBase,
                "dimension_names",
				getDimensionNames
			);
        }

        void setDimensionNames(const std::vector<std::string>& newDimensionNames) override {
			PYBIND11_OVERRIDE_PURE_NAME(
				void,
				ProposalBase,
                "dimension_names",
				setDimensionNames,
        newDimensionNames
			);
        }

        std::vector<std::string> getParameterNames() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::vector<std::string>,
				ProposalBase,
                "parameter_names",
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

        std::string getProposalName() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::string,
				ProposalBase,
                "name",
				getProposalName,
			);
        }

        double getStateNegativeLogLikelihood() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				double,
				ProposalBase,
                "state_negative_log_likelihood",
                getStateNegativeLogLikelihood
			);
        }

        double getProposalNegativeLogLikelihood() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				double,
				ProposalBase,
                "proposal_negative_log_likelihood",
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

        //const MatrixType& getA() const override {
		//	PYBIND11_OVERRIDE_PURE_NAME(
		//		const MatrixType&,
		//		ProposalBase,
        //        "A",
        //        getA
		//	);
        //}

        //const VectorType& getB() const override {
		//	PYBIND11_OVERRIDE_PURE_NAME(
		//		const MatrixType&,
		//		ProposalBase,
        //        "b",
        //        getB
		//	);
        //}

        std::unique_ptr<Proposal> copyProposal() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::unique_ptr<Proposal>,  // Return type
				ProposalBase,               // Parent class
                "__copy__",                 // Python function name
				copyProposal                    // C++ function name
            );
        }
	};

    class PyProposal : public Proposal {
    public:
        PyProposal() = default;

		PyProposal(py::object pyObj) : pyObj(std::move(pyObj)) {};

        VectorType& propose(hops::RandomNumberGenerator& rng) override {
            proposal = pyObj.attr("propose")(hopsy::RandomNumberGenerator(rng)).cast<VectorType>();
            return proposal;
        }

        VectorType& acceptProposal() override {
            state = pyObj.attr("accept_proposal")().cast<VectorType>();
            return state;
        }

        double computeLogAcceptanceProbability() override {
            return pyObj.attr("log_acceptance_probability")().cast<double>();
        }

        VectorType getProposal() const override {
            return pyObj.attr("proposal").cast<VectorType>();
        }

        VectorType getState() const override {
            return pyObj.attr("state").cast<VectorType>();
        }

        void setState(const VectorType& newState) override {
            pyObj.attr("state")(newState);
        }

        std::vector<std::string> getDimensionNames() const override {
            try {
              return pyObj.attr("dimension_names").cast<std::vector<std::string>>();
            }
            catch(...) {
               return {};
            }
        }

        void setDimensionNames(const std::vector<std::string>& newDimensionNames) override {
            try {
              pyObj.attr("dimension_names")(newDimensionNames);
            }
            catch(...) {
                return;
            }
        }

        std::vector<std::string> getParameterNames() const override {
            return pyObj.attr("parameter_names")().cast<std::vector<std::string>>();
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

        std::string getProposalName() const override {
            return pyObj.attr("get_name").cast<std::string>();
        }

        double getStateNegativeLogLikelihood() const override {
            try {
                return pyObj.attr("get_state_negative_log_likelihood")().cast<double>();
            }
            catch(...) {
                return 0;
            }
        }

        double getProposalNegativeLogLikelihood() const override {
            try {
                return pyObj.attr("get_proposal_negative_log_likelihood")().cast<double>();
            }
            catch(...) {
                return 0;
            }
        }

        bool hasNegativeLogLikelihood() const override {
            try {
                return pyObj.attr("has_negative_log_likelihood")().cast<double>();
            }
            catch(...) {
                return false;
            }
        }

        const MatrixType& getA() const override {
            throw std::runtime_error("Function not implemented.");
        }

        const VectorType& getB() const override {
            throw std::runtime_error("Function not implemented.");
        }

        std::unique_ptr<Proposal> copyProposal() const override {
            return std::make_unique<PyProposal>(PyProposal(pyObj));
        }

		py::object pyObj;

	private:
        VectorType proposal;
        VectorType state;
    };

    class ProposalWrapper : public Proposal {
    public:
		ProposalWrapper(const Proposal* proposal) {
            proposalPtr = proposal->copyProposal();
        }

        ProposalWrapper(std::shared_ptr<Proposal> proposal) {
            proposalPtr = proposal;
        }

        VectorType& propose(hops::RandomNumberGenerator& rng) override {
            return proposalPtr->propose(rng);
        }

        VectorType& propose(hops::RandomNumberGenerator& rng, const VectorType &activeSubspaces) override {
            return proposalPtr->propose(rng, activeSubspaces);
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

        std::vector<std::string> getDimensionNames() const override {
            return proposalPtr->getDimensionNames();
        }

        void setDimensionNames(const std::vector<std::string>& newDimensionNames) override {
            return proposalPtr->setDimensionNames(newDimensionNames);
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

        const MatrixType& getA() const override {
            return proposalPtr->getA();
        }

        const VectorType& getB() const override {
            return proposalPtr->getB();
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

        std::vector<std::string> getDimensionNames() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("dimension_names"));
            return proposal->getDimensionNames();
        }

        void setDimensionNames(const std::vector<std::string>& newDimensionNames) override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("set_dimension_names"));
            proposal->setDimensionNames(newDimensionNames);
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
            return proposal->hasNegativeLogLikelihood();
        }

        const MatrixType& getA() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("A"));
            return proposal->getA();
        }

        const VectorType& getB() const override {
            if (!proposal) throw std::runtime_error(uninitializedMethod("b"));
            return proposal->getB();
        }

        std::unique_ptr<Proposal> copyProposal() const override {
            return std::make_unique<UninitializedProposalWrapper<ProposalImpl, Args...>>(*this);
        }

        std::optional<ProposalImpl> proposal;
    private:
        std::string uninitializedMethod(const std::string& name) const {
            return "Tried to access " + name + " on an uninitialized proposal.";
        }

        bool isInitialized;
    };

    using AdaptiveMetropolisProposal = UninitializedProposalWrapper<
            hops::AdaptiveMetropolisProposal<MatrixType>, MatrixType, double, unsigned long>;
    using BallWalkProposal = UninitializedProposalWrapper<
            hops::BallWalkProposal<MatrixType, VectorType>, double>;
    using BilliardAdaptiveMetropolisProposal = UninitializedProposalWrapper<
            hops::BilliardAdaptiveMetropolisProposal<MatrixType>, MatrixType, double, unsigned long, long>;
    using BilliardMALAProposal = UninitializedProposalWrapper<
            hops::BilliardMALAProposal<ModelWrapper, MatrixType>, ModelWrapper, long, double>;
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
    using TruncatedGaussianProposal = UninitializedProposalWrapper<
            hops::TruncatedGaussianProposal<MatrixType, VectorType>, hops::Gaussian>;
    using UniformCoordinateHitAndRunProposal = UninitializedProposalWrapper<
			hops::CoordinateHitAndRunProposal<MatrixType, VectorType>>;
    using UniformHitAndRunProposal = UninitializedProposalWrapper<
			hops::HitAndRunProposal<MatrixType, VectorType>>;
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::AdaptiveMetropolisProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::BallWalkProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::BilliardAdaptiveMetropolisProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::BilliardMALAProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::CSmMALAProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::DikinWalkProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::GaussianCoordinateHitAndRunProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::GaussianHitAndRunProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::GaussianProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::PyProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::TruncatedGaussianProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::UniformCoordinateHitAndRunProposal);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::UniformHitAndRunProposal);

namespace hopsy {

    void addProposalParameters(py::module& m) {
        py::enum_<ProposalParameter>(m, "ProposalParameter", py::arithmetic(), "")
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
        template<typename ProposalType, typename Docs, typename ClassType>
        void addCommon(ClassType& prop) {
            prop.def("propose", [](ProposalType& self, hopsy::RandomNumberGenerator& rng) -> hops::VectorType& {
                            return self.propose(rng.rng);
                        },
                        Docs::propose
                )
                .def("accept_proposal", &ProposalType::acceptProposal, Docs::acceptProposal)
                .def_property_readonly("log_acceptance_probability", &ProposalType::computeLogAcceptanceProbability, Docs::logAcceptanceProbability)
                .def_property_readonly("proposal", &ProposalType::getProposal, Docs::proposal)
                .def_property("state", &ProposalType::getState, &ProposalType::setState, Docs::state)
                .def("_get_parameter", [] (const ProposalType& self,
                                           const ProposalParameter& parameter) {
                            return std::any_cast<double>(self.getParameter(parameter));
                        },
                        Docs::getParameter,
                        py::arg("param"))
                .def("_set_parameter", [] (ProposalType& self,
                                           const ProposalParameter& parameter,
                                           double value) {
                            return self.setParameter(parameter, std::any(value));
                        },
                        Docs::setParameter,
                        py::arg("param"),
                        py::arg("value"))
                .def_property_readonly("name", &ProposalType::getProposalName, Docs::name)
                .def_property_readonly("state_negative_log_likelihood", &ProposalType::getStateNegativeLogLikelihood,
                        Docs::stateNegativeLogLikelihood)
                .def_property_readonly("proposal_negative_log_likelihood", &ProposalType::getProposalNegativeLogLikelihood,
                        Docs::proposalNegativeLogLikelihood)
                .def_property_readonly("has_negative_log_likelihood", &ProposalType::hasNegativeLogLikelihood,
                        Docs::hasNegativeLogLikelihood)
                .def("deepcopy", &ProposalType::copyProposal, Docs::copyProposal)
            ;
        }

        template<typename ProposalType, typename ClassType, typename ParameterType = double>
        void addParameter(ClassType& prop, const hops::ProposalParameter &parameter, const char* parameterName, const char* doc = nullptr) {
            prop.def_property(parameterName,
                    [=] (const ProposalType& self) { return std::any_cast<ParameterType>(self.getParameter(parameter)); },
                    [=] (ProposalType& self, ParameterType value) { self.setParameter(parameter, value); },
                    doc
                )
            ;
        }
    }

    void addProposals(py::module &m) {
        // register abstract Proposal
        py::classh<Proposal, ProposalTrampoline<>> proposal(m, "Proposal", doc::Proposal::base);
        // common
        proposal::addCommon<Proposal, doc::Proposal>(proposal);


        // register AdaptiveMetropolisProposal
        py::classh<AdaptiveMetropolisProposal, Proposal, ProposalTrampoline<AdaptiveMetropolisProposal>> adaptiveMetropolisProposal(
                m, "AdaptiveMetropolisProposal", doc::AdaptiveMetropolisProposal::base);
        // constructor
        adaptiveMetropolisProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init([] (const Problem* problem,
                              double boundaryCushion,
                              double eps,
                              unsigned long warmUp) {
                        auto _problem(*problem);
                        MatrixType sqrtMve = computeSqrtMaximumVolumeEllipsoid(_problem);
                        auto proposal = AdaptiveMetropolisProposal::createFromProblem(&_problem, sqrtMve, eps, warmUp);
                        proposal.setParameter(ProposalParameter::BOUNDARY_CUSHION, boundaryCushion);
                        return proposal;
                    }),
                    doc::AdaptiveMetropolisProposal::__init__,
                    py::arg("problem"),
                    py::arg("boundary_cushion") = 0,
                    py::arg("eps") = 1.e-3,
                    py::arg("warm_up") = 100)
            .def(py::init([] (const Problem* problem,
                              const VectorType* startingPoint,
                              double boundaryCushion,
                              double eps,
                              unsigned long warmUp) {
                        auto _problem(*problem);
                        MatrixType sqrtMve = computeSqrtMaximumVolumeEllipsoid(_problem);
                        auto proposal = AdaptiveMetropolisProposal::create(&_problem, startingPoint, sqrtMve, eps, warmUp);
                        proposal.setParameter(ProposalParameter::BOUNDARY_CUSHION, boundaryCushion);
                        return proposal;
                    }),
                    doc::AdaptiveMetropolisProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point") = py::none(),
                    py::arg("boundary_cushion") = 0,
                    py::arg("eps") = 1.e-3,
                    py::arg("warm_up") = 100)
            ;
        // common
        proposal::addCommon<AdaptiveMetropolisProposal, doc::AdaptiveMetropolisProposal>(adaptiveMetropolisProposal);
        // parameters
        proposal::addParameter<AdaptiveMetropolisProposal>(
                adaptiveMetropolisProposal, ProposalParameter::BOUNDARY_CUSHION, "boundary_cushion", doc::AdaptiveMetropolisProposal::boundaryCushion);
        proposal::addParameter<AdaptiveMetropolisProposal>(
                adaptiveMetropolisProposal, ProposalParameter::EPSILON, "eps", doc::AdaptiveMetropolisProposal::epsilon);
        proposal::addParameter<AdaptiveMetropolisProposal, decltype(adaptiveMetropolisProposal), unsigned long>(
                adaptiveMetropolisProposal, ProposalParameter::WARM_UP, "warm_up", doc::AdaptiveMetropolisProposal::warmUp);
        // pickling
        adaptiveMetropolisProposal.def(py::pickle([] (const AdaptiveMetropolisProposal& self) {
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState(),
                                              self.proposal->getCholeskyOfMaximumVolumeEllipsoid(),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::BOUNDARY_CUSHION)),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::EPSILON)),
                                              std::any_cast<unsigned long>(self.proposal->getParameter(ProposalParameter::WARM_UP)));
                    },
                    [] (py::tuple t) {
                        if (t.size() != 7) throw std::runtime_error("Invalid state!");

                        auto p = AdaptiveMetropolisProposal(t[0].cast<MatrixType>(),
                                                            t[1].cast<VectorType>(),
                                                            t[2].cast<VectorType>(),
                                                            t[3].cast<MatrixType>(),
                                                            t[5].cast<double>(),
                                                            t[6].cast<unsigned long>());
                        p.setParameter(ProposalParameter::BOUNDARY_CUSHION, t[4].cast<double>());
                        return p;
                    })
                );



        // register BallWalkProposal
        py::classh<BallWalkProposal, Proposal, ProposalTrampoline<BallWalkProposal>> ballWalkProposal(
                m, "BallWalkProposal", doc::BallWalkProposal::base);
        // constructor
        ballWalkProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init(&BallWalkProposal::createFromProblem),
                    doc::BallWalkProposal::__init__,
                    py::arg("problem"),
                    py::arg("stepsize") = 1)
            .def(py::init(&BallWalkProposal::create),
                    doc::BallWalkProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point"),
                    py::arg("stepsize") = 1)
            ;
        // common
        proposal::addCommon<BallWalkProposal, doc::BallWalkProposal>(ballWalkProposal);
        // parameters
        proposal::addParameter<BallWalkProposal>(
                ballWalkProposal, ProposalParameter::STEP_SIZE, "stepsize", doc::BallWalkProposal::stepSize);
        // pickling
        ballWalkProposal.def(py::pickle([] (const BallWalkProposal& self) {
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState(),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::STEP_SIZE)));
                    },
                    [] (py::tuple t) {
                        if (t.size() != 4) throw std::runtime_error("Invalid state!");

                        return BallWalkProposal(t[0].cast<MatrixType>(),
                                                t[1].cast<VectorType>(),
                                                t[2].cast<VectorType>(),
                                                t[3].cast<double>());
                    })
                );



        // register BilliardAdaptiveMetropolisProposal
        py::classh<BilliardAdaptiveMetropolisProposal, Proposal, ProposalTrampoline<BilliardAdaptiveMetropolisProposal>> billiardBilliardAdaptiveMetropolisProposal(
                m, "BilliardAdaptiveMetropolisProposal", doc::BilliardAdaptiveMetropolisProposal::base);
        // constructor
        billiardBilliardAdaptiveMetropolisProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init([] (const Problem* problem,
                              double boundaryCushion,
                              double eps,
                              unsigned long warmUp,
                              long maxReflections) {
                        MatrixType sqrtMve = computeSqrtMaximumVolumeEllipsoid(*problem);
                        auto proposal = BilliardAdaptiveMetropolisProposal::createFromProblem(problem, sqrtMve, eps, warmUp, maxReflections);
                        proposal.setParameter(ProposalParameter::BOUNDARY_CUSHION, boundaryCushion);
                        return proposal;
                    }),
                    doc::BilliardAdaptiveMetropolisProposal::__init__,
                    py::arg("problem"),
                    py::arg("boundary_cushion") = 0,
                    py::arg("eps") = 1.e-3,
                    py::arg("warm_up") = 100,
                    py::arg("max_reflections") = 100)
            .def(py::init([] (const Problem* problem,
                              const VectorType* startingPoint,
                              double boundaryCushion,
                              double eps,
                              unsigned long warmUp,
                              long maxReflections) {
                        MatrixType sqrtMve = computeSqrtMaximumVolumeEllipsoid(*problem);
                        auto proposal = BilliardAdaptiveMetropolisProposal::create(problem, startingPoint, sqrtMve, eps, warmUp, maxReflections);
                        proposal.setParameter(ProposalParameter::BOUNDARY_CUSHION, boundaryCushion);
                        return proposal;
                    }),
                    doc::BilliardAdaptiveMetropolisProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point") = py::none(),
                    py::arg("boundary_cushion") = 0,
                    py::arg("eps") = 1.e-3,
                    py::arg("warm_up") = 100,
                    py::arg("max_reflections") = 100)
            ;
        // common
        proposal::addCommon<BilliardAdaptiveMetropolisProposal, doc::BilliardAdaptiveMetropolisProposal>(billiardBilliardAdaptiveMetropolisProposal);
        // parameters
        proposal::addParameter<BilliardAdaptiveMetropolisProposal>(
                billiardBilliardAdaptiveMetropolisProposal, ProposalParameter::BOUNDARY_CUSHION, "boundary_cushion", doc::BilliardAdaptiveMetropolisProposal::boundaryCushion);
        proposal::addParameter<BilliardAdaptiveMetropolisProposal>(
                billiardBilliardAdaptiveMetropolisProposal, ProposalParameter::EPSILON, "eps", doc::BilliardAdaptiveMetropolisProposal::epsilon);
        proposal::addParameter<BilliardAdaptiveMetropolisProposal, decltype(billiardBilliardAdaptiveMetropolisProposal), unsigned long>(
                billiardBilliardAdaptiveMetropolisProposal, ProposalParameter::WARM_UP, "warm_up", doc::BilliardAdaptiveMetropolisProposal::warmUp);
        proposal::addParameter<BilliardAdaptiveMetropolisProposal, decltype(billiardBilliardAdaptiveMetropolisProposal), long>(
                billiardBilliardAdaptiveMetropolisProposal, ProposalParameter::MAX_REFLECTIONS, "max_reflections", doc::BilliardAdaptiveMetropolisProposal::maxReflections);
        // pickling
        billiardBilliardAdaptiveMetropolisProposal.def(py::pickle([] (const BilliardAdaptiveMetropolisProposal& self) {
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState(),
                                              self.proposal->getCholeskyOfMaximumVolumeEllipsoid(),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::BOUNDARY_CUSHION)),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::EPSILON)),
                                              std::any_cast<unsigned long>(self.proposal->getParameter(ProposalParameter::WARM_UP)),
                                              std::any_cast<long>(self.proposal->getParameter(ProposalParameter::MAX_REFLECTIONS)));
                    },
                    [] (py::tuple t) {
                        if (t.size() != 8) throw std::runtime_error("Invalid state!");

                        auto p = BilliardAdaptiveMetropolisProposal(t[0].cast<MatrixType>(),
                                                            t[1].cast<VectorType>(),
                                                            t[2].cast<VectorType>(),
                                                            t[3].cast<MatrixType>(),
                                                            t[5].cast<double>(),
                                                            t[6].cast<unsigned long>(),
                                                            t[7].cast<unsigned long>());
                        p.setParameter(ProposalParameter::BOUNDARY_CUSHION, t[4].cast<double>());
                        return p;
                    })
                );



        // register BilliardMALAProposal
        py::classh<BilliardMALAProposal, Proposal, ProposalTrampoline<BilliardMALAProposal>> billiardmalaProposal(
                m, "BilliardMALAProposal", doc::BilliardMALAProposal::base);
        // constructor
        billiardmalaProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init([] (const Problem* problem,
                              double stepSize,
                              long maximumNumberOfReflections) -> BilliardMALAProposal {
                        if (problem) {
                            if (!problem->model) {
                                throw std::runtime_error("Cannot initialize hopsy.BilliardMALAProposal for "
                                        "uniform problem (problem.model == None).");
                            }

                            return BilliardMALAProposal::createFromProblem(
                                    problem, ModelWrapper(problem->model->copyModel()), maximumNumberOfReflections, stepSize);
                        } else {
                            throw std::runtime_error(std::string("Internal error in ") +
                                    std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
                        }
                    }),
                    doc::BilliardMALAProposal::__init__,
                    py::arg("problem"),
                    py::arg("stepsize") = 1.,
                    py::arg("max_reflections") = 100)
            .def(py::init([] (const Problem* problem,
                              const VectorType* startingPoint,
                              double stepSize,
                              long maximumNumberOfReflections) -> BilliardMALAProposal {
                        if (problem) {
                            if (!problem->model) {
                                throw std::runtime_error("Cannot initialize hopsy.BilliardMALAProposal for uniform problem (problem.model == None).");
                            }

                            return BilliardMALAProposal::create(problem, startingPoint, ModelWrapper(problem->model->copyModel()), maximumNumberOfReflections, stepSize);
                        } else {
                            throw std::runtime_error(std::string("Internal error in ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
                        }
                    }),
                    doc::BilliardMALAProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point"),
                    py::arg("stepsize") = 1.,
                    py::arg("max_reflections") = 100)
            ;
        // common
        proposal::addCommon<BilliardMALAProposal, doc::BilliardMALAProposal>(billiardmalaProposal);
        // parameters
        proposal::addParameter<BilliardMALAProposal>(
                billiardmalaProposal, ProposalParameter::MAX_REFLECTIONS, "max_reflections", doc::BilliardMALAProposal::maxReflections);
        proposal::addParameter<BilliardMALAProposal>(
                billiardmalaProposal, ProposalParameter::STEP_SIZE, "stepsize", doc::BilliardMALAProposal::stepSize);
        // pickling
        billiardmalaProposal.def(py::pickle([] (const BilliardMALAProposal& self) {
                        auto model = self.proposal->getModel()->copyModel().release();
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState(),
                                              model,
                                              std::any_cast<long>(self.proposal->getParameter(ProposalParameter::MAX_REFLECTIONS)),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::STEP_SIZE)));
                    },
                    [] (py::tuple t) {
                        if (t.size() != 6) throw std::runtime_error("Invalid state!");

                        return BilliardMALAProposal(t[0].cast<MatrixType>(),
                                               t[1].cast<VectorType>(),
                                               t[2].cast<VectorType>(),
                                               ModelWrapper(t[3].cast<Model*>()->copyModel()),
                                               t[4].cast<long>(),
                                               t[5].cast<double>());
                    })
                );

        // register CSmMALAProposal
        py::classh<CSmMALAProposal, Proposal, ProposalTrampoline<CSmMALAProposal>> csmmalaProposal(
                m, "CSmMALAProposal", doc::CSmMALAProposal::base);
        // constructor
        csmmalaProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init([] (const Problem* problem,
                              double stepSize,
                              double fisherWeight) -> CSmMALAProposal {
                        if (problem) {
                            if (!problem->model) {
                                throw std::runtime_error("Cannot initialize hopsy.CSmMALAProposal for "
                                        "uniform problem (problem.model == None).");
                            }

                            return CSmMALAProposal::createFromProblem(
                                    problem, ModelWrapper(problem->model->copyModel()), fisherWeight, stepSize);
                        } else {
                            throw std::runtime_error(std::string("Internal error in ") +
                                    std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
                        }
                    }),
                    doc::CSmMALAProposal::__init__,
                    py::arg("problem"),
                    py::arg("stepsize") = 1,
                    py::arg("fisher_weight") = 0.5)
            .def(py::init([] (const Problem* problem,
                              const VectorType* startingPoint,
                              double stepSize,
                              double fisherWeight) -> CSmMALAProposal {
                        if (problem) {
                            if (!problem->model) {
                                throw std::runtime_error("Cannot initialize hopsy.CSmMALAProposal for uniform problem (problem.model == None).");
                            }

                            return CSmMALAProposal::create(problem, startingPoint, ModelWrapper(problem->model->copyModel()), fisherWeight, stepSize);
                        } else {
                            throw std::runtime_error(std::string("Internal error in ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
                        }
                    }),
                    doc::CSmMALAProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point"),
                    py::arg("stepsize") = 1,
                    py::arg("fisher_weight") = 0.5)
            ;
        // common
        proposal::addCommon<CSmMALAProposal, doc::CSmMALAProposal>(csmmalaProposal);
        // parameters
        proposal::addParameter<CSmMALAProposal>(
                csmmalaProposal, ProposalParameter::FISHER_WEIGHT, "fisher_weight", doc::CSmMALAProposal::fisherWeight);
        proposal::addParameter<CSmMALAProposal>(
                csmmalaProposal, ProposalParameter::STEP_SIZE, "stepsize", doc::CSmMALAProposal::stepSize);
        // pickling
        csmmalaProposal.def(py::pickle([] (const CSmMALAProposal& self) {
                        auto model = self.proposal->getModel()->copyModel().release();
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState(),
                                              model,
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::FISHER_WEIGHT)),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::STEP_SIZE)));
                    },
                    [] (py::tuple t) {
                        if (t.size() != 6) throw std::runtime_error("Invalid state!");

                        return CSmMALAProposal(t[0].cast<MatrixType>(),
                                               t[1].cast<VectorType>(),
                                               t[2].cast<VectorType>(),
                                               ModelWrapper(t[3].cast<Model*>()->copyModel()),
                                               t[4].cast<double>(),
                                               t[5].cast<double>());
                    })
                );


        // register DikinWalkProposal
        py::classh<DikinWalkProposal, Proposal, ProposalTrampoline<DikinWalkProposal>> dikinWalkProposal(
                m, "DikinWalkProposal", doc::DikinWalkProposal::base);
        // constructor
        dikinWalkProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init([] (const Problem* problem,
                              double stepSize,
                              double boundaryCushion) {
                        auto proposal = DikinWalkProposal::createFromProblem(problem, stepSize);
                        proposal.setParameter(ProposalParameter::BOUNDARY_CUSHION, boundaryCushion);
                        return proposal;
                    }),
                    doc::DikinWalkProposal::__init__,
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
                    doc::DikinWalkProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point") = py::none(),
                    py::arg("stepsize") = 1,
                    py::arg("boundary_cushion") = 0)
            ;
        // common
        proposal::addCommon<DikinWalkProposal, doc::DikinWalkProposal>(dikinWalkProposal);
        // parameters
        proposal::addParameter<DikinWalkProposal>(
                dikinWalkProposal, ProposalParameter::BOUNDARY_CUSHION, "boundary_cushion", doc::DikinWalkProposal::boundaryCushion);
        proposal::addParameter<DikinWalkProposal>(
                dikinWalkProposal, ProposalParameter::STEP_SIZE, "stepsize", doc::DikinWalkProposal::stepSize);
        // pickling
        dikinWalkProposal.def(py::pickle([] (const DikinWalkProposal& self) {
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState(),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::STEP_SIZE)));
                    },
                    [] (py::tuple t) {
                        if (t.size() != 4) throw std::runtime_error("Invalid state!");

                        return DikinWalkProposal(t[0].cast<MatrixType>(),
                                             t[1].cast<VectorType>(),
                                             t[2].cast<VectorType>(),
                                             t[3].cast<double>());
                    })
                );



        // register GaussianCoordinateHitAndRun
        py::classh<GaussianCoordinateHitAndRunProposal, Proposal, ProposalTrampoline<GaussianCoordinateHitAndRunProposal>> gaussianCoordinateHitAndRunProposal(
                m, "GaussianCoordinateHitAndRunProposal", doc::GaussianCoordinateHitAndRunProposal::base);
        // constructor
        gaussianCoordinateHitAndRunProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init(&GaussianCoordinateHitAndRunProposal::createFromProblem),
                    doc::GaussianCoordinateHitAndRunProposal::__init__,
                    py::arg("problem"),
                    py::arg("stepsize") = 1)
            .def(py::init(&GaussianCoordinateHitAndRunProposal::create),
                    doc::GaussianCoordinateHitAndRunProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point"),
                    py::arg("stepsize") = 1)
            ;
		// common
        proposal::addCommon<GaussianCoordinateHitAndRunProposal, doc::GaussianCoordinateHitAndRunProposal>(gaussianCoordinateHitAndRunProposal);
        // parameters
        proposal::addParameter<GaussianCoordinateHitAndRunProposal>(
                gaussianCoordinateHitAndRunProposal, ProposalParameter::STEP_SIZE, "stepsize", doc::GaussianCoordinateHitAndRunProposal::stepSize);
        // pickling
        gaussianCoordinateHitAndRunProposal.def(py::pickle([] (const GaussianCoordinateHitAndRunProposal& self) {
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState(),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::STEP_SIZE)));
                    },
                    [] (py::tuple t) {
                        if (t.size() != 4) throw std::runtime_error("Invalid state!");

                        return GaussianCoordinateHitAndRunProposal(t[0].cast<MatrixType>(),
                                                                   t[1].cast<VectorType>(),
                                                                   t[2].cast<VectorType>(),
                                                                   t[3].cast<double>());
                    })
                );


        // register GaussianHitAndRun
        py::classh<GaussianHitAndRunProposal, Proposal, ProposalTrampoline<GaussianHitAndRunProposal>> gaussianHitAndRunProposal(
                m, "GaussianHitAndRunProposal", doc::GaussianCoordinateHitAndRunProposal::base);
        // constructor
        gaussianHitAndRunProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init(&GaussianHitAndRunProposal::createFromProblem),
                    doc::GaussianCoordinateHitAndRunProposal::__init__,
                    py::arg("problem"),
                    py::arg("stepsize") = 1)
            .def(py::init(&GaussianHitAndRunProposal::create),
                    doc::GaussianCoordinateHitAndRunProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point"),
                    py::arg("stepsize") = 1)
            ;
        // common
        proposal::addCommon<GaussianHitAndRunProposal, doc::GaussianHitAndRunProposal>(gaussianHitAndRunProposal);
        // parameters
        proposal::addParameter<GaussianHitAndRunProposal>(
                gaussianHitAndRunProposal, ProposalParameter::STEP_SIZE, "stepsize", doc::GaussianHitAndRunProposal::stepSize);
        // pickling
        gaussianHitAndRunProposal.def(py::pickle([] (const GaussianHitAndRunProposal& self) {
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState(),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::STEP_SIZE)));
                    },
                    [] (py::tuple t) {
                        if (t.size() != 4) throw std::runtime_error("Invalid state!");

                        return GaussianHitAndRunProposal(t[0].cast<MatrixType>(),
                                                         t[1].cast<VectorType>(),
                                                         t[2].cast<VectorType>(),
                                                         t[3].cast<double>());
                    })
                );


        // register GaussianProposal
        py::classh<GaussianProposal, Proposal, ProposalTrampoline<GaussianProposal>> gaussianProposal(
                m, "GaussianProposal", doc::GaussianProposal::base);
        // constructor
        gaussianProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init(&GaussianProposal::createFromProblem),
                    doc::GaussianProposal::__init__,
                    py::arg("problem"),
                    py::arg("stepsize") = 1)
            .def(py::init(&GaussianProposal::create),
                    doc::GaussianProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point"),
                    py::arg("stepsize") = 1)
            ;
        // common
        proposal::addCommon<GaussianProposal, doc::GaussianProposal>(gaussianProposal);
        // parameters
        proposal::addParameter<GaussianProposal>(
                gaussianProposal, ProposalParameter::STEP_SIZE, "stepsize", doc::GaussianProposal::stepSize);
        // pickling
        gaussianProposal.def(py::pickle([] (const GaussianProposal& self) {
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState(),
                                              std::any_cast<double>(self.proposal->getParameter(ProposalParameter::STEP_SIZE)));
                    },
                    [] (py::tuple t) {
                        if (t.size() != 4) throw std::runtime_error("Invalid state!");

                        return GaussianProposal(t[0].cast<MatrixType>(),
                                                t[1].cast<VectorType>(),
                                                t[2].cast<VectorType>(),
                                                t[3].cast<double>()
                                );
                    })
                );


        // register PyProposal
        py::classh<PyProposal, Proposal, ProposalTrampoline<PyProposal>>
                pyProposal(m, "PyProposal", doc::PyProposal::base);
        // constructor
        pyProposal.def(py::init<py::object>(), doc::PyProposal::__init__, py::arg("proposal"));
        // common
        proposal::addCommon<PyProposal, doc::PyProposal>(pyProposal);
        // pickling
        pyProposal.def(py::pickle([] (const PyProposal& self) {
                        return py::make_tuple(self.pyObj);
                    },
                    [] (py::tuple t) {
                        if (t.size() != 1) throw std::runtime_error("Invalid state!");

                        return PyProposal(t[0].cast<py::object>());
                    })
                );

        // register TruncatedGaussianProposal
        py::classh<TruncatedGaussianProposal, Proposal, ProposalTrampoline<TruncatedGaussianProposal>> truncatedgaussianProposal(
                m, "TruncatedGaussianProposal", doc::TruncatedGaussianProposal::base);
        // constructor
        truncatedgaussianProposal
                //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
                .def(py::init([] (const Problem* problem) -> TruncatedGaussianProposal {
                         if (problem) {
                             if (!problem->model) {
                                 throw std::runtime_error("Cannot initialize hopsy.TruncatedGaussianProposal for "
                                                          "uniform problem (problem.model == None).");
                             }
                             std::shared_ptr<hops::Model> modelPtr = problem->model->copyModel();
                             std::shared_ptr<Gaussian> casted = std::dynamic_pointer_cast<Gaussian>(modelPtr);
                             if (!casted) {
                                 throw std::runtime_error("Model is not Gaussian. Please reconsider.");
                             }
                             hops::Gaussian gaussian(casted->getMean(), casted->getCovariance());

                             return TruncatedGaussianProposal::createFromProblem(
                                     problem,
                                     gaussian
                                     );
                         } else {
                             throw std::runtime_error(std::string("Internal error in ") +
                                                      std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
                         }
                     }),
                     doc::TruncatedGaussianProposal::__init__,
                     py::arg("problem"))
                .def(py::init([] (const Problem* problem,
                                  const VectorType* startingPoint) -> TruncatedGaussianProposal {
                         if (problem) {
                             if (!problem->model) {
                                 throw std::runtime_error("Cannot initialize hopsy.TruncatedGaussianProposal for uniform problem (problem.model == None).");
                             }
                             std::shared_ptr<hops::Model> modelPtr = problem->model->copyModel();
                             std::shared_ptr<Gaussian> casted = std::dynamic_pointer_cast<Gaussian>(modelPtr);
                             if (!casted) {
                                 throw std::runtime_error("Model is not Gaussian. Please reconsider.");
                             }
                             hops::Gaussian gaussian(casted->getMean(), casted->getCovariance());

                             return TruncatedGaussianProposal::create(
                                     problem,
                                     startingPoint,
                                     gaussian
                             );
                         } else {
                             throw std::runtime_error(std::string("Internal error in ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "!!!");
                         }
                     }),
                     doc::TruncatedGaussianProposal::__init__,
                     py::arg("problem"),
                     py::arg("starting_point"));
        // common
        proposal::addCommon<TruncatedGaussianProposal, doc::TruncatedGaussianProposal>(truncatedgaussianProposal);
        // pickling
        truncatedgaussianProposal.def(py::pickle([] (const TruncatedGaussianProposal& self) {
                                           std::shared_ptr<Model> modelPtr = self.proposal->getModel()->copyModel();
                                           auto casted = std::dynamic_pointer_cast<hops::Gaussian>(modelPtr);
                                           if (!casted) {
                                             throw std::runtime_error("Model is not Gaussian. Please reconsider.");
                                           }

                                           return py::make_tuple(self.proposal->getA(),
                                                                 self.proposal->getB(),
                                                                 self.proposal->getState(),
                                                                 casted->getMean(),
                                                                 casted->getCovariance());
                                       },
                                       [] (py::tuple t) {
                                           if (t.size() != 5) throw std::runtime_error("Invalid state!");

                                           hops::Gaussian gaussian(t[3].cast<VectorType>(), t[4].cast<MatrixType>());

                                           return TruncatedGaussianProposal(t[0].cast<MatrixType>(),
                                                                  t[1].cast<VectorType>(),
                                                                  t[2].cast<VectorType>(),
                                                                  gaussian);
                                       })
        );

        // register UniformCoordinateHitAndRun
        py::classh<UniformCoordinateHitAndRunProposal, Proposal, ProposalTrampoline<UniformCoordinateHitAndRunProposal>> uniformCoordinateHitAndRunProposal(
                m, "UniformCoordinateHitAndRunProposal", doc::UniformCoordinateHitAndRunProposal::base);
        // constructor
        uniformCoordinateHitAndRunProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init(&UniformCoordinateHitAndRunProposal::createFromProblem),
                    doc::UniformCoordinateHitAndRunProposal::__init__,
                    py::arg("problem"))
            .def(py::init(&UniformCoordinateHitAndRunProposal::create),
                    doc::UniformCoordinateHitAndRunProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point"))
            ;
        // common
        proposal::addCommon<UniformCoordinateHitAndRunProposal, doc::UniformCoordinateHitAndRunProposal>(uniformCoordinateHitAndRunProposal);
        // pickling
        uniformCoordinateHitAndRunProposal.def(py::pickle([] (const UniformCoordinateHitAndRunProposal& self) {
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState());
                    },
                    [] (py::tuple t) {
                        if (t.size() != 3) throw std::runtime_error("Invalid state!");

                        return UniformCoordinateHitAndRunProposal(t[0].cast<MatrixType>(),
                                                                  t[1].cast<VectorType>(),
                                                                  t[2].cast<VectorType>());
                    })
                );


        // register UniformHitAndRun
        py::classh<UniformHitAndRunProposal, Proposal, ProposalTrampoline<UniformHitAndRunProposal>> uniformHitAndRunProposal(
                m, "UniformHitAndRunProposal", doc::UniformHitAndRunProposal::base);
        // constructor
        uniformHitAndRunProposal
            //.def(py::init<>()) # TODO solve re-initialization of empty proposals in markov chain before allowing default constructor
            .def(py::init(&UniformHitAndRunProposal::createFromProblem),
                    doc::UniformHitAndRunProposal::__init__,
                    py::arg("problem"))
            .def(py::init(&UniformHitAndRunProposal::create),
                    doc::UniformHitAndRunProposal::__init__,
                    py::arg("problem"),
                    py::arg("starting_point"))
            ;
        // common
        proposal::addCommon<UniformHitAndRunProposal, doc::UniformHitAndRunProposal>(uniformHitAndRunProposal);
        // pickling
        uniformHitAndRunProposal.def(py::pickle([] (const UniformHitAndRunProposal& self) {
                        return py::make_tuple(self.proposal->getA(),
                                              self.proposal->getB(),
                                              self.proposal->getState());
                    },
                    [] (py::tuple t) {
                        if (t.size() != 3) throw std::runtime_error("Invalid state!");

                        return UniformHitAndRunProposal(t[0].cast<MatrixType>(),
                                                        t[1].cast<VectorType>(),
                                                        t[2].cast<VectorType>());
                    })
                );

    }
} // namespace hopsy

#endif // HOPSY_PROPOSAL_HPP
