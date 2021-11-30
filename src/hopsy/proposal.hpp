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

#include "misc.hpp"

namespace py = pybind11;

namespace hopsy {
	using Proposal = hops::Proposal;
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Proposal);

namespace hopsy {
    template<typename ProposalBase = Proposal>
	class ProposalTrampoline : public ProposalBase, public py::trampoline_self_life_support {
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
			PYBIND11_OVERRIDE_PURE_NAME(
				std::unique_ptr<Proposal>,  // Return type 
				ProposalBase,               // Parent class
                "deepcopy",                 // Python function name
				deepCopy                    // C++ function name
            );
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

        std::unique_ptr<Proposal> deepCopy() const override {
            return pyObj.attr("deepcopy")().cast<std::unique_ptr<Proposal>>();
        }
	private:
		py::object pyObj;
    };


    using AdaptiveMetropolis = hops::AdaptiveMetropolisProposal<MatrixType, VectorType>;
    //using BallWalk = hops::BallWalk;
    using CoordinateHitAndRun = hops::CoordinateHitAndRunProposal<MatrixType, VectorType>;
    using CSmMALA = hops::CSmMALAProposal<MatrixType, VectorType>;
    using DikinWalk = hops::DikinProposal<MatrixType, VectorType>;
    using Gaussian = hops::GaussianProposal<MatrixType, VectorType>;
    using HitAndRun = hops::HitAndRunProposal<MatrixType, VectorType>;

    using MarkovChain = hops::MarkovChainPrototypeImpl;

} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::AdaptiveMetropolis);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::CoordinateHitAndRun);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::CSmMALA);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::DikinWalk);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Gaussian);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::HitAndRun);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::PyProposal);

#endif // HOPSY_PROPOSAL_HPP
