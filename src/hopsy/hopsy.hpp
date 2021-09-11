#ifndef HOPSY_HPP
#define HOPSY_HPP

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include "../../extern/hops/include/hops/hops.hpp"
#include "hopsy_linprog.hpp"

#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hopsy {
    class PyModel {
	public:
        using MatrixType = Eigen::MatrixXd;
        using VectorType = Eigen::VectorXd;

		PyModel(py::object pyObj) : pyObj(std::move(pyObj)) {};

		double computeNegativeLogLikelihood(const Eigen::VectorXd& x) const {
			return pyObj.attr("compute_negative_log_likelihood")(x).cast<double>();
		}

        Eigen::MatrixXd computeExpectedFisherInformation(const Eigen::VectorXd& x) const {
			return pyObj.attr("compute_expected_fisher_information")(x).cast<Eigen::MatrixXd>();
		}

        Eigen::VectorXd computeLogLikelihoodGradient(const Eigen::VectorXd& x) const {
			return pyObj.attr("compute_log_likelihood_gradient")(x).cast<Eigen::VectorXd>();
		}

	private:
		py::object pyObj;
	};

    class PyProposal {
    public:
        using StateType = Eigen::VectorXd;

        PyProposal() = default;

		PyProposal(py::object pyObj) : pyObj(std::move(pyObj)) {};

        void propose(hops::RandomNumberGenerator&) {
            pyObj.attr("propose")();
        }

        void acceptProposal() {
            pyObj.attr("accept_proposal")();
        }

        double computeLogAcceptanceProbability() {
            return pyObj.attr("compute_log_acceptance_probability")().cast<double>();
        }

        Eigen::VectorXd getState() const {
            return pyObj.attr("get_state")().cast<Eigen::VectorXd>();
        }

        void setState(Eigen::VectorXd newState) {
            pyObj.attr("set_state")(newState);
        }

        Eigen::VectorXd getProposal() const {
            return pyObj.attr("get_proposal")().cast<Eigen::VectorXd>();
        }

        double getStepSize() const {
            return pyObj.attr("get_stepsize")().cast<double>();
        }

        void setStepSize(double newStepSize) {
            pyObj.attr("set_stepsize")(newStepSize);
        }

        std::string getName() {
            return pyObj.attr("get_name")().cast<std::string>();
        }
	private:
		py::object pyObj;
    };


    typedef hops::DegenerateMultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd> DegenerateMultivariateGaussianModel;
    typedef hops::DynMultimodalModel<PyModel> MixtureModel;
    typedef hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd> MultivariateGaussianModel;
    typedef hops::RosenbrockModel<Eigen::MatrixXd, Eigen::VectorXd> RosenbrockModel;
    typedef hops::UniformDummyModel<Eigen::MatrixXd, Eigen::VectorXd> UniformModel;

    typedef hops::Problem<DegenerateMultivariateGaussianModel> DegenerateMultivariateGaussianProblem;
    typedef hops::Problem<MixtureModel> MixtureProblem;
    typedef hops::Problem<MultivariateGaussianModel> MultivariateGaussianProblem;
    typedef hops::Problem<RosenbrockModel> RosenbrockProblem;
    typedef hops::Problem<UniformModel> UniformProblem;
    typedef hops::Problem<PyModel> PyProblem;

	typedef hops::Run<DegenerateMultivariateGaussianModel> DegenerateMultivariateGaussianRun;
    typedef hops::Run<MixtureModel> MixtureRun;
    typedef hops::Run<MultivariateGaussianModel> MultivariateGaussianRun;
    typedef hops::Run<RosenbrockModel> RosenbrockRun;
    typedef hops::Run<UniformModel> UniformRun;
    typedef hops::Run<PyModel> PyRun;

	typedef hops::RunBase<DegenerateMultivariateGaussianModel, PyProposal> DegenerateMultivariateGaussianPyProposalRun;
    typedef hops::RunBase<MixtureModel, PyProposal> MixturePyProposalRun;
    typedef hops::RunBase<MultivariateGaussianModel, PyProposal> MultivariateGaussianPyProposalRun;
    typedef hops::RunBase<RosenbrockModel, PyProposal> RosenbrockPyProposalRun;
    typedef hops::RunBase<UniformModel, PyProposal> UniformPyProposalRun;
    typedef hops::RunBase<PyModel, PyProposal> PyPyProposalRun;


	hops::Problem<UniformModel> createUniformProblem(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
		return hops::Problem<UniformModel>(A, b);
	}

	hops::Problem<PyModel> createPyProblem(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, py::object& pyObj) {
		return hops::Problem<PyModel>(A, b, PyModel(pyObj));
	}


	template<typename T>
	hops::Problem<T> createProblem(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const T& t) {
		if constexpr(std::is_same<T, DegenerateMultivariateGaussianModel>::value) {
			return hops::Problem<DegenerateMultivariateGaussianModel>(A, b, t);
		}
		if constexpr(std::is_same<T, MixtureModel>::value) {
			return hops::Problem<MixtureModel>(A, b, t);
		}
		if constexpr(std::is_same<T, MultivariateGaussianModel>::value) {
			return hops::Problem<MultivariateGaussianModel>(A, b, t);
		}
		if constexpr(std::is_same<T, RosenbrockModel>::value) {
			return hops::Problem<RosenbrockModel>(A, b, t);
		}
		if constexpr(std::is_same<T, UniformModel>::value) {
			return hops::Problem<UniformModel>(A, b, t);
		}
		if constexpr(std::is_same<T, PyModel>::value) {
			return hops::Problem<PyModel>(A, b, t);
		}
	}


	template<typename T>
	hops::Run<T> createRun(const hops::Problem<T>& t, 
                           std::string chainTypeString = "HitAndRun", 
                           unsigned long numberOfSamples = 1000, 
                           unsigned long numberOfChains = 1,
                           std::vector<Eigen::VectorXd> startingPoints = std::vector<Eigen::VectorXd>()) {
		hops::MarkovChainType chainType;
		if (chainTypeString == "BallWalk") {
			chainType = hops::MarkovChainType::BallWalk;
		} else if (chainTypeString == "CoordinateHitAndRun") {
			chainType = hops::MarkovChainType::CoordinateHitAndRun;
		} else if (chainTypeString == "DikinWalk") {
			chainType = hops::MarkovChainType::DikinWalk;
		} else if (chainTypeString == "Gaussian") {
			chainType = hops::MarkovChainType::Gaussian;
		} else if (chainTypeString == "HitAndRun") {
			chainType = hops::MarkovChainType::HitAndRun;
		}

        // initialize missing starting points with chebyshev center
        if (startingPoints.size() < numberOfChains) {
            Eigen::VectorXd chebyshevCenter = computeChebyshevCenter(t);
            for (size_t i = startingPoints.size(); i < numberOfChains; ++i) {
                startingPoints.push_back(chebyshevCenter);
            }
        }

        auto run = hops::Run<T>(t, chainType, numberOfSamples, numberOfChains);
        run.setStartingPoints(startingPoints);
        return run;
	}

	template<typename T>
	hops::RunBase<T, PyProposal> createRunFromPyProposal(const hops::Problem<T>& t, 
                                                         PyProposal proposal, 
                                                         unsigned long numberOfSamples = 1000, 
                                                         unsigned long numberOfChains = 1,
                                                         std::vector<Eigen::VectorXd> startingPoints = std::vector<Eigen::VectorXd>()) {
        // initialize missing starting points with chebyshev center
        if (startingPoints.size() < numberOfChains) {
            Eigen::VectorXd chebyshevCenter = computeChebyshevCenter(t);
            for (size_t i = startingPoints.size(); i < numberOfChains; ++i) {
                startingPoints.push_back(chebyshevCenter);
            }
        }

        auto run = hops::RunBase<T, PyProposal>(t, proposal, numberOfSamples, numberOfChains);
        run.setStartingPoints(startingPoints);
        return run;
	}


	template<typename T>
	hops::RunBase<T, PyProposal> createRunFromPyObject(const hops::Problem<T>& t, 
                                                       py::object pyObj, 
                                                       unsigned long numberOfSamples = 1000, 
                                                       unsigned long numberOfChains = 1,
                                                       std::vector<Eigen::VectorXd> startingPoints = std::vector<Eigen::VectorXd>()) {
        return createRunFromPyProposal(t, PyProposal(pyObj), numberOfSamples, numberOfChains, startingPoints);
	}

    std::pair<Eigen::MatrixXd, Eigen::VectorXd> addBoxConstraintsToMatrixVector(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& lowerBounds, const Eigen::VectorXd& upperBounds) {
        Eigen::MatrixXd newA(A.rows() + 2*A.cols(), A.cols());
        Eigen::VectorXd newB(newA.rows());

        newA << A, Eigen::MatrixXd::Identity(A.cols(), A.cols()), -Eigen::MatrixXd::Identity(A.cols(), A.cols());
        newB << b, upperBounds, -lowerBounds;

        return {newA, newB};
    }

    std::pair<Eigen::MatrixXd, Eigen::VectorXd> addBoxConstraintsToMatrixVector(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, double lowerBound, double upperBound) {
        Eigen::VectorXd lowerBounds = lowerBound * Eigen::VectorXd::Ones(A.cols());
        Eigen::VectorXd upperBounds = upperBound * Eigen::VectorXd::Ones(A.cols());
        return addBoxConstraintsToMatrixVector(A, b, lowerBounds, upperBounds);
    }

    template<typename Problem>
    Problem addBoxConstraintsToProblem(const Problem& problem, const Eigen::VectorXd& lowerBounds, const Eigen::VectorXd& upperBounds) {
        auto[A, b] = addBoxConstraintsToMatrixVector(problem.getA(), problem.getB(), lowerBounds, upperBounds);
        Problem newProblem(problem);
        newProblem.setA(A);
        newProblem.setB(b);
        return newProblem;
    }

    template<typename Problem>
    Problem addBoxConstraintsToProblem(const Problem& problem, double lowerBound, double upperBound) {
        Eigen::VectorXd lowerBounds = lowerBound * Eigen::VectorXd::Ones(problem.getA().cols());
        Eigen::VectorXd upperBounds = upperBound * Eigen::VectorXd::Ones(problem.getA().cols());
        return addBoxConstraintsToProblem(problem, lowerBounds, upperBounds);
    }
}

#endif // HOPSY_HPP
