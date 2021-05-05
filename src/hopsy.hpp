#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include "../extern/hops/include/hops/hops.hpp"

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

		double calculateNegativeLogLikelihood(const Eigen::VectorXd& x) const {
			return pyObj.attr("calculate_negative_log_likelihood")(x).cast<double>();
		}

        Eigen::MatrixXd calculateExpectedFisherInformation(const Eigen::VectorXd& x) const {
			return pyObj.attr("calculate_expected_fisher_information")(x).cast<Eigen::MatrixXd>();
		}

        Eigen::VectorXd calculateLogLikelihoodGradient(const Eigen::VectorXd& x) const {
			return pyObj.attr("calculate_log_likelihood_gradient")(x).cast<Eigen::VectorXd>();
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

        double calculateLogAcceptanceProbability() {
            return pyObj.attr("get_log_acceptance_probability")().cast<double>();
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
    typedef hops::DynMultimodalModel<DegenerateMultivariateGaussianModel> MultimodalMultivariateGaussianModel;
    typedef hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd> MultivariateGaussianModel;
    typedef hops::RosenbrockModel<Eigen::MatrixXd, Eigen::VectorXd> RosenbrockModel;
    typedef hops::UniformDummyModel<Eigen::MatrixXd, Eigen::VectorXd> UniformModel;
    typedef hops::UniformDummyModel<Eigen::MatrixXd, Eigen::VectorXd> UniformModel;

    typedef hops::Problem<DegenerateMultivariateGaussianModel> DegenerateMultivariateGaussianProblem;
    typedef hops::Problem<MultimodalMultivariateGaussianModel> MultimodalMultivariateGaussianProblem;
    typedef hops::Problem<MultivariateGaussianModel> MultivariateGaussianProblem;
    typedef hops::Problem<RosenbrockModel> RosenbrockProblem;
    typedef hops::Problem<UniformModel> UniformProblem;
    typedef hops::Problem<PyModel> PyProblem;

	typedef hops::Run<DegenerateMultivariateGaussianModel> DegenerateMultivariateGaussianRun;
    typedef hops::Run<MultimodalMultivariateGaussianModel> MultimodalMultivariateGaussianRun;
    typedef hops::Run<MultivariateGaussianModel> MultivariateGaussianRun;
    typedef hops::Run<RosenbrockModel> RosenbrockRun;
    typedef hops::Run<UniformModel> UniformRun;
    typedef hops::Run<PyModel> PyRun;

	typedef hops::RunBase<DegenerateMultivariateGaussianModel, PyProposal> DegenerateMultivariateGaussianPyProposalRun;
    typedef hops::RunBase<MultimodalMultivariateGaussianModel, PyProposal> MultimodalMultivariateGaussianPyProposalRun;
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
		if constexpr(std::is_same<T, MultimodalMultivariateGaussianModel>::value) {
			return hops::Problem<MultimodalMultivariateGaussianModel>(A, b, t);
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
                           unsigned long numberOfChains = 1) {
		hops::MarkovChainType chainType;
		if (chainTypeString == "BallWalk") {
			chainType = hops::MarkovChainType::BallWalk;
		} else if (chainTypeString == "CoordinateHitAndRun") {
			chainType = hops::MarkovChainType::CoordinateHitAndRun;
		} else if (chainTypeString == "CSmMALA") {
			chainType = hops::MarkovChainType::CSmMALA;
		} else if (chainTypeString == "CSmMALANoGradient") {
			chainType = hops::MarkovChainType::CSmMALANoGradient;
		} else if (chainTypeString == "DikinWalk") {
			chainType = hops::MarkovChainType::DikinWalk;
		} else if (chainTypeString == "Gaussian") {
			chainType = hops::MarkovChainType::Gaussian;
		} else if (chainTypeString == "HitAndRun") {
			chainType = hops::MarkovChainType::HitAndRun;
		}

		if constexpr(std::is_same<T, DegenerateMultivariateGaussianModel>::value) {
			return hops::Run<DegenerateMultivariateGaussianModel>(t, chainType, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, MultimodalMultivariateGaussianModel>::value) {
			return hops::Run<MultimodalMultivariateGaussianModel>(t, chainType, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, MultivariateGaussianModel>::value) {
			return hops::Run<MultivariateGaussianModel>(t, chainType, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, RosenbrockModel>::value) {
			return hops::Run<RosenbrockModel>(t, chainType, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, UniformModel>::value) {
			return hops::Run<UniformModel>(t, chainType, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, PyModel>::value) {
			return hops::Run<PyModel>(t, chainType, numberOfSamples, numberOfChains);
		}
	}

	template<typename T>
	hops::RunBase<T, PyProposal> createRunFromPyProposal(const hops::Problem<T>& t, 
                                                         PyProposal proposal, 
                                                         unsigned long numberOfSamples = 1000, 
                                                         unsigned long numberOfChains = 1) {
		if constexpr(std::is_same<T, DegenerateMultivariateGaussianModel>::value) {
			return hops::RunBase<DegenerateMultivariateGaussianModel, PyProposal>(t, proposal, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, MultimodalMultivariateGaussianModel>::value) {
			return hops::RunBase<MultimodalMultivariateGaussianModel, PyProposal>(t, proposal, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, MultivariateGaussianModel>::value) {
			return hops::RunBase<MultivariateGaussianModel, PyProposal>(t, proposal, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, RosenbrockModel>::value) {
			return hops::RunBase<RosenbrockModel, PyProposal>(t, proposal, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, UniformModel>::value) {
			return hops::RunBase<UniformModel, PyProposal>(t, proposal, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, PyModel>::value) {
			return hops::RunBase<PyModel, PyProposal>(t, proposal, numberOfSamples, numberOfChains);
		}
	}


	template<typename T>
	hops::RunBase<T, PyProposal> createRunFromPyObject(const hops::Problem<T>& t, 
                                                      py::object pyObj, 
                                                      unsigned long numberOfSamples = 1000, 
                                                      unsigned long numberOfChains = 1) {
        return createRunFromPyProposal(t, PyProposal(pyObj), numberOfSamples, numberOfChains);
	}
}

