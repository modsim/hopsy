#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include "../extern/hops/include/hops/hops.hpp"

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
	hops::Run<T> Run(const hops::Problem<T>& t, 
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
}

PYBIND11_MODULE(hopsy, m) {
    //  
    // Model constructors
    // ==================
    //
    py::class_<hopsy::DegenerateMultivariateGaussianModel>(m, "DegenerateMultivariateGaussianModel")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>(),
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)))
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd, std::vector<long>>(),
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)),
                py::arg("inactives") = std::vector<long>());

    py::class_<hopsy::MultimodalMultivariateGaussianModel>(m, "MultimodalMultivariateGaussianModel")
        .def(py::init<std::vector<hopsy::DegenerateMultivariateGaussianModel>>());

    py::class_<hopsy::MultivariateGaussianModel>(m, "MultivariateGaussianModel")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>(),
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)));

    py::class_<hopsy::PyModel>(m, "PyModel")
        .def(py::init<py::object>());

    py::class_<hopsy::RosenbrockModel>(m, "RosenbrockModel")
        .def(py::init<double, Eigen::VectorXd>());

    py::class_<hopsy::UniformModel>(m, "UniformModel")
        .def(py::init<>());


    //  
    // Problem constructors
    // ====================
    //
    py::class_<hopsy::DegenerateMultivariateGaussianProblem>(m, "DegenerateMultivariateGaussianProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::DegenerateMultivariateGaussianModel>());

    py::class_<hopsy::MultimodalMultivariateGaussianProblem>(m, "MultimodalMultivariateGaussianProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::MultimodalMultivariateGaussianModel>());

    py::class_<hopsy::MultivariateGaussianProblem>(m, "MultivariateGaussianProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::MultivariateGaussianModel>());

    py::class_<hopsy::RosenbrockProblem>(m, "RosenbrockProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::RosenbrockModel>());

    py::class_<hopsy::UniformProblem>(m, "UniformProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd>())
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::UniformModel>());

    py::class_<hopsy::PyProblem>(m, "PyProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::PyModel>());


    //  
    // Problem factory method
    // ======================
    //
    // Instead of having to construct the correct problem from a given model, this method 
    // simulates a general problem constructor which then statically checks the passed model
    // type and returns the correctly instantiated problem type
    //
	m.def("Problem", &hopsy::createProblem<hopsy::DegenerateMultivariateGaussianModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::MultimodalMultivariateGaussianModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::MultivariateGaussianModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::PyModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::RosenbrockModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::UniformModel>);
	m.def("Problem", &hopsy::createUniformProblem);
	m.def("Problem", &hopsy::createPyProblem);


    py::class_<hopsy::DegenerateMultivariateGaussianRun>(m, "DegenerateMultivariateGaussianRun")
        .def(py::init<hopsy::DegenerateMultivariateGaussianProblem>())
        .def(py::init<hopsy::DegenerateMultivariateGaussianRun&>())
        .def("get_data", &hopsy::DegenerateMultivariateGaussianRun::getData)
        .def("init", &hopsy::DegenerateMultivariateGaussianRun::init)
        .def("sample", py::overload_cast<>(&hopsy::DegenerateMultivariateGaussianRun::sample))
        .def("sample", py::overload_cast<unsigned long, unsigned long>(
                    &hopsy::DegenerateMultivariateGaussianRun::sample), py::arg("number_of_samples"), py::arg("thinning") = 1)
        .def("set_problem", &hopsy::DegenerateMultivariateGaussianRun::setProblem)
        .def("get_problem", &hopsy::DegenerateMultivariateGaussianRun::getProblem)
        .def("set_starting_points", &hopsy::DegenerateMultivariateGaussianRun::setStartingPoints)
        .def("get_starting_points", &hopsy::DegenerateMultivariateGaussianRun::getStartingPoints)
        .def("set_markov_chain_type", &hopsy::DegenerateMultivariateGaussianRun::setMarkovChainType)
        .def("get_markovv_chain_type", &hopsy::DegenerateMultivariateGaussianRun::getMarkovChainType)
        .def("set_number_of_chains", &hopsy::DegenerateMultivariateGaussianRun::setNumberOfChains)
        .def("get_number_of_chains", &hopsy::DegenerateMultivariateGaussianRun::getNumberOfChains)
        .def("set_number_of_samples", &hopsy::DegenerateMultivariateGaussianRun::setNumberOfSamples)
        .def("get_number_of_samples", &hopsy::DegenerateMultivariateGaussianRun::getNumberOfSamples)
        .def("set_thinning", &hopsy::DegenerateMultivariateGaussianRun::setThinning)
        .def("get_thinning", &hopsy::DegenerateMultivariateGaussianRun::getThinning)
        .def("enable_rounding", &hopsy::DegenerateMultivariateGaussianRun::enableRounding)
        .def("disable_rounding", &hopsy::DegenerateMultivariateGaussianRun::disableRounding)
        .def("set_stepsize", &hopsy::DegenerateMultivariateGaussianRun::setStepSize)
        .def("get_stepsize", &hopsy::DegenerateMultivariateGaussianRun::getStepSize)
        .def("set_fisher_weight", &hopsy::DegenerateMultivariateGaussianRun::setFisherWeight)
        .def("get_fisher_weight", &hopsy::DegenerateMultivariateGaussianRun::getFisherWeight)
        .def("set_random_seed", &hopsy::DegenerateMultivariateGaussianRun::setRandomSeed)
        .def("get_random_seed", &hopsy::DegenerateMultivariateGaussianRun::getRandomSeed)
        .def("unset_sampling_until_convergence", &hopsy::DegenerateMultivariateGaussianRun::unsetSamplingUntilConvergence)
        .def("set_sampling_until_convergence", &hopsy::DegenerateMultivariateGaussianRun::setSamplingUntilConvergence)
        .def("get_diagnostics_threshold", &hopsy::DegenerateMultivariateGaussianRun::getDiagnosticsThreshold)
        .def("get_max_repetitions", &hopsy::DegenerateMultivariateGaussianRun::getMaxRepetitions);

    py::class_<hopsy::MultimodalMultivariateGaussianRun>(m, "MultimodalMultivariateGaussianRun")
        .def(py::init<hopsy::MultimodalMultivariateGaussianProblem>())
        .def(py::init<hopsy::MultimodalMultivariateGaussianRun&>())
        .def("get_data", &hopsy::MultimodalMultivariateGaussianRun::getData)
        .def("init", &hopsy::MultimodalMultivariateGaussianRun::init)
        .def("sample", py::overload_cast<>(&hopsy::MultimodalMultivariateGaussianRun::sample))
        .def("sample", py::overload_cast<unsigned long, unsigned long>(
                    &hopsy::MultimodalMultivariateGaussianRun::sample), py::arg("number_of_samples"), py::arg("thinning") = 1)
        .def("set_problem", &hopsy::MultimodalMultivariateGaussianRun::setProblem)
        .def("get_problem", &hopsy::MultimodalMultivariateGaussianRun::getProblem)
        .def("set_starting_points", &hopsy::MultimodalMultivariateGaussianRun::setStartingPoints)
        .def("get_starting_points", &hopsy::MultimodalMultivariateGaussianRun::getStartingPoints)
        .def("set_markov_chain_type", &hopsy::MultimodalMultivariateGaussianRun::setMarkovChainType)
        .def("get_markovv_chain_type", &hopsy::MultimodalMultivariateGaussianRun::getMarkovChainType)
        .def("set_number_of_chains", &hopsy::MultimodalMultivariateGaussianRun::setNumberOfChains)
        .def("get_number_of_chains", &hopsy::MultimodalMultivariateGaussianRun::getNumberOfChains)
        .def("set_number_of_samples", &hopsy::MultimodalMultivariateGaussianRun::setNumberOfSamples)
        .def("get_number_of_samples", &hopsy::MultimodalMultivariateGaussianRun::getNumberOfSamples)
        .def("set_thinning", &hopsy::MultimodalMultivariateGaussianRun::setThinning)
        .def("get_thinning", &hopsy::MultimodalMultivariateGaussianRun::getThinning)
        .def("set_stepsize", &hopsy::MultimodalMultivariateGaussianRun::setStepSize)
        .def("enable_rounding", &hopsy::MultimodalMultivariateGaussianRun::enableRounding)
        .def("disable_rounding", &hopsy::MultimodalMultivariateGaussianRun::disableRounding)
        .def("get_stepsize", &hopsy::MultimodalMultivariateGaussianRun::getStepSize)
        .def("set_fisher_weight", &hopsy::MultimodalMultivariateGaussianRun::setFisherWeight)
        .def("get_fisher_weight", &hopsy::MultimodalMultivariateGaussianRun::getFisherWeight)
        .def("set_random_seed", &hopsy::MultimodalMultivariateGaussianRun::setRandomSeed)
        .def("get_random_seed", &hopsy::MultimodalMultivariateGaussianRun::getRandomSeed)
        .def("unset_sampling_until_convergence", &hopsy::MultimodalMultivariateGaussianRun::unsetSamplingUntilConvergence)
        .def("set_sampling_until_convergence", &hopsy::MultimodalMultivariateGaussianRun::setSamplingUntilConvergence)
        .def("get_diagnostics_threshold", &hopsy::MultimodalMultivariateGaussianRun::getDiagnosticsThreshold)
        .def("get_max_repetitions", &hopsy::MultimodalMultivariateGaussianRun::getMaxRepetitions);

    py::class_<hopsy::MultivariateGaussianRun>(m, "MultivariateGaussianRun")
        .def(py::init<hopsy::MultivariateGaussianProblem>())
        //.def(py::init<hopsy::MultivariateGaussianRun&>())
        .def("get_data", &hopsy::MultivariateGaussianRun::getData)
        .def("init", &hopsy::MultivariateGaussianRun::init)
        .def("sample", py::overload_cast<>(&hopsy::MultivariateGaussianRun::sample))
        .def("sample", py::overload_cast<unsigned long, unsigned long>(
                    &hopsy::MultivariateGaussianRun::sample), py::arg("number_of_samples"), py::arg("thinning") = 1)
        .def("set_problem", &hopsy::MultivariateGaussianRun::setProblem)
        .def("get_problem", &hopsy::MultivariateGaussianRun::getProblem)
        .def("set_starting_points", &hopsy::MultivariateGaussianRun::setStartingPoints)
        .def("get_starting_points", &hopsy::MultivariateGaussianRun::getStartingPoints)
        .def("set_markov_chain_type", &hopsy::MultivariateGaussianRun::setMarkovChainType)
        .def("get_markovv_chain_type", &hopsy::MultivariateGaussianRun::getMarkovChainType)
        .def("set_number_of_chains", &hopsy::MultivariateGaussianRun::setNumberOfChains)
        .def("get_number_of_chains", &hopsy::MultivariateGaussianRun::getNumberOfChains)
        .def("set_number_of_samples", &hopsy::MultivariateGaussianRun::setNumberOfSamples)
        .def("get_number_of_samples", &hopsy::MultivariateGaussianRun::getNumberOfSamples)
        .def("set_thinning", &hopsy::MultivariateGaussianRun::setThinning)
        .def("get_thinning", &hopsy::MultivariateGaussianRun::getThinning)
        .def("enable_rounding", &hopsy::MultivariateGaussianRun::enableRounding)
        .def("disable_rounding", &hopsy::MultivariateGaussianRun::disableRounding)
        .def("set_stepsize", &hopsy::MultivariateGaussianRun::setStepSize)
        .def("get_stepsize", &hopsy::MultivariateGaussianRun::getStepSize)
        .def("set_fisher_weight", &hopsy::MultivariateGaussianRun::setFisherWeight)
        .def("get_fisher_weight", &hopsy::MultivariateGaussianRun::getFisherWeight)
        .def("set_random_seed", &hopsy::MultivariateGaussianRun::setRandomSeed)
        .def("get_random_seed", &hopsy::MultivariateGaussianRun::getRandomSeed)
        .def("unset_sampling_until_convergence", &hopsy::MultivariateGaussianRun::unsetSamplingUntilConvergence)
        .def("set_sampling_until_convergence", &hopsy::MultivariateGaussianRun::setSamplingUntilConvergence)
        .def("get_diagnostics_threshold", &hopsy::MultivariateGaussianRun::getDiagnosticsThreshold)
        .def("get_max_repetitions", &hopsy::MultivariateGaussianRun::getMaxRepetitions);

    py::class_<hopsy::RosenbrockRun>(m, "RosenbrockRun")
        .def(py::init<hopsy::RosenbrockProblem>())
        .def(py::init<hopsy::RosenbrockRun&>())
        .def("get_data", &hopsy::RosenbrockRun::getData)
        .def("init", &hopsy::RosenbrockRun::init)
        .def("sample", py::overload_cast<>(&hopsy::RosenbrockRun::sample))
        .def("sample", py::overload_cast<unsigned long, unsigned long>(
                    &hopsy::RosenbrockRun::sample), py::arg("number_of_samples"), py::arg("thinning") = 1)
        .def("set_problem", &hopsy::RosenbrockRun::setProblem)
        .def("get_problem", &hopsy::RosenbrockRun::getProblem)
        .def("set_starting_points", &hopsy::RosenbrockRun::setStartingPoints)
        .def("get_starting_points", &hopsy::RosenbrockRun::getStartingPoints)
        .def("set_markov_chain_type", &hopsy::RosenbrockRun::setMarkovChainType)
        .def("get_markovv_chain_type", &hopsy::RosenbrockRun::getMarkovChainType)
        .def("set_number_of_chains", &hopsy::RosenbrockRun::setNumberOfChains)
        .def("get_number_of_chains", &hopsy::RosenbrockRun::getNumberOfChains)
        .def("set_number_of_samples", &hopsy::RosenbrockRun::setNumberOfSamples)
        .def("get_number_of_samples", &hopsy::RosenbrockRun::getNumberOfSamples)
        .def("set_thinning", &hopsy::RosenbrockRun::setThinning)
        .def("get_thinning", &hopsy::RosenbrockRun::getThinning)
        .def("enable_rounding", &hopsy::RosenbrockRun::enableRounding)
        .def("disable_rounding", &hopsy::RosenbrockRun::disableRounding)
        .def("set_stepsize", &hopsy::RosenbrockRun::setStepSize)
        .def("get_stepsize", &hopsy::RosenbrockRun::getStepSize)
        .def("set_fisher_weight", &hopsy::RosenbrockRun::setFisherWeight)
        .def("get_fisher_weight", &hopsy::RosenbrockRun::getFisherWeight)
        .def("set_random_seed", &hopsy::RosenbrockRun::setRandomSeed)
        .def("get_random_seed", &hopsy::RosenbrockRun::getRandomSeed)
        .def("unset_sampling_until_convergence", &hopsy::RosenbrockRun::unsetSamplingUntilConvergence)
        .def("set_sampling_until_convergence", &hopsy::RosenbrockRun::setSamplingUntilConvergence)
        .def("get_diagnostics_threshold", &hopsy::RosenbrockRun::getDiagnosticsThreshold)
        .def("get_max_repetitions", &hopsy::RosenbrockRun::getMaxRepetitions);

    py::class_<hopsy::UniformRun>(m, "UniformRun")
        .def(py::init<hopsy::UniformProblem>())
        .def(py::init<hopsy::UniformRun&>())
        .def("get_data", &hopsy::UniformRun::getData)
        .def("init", &hopsy::UniformRun::init)
        .def("sample", py::overload_cast<>(&hopsy::UniformRun::sample))
        .def("sample", py::overload_cast<unsigned long, unsigned long>(
                    &hopsy::UniformRun::sample), py::arg("number_of_samples"), py::arg("thinning") = 1)
        .def("set_problem", &hopsy::UniformRun::setProblem)
        .def("get_problem", &hopsy::UniformRun::getProblem)
        .def("set_starting_points", &hopsy::UniformRun::setStartingPoints)
        .def("get_starting_points", &hopsy::UniformRun::getStartingPoints)
        .def("set_markov_chain_type", &hopsy::UniformRun::setMarkovChainType)
        .def("get_markovv_chain_type", &hopsy::UniformRun::getMarkovChainType)
        .def("set_number_of_chains", &hopsy::UniformRun::setNumberOfChains)
        .def("get_number_of_chains", &hopsy::UniformRun::getNumberOfChains)
        .def("set_number_of_samples", &hopsy::UniformRun::setNumberOfSamples)
        .def("get_number_of_samples", &hopsy::UniformRun::getNumberOfSamples)
        .def("set_thinning", &hopsy::UniformRun::setThinning)
        .def("get_thinning", &hopsy::UniformRun::getThinning)
        .def("enable_rounding", &hopsy::UniformRun::enableRounding)
        .def("disable_rounding", &hopsy::UniformRun::disableRounding)
        .def("set_stepsize", &hopsy::UniformRun::setStepSize)
        .def("get_stepsize", &hopsy::UniformRun::getStepSize)
        .def("set_fisher_weight", &hopsy::UniformRun::setFisherWeight)
        .def("get_fisher_weight", &hopsy::UniformRun::getFisherWeight)
        .def("set_random_seed", &hopsy::UniformRun::setRandomSeed)
        .def("get_random_seed", &hopsy::UniformRun::getRandomSeed)
        .def("unset_sampling_until_convergence", &hopsy::UniformRun::unsetSamplingUntilConvergence)
        .def("set_sampling_until_convergence", &hopsy::UniformRun::setSamplingUntilConvergence)
        .def("get_diagnostics_threshold", &hopsy::UniformRun::getDiagnosticsThreshold)
        .def("get_max_repetitions", &hopsy::UniformRun::getMaxRepetitions);

    py::class_<hopsy::PyRun>(m, "PyRun")
        .def(py::init<hopsy::PyProblem>())
        .def(py::init<hopsy::PyRun&>())
        .def("get_data", &hopsy::PyRun::getData)
        .def("init", &hopsy::PyRun::init)
        .def("sample", py::overload_cast<>(&hopsy::PyRun::sample))
        .def("sample", py::overload_cast<unsigned long, unsigned long>(
                    &hopsy::PyRun::sample), py::arg("number_of_samples"), py::arg("thinning") = 1)
        .def("set_problem", &hopsy::PyRun::setProblem)
        .def("get_problem", &hopsy::PyRun::getProblem)
        .def("set_starting_points", &hopsy::PyRun::setStartingPoints)
        .def("get_starting_points", &hopsy::PyRun::getStartingPoints)
        .def("set_markov_chain_type", &hopsy::PyRun::setMarkovChainType)
        .def("get_markovv_chain_type", &hopsy::PyRun::getMarkovChainType)
        .def("set_number_of_chains", &hopsy::PyRun::setNumberOfChains)
        .def("get_number_of_chains", &hopsy::PyRun::getNumberOfChains)
        .def("set_number_of_samples", &hopsy::PyRun::setNumberOfSamples)
        .def("get_number_of_samples", &hopsy::PyRun::getNumberOfSamples)
        .def("set_thinning", &hopsy::PyRun::setThinning)
        .def("get_thinning", &hopsy::PyRun::getThinning)
        .def("enable_rounding", &hopsy::PyRun::enableRounding)
        .def("disable_rounding", &hopsy::PyRun::disableRounding)
        .def("set_stepsize", &hopsy::PyRun::setStepSize)
        .def("get_stepsize", &hopsy::PyRun::getStepSize)
        .def("set_fisher_weight", &hopsy::PyRun::setFisherWeight)
        .def("get_fisher_weight", &hopsy::PyRun::getFisherWeight)
        .def("set_random_seed", &hopsy::PyRun::setRandomSeed)
        .def("get_random_seed", &hopsy::PyRun::getRandomSeed)
        .def("unset_sampling_until_convergence", &hopsy::PyRun::unsetSamplingUntilConvergence)
        .def("set_sampling_until_convergence", &hopsy::PyRun::setSamplingUntilConvergence)
        .def("get_diagnostics_threshold", &hopsy::PyRun::getDiagnosticsThreshold)
        .def("get_max_repetitions", &hopsy::PyRun::getMaxRepetitions);


	m.def("Run", &hopsy::Run<hopsy::DegenerateMultivariateGaussianModel>, 
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::Run<hopsy::MultimodalMultivariateGaussianModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::Run<hopsy::MultivariateGaussianModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::Run<hopsy::PyModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::Run<hopsy::RosenbrockModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::Run<hopsy::UniformModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);


    py::class_<hops::Data>(m, "Data")
        .def(py::init<>())
		.def("compute_expected_squared_jump_distance", &hops::Data::computeExpectedSquaredJumpDistance)
		.def("compute_effective_sample_size", &hops::Data::computeEffectiveSampleSize)
		.def("compute_potential_scale_reduction_factor", &hops::Data::computePotentialScaleReductionFactor)
		.def("get_chains", &hops::Data::getChains)
		.def("get_chain", &hops::Data::getChain)
		.def("get_dimension", &hops::Data::getDimension)
		.def("get_statistics", &hops::Data::getStatistics)
        .def("reset", &hops::Data::reset);

    py::class_<hops::ChainData>(m, "ChainData")
        .def(py::init<>())
		.def("get_acceptance_rates", &hops::ChainData::getAcceptanceRates)
		.def("get_negative_log_likelihood", &hops::ChainData::getNegativeLogLikelihood)
		.def("get_states", &hops::ChainData::getStates)
		.def("get_timestamps", &hops::ChainData::getTimestamps)
        .def("reset", &hops::ChainData::reset);

    py::class_<hops::StatisticsData>(m, "StatisticsData")
        .def(py::init<>())
		.def("get_expected_squared_jump_distance", &hops::StatisticsData::getExpectedSquaredJumpDistance)
		.def("get_effective_sample_size", &hops::StatisticsData::getEffectiveSampleSize)
		.def("get_potential_scale_reduction_factor", &hops::StatisticsData::getPotentialScaleReductionFactor);

    py::class_<hops::EmptyChainDataException>(m, "EmptyChainDataException");
    py::class_<hops::NoProblemProvidedException>(m, "NoProblemProvidedException");
    py::class_<hops::UninitializedDataFieldException>(m, "UninitializedDataFieldException");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
