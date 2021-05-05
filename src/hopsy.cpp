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

template<typename Problem, typename Run>
void addRunClassToModule(py::module& m, const char* name) {
    py::class_<Run>(m, name)
        .def(py::init<Problem>())
        .def(py::init<Run&>())
        .def("get_data", &Run::getData)
        .def("init", &Run::init)
        .def("sample", py::overload_cast<>(&Run::sample))
        .def("sample", py::overload_cast<unsigned long, unsigned long>(
                    &Run::sample), py::arg("number_of_samples"), py::arg("thinning") = 1)
        .def("set_problem", &Run::setProblem)
        .def("get_problem", &Run::getProblem)
        .def("set_starting_points", &Run::setStartingPoints)
        .def("get_starting_points", &Run::getStartingPoints)
        .def("set_markov_chain_type", &Run::setMarkovChainType)
        .def("get_markovv_chain_type", &Run::getMarkovChainType)
        .def("set_number_of_chains", &Run::setNumberOfChains)
        .def("get_number_of_chains", &Run::getNumberOfChains)
        .def("set_number_of_samples", &Run::setNumberOfSamples)
        .def("get_number_of_samples", &Run::getNumberOfSamples)
        .def("set_thinning", &Run::setThinning)
        .def("get_thinning", &Run::getThinning)
        .def("enable_rounding", &Run::enableRounding)
        .def("disable_rounding", &Run::disableRounding)
        .def("set_stepsize", &Run::setStepSize)
        .def("get_stepsize", &Run::getStepSize)
        .def("set_fisher_weight", &Run::setFisherWeight)
        .def("get_fisher_weight", &Run::getFisherWeight)
        .def("set_random_seed", &Run::setRandomSeed)
        .def("get_random_seed", &Run::getRandomSeed)
        .def("unset_sampling_until_convergence", &Run::unsetSamplingUntilConvergence)
        .def("set_sampling_until_convergence", &Run::setSamplingUntilConvergence)
        .def("get_diagnostics_threshold", &Run::getDiagnosticsThreshold)
        .def("get_max_repetitions", &Run::getMaxRepetitions);
}

PYBIND11_MODULE(hopsy, m) {
    //  
    // Model classes
    // =============
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
    // Problem classes
    // ===============
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
    // Instead of having to construct the correct problem from a given model manually, this method 
    // simulates a general problem constructor which then statically checks the passed model
    // type and returns the correctly instantiated problem object
    //
	m.def("Problem", &hopsy::createProblem<hopsy::DegenerateMultivariateGaussianModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::MultimodalMultivariateGaussianModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::MultivariateGaussianModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::PyModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::RosenbrockModel>);
	m.def("Problem", &hopsy::createProblem<hopsy::UniformModel>);
	m.def("Problem", &hopsy::createUniformProblem);
	m.def("Problem", &hopsy::createPyProblem);


    //  
    // Python proposal wrapper class
    // =============================
    //
    py::class_<hopsy::PyProposal>(m, "PyProposal")
        .def(py::init<py::object>());

    //  
    // Run classes
    // ===========
    //
    addRunClassToModule<hopsy::DegenerateMultivariateGaussianProblem, hopsy::DegenerateMultivariateGaussianRun>(m, "DegenerateMultivariateGaussianRun");
    addRunClassToModule<hopsy::MultimodalMultivariateGaussianProblem, hopsy::MultimodalMultivariateGaussianRun>(m, "MultimodalMultivariateGaussianRun");
    addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianRun>(m, "MultivariateGaussianRun");
    addRunClassToModule<hopsy::PyProblem, hopsy::PyRun>(m, "PyRun");
    addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockRun>(m, "RosenbrockRun");
    addRunClassToModule<hopsy::UniformProblem, hopsy::UniformRun>(m, "UniformRun");

    addRunClassToModule<hopsy::DegenerateMultivariateGaussianProblem, hopsy::DegenerateMultivariateGaussianPyProposalRun>(m, "DegenerateMultivariateGaussianPyProposalRun");
    addRunClassToModule<hopsy::MultimodalMultivariateGaussianProblem, hopsy::MultimodalMultivariateGaussianPyProposalRun>(m, "MultimodalMultivariateGaussianPyProposalRun");
    addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianPyProposalRun>(m, "MultivariateGaussianPyProposalRun");
    addRunClassToModule<hopsy::PyProblem, hopsy::PyPyProposalRun>(m, "PyPyProposalRun");
    addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockPyProposalRun>(m, "RosenbrockPyProposalRun");
    addRunClassToModule<hopsy::UniformProblem, hopsy::UniformPyProposalRun>(m, "UniformPyProposalRun");


    //  
    // Run factory method
    // ==================
    //
    // Instead of having to construct the correct run from a given problem manually, this method 
    // simulates a general run constructor which then statically checks the passed problem
    // type and returns the correctly instantiated run object
    //
	m.def("Run", &hopsy::createRun<hopsy::DegenerateMultivariateGaussianModel>, 
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::MultimodalMultivariateGaussianModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::MultivariateGaussianModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::PyModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::RosenbrockModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::UniformModel>,
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);

	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::DegenerateMultivariateGaussianModel>, 
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MultimodalMultivariateGaussianModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MultivariateGaussianModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::PyModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::RosenbrockModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::UniformModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);

	m.def("Run", &hopsy::createRunFromPyObject<hopsy::DegenerateMultivariateGaussianModel>, 
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::MultimodalMultivariateGaussianModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::MultivariateGaussianModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::PyModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::RosenbrockModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::UniformModel>,
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);


    //  
    // Data classes
    // ============
    //
    // Instead of having to construct the correct run from a given problem manually, this method 
    // simulates a general run constructor which then statically checks the passed problem
    // type and returns the correctly instantiated run object
    //
    py::class_<hops::Data>(m, "Data")
        .def(py::init<>())
        .def("get_states", &hops::Data::getStates)
        .def("reset", &hops::Data::reset)
        .def("write", &hops::Data::write);

    // check: https://stackoverflow.com/questions/49452957/overload-cast-fails-in-a-specific-case
    using computeStatisticsSignature = Eigen::VectorXd(hops::Data&);
    m.def("compute_acceptance_rate", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeAcceptanceRate));
    m.def("compute_effective_sample_size", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeEffectiveSampleSize));
    m.def("compute_expected_squared_jump_distance", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeExpectedSquaredJumpDistance));
    m.def("compute_potential_scale_reduction_factor", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computePotentialScaleReductionFactor));
    m.def("compute_total_time_taken", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeTotalTimeTaken));

    py::class_<hops::ChainData>(m, "ChainData")
        .def(py::init<>())
		.def("get_acceptance_rates", &hops::ChainData::getAcceptanceRates)
		.def("get_negative_log_likelihood", &hops::ChainData::getNegativeLogLikelihood)
		.def("get_states", &hops::ChainData::getStates)
		.def("get_timestamps", &hops::ChainData::getTimestamps)
        .def("reset", &hops::ChainData::reset);

    py::class_<hops::EmptyChainDataException>(m, "EmptyChainDataException");
    py::class_<hops::NoProblemProvidedException>(m, "NoProblemProvidedException");
    py::class_<hops::UninitializedDataFieldException>(m, "UninitializedDataFieldException");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
