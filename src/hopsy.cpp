#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hopsy.hpp"

#include <Eigen/Core>

#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

template<typename Problem, typename Run>
void addRunClassToModule(py::module& m, const char* name, const char* doc) {
    py::class_<Run>(m, name, doc)
        .def(py::init<Problem>(),
				R"pbdoc()pbdoc")
        .def(py::init<Run&>(),
				R"pbdoc()pbdoc")
        .def("get_data", &Run::getData,
				R"pbdoc()pbdoc")
        .def("init", &Run::init,
				R"pbdoc()pbdoc")
        .def("sample", py::overload_cast<>(&Run::sample),
				R"pbdoc()pbdoc")
        .def("sample", py::overload_cast<unsigned long, unsigned long>(&Run::sample),
                R"pbdoc()pbdoc", 
                py::arg("number_of_samples"), py::arg("thinning") = 1)
        .def("set_problem", &Run::setProblem,
				R"pbdoc()pbdoc")
        .def("get_problem", &Run::getProblem,
				R"pbdoc()pbdoc")
        .def("set_starting_points", &Run::setStartingPoints,
				R"pbdoc()pbdoc")
        .def("get_starting_points", &Run::getStartingPoints,
				R"pbdoc()pbdoc")
        .def("set_markov_chain_type", &Run::setMarkovChainType,
				R"pbdoc()pbdoc")
        .def("get_markovv_chain_type", &Run::getMarkovChainType,
				R"pbdoc()pbdoc")
        .def("set_number_of_chains", &Run::setNumberOfChains,
				R"pbdoc()pbdoc")
        .def("get_number_of_chains", &Run::getNumberOfChains,
				R"pbdoc()pbdoc")
        .def("set_number_of_samples", &Run::setNumberOfSamples,
				R"pbdoc()pbdoc")
        .def("get_number_of_samples", &Run::getNumberOfSamples,
				R"pbdoc()pbdoc")
        .def("set_thinning", &Run::setThinning,
				R"pbdoc()pbdoc")
        .def("get_thinning", &Run::getThinning,
				R"pbdoc()pbdoc")
        .def("enable_rounding", &Run::enableRounding,
				R"pbdoc()pbdoc")
        .def("disable_rounding", &Run::disableRounding,
				R"pbdoc()pbdoc")
        .def("set_stepsize", &Run::setStepSize,
				R"pbdoc()pbdoc")
        .def("get_stepsize", &Run::getStepSize,
				R"pbdoc()pbdoc")
        .def("set_fisher_weight", &Run::setFisherWeight,
				R"pbdoc()pbdoc")
        .def("get_fisher_weight", &Run::getFisherWeight,
				R"pbdoc()pbdoc")
        .def("set_random_seed", &Run::setRandomSeed,
				R"pbdoc()pbdoc")
        .def("get_random_seed", &Run::getRandomSeed,
				R"pbdoc()pbdoc")
        .def("unset_sampling_until_convergence", &Run::unsetSamplingUntilConvergence,
				R"pbdoc()pbdoc")
        .def("set_sampling_until_convergence", &Run::setSamplingUntilConvergence,
				R"pbdoc()pbdoc")
        .def("get_diagnostics_threshold", &Run::getDiagnosticsThreshold,
				R"pbdoc()pbdoc")
        .def("get_max_repetitions", &Run::getMaxRepetitions,
				R"pbdoc()pbdoc");
}

PYBIND11_MODULE(hopsy, m) {
    m.doc() = R"pbdoc(
        hopsy -         
   		-----------------------
		A python interface for HOPS - the Highly Optimized toolbox for Polytope Sampling.
		Built using pybind11.

        .. currentmodule:: hopsy

        .. autosummary::
            :toctree: _generate
		    DegenerateMultivariateGaussianModel
			MultimodalMultivariateGaussianModel
			MultivariateGaussianModel
			PyModel
            RosenbrockModel
            UniformModel
    )pbdoc";

    // Model classes
    // =============
    //
    py::class_<hopsy::DegenerateMultivariateGaussianModel>(m, "DegenerateMultivariateGaussianModel",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>(),
                R"pbdoc()pbdoc",
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)))
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd, std::vector<long>>(),
                R"pbdoc()pbdoc",
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)),
                py::arg("inactives") = std::vector<long>());

    py::class_<hopsy::MultimodalMultivariateGaussianModel>(m, "MultimodalMultivariateGaussianModel",
				R"pbdoc()pbdoc")
        .def(py::init<std::vector<hopsy::DegenerateMultivariateGaussianModel>>(),
                R"pbdoc()pbdoc");

    py::class_<hopsy::MultivariateGaussianModel>(m, "MultivariateGaussianModel",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>(),
                R"pbdoc()pbdoc",
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)));

    py::class_<hopsy::PyModel>(m, "PyModel",
				R"pbdoc()pbdoc")
        .def(py::init<py::object>(),
                R"pbdoc()pbdoc");

    py::class_<hopsy::RosenbrockModel>(m, "RosenbrockModel",
				R"pbdoc()pbdoc")
        .def(py::init<double, Eigen::VectorXd>(),
                R"pbdoc()pbdoc");

    py::class_<hopsy::UniformModel>(m, "UniformModel",
				R"pbdoc()pbdoc")
        .def(py::init<>(),
                R"pbdoc()pbdoc");


    //  
    // Problem classes
    // ===============
    //
    py::class_<hopsy::DegenerateMultivariateGaussianProblem>(m, "DegenerateMultivariateGaussianProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::DegenerateMultivariateGaussianModel>(),
                R"pbdoc()pbdoc");

    py::class_<hopsy::MultimodalMultivariateGaussianProblem>(m, "MultimodalMultivariateGaussianProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::MultimodalMultivariateGaussianModel>(),
                R"pbdoc()pbdoc");

    py::class_<hopsy::MultivariateGaussianProblem>(m, "MultivariateGaussianProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::MultivariateGaussianModel>(),
                R"pbdoc()pbdoc");

    py::class_<hopsy::RosenbrockProblem>(m, "RosenbrockProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::RosenbrockModel>(),
                R"pbdoc()pbdoc");

    py::class_<hopsy::UniformProblem>(m, "UniformProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd>(),
                R"pbdoc()pbdoc",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::UniformModel>(),
                R"pbdoc()pbdoc");

    py::class_<hopsy::PyProblem>(m, "PyProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::PyModel>(),
                R"pbdoc()pbdoc");


    //  
    // Problem factory method
    // ======================
    //
    // Instead of having to construct the correct problem from a given model manually, this method 
    // simulates a general problem constructor which then statically checks the passed model
    // type and returns the correctly instantiated problem object
    //
	m.def("Problem", &hopsy::createProblem<hopsy::DegenerateMultivariateGaussianModel>,
                R"pbdoc()pbdoc");
	m.def("Problem", &hopsy::createProblem<hopsy::MultimodalMultivariateGaussianModel>,
                R"pbdoc()pbdoc");
	m.def("Problem", &hopsy::createProblem<hopsy::MultivariateGaussianModel>,
                R"pbdoc()pbdoc");
	m.def("Problem", &hopsy::createProblem<hopsy::PyModel>,
                R"pbdoc()pbdoc");
	m.def("Problem", &hopsy::createProblem<hopsy::RosenbrockModel>,
                R"pbdoc()pbdoc");
	m.def("Problem", &hopsy::createProblem<hopsy::UniformModel>,
                R"pbdoc()pbdoc");
	m.def("Problem", &hopsy::createUniformProblem,
                R"pbdoc()pbdoc");
	m.def("Problem", &hopsy::createPyProblem,
                R"pbdoc()pbdoc");


    //  
    // Python proposal wrapper class
    // =============================
    //
    py::class_<hopsy::PyProposal>(m, "PyProposal",
				R"pbdoc()pbdoc")
        .def(py::init<py::object>(),
				R"pbdoc()pbdoc");

    //  
    // Run classes
    // ===========
    //
    addRunClassToModule<hopsy::DegenerateMultivariateGaussianProblem, hopsy::DegenerateMultivariateGaussianRun>(m, "DegenerateMultivariateGaussianRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::MultimodalMultivariateGaussianProblem, hopsy::MultimodalMultivariateGaussianRun>(m, "MultimodalMultivariateGaussianRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianRun>(m, "MultivariateGaussianRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::PyProblem, hopsy::PyRun>(m, "PyRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockRun>(m, "RosenbrockRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::UniformProblem, hopsy::UniformRun>(m, "UniformRun",
                R"pbdoc()pbdoc");

    addRunClassToModule<hopsy::DegenerateMultivariateGaussianProblem, hopsy::DegenerateMultivariateGaussianPyProposalRun>(m, "DegenerateMultivariateGaussianPyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::MultimodalMultivariateGaussianProblem, hopsy::MultimodalMultivariateGaussianPyProposalRun>(m, "MultimodalMultivariateGaussianPyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianPyProposalRun>(m, "MultivariateGaussianPyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::PyProblem, hopsy::PyPyProposalRun>(m, "PyPyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockPyProposalRun>(m, "RosenbrockPyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::UniformProblem, hopsy::UniformPyProposalRun>(m, "UniformPyProposalRun",
                R"pbdoc()pbdoc");


    //  
    // Run factory method
    // ==================
    //
    // Instead of having to construct the correct run from a given problem manually, this method 
    // simulates a general run constructor which then statically checks the passed problem
    // type and returns the correctly instantiated run object
    //
	m.def("Run", &hopsy::createRun<hopsy::DegenerateMultivariateGaussianModel>, 
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::MultimodalMultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::MultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::PyModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::RosenbrockModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::UniformModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("chainType") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);

	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::DegenerateMultivariateGaussianModel>, 
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MultimodalMultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::PyModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::RosenbrockModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::UniformModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);

	m.def("Run", &hopsy::createRunFromPyObject<hopsy::DegenerateMultivariateGaussianModel>, 
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::MultimodalMultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::MultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::PyModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::RosenbrockModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::UniformModel>,
            	R"pbdoc()pbdoc",
            py::arg("problem"), py::arg("proposal_algorithm"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);


    //  
    // Data classes
    // ============
    //
    // Instead of having to construct the correct run from a given problem manually, this method 
    // simulates a general run constructor which then statically checks the passed problem
    // type and returns the correctly instantiated run object
    //
    py::class_<hops::Data>(m, "Data",
                R"pbdoc()pbdoc")
        .def(py::init<>())
        .def("get_states", &hops::Data::getStates,
				R"pbdoc()pbdoc")
        .def("reset", &hops::Data::reset,
				R"pbdoc()pbdoc")
        .def("write", &hops::Data::write,
				R"pbdoc()pbdoc");

    // check: https://stackoverflow.com/questions/49452957/overload-cast-fails-in-a-specific-case
    using computeStatisticsSignature = Eigen::VectorXd(hops::Data&);
    m.def("compute_acceptance_rate", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeAcceptanceRate),
				R"pbdoc()pbdoc");
    m.def("compute_effective_sample_size", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeEffectiveSampleSize),
				R"pbdoc()pbdoc");
    m.def("compute_expected_squared_jump_distance", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeExpectedSquaredJumpDistance),
				R"pbdoc()pbdoc");
    m.def("compute_potential_scale_reduction_factor", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computePotentialScaleReductionFactor),
				R"pbdoc()pbdoc");
    m.def("compute_total_time_taken", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeTotalTimeTaken),
				R"pbdoc()pbdoc");

    py::class_<hops::ChainData>(m, "ChainData",
                R"pbdoc()pbdoc")
        .def(py::init<>(),
				R"pbdoc()pbdoc")
		.def("get_acceptance_rates", &hops::ChainData::getAcceptanceRates,
				R"pbdoc()pbdoc")
		.def("get_negative_log_likelihood", &hops::ChainData::getNegativeLogLikelihood,
				R"pbdoc()pbdoc")
		.def("get_states", &hops::ChainData::getStates,
				R"pbdoc()pbdoc")
		.def("get_timestamps", &hops::ChainData::getTimestamps,
				R"pbdoc()pbdoc")
        .def("reset", &hops::ChainData::reset,
				R"pbdoc()pbdoc");

    py::class_<hops::EmptyChainDataException>(m, "EmptyChainDataException");
    py::class_<hops::NoProblemProvidedException>(m, "NoProblemProvidedException");
    py::class_<hops::UninitializedDataFieldException>(m, "UninitializedDataFieldException");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
