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
                R"pbdoc()pbdoc",
                py::arg("components"));

    py::class_<hopsy::MultivariateGaussianModel>(m, "MultivariateGaussianModel",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>(),
                R"pbdoc()pbdoc",
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)));

    py::class_<hopsy::PyModel>(m, "PyModel",
				R"pbdoc()pbdoc")
        .def(py::init<py::object>(),
                R"pbdoc()pbdoc",
                py::arg("model"));

    py::class_<hopsy::RosenbrockModel>(m, "RosenbrockModel",
				R"pbdoc()pbdoc")
        .def(py::init<double, Eigen::VectorXd>());

    py::class_<hopsy::UniformModel>(m, "UniformModel",
				R"pbdoc(
The ``hopsy.UniformModel`` defines the uniform target distribution on the polytope

.. math::
   \pi(x) := \frac{1}{Z} \mathbf{1}_{\mathcal{P}}(x)

where

.. math::
   Z = \int_{\mathcal{P}} \mathbf{1}_{\mathcal{P}}(x) \mathrm{d}x
                )pbdoc")
        .def(py::init<>());


    //  
    // Problem classes
    // ===============
    //
    py::class_<hopsy::DegenerateMultivariateGaussianProblem>(m, "DegenerateMultivariateGaussianProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::DegenerateMultivariateGaussianModel>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));

    py::class_<hopsy::MultimodalMultivariateGaussianProblem>(m, "MultimodalMultivariateGaussianProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::MultimodalMultivariateGaussianModel>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));

    py::class_<hopsy::MultivariateGaussianProblem>(m, "MultivariateGaussianProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::MultivariateGaussianModel>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));

    py::class_<hopsy::RosenbrockProblem>(m, "RosenbrockProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::RosenbrockModel>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));

    py::class_<hopsy::UniformProblem>(m, "UniformProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"))
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::UniformModel>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));

    py::class_<hopsy::PyProblem>(m, "PyProblem",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::PyModel>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));


    //  
    // Problem factory method
    // ======================
    //
    // Instead of having to construct the correct problem from a given model manually, this method 
    // simulates a general problem constructor which then statically checks the passed model
    // type and returns the correctly instantiated problem object
    //
	m.def("Problem", &hopsy::createProblem<hopsy::DegenerateMultivariateGaussianModel>,
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));
	m.def("Problem", &hopsy::createProblem<hopsy::MultimodalMultivariateGaussianModel>,
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));
	m.def("Problem", &hopsy::createProblem<hopsy::MultivariateGaussianModel>,
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));
	m.def("Problem", &hopsy::createProblem<hopsy::PyModel>,
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));
	m.def("Problem", &hopsy::createProblem<hopsy::RosenbrockModel>,
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));
	m.def("Problem", &hopsy::createProblem<hopsy::UniformModel>,
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));
	m.def("Problem", &hopsy::createUniformProblem,
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"));
	m.def("Problem", &hopsy::createPyProblem,
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"));


    //  
    // Python proposal wrapper class
    // =============================
    //
    py::class_<hopsy::PyProposal>(m, "PyProposal",
				R"pbdoc()pbdoc")
        .def(py::init<py::object>(),
				R"pbdoc()pbdoc",
                py::arg("proposal"));

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
            	py::arg("problem"), py::arg("proposal_name") = "HitAndRun", py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRun<hopsy::MultimodalMultivariateGaussianModel>);
	m.def("Run", &hopsy::createRun<hopsy::MultivariateGaussianModel>);
	m.def("Run", &hopsy::createRun<hopsy::PyModel>);
	m.def("Run", &hopsy::createRun<hopsy::RosenbrockModel>);
	m.def("Run", &hopsy::createRun<hopsy::UniformModel>);

	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::DegenerateMultivariateGaussianModel>, 
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MultimodalMultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::PyModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::RosenbrockModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::UniformModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);

	m.def("Run", &hopsy::createRunFromPyObject<hopsy::DegenerateMultivariateGaussianModel>, 
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::MultimodalMultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::MultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::PyModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::RosenbrockModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::UniformModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), py::arg("proposal"), py::arg("number_of_samples") = 1000, py::arg("number_of_chains") = 1);


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
				R"pbdoc(
                Compute the average acceptance rate of the chains in ``data``. 
                Acceptance rates are returned in an ``m`` x ``1`` column vector, 
                where ``m`` is the number of chains stored in ``data``.

                The acceptance rate is
                actually also logged after every chain iteration and stored in the ChainData,
                but this initializes the acceptance_rate field inside the Data object and thus
                allows to discard the samples.

                Parameters
                ----------
                )pbdoc", 
                py::arg("data"));
    m.def("compute_effective_sample_size", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeEffectiveSampleSize),
				R"pbdoc(
                Compute the effective sample size of the chains in ``data``. 
                The effective sample size is computed for every dimension individually and is then
                returned in an ``m`` x ``1`` column vector, 
                where ``m`` is the dimension of the states.

                Parameters
                ----------
                )pbdoc", 
                py::arg("data"));
    m.def("compute_expected_squared_jump_distance", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeExpectedSquaredJumpDistance),
				R"pbdoc(
                Compute the expected squared jump distance of the chains in ``data``. 
                The expected squared jump distance is computed for every chain individually and is then
                returned in an ``m`` x ``1`` column vector, 
                where ``m`` is the number of chains stored in ``data``.

                Parameters
                ----------
                )pbdoc", 
                py::arg("data"));
    m.def("compute_potential_scale_reduction_factor", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computePotentialScaleReductionFactor),
				R"pbdoc(
                Compute the potential scale reduction factor (also known as R-hat) 
                of the chains in ``data``. 
                The potential scale reduction factor is computed for every dimension individually and is then
                returned in an ``m`` x ``1`` column vector, 
                where ``m`` is the dimension of the states.

                Parameters
                ----------
                )pbdoc", 
                py::arg("data"));
    m.def("compute_total_time_taken", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeTotalTimeTaken),
				R"pbdoc(
                Compute the total time taken of the chains in ``data``. 
                Times are returned in an ``m`` x ``1`` column vector, 
                where ``m`` is the number of chains stored in ``data``.

                Timestamps are actually also logged after every chain iteration and stored in the ChainData,
                so this function just takes the difference of the last and first timestamp.
                It also initializes the total_time_taken field inside the Data object and thus
                allows to discard the samples.

                Parameters
                ----------
                )pbdoc", 
                py::arg("data"));

    py::class_<hops::ChainData>(m, "ChainData",
                R"pbdoc()pbdoc")
        .def(py::init<>())
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
