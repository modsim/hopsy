#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Utility/Data.hpp"
#include "hopsy.hpp"
#include "doc.hpp"

#include <Eigen/Core>

#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

template<typename Problem, typename Run>
void addRunClassToModule(py::module& m, const char* name, const char* doc) {
    py::class_<Run>(m, name, doc)
        .def(py::init<Problem>())
        .def("init", &Run::init)
        .def("sample", py::overload_cast<>(&Run::sample))
        .def("sample", py::overload_cast<unsigned long, unsigned long>(&Run::sample), 
                py::arg("n_samples"), py::arg("thinning") = 1)
        .def_property_readonly("data", &Run::getData)
        .def_property("problem", &Run::getProblem, &Run::setProblem)
        .def_property("starting_points", &Run::getStartingPoints, &Run::setStartingPoints)
        .def_property("n_chains", &Run::getNumberOfChains, &Run::setNumberOfChains)
        .def_property("n_samples", &Run::getNumberOfSamples, &Run::setNumberOfSamples)
        .def_property("thinning", &Run::getThinning, &Run::setThinning)
        .def_property("stepsize", &Run::getStepSize, &Run::setStepSize)
        .def_property("fisher_weight", &Run::getFisherWeight, &Run::setFisherWeight)
        .def_property("random_seed", &Run::getRandomSeed, &Run::setRandomSeed)
        .def_property("sample_until_convergence", &Run::getSamplingUntilConvergence, &Run::setSamplingUntilConvergence)
        .def_property("convergence_threshold", &Run::getConvergenceThreshold, &Run::setConvergenceThreshold)
        .def_property("diagnostics_threshold", &Run::getConvergenceThreshold, &Run::setConvergenceThreshold)
        .def_property("max_repetitions", &Run::getMaxRepetitions, &Run::setMaxRepetitions);
}

template<typename Model, typename Problem>
void addProblemClassToModule(py::module& m, const char* name, const char* doc) {
    py::class_<Problem>(m, name, doc)
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Model>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"))
        .def_property("A", &Problem::getA, &Problem::setA)
        .def_property("b", &Problem::getB, &Problem::setB)
        .def_property("starting_point", &Problem::getStartingPoint, &Problem::setStartingPoint)
        .def_property("unrounding_transformation", &Problem::getUnroundingTransformation, &Problem::setUnroundingTransformation)
        .def_property("unrounding_shift", &Problem::getUnroundingShift, &Problem::setUnroundingShift)
        ;
}

template<typename Run, typename Target>
void addTuningMethodToModule(py::module& m, const char* doc) {
    m.def("tune", &hopsy::tune<Run, Target>,
          doc,
          py::arg("run"),
          py::arg("target"),
          py::arg("method") = "ThompsonSampling",
          py::arg("n_test_samples") = 100,
          py::arg("n_posterior_updates") = 100,
          py::arg("n_pure_sampling") = 1,
          py::arg("n_convergence_threshold") = 5,
          py::arg("gridsize") = 101,
          py::arg("lower_grid_bound") = 1e-5,
          py::arg("upupp_grid_bound") = 1e5,
          py::arg("smoothing_length") = 0.5,
          py::arg("random_seed") = 0,
          py::arg("record_data") = true);
}


PYBIND11_MODULE(_hopsy, m) {
    //
    // Model classes
    // =============
    //
    py::class_<hopsy::DegenerateMultivariateGaussianModel>(m, "DegenerateMultivariateGaussianModel",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd, std::vector<long>>(),
                R"pbdoc()pbdoc",
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)),
                py::arg("inactives") = std::vector<long>())
        .def("compute_negative_log_likelihood", &hopsy::DegenerateMultivariateGaussianModel::computeNegativeLogLikelihood, py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::DegenerateMultivariateGaussianModel::computeLogLikelihoodGradient, py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::DegenerateMultivariateGaussianModel::computeExpectedFisherInformation, py::arg("x"));

    py::class_<hopsy::MixtureModel>(m, "MixtureModel",
				R"pbdoc()pbdoc")
        .def(py::init<std::vector<hopsy::PyModel>>(),
                R"pbdoc()pbdoc",
                py::arg("components"))
        .def(py::init<std::vector<hopsy::PyModel>, std::vector<double>>(),
                R"pbdoc()pbdoc",
                py::arg("components"),
                py::arg("weights"))
        .def("compute_negative_log_likelihood", &hopsy::MixtureModel::computeNegativeLogLikelihood, py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::MixtureModel::computeLogLikelihoodGradient, py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::MixtureModel::computeExpectedFisherInformation, py::arg("x"));

    py::class_<hopsy::MultivariateGaussianModel>(m, "MultivariateGaussianModel",
				R"pbdoc()pbdoc")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>(),
                R"pbdoc()pbdoc",
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)))
        .def("compute_negative_log_likelihood", &hopsy::MultivariateGaussianModel::computeNegativeLogLikelihood, py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::MultivariateGaussianModel::computeLogLikelihoodGradient, py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::MultivariateGaussianModel::computeExpectedFisherInformation, py::arg("x"));

    py::class_<hopsy::PyModel>(m, "PyModel",
				R"pbdoc()pbdoc")
        .def(py::init<py::object>(),
                R"pbdoc()pbdoc",
                py::arg("model"))
        .def("compute_negative_log_likelihood", &hopsy::PyModel::computeNegativeLogLikelihood, py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::PyModel::computeLogLikelihoodGradient, py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::PyModel::computeExpectedFisherInformation, py::arg("x"));

    py::class_<hopsy::RosenbrockModel>(m, "RosenbrockModel",
				R"pbdoc()pbdoc")
        .def(py::init<double, Eigen::VectorXd>(),
                R"pbdoc()pbdoc",
                py::arg("scale"),
                py::arg("shift"))
        .def("compute_negative_log_likelihood", &hopsy::RosenbrockModel::computeNegativeLogLikelihood, py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::RosenbrockModel::computeLogLikelihoodGradient, py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::RosenbrockModel::computeExpectedFisherInformation, py::arg("x"));

    py::class_<hopsy::UniformModel>(m, "UniformModel",
				R"pbdoc()pbdoc")
        .def(py::init<>())
        .def("compute_negative_log_likelihood", &hopsy::UniformModel::computeNegativeLogLikelihood, py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::UniformModel::computeLogLikelihoodGradient, py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::UniformModel::computeExpectedFisherInformation, py::arg("x"));


    //  
    // Problem classes
    // ===============
    //
    addProblemClassToModule<hopsy::DegenerateMultivariateGaussianModel, hopsy::DegenerateMultivariateGaussianProblem>(
                m, "DegenerateMultivariateGaussianProblem", 
                R"pbdoc()pbdoc");

    addProblemClassToModule<hopsy::MixtureModel, hopsy::MixtureProblem>(
                m, "MixtureProblem", 
                R"pbdoc()pbdoc");

    addProblemClassToModule<hopsy::MultivariateGaussianModel, hopsy::MultivariateGaussianProblem>(
                m, "MultivariateGaussianProblem", 
                R"pbdoc()pbdoc");

    addProblemClassToModule<hopsy::PyModel, hopsy::PyProblem>(
                m, "PyProblem", 
                R"pbdoc()pbdoc");

    addProblemClassToModule<hopsy::RosenbrockModel, hopsy::RosenbrockProblem>(
                m, "RosenbrockProblem", 
                R"pbdoc()pbdoc");

    py::class_<hopsy::UniformProblem>(
                m, "UniformProblem",
                R"pbdoc()pbdoc")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"))
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::UniformModel>(),
                R"pbdoc()pbdoc",
                py::arg("A"),
                py::arg("b"),
                py::arg("model"))
        .def_property("A", &hopsy::UniformProblem::getA, &hopsy::UniformProblem::setA)
        .def_property("b", &hopsy::UniformProblem::getB, &hopsy::UniformProblem::setB)
        .def_property("starting_point", &hopsy::UniformProblem::getStartingPoint, &hopsy::UniformProblem::setStartingPoint)
        .def_property("unrounding_transformation", &hopsy::UniformProblem::getUnroundingTransformation, &hopsy::UniformProblem::setUnroundingTransformation)
        .def_property("unrounding_shift", &hopsy::UniformProblem::getUnroundingShift, &hopsy::UniformProblem::setUnroundingShift);


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
	m.def("Problem", &hopsy::createProblem<hopsy::MixtureModel>,
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
    // Box constraints convenience function
    // ====================================
    //
    m.def("add_box_constraints", 
            py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
                &hopsy::addBoxConstraintsToMatrixVector), 
            py::arg("A"),
            py::arg("b"),
            py::arg("lower_bounds"),
            py::arg("upper_bounds"));
    m.def("add_box_constraints", 
            py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&, double, double>(
                &hopsy::addBoxConstraintsToMatrixVector), 
            py::arg("A"),
            py::arg("b"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::DegenerateMultivariateGaussianProblem&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
                &hopsy::addBoxConstraintsToProblem<hopsy::DegenerateMultivariateGaussianProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::DegenerateMultivariateGaussianProblem&, double, double>(
                &hopsy::addBoxConstraintsToProblem<hopsy::DegenerateMultivariateGaussianProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::MixtureProblem&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
                &hopsy::addBoxConstraintsToProblem<hopsy::MixtureProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::MixtureProblem&, double, double>(
                &hopsy::addBoxConstraintsToProblem<hopsy::MixtureProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::MultivariateGaussianProblem&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
                &hopsy::addBoxConstraintsToProblem<hopsy::MultivariateGaussianProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::MultivariateGaussianProblem&, double, double>(
                &hopsy::addBoxConstraintsToProblem<hopsy::MultivariateGaussianProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::PyProblem&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
                &hopsy::addBoxConstraintsToProblem<hopsy::PyProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::PyProblem&, double, double>(
                &hopsy::addBoxConstraintsToProblem<hopsy::PyProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::RosenbrockProblem&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
                &hopsy::addBoxConstraintsToProblem<hopsy::RosenbrockProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::RosenbrockProblem&, double, double>(
                &hopsy::addBoxConstraintsToProblem<hopsy::RosenbrockProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::UniformProblem&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
                &hopsy::addBoxConstraintsToProblem<hopsy::UniformProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::UniformProblem&, double, double>(
                &hopsy::addBoxConstraintsToProblem<hopsy::UniformProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));


    //
    // Problem rounding with PolyRound
    // ===============================
    //
    m.def("round", &hopsy::round<hopsy::DegenerateMultivariateGaussianModel>, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::MixtureModel>, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::MultivariateGaussianModel>, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::PyModel>, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::RosenbrockModel>, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::UniformModel>, py::arg("problem"));


    //
    // Chebyshev center with PolyRound
    // ===============================
    //
    m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::DegenerateMultivariateGaussianModel>, py::arg("problem"));
    m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::MixtureModel>, py::arg("problem"));
    m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::MultivariateGaussianModel>, py::arg("problem"));
    m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::PyModel>, py::arg("problem"));
    m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::RosenbrockModel>, py::arg("problem"));
    m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::UniformModel>, py::arg("problem"));



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
    addRunClassToModule<hopsy::DegenerateMultivariateGaussianProblem, hopsy::DegenerateMultivariateGaussianRun>(
                m, "DegenerateMultivariateGaussianRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::MixtureProblem, hopsy::MixtureRun>(
                m, "MixtureRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianRun>(
                m, "MultivariateGaussianRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::PyProblem, hopsy::PyRun>(
                m, "PyRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockRun>(
                m, "RosenbrockRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::UniformProblem, hopsy::UniformRun>(
                m, "UniformRun",
                R"pbdoc()pbdoc");

    addRunClassToModule<hopsy::DegenerateMultivariateGaussianProblem, hopsy::DegenerateMultivariateGaussianPyProposalRun>(
                m, "DegenerateMultivariateGaussianPyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::MixtureProblem, hopsy::MixturePyProposalRun>(
                m, "MixturePyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianPyProposalRun>(
                m, "MultivariateGaussianPyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::PyProblem, hopsy::PyPyProposalRun>(
                m, "PyPyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockPyProposalRun>(
                m, "RosenbrockPyProposalRun",
                R"pbdoc()pbdoc");
    addRunClassToModule<hopsy::UniformProblem, hopsy::UniformPyProposalRun>(
                m, "UniformPyProposalRun",
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
            	py::arg("problem"), 
                py::arg("proposal") = "HitAndRun", 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRun<hopsy::MixtureModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal") = "HitAndRun", 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRun<hopsy::MultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal") = "HitAndRun", 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRun<hopsy::PyModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal") = "HitAndRun", 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRun<hopsy::RosenbrockModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal") = "HitAndRun", 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRun<hopsy::UniformModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal") = "HitAndRun", 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());

	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::DegenerateMultivariateGaussianModel>, 
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MixtureModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::PyModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::RosenbrockModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyProposal<hopsy::UniformModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());

	m.def("Run", &hopsy::createRunFromPyObject<hopsy::DegenerateMultivariateGaussianModel>, 
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::MixtureModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::MultivariateGaussianModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::PyModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::RosenbrockModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	m.def("Run", &hopsy::createRunFromPyObject<hopsy::UniformModel>,
            	R"pbdoc()pbdoc",
            	py::arg("problem"), 
                py::arg("proposal"), 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>());


    //  
    // Run tuning
    // ==========
    //
    //

    py::class_<hopsy::AcceptanceRateTarget>(m, "AcceptanceRateTarget", R"pbdoc()pbdoc")
        .def(py::init(
                    [] (double acceptanceRate) { 
                        hopsy::AcceptanceRateTarget tmp; 
                        tmp.acceptanceRateTargetValue = acceptanceRate; 
                        return tmp;
                    }), 
            py::arg("acceptance_rate") = .234)
        .def_readwrite("acceptance_rate", &hopsy::AcceptanceRateTarget::acceptanceRateTargetValue);

    py::class_<hopsy::ExpectedSquaredJumpDistanceTarget>(m, "ExpectedSquaredJumpDistanceTarget", R"pbdoc()pbdoc")
        .def(py::init(
                    [] (bool considerTimeCost) { 
                        hopsy::ExpectedSquaredJumpDistanceTarget tmp; 
                        tmp.considerTimeCost = considerTimeCost; 
                        return tmp;
                    }), 
            py::arg("consider_time_cost") = false)
        .def_readwrite("consider_time_cost", &hopsy::ExpectedSquaredJumpDistanceTarget::considerTimeCost);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MixtureRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MultivariateGaussianRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::RosenbrockRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::UniformRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::PyRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianPyProposalRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MixturePyProposalRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MultivariateGaussianPyProposalRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::RosenbrockPyProposalRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::UniformPyProposalRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::PyPyProposalRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tuning);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MixtureRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MultivariateGaussianRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::RosenbrockRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::UniformRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::PyRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MixturePyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MultivariateGaussianPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::RosenbrockPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::UniformPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::PyPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m, hopsy::doc::tuning);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MixtureRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MultivariateGaussianRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::RosenbrockRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::UniformRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::PyRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianPyProposalRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MixturePyProposalRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::MultivariateGaussianPyProposalRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::RosenbrockPyProposalRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::UniformPyProposalRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);
    addTuningMethodToModule<hopsy::PyPyProposalRun, hopsy::PyTuningTarget>(m, hopsy::doc::tuning);


    //  
    // Data classes
    // ============
    //
    //
    py::class_<hops::Data>(m, "Data",
                R"pbdoc()pbdoc")
        .def(py::init<>())
        //.def(py::init(&hopsy::constructDataFromSimpleData), py::arg("simple_data"))
		.def_property_readonly("n_chains",
                [] (const hops::Data& self) {
                    return self.chains.size();
                },
                R"pbdoc()pbdoc")
		.def_property_readonly("n_samples",
                [] (const hops::Data& self) {
                    return ( self.chains.size() ? self.chains[0].states->size() : 0 );
                },
                R"pbdoc()pbdoc")
		.def_property_readonly("dims", 
                [] (const hops::Data& self) {
                    size_t numberOfSamples = ( self.chains.size() ? self.chains[0].states->size() : 0 );
                    return static_cast<size_t>( numberOfSamples ? self.chains[0].states->at(0).size() : 0);
                },
                R"pbdoc()pbdoc")
		.def_property_readonly("shape",
                [] (const hops::Data& self) -> std::tuple<size_t, size_t, size_t> {
                    size_t numberOfSamples = ( self.chains.size() ? self.chains[0].states->size() : 0 );
                    size_t dim = static_cast<size_t>( numberOfSamples ? self.chains[0].states->at(0).size() : 0);
                    return {self.chains.size(), numberOfSamples, dim};
                },
                R"pbdoc()pbdoc")
		.def_property(
                "acceptance_rates", 
                [] (hops::Data& self) -> py::array { return py::cast(self.getAcceptanceRates()); }, 
                &hops::Data::setAcceptanceRates,
				R"pbdoc()pbdoc")
		.def_property(
                "negative_log_likelihood", 
                [] (hops::Data& self) -> py::array { return py::cast(self.getNegativeLogLikelihood()); }, 
                &hops::Data::setNegativeLogLikelihood,
				R"pbdoc()pbdoc")
		.def_property(
                "states", 
                [] (hops::Data& self) -> py::array { return py::cast(self.getStates()); }, 
                &hops::Data::setStates,
				R"pbdoc()pbdoc")
		.def_property(
                "timestamps", 
                [] (hops::Data& self) -> py::array { return py::cast(self.getTimestamps()); }, 
                &hops::Data::setTimestamps,
				R"pbdoc()pbdoc")
        .def("flatten", &hops::Data::flatten)
        .def("subsample", &hops::Data::subsample, 
                py::arg("n_subsamples"), 
                py::arg("seed"))
        .def("thin", &hops::Data::thin, 
                py::arg("thinning"))
        .def("__getitem__", py::overload_cast<const hops::Data&, const py::slice&>(&hopsy::getDataItem), 
                py::arg("slice"))
        .def("__getitem__", py::overload_cast<const hops::Data&, const std::tuple<py::slice, py::slice>&>(&hopsy::getDataItem), 
                py::arg("slices"))
        .def("reset", &hops::Data::reset,
				R"pbdoc()pbdoc")
        .def("write", &hops::Data::write,
            	py::arg("path"), 
                py::arg("discard_raw") = false, 
				R"pbdoc()pbdoc");

    // check: https://stackoverflow.com/questions/49452957/overload-cast-fails-in-a-specific-case
    using computeStatisticsSignature = Eigen::VectorXd(hops::Data&);
    m.def("compute_acceptance_rate", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeAcceptanceRate),
				R"pbdoc(Compute the average acceptance rate of the chains in ``data``. 
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
				R"pbdoc(Compute the effective sample size of the chains in ``data``. 
                The effective sample size is computed for every dimension individually and is then
                returned in an ``m`` x ``1`` column vector, 
                where ``m`` is the dimension of the states.

                Parameters
                ----------
                )pbdoc", 
                py::arg("data"));

    using computeExpectedSquaredJumpDistanceSignature = std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>(const hops::Data&, const Eigen::MatrixXd&);
    m.def("compute_expected_squared_jump_distance", py::overload_cast<const hops::Data&, const Eigen::MatrixXd&>(
                (computeExpectedSquaredJumpDistanceSignature*)&hops::computeExpectedSquaredJumpDistanceIncrementally),
				R"pbdoc(Compute the expected squared jump distance of the chains in ``data``. 
                The expected squared jump distance is computed for every chain individually and is then
                returned in an ``m`` x ``1`` column vector, 
                where ``m`` is the number of chains stored in ``data``.

                Parameters
                ----------
                )pbdoc", 
                py::arg("data"), py::arg("sqrt_covariance") = Eigen::MatrixXd(0, 0));

    using computeExpectedSquaredJumpDistanceIncrementallySignature = std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>(
            const hops::Data&, const hops::IntermediateExpectedSquaredJumpDistanceResults_&, const Eigen::MatrixXd&);
    m.def("compute_expected_squared_jump_distance", py::overload_cast<const hops::Data&, const hops::IntermediateExpectedSquaredJumpDistanceResults_&, const Eigen::MatrixXd&>(
                (computeExpectedSquaredJumpDistanceIncrementallySignature*)&hops::computeExpectedSquaredJumpDistanceIncrementally),
				R"pbdoc(Compute the expected squared jump distance of the chains in ``data``. 
                The expected squared jump distance is computed for every chain individually and is then
                returned in an ``m`` x ``1`` column vector, 
                where ``m`` is the number of chains stored in ``data``.

                Parameters
                ----------
                )pbdoc", 
                py::arg("data"), py::arg("intermediate_result"), py::arg("sqrt_covariance") = Eigen::MatrixXd(0, 0));

    using computeExpectedSquaredJumpDistanceEverySignature = Eigen::MatrixXd(const hops::Data&, size_t, const Eigen::MatrixXd&);
    m.def("compute_expected_squared_jump_distance", py::overload_cast<const hops::Data&, size_t, const Eigen::MatrixXd&>(
                (computeExpectedSquaredJumpDistanceEverySignature*)&hops::computeExpectedSquaredJumpDistanceEvery),
				R"pbdoc(
                Compute the expected squared jump distance of the chains in ``data``. 
                The expected squared jump distance is computed for every chain individually and is then
                returned in an ``m`` x ``1`` column vector, 
                where ``m`` is the number of chains stored in ``data``.

                Parameters
                ----------
                )pbdoc", 
                py::arg("data"), py::arg("every"), py::arg("sqrt_covariance") = Eigen::MatrixXd(0, 0));

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
		.def_property_readonly(
                "acceptance_rates", &hops::ChainData::getAcceptanceRates,
				R"pbdoc()pbdoc")
		.def_property_readonly(
                "negative_log_likelihood", &hops::ChainData::getNegativeLogLikelihood,
				R"pbdoc()pbdoc")
		.def_property_readonly(
                "states", &hops::ChainData::getStates,
				R"pbdoc()pbdoc")
		.def_property_readonly(
                "timestamps", &hops::ChainData::getTimestamps,
				R"pbdoc()pbdoc")
        .def(
                "reset", &hops::ChainData::reset,
				R"pbdoc()pbdoc");

    py::class_<hops::EmptyChainDataException>(m, "EmptyChainDataException");
    py::class_<hops::NoProblemProvidedException>(m, "NoProblemProvidedException");
    py::class_<hops::MissingStartingPointsException>(m, "MissingStartingPointsException");
    py::class_<hops::UninitializedDataFieldException>(m, "UninitializedDataFieldException");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
