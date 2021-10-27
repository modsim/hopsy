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
void addRunClassToModule(py::module& m, const char* name, const char* doc = "") {
    py::class_<Run>(m, name, doc)
        .def(py::init<Problem>(),
				hopsy::doc::Run::__init__)
        .def("init", &Run::init,
				hopsy::doc::Run::init)
        .def("sample", py::overload_cast<>(&Run::sample),
                hopsy::doc::Run::sample)
        .def("sample", py::overload_cast<unsigned long, unsigned long>(&Run::sample), 
                hopsy::doc::Run::sample,
                py::arg("n_samples"), py::arg("thinning") = 1)
        .def_property_readonly("data", &Run::getData,
				hopsy::doc::Run::data)
        .def_property("problem", &Run::getProblem, &Run::setProblem,
				hopsy::doc::Run::problem)
        .def_property("starting_points", &Run::getStartingPoints, &Run::setStartingPoints,
				hopsy::doc::Run::startingPoints)
        .def_property("n_chains", &Run::getNumberOfChains, &Run::setNumberOfChains,
				hopsy::doc::Run::numberOfChains)
        .def_property("n_samples", &Run::getNumberOfSamples, &Run::setNumberOfSamples,
				hopsy::doc::Run::numberOfSamples)
        .def_property("thinning", &Run::getThinning, &Run::setThinning,
				hopsy::doc::Run::thinning)
        .def_property("stepsize", &Run::getStepSize, &Run::setStepSize,
				hopsy::doc::Run::stepSize)
        .def_property("fisher_weight", &Run::getFisherWeight, &Run::setFisherWeight,
				hopsy::doc::Run::fisherWeight)
        .def_property("random_seed", &Run::getRandomSeed, &Run::setRandomSeed,
				hopsy::doc::Run::randomSeed)
        .def_property("sample_until_convergence", &Run::getSamplingUntilConvergence, &Run::setSamplingUntilConvergence,
				hopsy::doc::Run::samplingUntilConvergence)
        .def_property("diagnostics_threshold", &Run::getConvergenceThreshold, &Run::setConvergenceThreshold,
				hopsy::doc::Run::convergenceThreshold)
        .def_property("max_repetitions", &Run::getMaxRepetitions, &Run::setMaxRepetitions,
				hopsy::doc::Run::maxRepetitions);
}

template<typename Model>
void overloadCreateRun(py::module& m, const char* doc = "") {
    m.def("Run", &hopsy::createRun<Model>, 
            	doc,
            	py::arg("problem"), 
                py::arg("proposal") = "HitAndRun", 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>(),
                py::arg("thinning") = 1,
                py::arg("stepsize") = 1,
                py::arg("fisher_weight") = 0.5,
                py::arg("random_seed") = 0,
                py::arg("sample_until_convergence") = false,
                py::arg("diagnostics_threshold") = 1.05,
                py::arg("max_repetitions") = 10);
}

template<typename Model>
void overloadCreateRunFromPyProposal(py::module& m, const char* doc = "") {
    m.def("Run", &hopsy::createRunFromPyProposal<Model>, 
            	doc,
            	py::arg("problem"), 
                py::arg("proposal") = "HitAndRun", 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>(),
                py::arg("thinning") = 1,
                py::arg("stepsize") = 1,
                py::arg("fisher_weight") = 0.5,
                py::arg("random_seed") = 0,
                py::arg("sample_until_convergence") = false,
                py::arg("diagnostics_threshold") = 1.05,
                py::arg("max_repetitions") = 10);
}

template<typename Model>
void overloadCreateRunFromPyObject(py::module& m, const char* doc = "") {
    m.def("Run", &hopsy::createRunFromPyObject<Model>, 
            	doc,
            	py::arg("problem"), 
                py::arg("proposal") = "HitAndRun", 
                py::arg("n_samples") = 1000, 
                py::arg("n_chains") = 1, 
                py::arg("starting_points") = std::vector<Eigen::VectorXd>(),
                py::arg("thinning") = 1,
                py::arg("stepsize") = 1,
                py::arg("fisher_weight") = 0.5,
                py::arg("random_seed") = 0,
                py::arg("sample_until_convergence") = false,
                py::arg("diagnostics_threshold") = 1.05,
                py::arg("max_repetitions") = 10);
}

template<typename Model, typename Problem>
void addProblemClassToModule(py::module& m, const char* name, const char* doc = "") {
    py::class_<Problem>(m, name, doc)
        .def(py::init(
                    [] (Eigen::MatrixXd A, 
                        Eigen::VectorXd b, 
                        Model model, 
                        Eigen::VectorXd startingPoint, 
                        Eigen::MatrixXd unroundingTransformation, 
                        Eigen::VectorXd unroundingShift
                    ) -> Problem {
                        Problem problem{A, b, model};
                        problem.setStartingPoint(startingPoint);
                        problem.setUnroundingTransformation(unroundingTransformation);
                        problem.setUnroundingShift(unroundingShift);

                        return problem;
                    }
                ),
                hopsy::doc::Problem::__init__,
                py::arg("A"),
                py::arg("b"),
                py::arg("model"),
                py::arg("starting_point") = Eigen::VectorXd(0),
                py::arg("transformation") = Eigen::MatrixXd(0, 0),
                py::arg("shift") = Eigen::VectorXd(0))
        .def_property("A", &Problem::getA, &Problem::setA,
				hopsy::doc::Problem::A)
        .def_property("b", &Problem::getB, &Problem::setB,
				hopsy::doc::Problem::b)
        .def_property_readonly("model", &Problem::getModel,
				hopsy::doc::Problem::model)
        .def_property("starting_point", &Problem::getStartingPoint, &Problem::setStartingPoint,
				hopsy::doc::Problem::startingPoint)
        .def_property("transformation", &Problem::getUnroundingTransformation, &Problem::setUnroundingTransformation,
				hopsy::doc::Problem::unroundingTransformation)
        .def_property("shift", &Problem::getUnroundingShift, &Problem::setUnroundingShift,
				hopsy::doc::Problem::unroundingShift)
        ;
}

template<typename Run, typename Target>
void addTuningMethodToModule(py::module& m, const char* doc = "") {
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
          py::arg("upper_grid_bound") = 1e5,
          py::arg("smoothing_length") = 0.5,
          py::arg("random_seed") = 0,
          py::arg("record_data") = true);
}


PYBIND11_MODULE(_hopsy, m) {
    py::options options;
    options.disable_function_signatures();

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // 
    // OpenMP controls
    // ===============
    //
    m.attr("n_threads") = hops::numberOfThreads;

    //
    // Model classes
    // =============
    //
    py::class_<hopsy::DegenerateMultivariateGaussianModel>(m, "MultivariateGaussian", 
            hopsy::doc::DegenerateMultivariateGaussianModel::base)
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd, std::vector<long>>(),
                hopsy::doc::DegenerateMultivariateGaussianModel::__init__,
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)),
                py::arg("inactives") = std::vector<long>())
        .def("compute_negative_log_likelihood", &hopsy::DegenerateMultivariateGaussianModel::computeNegativeLogLikelihood, 
                hopsy::doc::DegenerateMultivariateGaussianModel::computeNegativeLogLikelihood,
                py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::DegenerateMultivariateGaussianModel::computeLogLikelihoodGradient, 
                hopsy::doc::DegenerateMultivariateGaussianModel::computeLogLikelihoodGradient,
                py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::DegenerateMultivariateGaussianModel::computeExpectedFisherInformation, 
                hopsy::doc::DegenerateMultivariateGaussianModel::computeExpectedFisherInformation,
                py::arg("x"))
        ;

    py::class_<hopsy::GaussianMixtureModel>(m, "GaussianMixtureModel",
				hopsy::doc::GaussianMixtureModel::base)
        .def(py::init<std::vector<hopsy::DegenerateMultivariateGaussianModel>>(),
                hopsy::doc::GaussianMixtureModel::__init__,
                py::arg("components") = std::vector<hopsy::DegenerateMultivariateGaussianModel>())
        .def(py::init<std::vector<hopsy::DegenerateMultivariateGaussianModel>, std::vector<double>>(),
                py::arg("components"),
                py::arg("weights"))
        .def("compute_negative_log_likelihood", &hopsy::GaussianMixtureModel::computeNegativeLogLikelihood, 
                hopsy::doc::GaussianMixtureModel::computeNegativeLogLikelihood,
                py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::GaussianMixtureModel::computeLogLikelihoodGradient, 
                hopsy::doc::GaussianMixtureModel::computeLogLikelihoodGradient,
                py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::GaussianMixtureModel::computeExpectedFisherInformation, 
                hopsy::doc::GaussianMixtureModel::computeExpectedFisherInformation,
                py::arg("x"));

    py::class_<hopsy::MixtureModel>(m, "MixtureModel",
				hopsy::doc::MixtureModel::base)
        .def(py::init<std::vector<hopsy::PyModel>>(),
                hopsy::doc::MixtureModel::__init__,
                py::arg("components") = std::vector<hopsy::PyModel>())
        .def(py::init<std::vector<hopsy::PyModel>, std::vector<double>>(),
                py::arg("components"),
                py::arg("weights"))
        .def("compute_negative_log_likelihood", &hopsy::MixtureModel::computeNegativeLogLikelihood, 
                hopsy::doc::MixtureModel::computeNegativeLogLikelihood,
                py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::MixtureModel::computeLogLikelihoodGradient, 
                hopsy::doc::MixtureModel::computeLogLikelihoodGradient,
                py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::MixtureModel::computeExpectedFisherInformation, 
                hopsy::doc::MixtureModel::computeExpectedFisherInformation,
                py::arg("x"));

    //py::class_<hopsy::MultivariateGaussianModel>(m, "MultivariateGaussian",
	//			hopsy::doc::MultivariateGaussianModel::base)
    //    .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>(),
    //            hopsy::doc::MultivariateGaussianModel::__init__,
    //            py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
    //            py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)))
    //    .def("compute_negative_log_likelihood", &hopsy::MultivariateGaussianModel::computeNegativeLogLikelihood, 
    //            hopsy::doc::MultivariateGaussianModel::computeNegativeLogLikelihood,
	//			py::arg("x"))
    //    .def("compute_log_likelihood_gradient", &hopsy::MultivariateGaussianModel::computeLogLikelihoodGradient, 
    //            hopsy::doc::MultivariateGaussianModel::computeLogLikelihoodGradient,
	//			py::arg("x"))
    //    .def("compute_expected_fisher_information", &hopsy::MultivariateGaussianModel::computeExpectedFisherInformation, 
    //            hopsy::doc::MultivariateGaussianModel::computeExpectedFisherInformation,
	//			py::arg("x"));

    py::class_<hopsy::PyModel>(m, "PyModel",
				hopsy::doc::PyModel::base)
        .def(py::init<py::object>(),
                hopsy::doc::PyModel::__init__,
                py::arg("model"))
        .def("compute_negative_log_likelihood", &hopsy::PyModel::computeNegativeLogLikelihood, 
                hopsy::doc::PyModel::computeNegativeLogLikelihood,
				py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::PyModel::computeLogLikelihoodGradient, 
                hopsy::doc::PyModel::computeLogLikelihoodGradient,
				py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::PyModel::computeExpectedFisherInformation, 
                hopsy::doc::PyModel::computeExpectedFisherInformation,
				py::arg("x"));

    py::class_<hopsy::RosenbrockModel>(m, "Rosenbrock",
				hopsy::doc::RosenbrockModel::base)
        .def(py::init<double, Eigen::VectorXd>(),
                hopsy::doc::RosenbrockModel::__init__,
                py::arg("scale") = 1,
                py::arg("shift") = Eigen::VectorXd::Zero(1))
        .def("compute_negative_log_likelihood", &hopsy::RosenbrockModel::computeNegativeLogLikelihood, 
                hopsy::doc::RosenbrockModel::computeNegativeLogLikelihood,
				py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::RosenbrockModel::computeLogLikelihoodGradient, 
                hopsy::doc::RosenbrockModel::computeLogLikelihoodGradient,
				py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::RosenbrockModel::computeExpectedFisherInformation, 
                hopsy::doc::RosenbrockModel::computeExpectedFisherInformation,
				py::arg("x"));

    py::class_<hopsy::UniformModel>(m, "Uniform",
				hopsy::doc::UniformModel::base)
        .def(py::init<>())
        .def("compute_negative_log_likelihood", &hopsy::UniformModel::computeNegativeLogLikelihood, 
                hopsy::doc::UniformModel::computeNegativeLogLikelihood,
				py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::UniformModel::computeLogLikelihoodGradient, 
                hopsy::doc::UniformModel::computeLogLikelihoodGradient,
				py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::UniformModel::computeExpectedFisherInformation, 
                hopsy::doc::UniformModel::computeExpectedFisherInformation,
				py::arg("x"));


    //  
    // Problem classes
    // ===============
    //
    addProblemClassToModule<hopsy::DegenerateMultivariateGaussianModel, hopsy::DegenerateMultivariateGaussianProblem>(
                m, "MultivariateGaussianProblem", 
                hopsy::doc::Problem::base);

    addProblemClassToModule<hopsy::GaussianMixtureModel, hopsy::GaussianMixtureProblem>(
                m, "GaussianMixtureProblem", 
                hopsy::doc::Problem::base);

    addProblemClassToModule<hopsy::MixtureModel, hopsy::MixtureProblem>(
                m, "MixtureProblem", 
                hopsy::doc::Problem::base);

    //addProblemClassToModule<hopsy::MultivariateGaussianModel, hopsy::MultivariateGaussianProblem>(
    //            m, "MultivariateGaussianProblem", 
    //            hopsy::doc::Problem::base);

    addProblemClassToModule<hopsy::PyModel, hopsy::PyProblem>(
                m, "PyProblem", 
                hopsy::doc::Problem::base);

    addProblemClassToModule<hopsy::RosenbrockModel, hopsy::RosenbrockProblem>(
                m, "RosenbrockProblem", 
                hopsy::doc::Problem::base);

    py::class_<hopsy::UniformProblem>(
                m, "UniformProblem",
                hopsy::doc::Problem::base)
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd>(),
                hopsy::doc::Problem::__init__,
                py::arg("A"),
                py::arg("b"))
        .def(py::init(
                    [] (Eigen::MatrixXd A, 
                        Eigen::VectorXd b, 
                        hopsy::UniformModel model, 
                        Eigen::VectorXd startingPoint, 
                        Eigen::MatrixXd unroundingTransformation, 
                        Eigen::VectorXd unroundingShift
                    ) -> hopsy::UniformProblem {
                        hopsy::UniformProblem problem{A, b, model};

                        problem.setStartingPoint(startingPoint);
                        problem.setUnroundingTransformation(unroundingTransformation);
                        problem.setUnroundingShift(unroundingShift);

                        return problem;
                    }
                ),
                hopsy::doc::Problem::__init__,
                py::arg("A"),
                py::arg("b"),
                py::arg("model"),
                py::arg("starting_point") = Eigen::VectorXd(0),
                py::arg("transformation") = Eigen::MatrixXd(0, 0),
                py::arg("shift") = Eigen::VectorXd(0))
        .def_property("A", &hopsy::UniformProblem::getA, &hopsy::UniformProblem::setA,
				hopsy::doc::Problem::A)
        .def_property("b", &hopsy::UniformProblem::getB, &hopsy::UniformProblem::setB,
				hopsy::doc::Problem::b)
        .def_property_readonly("model", &hopsy::UniformProblem::getModel,
				hopsy::doc::Problem::model)
        .def_property("starting_point", &hopsy::UniformProblem::getStartingPoint, &hopsy::UniformProblem::setStartingPoint,
				hopsy::doc::Problem::startingPoint)
        .def_property("transformation", &hopsy::UniformProblem::getUnroundingTransformation, &hopsy::UniformProblem::setUnroundingTransformation,
				hopsy::doc::Problem::unroundingTransformation)
        .def_property("shift", &hopsy::UniformProblem::getUnroundingShift, &hopsy::UniformProblem::setUnroundingShift,
				hopsy::doc::Problem::unroundingShift)
        ;


    //  
    // Problem factory method
    // ======================
    //
    // Instead of having to construct the correct problem from a given model manually, this method 
    // simulates a general problem constructor which then statically checks the passed model
    // type and returns the correctly instantiated problem object
    //
	//m.def("Problem", &hopsy::createProblem<hopsy::DegenerateMultivariateGaussianModel>,
    //            R"pbdoc()pbdoc",
    //            py::arg("A"),
    //            py::arg("b"),
    //            py::arg("model"),
    //            py::arg("starting_point") = Eigen::VectorXd(0, 0),
    //            py::arg("unrounding_transformation") = Eigen::MatrixXd(0, 0),
    //            py::arg("unrounding_shift") = Eigen::VectorXd(0, 0));
	//m.def("Problem", &hopsy::createProblem<hopsy::MixtureModel>,
    //            R"pbdoc()pbdoc",
    //            py::arg("A"),
    //            py::arg("b"),
    //            py::arg("model"),
    //            py::arg("starting_point") = Eigen::VectorXd(0, 0),
    //            py::arg("unrounding_transformation") = Eigen::MatrixXd(0, 0),
    //            py::arg("unrounding_shift") = Eigen::VectorXd(0, 0));
	//m.def("Problem", &hopsy::createProblem<hopsy::MultivariateGaussianModel>,
    //            R"pbdoc()pbdoc",
    //            py::arg("A"),
    //            py::arg("b"),
    //            py::arg("model"),
    //            py::arg("starting_point") = Eigen::VectorXd(0, 0),
    //            py::arg("unrounding_transformation") = Eigen::MatrixXd(0, 0),
    //            py::arg("unrounding_shift") = Eigen::VectorXd(0, 0));
	//m.def("Problem", &hopsy::createProblem<hopsy::PyModel>,
    //            R"pbdoc()pbdoc",
    //            py::arg("A"),
    //            py::arg("b"),
    //            py::arg("model"),
    //            py::arg("starting_point") = Eigen::VectorXd(0, 0),
    //            py::arg("unrounding_transformation") = Eigen::MatrixXd(0, 0),
    //            py::arg("unrounding_shift") = Eigen::VectorXd(0, 0));
	//m.def("Problem", &hopsy::createProblem<hopsy::RosenbrockModel>,
    //            R"pbdoc()pbdoc",
    //            py::arg("A"),
    //            py::arg("b"),
    //            py::arg("model"),
    //            py::arg("starting_point") = Eigen::VectorXd(0, 0),
    //            py::arg("unrounding_transformation") = Eigen::MatrixXd(0, 0),
    //            py::arg("unrounding_shift") = Eigen::VectorXd(0, 0));
	//m.def("Problem", &hopsy::createProblem<hopsy::UniformModel>,
    //            R"pbdoc()pbdoc",
    //            py::arg("A"),
    //            py::arg("b"),
    //            py::arg("model"),
    //            py::arg("starting_point") = Eigen::VectorXd(0, 0),
    //            py::arg("unrounding_transformation") = Eigen::MatrixXd(0, 0),
    //            py::arg("unrounding_shift") = Eigen::VectorXd(0, 0));
	//m.def("Problem", &hopsy::createUniformProblem,
    //            R"pbdoc()pbdoc",
    //            py::arg("A"),
    //            py::arg("b"));
	//m.def("Problem", &hopsy::createPyProblem,
    //            R"pbdoc()pbdoc",
    //            py::arg("A"),
    //            py::arg("b"),
    //            py::arg("model"),
    //            py::arg("starting_point") = Eigen::VectorXd(0, 0),
    //            py::arg("unrounding_transformation") = Eigen::MatrixXd(0, 0),
    //            py::arg("unrounding_shift") = Eigen::VectorXd(0, 0));


    //
    // Box constraints convenience function
    // ====================================
    //
    //m.def("add_box_constraints", 
    //        py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
    //            &hopsy::addBoxConstraintsToMatrixVector), 
    //        hopsy::doc::addBoxConstraintsToMatrixVector,
    //        py::arg("A"),
    //        py::arg("b"),
    //        py::arg("lower_bounds"),
    //        py::arg("upper_bounds"));
    //m.def("add_box_constraints", 
    //        py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&, double, double>(
    //            &hopsy::addBoxConstraintsToMatrixVector), 
    //        py::arg("A"),
    //        py::arg("b"),
    //        py::arg("lower_bound"),
    //        py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::DegenerateMultivariateGaussianProblem&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
                &hopsy::addBoxConstraintsToProblem<hopsy::DegenerateMultivariateGaussianProblem>), 
            hopsy::doc::addBoxConstraintsToProblem,
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
            py::overload_cast<const hopsy::GaussianMixtureProblem&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
                &hopsy::addBoxConstraintsToProblem<hopsy::GaussianMixtureProblem>), 
            py::arg("problem"),
            py::arg("lower_bound"),
            py::arg("upper_bound"));
    m.def("add_box_constraints", 
            py::overload_cast<const hopsy::GaussianMixtureProblem&, double, double>(
                &hopsy::addBoxConstraintsToProblem<hopsy::GaussianMixtureProblem>), 
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
    //m.def("add_box_constraints", 
    //        py::overload_cast<const hopsy::MultivariateGaussianProblem&, const Eigen::VectorXd&, const Eigen::VectorXd&>(
    //            &hopsy::addBoxConstraintsToProblem<hopsy::MultivariateGaussianProblem>), 
    //        hopsy::doc::addBoxConstraintsToProblem,
    //        py::arg("problem"),
    //        py::arg("lower_bound"),
    //        py::arg("upper_bound"));
    //m.def("add_box_constraints", 
    //        py::overload_cast<const hopsy::MultivariateGaussianProblem&, double, double>(
    //            &hopsy::addBoxConstraintsToProblem<hopsy::MultivariateGaussianProblem>), 
    //        hopsy::doc::addBoxConstraintsToProblem,
    //        py::arg("problem"),
    //        py::arg("lower_bound"),
    //        py::arg("upper_bound"));
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
    m.def("round", &hopsy::round<hopsy::DegenerateMultivariateGaussianModel>, hopsy::doc::round, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::GaussianMixtureModel>, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::MixtureModel>, py::arg("problem"));
    //m.def("round", &hopsy::round<hopsy::MultivariateGaussianModel>, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::PyModel>, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::RosenbrockModel>, py::arg("problem"));
    m.def("round", &hopsy::round<hopsy::UniformModel>, py::arg("problem"));


    //
    // Chebyshev center with PolyRound
    // ===============================
    //
    m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::DegenerateMultivariateGaussianModel>, hopsy::doc::computeChebyshevCenter, py::arg("problem"));
    m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::GaussianMixtureModel>, py::arg("problem"));
    m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::MixtureModel>, py::arg("problem"));
    //m.def("compute_chebyshev_center", &hopsy::computeChebyshevCenter<hopsy::MultivariateGaussianModel>, py::arg("problem"));
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
                m, "MultivariateGaussianRun",
                hopsy::doc::Run::base);
    addRunClassToModule<hopsy::GaussianMixtureProblem, hopsy::GaussianMixtureRun>(
                m, "GaussianMixtureRun",
                hopsy::doc::Run::base);
    addRunClassToModule<hopsy::MixtureProblem, hopsy::MixtureRun>(
                m, "MixtureRun",
                hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianRun>(
    //            m, "MultivariateGaussianRun",
    //            hopsy::doc::Run::base);
    addRunClassToModule<hopsy::PyProblem, hopsy::PyRun>(
                m, "PyRun",
                hopsy::doc::Run::base);
    addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockRun>(
                m, "RosenbrockRun",
                hopsy::doc::Run::base);
    addRunClassToModule<hopsy::UniformProblem, hopsy::UniformRun>(
                m, "UniformRun",
                hopsy::doc::Run::base);

    addRunClassToModule<hopsy::DegenerateMultivariateGaussianProblem, hopsy::DegenerateMultivariateGaussianPyProposalRun>(
                m, "MultivariateGaussianPyProposalRun",
                hopsy::doc::Run::base);
    addRunClassToModule<hopsy::GaussianMixtureProblem, hopsy::GaussianMixturePyProposalRun>(
                m, "GaussianMixturePyProposalRun",
                hopsy::doc::Run::base);
    addRunClassToModule<hopsy::MixtureProblem, hopsy::MixturePyProposalRun>(
                m, "MixturePyProposalRun",
                hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianPyProposalRun>(
    //            m, "MultivariateGaussianPyProposalRun",
    //            hopsy::doc::Run::base);
    addRunClassToModule<hopsy::PyProblem, hopsy::PyPyProposalRun>(
                m, "PyPyProposalRun",
                hopsy::doc::Run::base);
    addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockPyProposalRun>(
                m, "RosenbrockPyProposalRun",
                hopsy::doc::Run::base);
    addRunClassToModule<hopsy::UniformProblem, hopsy::UniformPyProposalRun>(
                m, "UniformPyProposalRun",
                hopsy::doc::Run::base);


    //  
    // Run factory method
    // ==================
    //
    // Instead of having to construct the correct run from a given problem manually, this method 
    // simulates a general run constructor which then statically checks the passed problem
    // type and returns the correctly instantiated run object
    //
    
    overloadCreateRun<hopsy::DegenerateMultivariateGaussianModel>(m, hopsy::doc::Run::__init__);
    overloadCreateRun<hopsy::GaussianMixtureModel>(m);
    overloadCreateRun<hopsy::MixtureModel>(m);
    //overloadCreateRun<hopsy::MultivariateGaussianModel>(m);
    overloadCreateRun<hopsy::PyModel>(m);
    overloadCreateRun<hopsy::RosenbrockModel>(m);
    overloadCreateRun<hopsy::UniformRun>(m);

    overloadCreateRunFromPyProposal<hopsy::DegenerateMultivariateGaussianModel>(m);
    overloadCreateRunFromPyProposal<hopsy::GaussianMixtureModel>(m);
    overloadCreateRunFromPyProposal<hopsy::MixtureModel>(m);
    //overloadCreateRunFromPyProposal<hopsy::MultivariateGaussianModel>(m);
    overloadCreateRunFromPyProposal<hopsy::PyModel>(m);
    overloadCreateRunFromPyProposal<hopsy::RosenbrockModel>(m);
    overloadCreateRunFromPyProposal<hopsy::UniformRun>(m);

    overloadCreateRunFromPyObject<hopsy::DegenerateMultivariateGaussianModel>(m);
    overloadCreateRunFromPyObject<hopsy::GaussianMixtureModel>(m);
    overloadCreateRunFromPyObject<hopsy::MixtureModel>(m);
    //overloadCreateRunFromPyObject<hopsy::MultivariateGaussianModel>(m);
    overloadCreateRunFromPyObject<hopsy::PyModel>(m);
    overloadCreateRunFromPyObject<hopsy::RosenbrockModel>(m);
    overloadCreateRunFromPyObject<hopsy::UniformRun>(m);

	//m.def("Run", &hopsy::createRunFromPyProposal<hopsy::DegenerateMultivariateGaussianModel>, 
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MixtureModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyProposal<hopsy::MultivariateGaussianModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyProposal<hopsy::PyModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyProposal<hopsy::RosenbrockModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyProposal<hopsy::UniformModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());

	//m.def("Run", &hopsy::createRunFromPyObject<hopsy::DegenerateMultivariateGaussianModel>, 
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyObject<hopsy::MixtureModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyObject<hopsy::MultivariateGaussianModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyObject<hopsy::PyModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyObject<hopsy::RosenbrockModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());
	//m.def("Run", &hopsy::createRunFromPyObject<hopsy::UniformModel>,
    //        	R"pbdoc()pbdoc",
    //        	py::arg("problem"), 
    //            py::arg("proposal"), 
    //            py::arg("n_samples") = 1000, 
    //            py::arg("n_chains") = 1, 
    //            py::arg("starting_points") = std::vector<Eigen::VectorXd>());


    //  
    // Run tuning
    // ==========
    //
    //

    py::class_<hopsy::AcceptanceRateTarget>(m, "AcceptanceRateTarget", hopsy::doc::AcceptanceRateTarget::base)
        .def(py::init(
                    [] (double acceptanceRate) { 
                        hopsy::AcceptanceRateTarget tmp; 
                        tmp.acceptanceRateTargetValue = acceptanceRate; 
                        return tmp;
                    }), 
            hopsy::doc::AcceptanceRateTarget::__init__, 
            py::arg("acceptance_rate") = .234)
        .def_readwrite("acceptance_rate", &hopsy::AcceptanceRateTarget::acceptanceRateTargetValue, hopsy::doc::AcceptanceRateTarget::acceptanceRate);

    py::class_<hopsy::ExpectedSquaredJumpDistanceTarget>(m, "ExpectedSquaredJumpDistanceTarget", hopsy::doc::ExpectedSquaredJumpDistanceTarget::base)
        .def(py::init(
                    [] (unsigned long lags, bool considerTimeCost) { 
                        hopsy::ExpectedSquaredJumpDistanceTarget tmp; 
                        std::vector<unsigned long> _lags;

                        for (unsigned long i = 0; i < lags; ++i) {
                            _lags.push_back(i);
                        }

                        tmp.lags = _lags;
                        tmp.considerTimeCost = considerTimeCost; 
                        return tmp;
                    }), 
            hopsy::doc::ExpectedSquaredJumpDistanceTarget::__init__, 
            py::arg("lags") = 1,
            py::arg("consider_time_cost") = false)
        .def(py::init(
                    [] (std::vector<unsigned long> lags, bool considerTimeCost) { 
                        hopsy::ExpectedSquaredJumpDistanceTarget tmp; 
                        tmp.lags = lags;
                        tmp.considerTimeCost = considerTimeCost; 
                        return tmp;
                    }), 
            py::arg("lags") = std::vector<unsigned long>{1},
            py::arg("consider_time_cost") = false)
        .def_readwrite("lags", &hopsy::ExpectedSquaredJumpDistanceTarget::lags, hopsy::doc::ExpectedSquaredJumpDistanceTarget::lags)
        .def_readwrite("consider_time_cost", &hopsy::ExpectedSquaredJumpDistanceTarget::considerTimeCost, hopsy::doc::ExpectedSquaredJumpDistanceTarget::considerTimeCost);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tune);
    addTuningMethodToModule<hopsy::GaussianMixtureRun, hopsy::AcceptanceRateTarget>(m);
    addTuningMethodToModule<hopsy::MixtureRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::MultivariateGaussianRun, hopsy::AcceptanceRateTarget>(m);
    addTuningMethodToModule<hopsy::RosenbrockRun, hopsy::AcceptanceRateTarget>(m);
    addTuningMethodToModule<hopsy::UniformRun, hopsy::AcceptanceRateTarget>(m);
    addTuningMethodToModule<hopsy::PyRun, hopsy::AcceptanceRateTarget>(m);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianPyProposalRun, hopsy::AcceptanceRateTarget>(m);
    addTuningMethodToModule<hopsy::GaussianMixturePyProposalRun, hopsy::AcceptanceRateTarget>(m);
    addTuningMethodToModule<hopsy::MixturePyProposalRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::MultivariateGaussianPyProposalRun, hopsy::AcceptanceRateTarget>(m);
    addTuningMethodToModule<hopsy::RosenbrockPyProposalRun, hopsy::AcceptanceRateTarget>(m);
    addTuningMethodToModule<hopsy::UniformPyProposalRun, hopsy::AcceptanceRateTarget>(m);
    addTuningMethodToModule<hopsy::PyPyProposalRun, hopsy::AcceptanceRateTarget>(m);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::GaussianMixtureRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::MixtureRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::MultivariateGaussianRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::RosenbrockRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::UniformRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::PyRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::GaussianMixturePyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::MixturePyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::MultivariateGaussianPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::RosenbrockPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::UniformPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    addTuningMethodToModule<hopsy::PyPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::GaussianMixtureRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::MixtureRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::MultivariateGaussianRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::RosenbrockRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::UniformRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::PyRun, hopsy::PyTuningTarget>(m);

    addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianPyProposalRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::GaussianMixturePyProposalRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::MixturePyProposalRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::MultivariateGaussianPyProposalRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::RosenbrockPyProposalRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::UniformPyProposalRun, hopsy::PyTuningTarget>(m);
    addTuningMethodToModule<hopsy::PyPyProposalRun, hopsy::PyTuningTarget>(m);


    //  
    // Data classes
    // ============
    //
    //
    py::class_<hops::Data>(m, "Data",
                hopsy::doc::Data::base)
        .def(py::init<>(), hopsy::doc::Data::__init__)
        //.def(py::init(&hopsy::constructDataFromSimpleData), py::arg("simple_data"))
		.def_property_readonly("n_chains",
                [] (const hops::Data& self) {
                    return self.chains.size();
                },
                hopsy::doc::Data::numberOfChains)
		.def_property_readonly("n_samples",
                [] (const hops::Data& self) {
                    return ( self.chains.size() ? self.chains[0].states->size() : 0 );
                },
                hopsy::doc::Data::numberOfSamples)
		.def_property_readonly("dims", 
                [] (const hops::Data& self) {
                    size_t numberOfSamples = ( self.chains.size() ? self.chains[0].states->size() : 0 );
                    return static_cast<size_t>( numberOfSamples ? self.chains[0].states->at(0).size() : 0);
                },
                hopsy::doc::Data::dims)
		.def_property_readonly("shape",
                [] (const hops::Data& self) -> std::tuple<size_t, size_t, size_t> {
                    size_t numberOfSamples = ( self.chains.size() ? self.chains[0].states->size() : 0 );
                    size_t dim = static_cast<size_t>( numberOfSamples ? self.chains[0].states->at(0).size() : 0);
                    return {self.chains.size(), numberOfSamples, dim};
                },
                hopsy::doc::Data::shape)
		.def_property(
                "acceptance_rates", 
                [] (hops::Data& self) -> py::array { return py::cast(self.getAcceptanceRates()); }, 
                &hops::Data::setAcceptanceRates,
				hopsy::doc::Data::acceptanceRates)
		.def_property(
                "negative_log_likelihood", 
                [] (hops::Data& self) -> py::array { return py::cast(self.getNegativeLogLikelihood()); }, 
                &hops::Data::setNegativeLogLikelihood,
				hopsy::doc::Data::negativeLogLikelihood)
		.def_property(
                "states", 
                [] (hops::Data& self) -> py::array { return py::cast(self.getStates()); }, 
                &hops::Data::setStates,
				hopsy::doc::Data::states)
		.def_property(
                "timestamps", 
                [] (hops::Data& self) -> py::array { return py::cast(self.getTimestamps()); }, 
                &hops::Data::setTimestamps,
				hopsy::doc::Data::timestamps)
        .def("flatten", &hops::Data::flatten,
                hopsy::doc::Data::flatten)
        .def("subsample", &hops::Data::subsample, 
                hopsy::doc::Data::subsample,
                py::arg("n_subsamples"), 
                py::arg("seed"))
        .def("thin", &hops::Data::thin, 
                hopsy::doc::Data::thin,
                py::arg("thinning"))
        .def("__getitem__", py::overload_cast<const hops::Data&, const py::slice&>(&hopsy::getDataItem), 
                hopsy::doc::Data::__getitem__,
                py::arg("slice"))
        .def("__getitem__", py::overload_cast<const hops::Data&, const std::tuple<py::slice, py::slice>&>(&hopsy::getDataItem), 
                hopsy::doc::Data::__getitem__,
                py::arg("slices"))
        .def("reset", &hops::Data::reset,
				hopsy::doc::Data::reset)
        .def("write", &hops::Data::write, 
				hopsy::doc::Data::write,
            	py::arg("path"), 
                py::arg("discard_raw") = false);

    // check: https://stackoverflow.com/questions/49452957/overload-cast-fails-in-a-specific-case
    using computeStatisticsSignature = Eigen::VectorXd(hops::Data&);
    m.def("compute_acceptance_rate", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeAcceptanceRate),
                hopsy::doc::computeAcceptanceRate,
                py::arg("data"));
    m.def("compute_effective_sample_size", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeEffectiveSampleSize),
                py::arg("data"));

    using computeExpectedSquaredJumpDistanceSignature = std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>(const hops::Data&, const Eigen::MatrixXd&);
    m.def("compute_expected_squared_jump_distance", py::overload_cast<const hops::Data&, const Eigen::MatrixXd&>(
                (computeExpectedSquaredJumpDistanceSignature*)&hops::computeExpectedSquaredJumpDistanceIncrementally),
                hopsy::doc::computeExpectedSquaredJumpDistance,
                py::arg("data"), py::arg("sqrt_covariance") = Eigen::MatrixXd(0, 0));

    using computeExpectedSquaredJumpDistanceIncrementallySignature = std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>(
            const hops::Data&, const hops::IntermediateExpectedSquaredJumpDistanceResults_&, const Eigen::MatrixXd&);
    m.def("compute_expected_squared_jump_distance", py::overload_cast<const hops::Data&, const hops::IntermediateExpectedSquaredJumpDistanceResults_&, const Eigen::MatrixXd&>(
                (computeExpectedSquaredJumpDistanceIncrementallySignature*)&hops::computeExpectedSquaredJumpDistanceIncrementally),
                //hopsy::doc::computeExpectedSquaredJumpDistanceNoIndex,
                py::arg("data"), py::arg("intermediate_result"), py::arg("sqrt_covariance") = Eigen::MatrixXd(0, 0));

    using computeExpectedSquaredJumpDistanceEverySignature = Eigen::MatrixXd(const hops::Data&, size_t, const Eigen::MatrixXd&);
    m.def("compute_expected_squared_jump_distance", py::overload_cast<const hops::Data&, size_t, const Eigen::MatrixXd&>(
                (computeExpectedSquaredJumpDistanceEverySignature*)&hops::computeExpectedSquaredJumpDistanceEvery),
                //hopsy::doc::computeExpectedSquaredJumpDistanceNoIndex,
                py::arg("data"), py::arg("every"), py::arg("sqrt_covariance") = Eigen::MatrixXd(0, 0));

    m.def("compute_potential_scale_reduction_factor", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computePotentialScaleReductionFactor),
                hopsy::doc::computePotentialScaleReductionFactor,
                py::arg("data"));
    m.def("compute_total_time_taken", py::overload_cast<hops::Data&>(
                (computeStatisticsSignature*)&hops::computeTotalTimeTaken),
                hopsy::doc::computeTotalTimeTaken,
                py::arg("data"));

    //py::class_<hops::ChainData>(m, "ChainData",
    //            R"pbdoc()pbdoc")
    //    .def(py::init<>())
	//	.def_property_readonly(
    //            "acceptance_rates", &hops::ChainData::getAcceptanceRates,
	//			R"pbdoc()pbdoc")
	//	.def_property_readonly(
    //            "negative_log_likelihood", &hops::ChainData::getNegativeLogLikelihood,
	//			R"pbdoc()pbdoc")
	//	.def_property_readonly(
    //            "states", &hops::ChainData::getStates,
	//			R"pbdoc()pbdoc")
	//	.def_property_readonly(
    //            "timestamps", &hops::ChainData::getTimestamps,
	//			R"pbdoc()pbdoc")
    //    .def(
    //            "reset", &hops::ChainData::reset,
	//			R"pbdoc()pbdoc");

    py::class_<hops::EmptyChainDataException>(m, "EmptyChainDataException");
    py::class_<hops::NoProblemProvidedException>(m, "NoProblemProvidedException");
    py::class_<hops::MissingStartingPointsException>(m, "MissingStartingPointsException");
    py::class_<hops::UninitializedDataFieldException>(m, "UninitializedDataFieldException");
}
