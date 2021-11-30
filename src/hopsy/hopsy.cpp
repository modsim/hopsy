#include <memory>
#include <string>

#include <Eigen/Core>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "doc.hpp"
#include "misc.hpp"
#include "model.hpp"
#include "proposal.hpp"
#include "random.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

//template<typename Problem, typename Run>
//void addRunClassToModule(py::module& m, const char* name, const char* doc = "") {
//    py::class_<Run>(m, name, doc)
//        .def(py::init<Problem>(),
//				hopsy::doc::Run::__init__)
//        .def("init", &Run::init,
//				hopsy::doc::Run::init)
//        .def("sample", py::overload_cast<>(&Run::sample),
//                hopsy::doc::Run::sample)
//        .def("sample", py::overload_cast<unsigned long, unsigned long>(&Run::sample), 
//                hopsy::doc::Run::sample,
//                py::arg("n_samples"), py::arg("thinning") = 1)
//        .def_property_readonly("data", &Run::getData,
//				hopsy::doc::Run::data)
//        .def_property("problem", &Run::getProblem, &Run::setProblem,
//				hopsy::doc::Run::problem)
//        .def_property("starting_points", &Run::getStartingPoints, &Run::setStartingPoints,
//				hopsy::doc::Run::startingPoints)
//        .def_property("n_chains", &Run::getNumberOfChains, &Run::setNumberOfChains,
//				hopsy::doc::Run::numberOfChains)
//        .def_property("n_samples", &Run::getNumberOfSamples, &Run::setNumberOfSamples,
//				hopsy::doc::Run::numberOfSamples)
//        .def_property("thinning", &Run::getThinning, &Run::setThinning,
//				hopsy::doc::Run::thinning)
//        .def_property("stepsize", &Run::getStepSize, &Run::setStepSize,
//				hopsy::doc::Run::stepSize)
//        .def_property("fisher_weight", &Run::getFisherWeight, &Run::setFisherWeight,
//				hopsy::doc::Run::fisherWeight)
//        .def_property("random_seed", &Run::getRandomSeed, &Run::setRandomSeed,
//				hopsy::doc::Run::randomSeed)
//        .def_property("sample_until_convergence", &Run::getSamplingUntilConvergence, &Run::setSamplingUntilConvergence,
//				hopsy::doc::Run::samplingUntilConvergence)
//        .def_property("diagnostics_threshold", &Run::getConvergenceThreshold, &Run::setConvergenceThreshold,
//				hopsy::doc::Run::convergenceThreshold)
//        .def_property("max_repetitions", &Run::getMaxRepetitions, &Run::setMaxRepetitions,
//				hopsy::doc::Run::maxRepetitions);
//}
//
//template<typename Model>
//void overloadCreateRun(py::module& m, const char* doc = "") {
//    m.def("Run", &hopsy::createRun<Model>, 
//            	doc,
//            	py::arg("problem"), 
//                py::arg("proposal") = "HitAndRun", 
//                py::arg("n_samples") = 1000, 
//                py::arg("n_chains") = 1, 
//                py::arg("starting_points") = std::vector<Eigen::VectorXd>(),
//                py::arg("thinning") = 1,
//                py::arg("stepsize") = 1,
//                py::arg("fisher_weight") = 0.5,
//                py::arg("random_seed") = 0,
//                py::arg("sample_until_convergence") = false,
//                py::arg("diagnostics_threshold") = 1.05,
//                py::arg("max_repetitions") = 10);
//}
//
//template<typename Model>
//void overloadCreateRunFromPyProposal(py::module& m, const char* doc = "") {
//    m.def("Run", &hopsy::createRunFromPyProposal<Model>, 
//            	doc,
//            	py::arg("problem"), 
//                py::arg("proposal") = "HitAndRun", 
//                py::arg("n_samples") = 1000, 
//                py::arg("n_chains") = 1, 
//                py::arg("starting_points") = std::vector<Eigen::VectorXd>(),
//                py::arg("thinning") = 1,
//                py::arg("stepsize") = 1,
//                py::arg("fisher_weight") = 0.5,
//                py::arg("random_seed") = 0,
//                py::arg("sample_until_convergence") = false,
//                py::arg("diagnostics_threshold") = 1.05,
//                py::arg("max_repetitions") = 10);
//}
//
//template<typename Model>
//void overloadCreateRunFromPyObject(py::module& m, const char* doc = "") {
//    m.def("Run", &hopsy::createRunFromPyObject<Model>, 
//            	doc,
//            	py::arg("problem"), 
//                py::arg("proposal") = "HitAndRun", 
//                py::arg("n_samples") = 1000, 
//                py::arg("n_chains") = 1, 
//                py::arg("starting_points") = std::vector<Eigen::VectorXd>(),
//                py::arg("thinning") = 1,
//                py::arg("stepsize") = 1,
//                py::arg("fisher_weight") = 0.5,
//                py::arg("random_seed") = 0,
//                py::arg("sample_until_convergence") = false,
//                py::arg("diagnostics_threshold") = 1.05,
//                py::arg("max_repetitions") = 10);
//}
//
//template<typename Run, typename Target>
//void addTuningMethodToModule(py::module& m, const char* doc = "") {
//    m.def("tune", &hopsy::tune<Run, Target>,
//          doc,
//          py::arg("run"),
//          py::arg("target"),
//          py::arg("method") = "ThompsonSampling",
//          py::arg("n_test_samples") = 100,
//          py::arg("n_posterior_updates") = 100,
//          py::arg("n_pure_sampling") = 1,
//          py::arg("n_convergence_threshold") = 5,
//          py::arg("gridsize") = 101,
//          py::arg("lower_grid_bound") = 1e-5,
//          py::arg("upper_grid_bound") = 1e5,
//          py::arg("smoothing_length") = 0.5,
//          py::arg("random_seed") = 0,
//          py::arg("record_data") = true);
//}

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
    //m.attr("n_threads") = hops::numberOfThreads;

    //
    // Model classes
    // =============
    //
    py::classh<hopsy::Model, hopsy::ModelTrampoline<>>(m, "Model", 
            hopsy::doc::Model::base)
        .def(py::init<>(), hopsy::doc::Model::__init__)
        .def("compute_negative_log_likelihood", &hopsy::Model::computeNegativeLogLikelihood, 
                hopsy::doc::Model::computeNegativeLogLikelihood,
                py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::Model::computeLogLikelihoodGradient, 
                hopsy::doc::Model::computeLogLikelihoodGradient,
                py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::Model::computeExpectedFisherInformation, 
                hopsy::doc::Model::computeExpectedFisherInformation,
                py::arg("x"))
        .def("__repr__", [] (const hopsy::Model& self) -> std::string {
                    std::string repr = "hopsy.Model()";
                    return repr;
                })
        ;

    py::classh<hopsy::DegenerateGaussian, hopsy::Model, hopsy::ModelTrampoline<hopsy::DegenerateGaussian>>(m, "Gaussian", 
            hopsy::doc::DegenerateGaussian::base)
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd, std::vector<long>>(),
                hopsy::doc::DegenerateGaussian::__init__,
                py::arg("mean") = Eigen::VectorXd(Eigen::VectorXd::Zero(2)), 
                py::arg("covariance") = Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2)),
                py::arg("inactives") = std::vector<long>())
        .def("compute_negative_log_likelihood", &hopsy::DegenerateGaussian::computeNegativeLogLikelihood, 
                hopsy::doc::DegenerateGaussian::computeNegativeLogLikelihood,
                py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::DegenerateGaussian::computeLogLikelihoodGradient, 
                hopsy::doc::DegenerateGaussian::computeLogLikelihoodGradient,
                py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::DegenerateGaussian::computeExpectedFisherInformation, 
                hopsy::doc::DegenerateGaussian::computeExpectedFisherInformation,
                py::arg("x"))
        .def("__repr__", [] (const hopsy::DegenerateGaussian& self) -> std::string {
                    std::string repr = "hopsy.Gaussian(";
                    //repr += "mean=" + py::cast<std::string>(py::cast(self.mean).attr("__repr__")()) + ", ";
                    //repr += "covariance=" + py::cast<std::string>(py::cast(self.covariance).attr("__repr__")()) + ", ";
                    //repr += "inactives=" + py::cast<std::string>(py::cast(self.inactives).attr("__repr__")());
                    repr += ")";
                    return repr;
                })
        ;

    py::classh<hopsy::Mixture, hopsy::Model, hopsy::ModelTrampoline<hopsy::Mixture>>(m, "Mixture",
				hopsy::doc::Mixture::base)
        .def(py::init<std::vector<std::shared_ptr<hopsy::Model>>>(),
                hopsy::doc::Mixture::__init__,
                py::arg("components") = std::vector<std::shared_ptr<hopsy::Model>>())
        .def(py::init<std::vector<std::shared_ptr<hopsy::Model>>, std::vector<double>>(),
                py::arg("components"),
                py::arg("weights"))
        .def("compute_negative_log_likelihood", &hopsy::Mixture::computeNegativeLogLikelihood, 
                hopsy::doc::Mixture::computeNegativeLogLikelihood,
                py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::Mixture::computeLogLikelihoodGradient, 
                hopsy::doc::Mixture::computeLogLikelihoodGradient,
                py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::Mixture::computeExpectedFisherInformation, 
                hopsy::doc::Mixture::computeExpectedFisherInformation,
                py::arg("x"))
        .def("__repr__", [] (const hopsy::Mixture& self) -> std::string {
                    std::string repr = "hopsy.Mixture(";
                    //repr += "components=[";
                    //for (auto& component : self.modelComponents) {
                    //    repr += component.__repr__() + ", ";
                    //}
                    //repr += "], ";
                    //repr += "weights=[";
                    //for (auto& weight : self.weights) {
                    //    std::string str = std::to_string(weight);
                    //    str.erase(str.find_last_not_of('0') + 1, std::string::npos); // remove trailing zeros
                    //    repr += str + ", ";
                    //}
                    //repr += "]";
                    repr += ")";
                    return repr;
                })
        ;

    py::classh<hopsy::PyModel, hopsy::Model, hopsy::ModelTrampoline<hopsy::PyModel>>(m, "PyModel",
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
                py::arg("x"))
        .def("__repr__", [] (const hopsy::PyModel& self) -> std::string {
                    std::string repr = "hopsy.PyModel(";
                    repr += ")";
                    return repr;
                })
        ;

    py::classh<hopsy::Rosenbrock, hopsy::Model, hopsy::ModelTrampoline<hopsy::Rosenbrock>>(m, "Rosenbrock",
				hopsy::doc::Rosenbrock::base)
        .def(py::init<double, Eigen::VectorXd>(),
                hopsy::doc::Rosenbrock::__init__,
                py::arg("scale") = 1,
                py::arg("shift") = Eigen::VectorXd::Zero(1))
        .def("compute_negative_log_likelihood", &hopsy::Rosenbrock::computeNegativeLogLikelihood, 
                hopsy::doc::Rosenbrock::computeNegativeLogLikelihood,
				py::arg("x"))
        .def("compute_log_likelihood_gradient", &hopsy::Rosenbrock::computeLogLikelihoodGradient, 
                hopsy::doc::Rosenbrock::computeLogLikelihoodGradient,
				py::arg("x"))
        .def("compute_expected_fisher_information", &hopsy::Rosenbrock::computeExpectedFisherInformation, 
                hopsy::doc::Rosenbrock::computeExpectedFisherInformation,
				py::arg("x"))
        .def("__repr__", [] (const hopsy::Rosenbrock& self) -> std::string {
                    std::string repr = "hopsy.Rosenbrock(";
                    //std::string str = std::to_string(self.scaleParameter);
                    //str.erase(str.find_last_not_of('0') + 1, std::string::npos); // remove trailing zeros
                    //repr += "scale=" + str + ", ";
                    //repr += "shift=" + py::cast<std::string>(py::cast(self.shiftParameter).attr("__repr__")());
                    repr += ")";
                    return repr;
                })
        ;

    //py::class_<hopsy::UniformModel>(m, "Uniform",
	//			hopsy::doc::UniformModel::base)
    //    .def(py::init<>())
    //    .def("compute_negative_log_likelihood", &hopsy::UniformModel::computeNegativeLogLikelihood, 
    //            hopsy::doc::UniformModel::computeNegativeLogLikelihood,
	//			py::arg("x"))
    //    .def("compute_log_likelihood_gradient", &hopsy::UniformModel::computeLogLikelihoodGradient, 
    //            hopsy::doc::UniformModel::computeLogLikelihoodGradient,
	//			py::arg("x"))
    //    .def("compute_expected_fisher_information", &hopsy::UniformModel::computeExpectedFisherInformation, 
    //            hopsy::doc::UniformModel::computeExpectedFisherInformation,
    //            py::arg("x"))
    //    .def("__repr__", [] (const hopsy::UniformModel& self) -> std::string {
    //                std::string repr = "hopsy.Uniform()";
    //                return repr;
    //            })
    //    ;


    py::class_<hopsy::RandomNumberGenerator>(m, "RandomNumberGenerator")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"))
        .def(py::init<unsigned int, unsigned int>(), py::arg("seed"), py::arg("stream"))
        .def("__call__", [] (hopsy::RandomNumberGenerator& self) { return self(); })
        .def("__repr__", &hopsy::RandomNumberGenerator::__repr__)
        ;

    py::class_<hopsy::Uniform>(m, "Uniform")
        .def(py::init<double, double>(), py::arg("a") = 0, py::arg("b") = 1)
        .def("__call__", [] (hopsy::Uniform& self, hopsy::RandomNumberGenerator& rng) -> double { 
                return self(rng.rng); 
            })
        .def("__repr__", [] (hopsy::Uniform& self) -> std::string {
                std::string repr = "hopsy.Uniform(";
                repr += "a=" + std::to_string(self.a()) + ", ";
                repr += "b=" + std::to_string(self.b()) + ")";
                return repr;
            })
        ;

    py::class_<hopsy::Normal>(m, "Normal")
        .def(py::init<double, double>(), py::arg("mean") = 0, py::arg("stddev") = 1)
        .def("__call__", [] (hopsy::Normal& self, hopsy::RandomNumberGenerator& rng) -> double { 
                return self(rng.rng); 
            })
        .def("__repr__", [] (hopsy::Normal& self) -> std::string {
                std::string repr = "hopsy.Normal(";
                repr += "mean=" + std::to_string(self.mean()) + ", ";
                repr += "stddev=" + std::to_string(self.stddev()) + ")";
                return repr;
            })
        ;

    //  
    // Python proposal wrapper class
    // =============================
    //
    
    //py::class_<hopsy::Proposal, hopsy::ProposalTrampoline<>>(m, "Proposal")
    py::classh<hopsy::Proposal, hopsy::ProposalTrampoline<>>(m, "Proposal")
        .def(py::init<>())
        .def("propose", &hopsy::Proposal::propose)
        .def("accept_proposal", &hopsy::Proposal::acceptProposal)
        .def_property_readonly("proposal", &hopsy::Proposal::getProposal)
        .def_property("state", &hopsy::Proposal::getState, &hopsy::Proposal::setState)
        .def_property("stepsize", &hopsy::Proposal::getStepSize, &hopsy::Proposal::setStepSize)
        .def("has_stepsize", &hopsy::Proposal::hasStepSize)
        .def_property_readonly("proposal_name", &hopsy::Proposal::getProposalName)
        .def_property_readonly("negative_log_likelihood", &hopsy::Proposal::getNegativeLogLikelihood);
    
    py::classh<hopsy::PyProposal, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::PyProposal>>(m, "PyProposal")
        .def(py::init<py::object>(),
                py::arg("proposal"))
        .def("propose", [](hopsy::PyProposal& self, hopsy::RandomNumberGenerator& rng) -> std::pair<double, hopsy::VectorType> { 
                return self.propose(rng.rng); 
            })
        .def("accept_proposal", &hopsy::PyProposal::acceptProposal)
        .def_property_readonly("proposal", &hopsy::PyProposal::getProposal)
        .def_property("state", &hopsy::PyProposal::getState, &hopsy::PyProposal::setState)
        .def_property("stepsize", &hopsy::PyProposal::getStepSize, &hopsy::PyProposal::setStepSize)
        .def("has_stepsize", &hopsy::PyProposal::hasStepSize)
        .def_property_readonly("proposal_name", &hopsy::PyProposal::getProposalName)
        .def_property_readonly("negative_log_likelihood", &hopsy::PyProposal::getNegativeLogLikelihood);

    py::classh<hopsy::AdaptiveMetropolis, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::AdaptiveMetropolis>>(m, "AdaptiveMetropolis")
        .def(py::init<hopsy::MatrixType, hopsy::VectorType, hopsy::VectorType>(),
                py::arg("A"),
                py::arg("b"),
                py::arg("state"))
        .def("propose", [](hopsy::AdaptiveMetropolis& self, hopsy::RandomNumberGenerator& rng) -> std::pair<double, hopsy::VectorType> { 
                return self.propose(rng.rng); 
            })
        .def("accept_proposal", &hopsy::AdaptiveMetropolis::acceptProposal)
        .def_property_readonly("proposal", &hopsy::AdaptiveMetropolis::getProposal)
        .def_property("state", &hopsy::AdaptiveMetropolis::getState, &hopsy::AdaptiveMetropolis::setState)
        .def_property("stepsize", &hopsy::AdaptiveMetropolis::getStepSize, &hopsy::AdaptiveMetropolis::setStepSize)
        .def("has_stepsize", &hopsy::AdaptiveMetropolis::hasStepSize)
        .def_property_readonly("proposal_name", &hopsy::AdaptiveMetropolis::getProposalName)
        .def_property_readonly("negative_log_likelihood", &hopsy::AdaptiveMetropolis::getNegativeLogLikelihood);

    py::classh<hopsy::Gaussian, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::Gaussian>>(m, "GaussianProposal")
        .def(py::init<hopsy::MatrixType, hopsy::VectorType, hopsy::VectorType>(),
                py::arg("A"),
                py::arg("b"),
                py::arg("state"))
        .def("propose", [](hopsy::Gaussian& self, hopsy::RandomNumberGenerator& rng) -> std::pair<double, hopsy::VectorType> { 
                return self.propose(rng.rng); 
            })
        .def("accept_proposal", &hopsy::Gaussian::acceptProposal)
        .def_property_readonly("proposal", &hopsy::Gaussian::getProposal)
        .def_property("state", &hopsy::Gaussian::getState, &hopsy::Gaussian::setState)
        .def_property("stepsize", &hopsy::Gaussian::getStepSize, &hopsy::Gaussian::setStepSize)
        .def("has_stepsize", &hopsy::Gaussian::hasStepSize)
        .def_property_readonly("proposal_name", &hopsy::Gaussian::getProposalName)
        .def_property_readonly("negative_log_likelihood", &hopsy::Gaussian::getNegativeLogLikelihood);


    //py::class_<hopsy::MarkovChain>(m, "hopsMarkovChain")
    //    .def(py::init<std::shared_ptr<hopsy::Proposal>>(),
    //            py::arg("proposal"))
    //    .def("draw", [] (hopsy::MarkovChain& self, hopsy::RandomNumberGenerator& rng, long thinning) -> std::pair<double, hopsy::VectorType> { 
    //            return self.draw(rng.rng, thinning); 
    //        })
    //    .def_property("proposal_distribution", 
    //            [] (hopsy::MarkovChain& self) -> std::shared_ptr<hopsy::Proposal> {
    //                return self.proposalDistribution;
    //            },
    //            [] (hopsy::MarkovChain& self, std::shared_ptr<hopsy::Proposal> proposalDistribution) -> void {
    //                self.proposalDistribution = proposalDistribution->deepCopy();
    //            })
    //    .def_property("state", &hopsy::MarkovChain::getState, &hopsy::MarkovChain::setState)
    //    .def_property("parameter", &hopsy::MarkovChain::getProposalParameter, &hopsy::MarkovChain::setProposalParameter)
    //;

    //  
    // Run classes
    // ===========
    //
    //addRunClassToModule<hopsy::DegenerateMultivariateGaussianProblem, hopsy::DegenerateMultivariateGaussianRun>(
    //            m, "MultivariateGaussianRun",
    //            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::GaussianMixtureProblem, hopsy::GaussianMixtureRun>(
    //            m, "GaussianMixtureRun",
    //            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::MixtureProblem, hopsy::MixtureRun>(
    //            m, "MixtureRun",
    //            hopsy::doc::Run::base);
    ////addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianRun>(
    ////            m, "MultivariateGaussianRun",
    ////            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::PyProblem, hopsy::PyRun>(
    //            m, "PyModelRun",
    //            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockRun>(
    //            m, "RosenbrockRun",
    //            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::UniformProblem, hopsy::UniformRun>(
    //            m, "UniformRun",
    //            hopsy::doc::Run::base);

    //addRunClassToModule<hopsy::DegenerateMultivariateGaussianProblem, hopsy::DegenerateMultivariateGaussianPyProposalRun>(
    //            m, "MultivariateGaussianPyProposalRun",
    //            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::GaussianMixtureProblem, hopsy::GaussianMixturePyProposalRun>(
    //            m, "GaussianMixturePyProposalRun",
    //            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::MixtureProblem, hopsy::MixturePyProposalRun>(
    //            m, "MixturePyProposalRun",
    //            hopsy::doc::Run::base);
    ////addRunClassToModule<hopsy::MultivariateGaussianProblem, hopsy::MultivariateGaussianPyProposalRun>(
    ////            m, "MultivariateGaussianPyProposalRun",
    ////            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::PyProblem, hopsy::PyPyProposalRun>(
    //            m, "PyModelPyProposalRun",
    //            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::RosenbrockProblem, hopsy::RosenbrockPyProposalRun>(
    //            m, "RosenbrockPyProposalRun",
    //            hopsy::doc::Run::base);
    //addRunClassToModule<hopsy::UniformProblem, hopsy::UniformPyProposalRun>(
    //            m, "UniformPyProposalRun",
    //            hopsy::doc::Run::base);


    //  
    // Run factory method
    // ==================
    //
    // Instead of having to construct the correct run from a given problem manually, this method 
    // simulates a general run constructor which then statically checks the passed problem
    // type and returns the correctly instantiated run object
    //
    
    //overloadCreateRun<hopsy::DegenerateMultivariateGaussianModel>(m, hopsy::doc::Run::__init__);
    //overloadCreateRun<hopsy::GaussianMixtureModel>(m);
    //overloadCreateRun<hopsy::MixtureModel>(m);
    ////overloadCreateRun<hopsy::MultivariateGaussianModel>(m);
    //overloadCreateRun<hopsy::PyModel>(m);
    //overloadCreateRun<hopsy::RosenbrockModel>(m);
    //overloadCreateRun<hopsy::UniformModel>(m);

    //overloadCreateRunFromPyProposal<hopsy::DegenerateMultivariateGaussianModel>(m);
    //overloadCreateRunFromPyProposal<hopsy::GaussianMixtureModel>(m);
    //overloadCreateRunFromPyProposal<hopsy::MixtureModel>(m);
    ////overloadCreateRunFromPyProposal<hopsy::MultivariateGaussianModel>(m);
    //overloadCreateRunFromPyProposal<hopsy::PyModel>(m);
    //overloadCreateRunFromPyProposal<hopsy::RosenbrockModel>(m);
    //overloadCreateRunFromPyProposal<hopsy::UniformModel>(m);

    //overloadCreateRunFromPyObject<hopsy::DegenerateMultivariateGaussianModel>(m);
    //overloadCreateRunFromPyObject<hopsy::GaussianMixtureModel>(m);
    //overloadCreateRunFromPyObject<hopsy::MixtureModel>(m);
    ////overloadCreateRunFromPyObject<hopsy::MultivariateGaussianModel>(m);
    //overloadCreateRunFromPyObject<hopsy::PyModel>(m);
    //overloadCreateRunFromPyObject<hopsy::RosenbrockModel>(m);
    //overloadCreateRunFromPyObject<hopsy::UniformModel>(m);

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

    //py::class_<hopsy::AcceptanceRateTarget>(m, "AcceptanceRateTarget", hopsy::doc::AcceptanceRateTarget::base)
    //    .def(py::init(
    //                [] (double acceptanceRate) { 
    //                    hopsy::AcceptanceRateTarget tmp; 
    //                    tmp.acceptanceRateTargetValue = acceptanceRate; 
    //                    return tmp;
    //                }), 
    //        hopsy::doc::AcceptanceRateTarget::__init__, 
    //        py::arg("acceptance_rate") = .234)
    //    .def_readwrite("acceptance_rate", &hopsy::AcceptanceRateTarget::acceptanceRateTargetValue, hopsy::doc::AcceptanceRateTarget::acceptanceRate);

    //py::class_<hopsy::ExpectedSquaredJumpDistanceTarget>(m, "ExpectedSquaredJumpDistanceTarget", hopsy::doc::ExpectedSquaredJumpDistanceTarget::base)
    //    .def(py::init(
    //                [] (unsigned long lags, bool considerTimeCost) { 
    //                    hopsy::ExpectedSquaredJumpDistanceTarget tmp; 
    //                    std::vector<unsigned long> _lags;

    //                    for (unsigned long i = 0; i < lags; ++i) {
    //                        _lags.push_back(i);
    //                    }

    //                    tmp.lags = _lags;
    //                    tmp.considerTimeCost = considerTimeCost; 
    //                    return tmp;
    //                }), 
    //        hopsy::doc::ExpectedSquaredJumpDistanceTarget::__init__, 
    //        py::arg("lags") = 1,
    //        py::arg("consider_time_cost") = false)
    //    .def(py::init(
    //                [] (std::vector<unsigned long> lags, bool considerTimeCost) { 
    //                    hopsy::ExpectedSquaredJumpDistanceTarget tmp; 
    //                    tmp.lags = lags;
    //                    tmp.considerTimeCost = considerTimeCost; 
    //                    return tmp;
    //                }), 
    //        py::arg("lags") = std::vector<unsigned long>{1},
    //        py::arg("consider_time_cost") = false)
    //    .def_readwrite("lags", &hopsy::ExpectedSquaredJumpDistanceTarget::lags, hopsy::doc::ExpectedSquaredJumpDistanceTarget::lags)
    //    .def_readwrite("consider_time_cost", &hopsy::ExpectedSquaredJumpDistanceTarget::considerTimeCost, hopsy::doc::ExpectedSquaredJumpDistanceTarget::considerTimeCost);

    //addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianRun, hopsy::AcceptanceRateTarget>(m, hopsy::doc::tune);
    //addTuningMethodToModule<hopsy::GaussianMixtureRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::MixtureRun, hopsy::AcceptanceRateTarget>(m);
    ////addTuningMethodToModule<hopsy::MultivariateGaussianRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::RosenbrockRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::UniformRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::PyRun, hopsy::AcceptanceRateTarget>(m);

    //addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianPyProposalRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::GaussianMixturePyProposalRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::MixturePyProposalRun, hopsy::AcceptanceRateTarget>(m);
    ////addTuningMethodToModule<hopsy::MultivariateGaussianPyProposalRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::RosenbrockPyProposalRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::UniformPyProposalRun, hopsy::AcceptanceRateTarget>(m);
    //addTuningMethodToModule<hopsy::PyPyProposalRun, hopsy::AcceptanceRateTarget>(m);

    //addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::GaussianMixtureRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::MixtureRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    ////addTuningMethodToModule<hopsy::MultivariateGaussianRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::RosenbrockRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::UniformRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::PyRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);

    //addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::GaussianMixturePyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::MixturePyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    ////addTuningMethodToModule<hopsy::MultivariateGaussianPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::RosenbrockPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::UniformPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);
    //addTuningMethodToModule<hopsy::PyPyProposalRun, hopsy::ExpectedSquaredJumpDistanceTarget>(m);

    //addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::GaussianMixtureRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::MixtureRun, hopsy::PyTuningTarget>(m);
    ////addTuningMethodToModule<hopsy::MultivariateGaussianRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::RosenbrockRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::UniformRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::PyRun, hopsy::PyTuningTarget>(m);

    //addTuningMethodToModule<hopsy::DegenerateMultivariateGaussianPyProposalRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::GaussianMixturePyProposalRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::MixturePyProposalRun, hopsy::PyTuningTarget>(m);
    ////addTuningMethodToModule<hopsy::MultivariateGaussianPyProposalRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::RosenbrockPyProposalRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::UniformPyProposalRun, hopsy::PyTuningTarget>(m);
    //addTuningMethodToModule<hopsy::PyPyProposalRun, hopsy::PyTuningTarget>(m);


    ////  
    //// Data classes
    //// ============
    ////
    ////
    //py::class_<hops::Data>(m, "Data",
    //            hopsy::doc::Data::base)
    //    .def(py::init<>(), hopsy::doc::Data::__init__)
    //    //.def(py::init(&hopsy::constructDataFromSimpleData), py::arg("simple_data"))
	//	.def_property_readonly("n_chains",
    //            [] (const hops::Data& self) {
    //                return self.chains.size();
    //            },
    //            hopsy::doc::Data::numberOfChains)
	//	.def_property_readonly("n_samples",
    //            [] (const hops::Data& self) {
    //                return ( self.chains.size() ? self.chains[0].states->size() : 0 );
    //            },
    //            hopsy::doc::Data::numberOfSamples)
	//	.def_property_readonly("dims", 
    //            [] (const hops::Data& self) {
    //                size_t numberOfSamples = ( self.chains.size() ? self.chains[0].states->size() : 0 );
    //                return static_cast<size_t>( numberOfSamples ? self.chains[0].states->at(0).size() : 0);
    //            },
    //            hopsy::doc::Data::dims)
	//	.def_property_readonly("shape",
    //            [] (const hops::Data& self) -> std::tuple<size_t, size_t, size_t> {
    //                size_t numberOfSamples = ( self.chains.size() ? self.chains[0].states->size() : 0 );
    //                size_t dim = static_cast<size_t>( numberOfSamples ? self.chains[0].states->at(0).size() : 0);
    //                return {self.chains.size(), numberOfSamples, dim};
    //            },
    //            hopsy::doc::Data::shape)
	//	.def_property(
    //            "acceptance_rates", 
    //            [] (hops::Data& self) -> py::array { return py::cast(self.getAcceptanceRates()); }, 
    //            &hops::Data::setAcceptanceRates,
	//			hopsy::doc::Data::acceptanceRates)
	//	.def_property(
    //            "negative_log_likelihood", 
    //            [] (hops::Data& self) -> py::array { return py::cast(self.getNegativeLogLikelihood()); }, 
    //            &hops::Data::setNegativeLogLikelihood,
	//			hopsy::doc::Data::negativeLogLikelihood)
	//	.def_property(
    //            "states", 
    //            [] (hops::Data& self) -> py::array { return py::cast(self.getStates()); }, 
    //            &hops::Data::setStates,
	//			hopsy::doc::Data::states)
	//	.def_property(
    //            "timestamps", 
    //            [] (hops::Data& self) -> py::array { return py::cast(self.getTimestamps()); }, 
    //            &hops::Data::setTimestamps,
	//			hopsy::doc::Data::timestamps)
    //    .def("flatten", &hops::Data::flatten,
    //            hopsy::doc::Data::flatten)
    //    .def("subsample", &hops::Data::subsample, 
    //            hopsy::doc::Data::subsample,
    //            py::arg("n_subsamples"), 
    //            py::arg("seed"))
    //    .def("thin", &hops::Data::thin, 
    //            hopsy::doc::Data::thin,
    //            py::arg("thinning"))
    //    .def("__getitem__", py::overload_cast<const hops::Data&, const py::slice&>(&hopsy::getDataItem), 
    //            hopsy::doc::Data::__getitem__,
    //            py::arg("slice"))
    //    .def("__getitem__", py::overload_cast<const hops::Data&, const std::tuple<py::slice, py::slice>&>(&hopsy::getDataItem), 
    //            hopsy::doc::Data::__getitem__,
    //            py::arg("slices"))
    //    .def("reset", &hops::Data::reset,
	//			hopsy::doc::Data::reset)
    //    .def("write", &hops::Data::write, 
	//			hopsy::doc::Data::write,
    //        	py::arg("path"), 
    //            py::arg("discard_raw") = false);

    //// check: https://stackoverflow.com/questions/49452957/overload-cast-fails-in-a-specific-case
    //using computeStatisticsSignature = Eigen::VectorXd(hops::Data&);
    //m.def("compute_acceptance_rate", py::overload_cast<hops::Data&>(
    //            (computeStatisticsSignature*)&hops::computeAcceptanceRate),
    //            hopsy::doc::computeAcceptanceRate,
    //            py::arg("data"));
    //m.def("compute_effective_sample_size", py::overload_cast<hops::Data&>(
    //            (computeStatisticsSignature*)&hops::computeEffectiveSampleSize),
    //            py::arg("data"));

    //using computeExpectedSquaredJumpDistanceSignature = std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>(const hops::Data&, const Eigen::MatrixXd&);
    //m.def("compute_expected_squared_jump_distance", py::overload_cast<const hops::Data&, const Eigen::MatrixXd&>(
    //            (computeExpectedSquaredJumpDistanceSignature*)&hops::computeExpectedSquaredJumpDistanceIncrementally),
    //            hopsy::doc::computeExpectedSquaredJumpDistance,
    //            py::arg("data"), py::arg("sqrt_covariance") = Eigen::MatrixXd(0, 0));

    //using computeExpectedSquaredJumpDistanceIncrementallySignature = std::tuple<Eigen::VectorXd, hops::IntermediateExpectedSquaredJumpDistanceResults_>(
    //        const hops::Data&, const hops::IntermediateExpectedSquaredJumpDistanceResults_&, const Eigen::MatrixXd&);
    //m.def("compute_expected_squared_jump_distance", py::overload_cast<const hops::Data&, const hops::IntermediateExpectedSquaredJumpDistanceResults_&, const Eigen::MatrixXd&>(
    //            (computeExpectedSquaredJumpDistanceIncrementallySignature*)&hops::computeExpectedSquaredJumpDistanceIncrementally),
    //            //hopsy::doc::computeExpectedSquaredJumpDistanceNoIndex,
    //            py::arg("data"), py::arg("intermediate_result"), py::arg("sqrt_covariance") = Eigen::MatrixXd(0, 0));

    //using computeExpectedSquaredJumpDistanceEverySignature = Eigen::MatrixXd(const hops::Data&, size_t, const Eigen::MatrixXd&);
    //m.def("compute_expected_squared_jump_distance", py::overload_cast<const hops::Data&, size_t, const Eigen::MatrixXd&>(
    //            (computeExpectedSquaredJumpDistanceEverySignature*)&hops::computeExpectedSquaredJumpDistanceEvery),
    //            //hopsy::doc::computeExpectedSquaredJumpDistanceNoIndex,
    //            py::arg("data"), py::arg("every"), py::arg("sqrt_covariance") = Eigen::MatrixXd(0, 0));

    //m.def("compute_potential_scale_reduction_factor", py::overload_cast<hops::Data&>(
    //            (computeStatisticsSignature*)&hops::computePotentialScaleReductionFactor),
    //            hopsy::doc::computePotentialScaleReductionFactor,
    //            py::arg("data"));
    //m.def("compute_total_time_taken", py::overload_cast<hops::Data&>(
    //            (computeStatisticsSignature*)&hops::computeTotalTimeTaken),
    //            hopsy::doc::computeTotalTimeTaken,
    //            py::arg("data"));

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
}
