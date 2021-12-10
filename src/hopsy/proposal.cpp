#include "proposal.hpp"

void addProposals(py::module &m) {
    // register abstract Proposal
    py::classh<hopsy::Proposal, hopsy::ProposalTrampoline<>> proposal(m, "Proposal");
    // common
    proposal::addCommon<hopsy::Proposal>(m, proposal);


    // register AdaptiveMetropolisProposal
    py::classh<hopsy::AdaptiveMetropolisProposal, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::AdaptiveMetropolisProposal>> adaptiveMetropolisProposal(
            m, "AdaptiveMetropolisProposal");
    // constructor
    adaptiveMetropolisProposal.def(py::init<hopsy::MatrixType, hopsy::VectorType, hopsy::VectorType, double, double, long>(), 
            py::arg("A"), py::arg("b"), py::arg("starting_point"), py::arg("stepsize") = 1, py::arg("eps") = 1.e-3, py::arg("warm_up") = 100);
    // common
    proposal::addCommon<hopsy::AdaptiveMetropolisProposal>(m, adaptiveMetropolisProposal);
    // parameters
    proposal::addParameter<hopsy::AdaptiveMetropolisProposal>(
            m, adaptiveMetropolisProposal, hops::ProposalParameter::BOUNDARY_CUSHION, "boundary_cushion");
    proposal::addParameter<hopsy::AdaptiveMetropolisProposal>(
            m, adaptiveMetropolisProposal, hops::ProposalParameter::EPSILON, "eps");
    proposal::addParameter<hopsy::AdaptiveMetropolisProposal>(
            m, adaptiveMetropolisProposal, hops::ProposalParameter::STEP_SIZE, "stepsize");
    proposal::addParameter<hopsy::AdaptiveMetropolisProposal, decltype(adaptiveMetropolisProposal), long>(
            m, adaptiveMetropolisProposal, hops::ProposalParameter::WARM_UP, "warm_up");
    
    
    // register GaussianProposal
    py::classh<hopsy::BallWalkProposal, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::BallWalkProposal>> ballWalkProposal(
            m, "BallWalkProposal");
    // constructor
    ballWalkProposal.def(py::init<hopsy::MatrixType, hopsy::VectorType, hopsy::VectorType, double>(), 
            py::arg("A"), py::arg("b"), py::arg("starting_point"), py::arg("stepsize") = 1);
    // common
    proposal::addCommon<hopsy::BallWalkProposal>(m, ballWalkProposal);
    // parameters
    proposal::addParameter<hopsy::BallWalkProposal>(
            m, ballWalkProposal, hops::ProposalParameter::STEP_SIZE, "stepsize");


    // register CSmMALAProposal
    py::classh<hopsy::CSmMALAProposal, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::CSmMALAProposal>> csmmalaProposal(
            m, "CSmMALAProposal");
    // constructor
    csmmalaProposal.def(py::init([] (std::unique_ptr<hopsy::Model> model, 
                                     const hopsy::MatrixType& A, 
                                     const hopsy::VectorType& b, 
                                     const hopsy::VectorType& x, 
                                     double fisherWeight,
                                     double stepSize) -> hopsy::CSmMALAProposal {
                return hopsy::CSmMALAProposal(hopsy::ModelWrapper(model), A, b, x, fisherWeight, stepSize);
            }), 
            py::arg("model"), py::arg("A"), py::arg("b"), py::arg("starting_point"), py::arg("fisher_weight") = 1, py::arg("stepsize") = 1);
    csmmalaProposal.def(py::init([] (py::object model, 
                                     const hopsy::MatrixType& A, 
                                     const hopsy::VectorType& b, 
                                     const hopsy::VectorType& x, 
                                     double fisherWeight,
                                     double stepSize) -> hopsy::CSmMALAProposal {
                return hopsy::CSmMALAProposal(hopsy::ModelWrapper(hopsy::PyModel(model).copyModel()), A, b, x, fisherWeight, stepSize);
            }), 
            py::arg("model"), py::arg("A"), py::arg("b"), py::arg("starting_point"), py::arg("fisher_weight") = 1, py::arg("stepsize") = 1);
    // common
    proposal::addCommon<hopsy::CSmMALAProposal>(m, csmmalaProposal);
    // parameters
    proposal::addParameter<hopsy::CSmMALAProposal>(
            m, csmmalaProposal, hops::ProposalParameter::FISHER_WEIGHT, "fisher_weight");
    proposal::addParameter<hopsy::CSmMALAProposal>(
            m, csmmalaProposal, hops::ProposalParameter::STEP_SIZE, "stepsize");
    
    
    // register DikinWalkProposal
    py::classh<hopsy::DikinWalkProposal, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::DikinWalkProposal>> dikinWalkProposal(
            m, "DikinWalkProposal");
    // constructor
    dikinWalkProposal.def(py::init<hopsy::MatrixType, hopsy::VectorType, hopsy::VectorType, double>(), 
            py::arg("A"), py::arg("b"), py::arg("starting_point"), py::arg("stepsize") = 1);
    // common
    proposal::addCommon<hopsy::DikinWalkProposal>(m, dikinWalkProposal);
    // parameters
    proposal::addParameter<hopsy::DikinWalkProposal>(
            m, dikinWalkProposal, hops::ProposalParameter::BOUNDARY_CUSHION, "boundary_cushion");
    proposal::addParameter<hopsy::DikinWalkProposal>(
            m, dikinWalkProposal, hops::ProposalParameter::STEP_SIZE, "stepsize");
    
    
    // register GaussianCoordinateHitAndRun
    py::classh<hopsy::GaussianCoordinateHitAndRunProposal, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::GaussianCoordinateHitAndRunProposal>> gaussianCoordinateHitAndRunProposal(
            m, "GaussianCoordinateHitAndRunProposal");
    // constructor
    gaussianCoordinateHitAndRunProposal.def(py::init<hopsy::MatrixType, hopsy::VectorType, hopsy::VectorType>(), 
            py::arg("A"), py::arg("b"), py::arg("starting_point"));
    // common
    proposal::addCommon<hopsy::GaussianCoordinateHitAndRunProposal>(m, gaussianCoordinateHitAndRunProposal);
    // parameters
    proposal::addParameter<hopsy::GaussianCoordinateHitAndRunProposal>(
            m, gaussianCoordinateHitAndRunProposal, hops::ProposalParameter::STEP_SIZE, "stepsize");


    // register GaussianHitAndRun
    py::classh<hopsy::GaussianHitAndRunProposal, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::GaussianHitAndRunProposal>> gaussianHitAndRunProposal(
            m, "GaussianHitAndRunProposal");
    // constructor
    gaussianHitAndRunProposal.def(py::init<hopsy::MatrixType, hopsy::VectorType, hopsy::VectorType>(), 
            py::arg("A"), py::arg("b"), py::arg("starting_point"));
    // common
    proposal::addCommon<hopsy::GaussianHitAndRunProposal>(m, gaussianHitAndRunProposal);
    // parameters
    proposal::addParameter<hopsy::GaussianHitAndRunProposal>(
            m, gaussianHitAndRunProposal, hops::ProposalParameter::STEP_SIZE, "stepsize");


    // register GaussianProposal
    py::classh<hopsy::GaussianProposal, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::GaussianProposal>> gaussianProposal(
            m, "GaussianProposal");
    // constructor
    gaussianProposal.def(py::init<hopsy::MatrixType, hopsy::VectorType, hopsy::VectorType, double>(), 
            py::arg("A"), py::arg("b"), py::arg("starting_point"), py::arg("stepsize") = 1);
    // common
    proposal::addCommon<hopsy::GaussianProposal>(m, gaussianProposal);
    // parameters
    proposal::addParameter<hopsy::GaussianProposal>(
            m, gaussianProposal, hops::ProposalParameter::STEP_SIZE, "stepsize");


    // register PyProposal
    py::classh<hopsy::PyProposal, hopsy::Proposal, hopsy::ProposalTrampoline<hopsy::PyProposal>> 
        pyProposal(m, "PyProposal");
    // constructor
    pyProposal.def(py::init<py::object>(), py::arg("proposal"));
    // common
    proposal::addCommon<hopsy::PyProposal>(m, pyProposal);
}
