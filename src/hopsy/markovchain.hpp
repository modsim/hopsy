#ifndef HOPSY_MARKOVCHAIN_HPP
#define HOPSY_MARKOVCHAIN_HPP

#include <set>
#include <memory>
#include <stdexcept>
#include <utility>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "../../extern/hops/src/hops/hops.hpp"

#include "doc.hpp"
#include "misc.hpp"
#include "model.hpp"
#include "problem.hpp"
#include "proposal.hpp"
#include "random.hpp"
#include "transformation.hpp"

namespace py = pybind11;

namespace hopsy {

    namespace {
        template<typename ProposalImpl>
        inline std::unique_ptr <hops::MarkovChain> proposalImplToMarkovChain(
                const ProposalImpl &proposalImpl,
                const std::shared_ptr <Model> model,
                std::optional <hopsy::RandomNumberGenerator> parallelTemperingSyncRng,
                double exchangeAttemptProbability) {

            if (model) {
                if (parallelTemperingSyncRng) {
                    auto wrappedProposalAndModel = hops::ModelMixin(proposalImpl,
                                                                    hops::Coldness(hopsy::ModelWrapper(model)));
                    auto mc = hops::MarkovChainAdapter(
                            hops::ParallelTempering(
                                    hops::MetropolisHastingsFilter(wrappedProposalAndModel),
                                    parallelTemperingSyncRng.value().rng,
                                    exchangeAttemptProbability)
                    );
                    return std::make_unique<decltype(mc)>(mc);
                }
                auto wrappedProposalAndModel = hops::ModelMixin(proposalImpl, hopsy::ModelWrapper(model));
                auto mc = hops::MarkovChainAdapter(hops::MetropolisHastingsFilter(wrappedProposalAndModel));
                return std::make_unique<decltype(mc)>(mc);
            } else {
                auto mc = hops::MarkovChainAdapter(hops::MetropolisHastingsFilter(proposalImpl));
                return std::make_unique<decltype(mc)>(mc);
            }
        }

        static inline auto wrapProposal(std::shared_ptr <Proposal> proposal,
                                        const LinearTransformation &transformation) {
            return hops::StateTransformation(
                    hopsy::ProposalWrapper(proposal),
                    transformation
            );
        }

        static inline auto wrapProposal(std::shared_ptr <Proposal> proposal) {
            return hopsy::ProposalWrapper(proposal);
        }
    }

    class MarkovChain : public hops::MarkovChain {
    public:
        MarkovChain() = default;

        MarkovChain(const Proposal *proposal,
                    const std::optional <LinearTransformation> &transformation = std::nullopt,
                    std::optional <RandomNumberGenerator> parallelTemperingSyncRng = std::nullopt,
                    double exchangeAttemptProbability = 0.1) :
                proposal(nullptr),
                model(nullptr),
                parallelTemperingSyncRng(parallelTemperingSyncRng),
                exchangeAttemptProbability(exchangeAttemptProbability),
                transformation(transformation) {
            createMarkovChain(this,
                              proposal,
                              this->model,
                              this->transformation,
                              this->parallelTemperingSyncRng,
                              this->exchangeAttemptProbability);
        }

        MarkovChain(const Proposal *proposal,
                    const std::unique_ptr <Model> &model,
                    const std::optional <LinearTransformation> &transformation = std::nullopt,
                    std::optional <RandomNumberGenerator> parallelTemperingSyncRng = std::nullopt,
                    double exchangeAttemptProbability = 0.1) :
                proposal(nullptr),
                parallelTemperingSyncRng(parallelTemperingSyncRng),
                exchangeAttemptProbability(exchangeAttemptProbability),
                transformation(transformation) {
            if (model) this->model = model->copyModel();
            createMarkovChain(this,
                              proposal,
                              this->model,
                              transformation,
                              parallelTemperingSyncRng,
                              exchangeAttemptProbability);
        }

        std::pair<double, VectorType>
        draw(hops::RandomNumberGenerator &randomNumberGenerator, long thinning = 1) override {
            auto draw = markovChain->draw(randomNumberGenerator, thinning);
            return draw;
        }

        VectorType getState() const override {
            return markovChain->getState();
        }

        void setState(const VectorType &x) override {
            return markovChain->setState(x);
        }

        double getStateNegativeLogLikelihood() override {
            return markovChain->getStateNegativeLogLikelihood();
        }

        double getStateLogDensity() {
            return -markovChain->getStateNegativeLogLikelihood();
        }

        std::any getParameter(const hops::ProposalParameter &parameter) const override {
            return markovChain->getParameter(parameter);
        }

        void setParameter(const hops::ProposalParameter &parameter, const std::any &value) override {
            return markovChain->setParameter(parameter, value);
        }

        std::variant <std::shared_ptr<Proposal>, std::shared_ptr<PyProposal>> getProposal() {
            if (proposal) {
                std::shared_ptr <PyProposal> pyProposalPtr = std::dynamic_pointer_cast<PyProposal>(proposal);
                if (pyProposalPtr) {
                    return pyProposalPtr;
                    }
            }
            return proposal;
        }

        void setProposal(std::variant<Proposal *, py::object> proposal) {
            std::unique_ptr <Proposal> proposalPtr = nullptr;
            try {
                proposalPtr = std::make_unique<hopsy::PyProposal>(std::get<py::object>(proposal));
            }
            catch (std::bad_variant_access &) {
                proposalPtr = std::get<Proposal *>(proposal)->copyProposal();
            }
            createMarkovChain(this,
                              proposalPtr.get(),
                              this->model,
                              this->transformation,
                              this->parallelTemperingSyncRng,
                              this->exchangeAttemptProbability);
        }

        /**
         * @@brief only used in parallel tempering runs to determine how often parallel chain exchange.
         */
        void setExchangeAttemptProbability(double newExchangeAttemptProbability) {
            createMarkovChain(this,
                              this->proposal.get(),
                              this->model,
                              this->transformation,
                              this->parallelTemperingSyncRng,
                              newExchangeAttemptProbability);
        }

        double getExchangeAttemptProbability() {
            return this->getExchangeAttemptProbability();
        }

        std::variant<py::object, Model *> getModel() const {
            if (model) {
                Model *modelPtr = model.get();
                auto pyModelPtr = dynamic_cast<hopsy::PyModel *>(modelPtr);
                if (pyModelPtr) {
                    return pyModelPtr->pyObj;
                }
                return modelPtr;
            } else {
                return nullptr;
            }
        }

        void setModel(std::variant <py::object, std::shared_ptr<Model>> model) {
            if (this->proposal->hasNegativeLogLikelihood()) {
                throw std::runtime_error(
                        "Warning: proposal will not receive new model object. "
                        "Fix this issue by creating a new MarkovChain with a new proposal containing the new model.");
            }
            std::shared_ptr <Model> modelPtr;
            try {
                py::object object = std::get<py::object>(model);
                modelPtr = std::make_shared<PyModel>(PyModel(object));
            }
            catch (std::bad_variant_access &) {
                modelPtr = std::get < std::shared_ptr < Model >> (model);
            }
            createMarkovChain(this, this->proposal.get(), modelPtr, this->transformation,
                              this->parallelTemperingSyncRng, this->exchangeAttemptProbability);
        }

        std::shared_ptr <hops::MarkovChain> &getMarkovChain() {
            return markovChain;
        }

        Problem getProblem() const {
            py::object handle = py::cast(proposal->copyProposal().get());
            if (transformation) {
                return Problem(proposal->getA(),
                               proposal->getB(),
                               (model ? model->copyModel() : std::unique_ptr<Model>(nullptr)),
                               std::optional(proposal->getState()),
                               std::optional(transformation->getMatrix()),
                               std::optional(transformation->getShift()));
            } else {
                return Problem(proposal->getA(),
                               proposal->getB(),
                               (model ? model->copyModel() : std::unique_ptr<Model>(nullptr)),
                               proposal->getState(),
                               std::optional<MatrixType>(),
                               std::optional<VectorType>());
            }
        }

        std::shared_ptr <Proposal> proposal;
        std::shared_ptr <Model> model;
        std::optional <hopsy::RandomNumberGenerator> parallelTemperingSyncRng;
        double exchangeAttemptProbability;

    private:
        std::shared_ptr <hops::MarkovChain> markovChain;
        std::optional <LinearTransformation> transformation;


        /**
         * @brief Initialized mc pointer.
         * @param mc
         * @param proposal
         * @param model
         * @param transformation
         * @param parallelTemperingSyncRng if optional is not empty, chain will be created with parallel tempering.
         *                                  Note that parallel tempering requires MPI communicator set up by hops.
         *                                  The MPI_COMM_WORLD communicator is used.
         */
        static inline void createMarkovChain(MarkovChain *mc,
                                             const Proposal *proposal,
                                             const std::shared_ptr <Model> model,
                                             const std::optional <LinearTransformation> &transformation,
                                             std::optional <RandomNumberGenerator> parallelTemperingSyncRng = std::nullopt,
                                             double exchangeAttemptProbability = 0.1) {
            if (!proposal) {
                std::invalid_argument("Passing nullptr for proposal is invalid.");
            }

            mc->proposal = proposal->copyProposal();
            mc->model = model;
            mc->transformation = transformation;
            mc->parallelTemperingSyncRng = parallelTemperingSyncRng;

            if (mc->model && mc->transformation) {
                if (mc->proposal->hasNegativeLogLikelihood()) {
                    throw std::invalid_argument(
                            "Model & Transformation not valid for this proposal type. Try again without rounding the problem.");
                }
                auto markovChain = proposalImplToMarkovChain(
                        wrapProposal(mc->proposal, mc->transformation.value()),
                        mc->model,
                        mc->parallelTemperingSyncRng,
                        exchangeAttemptProbability
                );

                mc->markovChain = std::move(markovChain);
                return;
            } else if (mc->model && !mc->transformation) {
                // If proposal has negativeLogLikelihood, it knows the model and the model wrapper is not required.
                if (mc->proposal->hasNegativeLogLikelihood()) {
                    // proposal knows model already
                    if (mc->parallelTemperingSyncRng) {
                        throw std::runtime_error("Parallel Tempering not yet supported for this proposal type.");
                    }
                    auto markovChain = proposalImplToMarkovChain(wrapProposal(mc->proposal),
                                                                 nullptr,
                                                                 std::nullopt,
                                                                 exchangeAttemptProbability);

                    mc->markovChain = std::move(markovChain);
                    return;
                } else {
                    auto markovChain = proposalImplToMarkovChain(
                            wrapProposal(mc->proposal),
                            mc->model,
                            mc->parallelTemperingSyncRng,
                            exchangeAttemptProbability
                    );

                    mc->markovChain = std::move(markovChain);
                    return;
                }
            } else if (!mc->model && mc->transformation) {
                if (mc->proposal->isSymmetric()) {
                    auto tmp = hops::MarkovChainAdapter(
                            hops::NoOpDrawAdapter(
                                    hops::StateTransformation(
                                            hopsy::ProposalWrapper(mc->proposal),
                                            *transformation
                                    )
                            )
                    );

                    mc->markovChain = std::make_unique<decltype(tmp)>(tmp);
                    return;
                } else {
                    auto tmp = hops::MarkovChainAdapter(
                            hops::MetropolisHastingsFilter(
                                    hops::StateTransformation(
                                            hopsy::ProposalWrapper(proposal),
                                            transformation.value()
                                    )
                            )
                    );
                    mc->markovChain = std::make_unique<decltype(tmp)>(tmp);
                    return;
                }
            } else {
                // Case: no model and no transformation
                if (proposal->isSymmetric()) {
                    auto tmp = hops::MarkovChainAdapter(
                            hops::NoOpDrawAdapter(
                                    hopsy::ProposalWrapper(mc->proposal)
                            )
                    );

                    mc->markovChain = std::make_unique<decltype(tmp)>(tmp);
                    return;
                } else {
                    auto tmp = hops::MarkovChainAdapter(
                            hops::MetropolisHastingsFilter(
                                    hopsy::ProposalWrapper(mc->proposal)
                            )
                    );
                    mc->markovChain = std::make_unique<decltype(tmp)>(tmp);
                    return;
                }
            }
        }
    };
}// namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::MarkovChain);

namespace hopsy {
    template<typename MarkovChainBase = MarkovChain>
    class MarkovChainTrampoline : public MarkovChainBase, public py::trampoline_self_life_support {
        std::pair<double, VectorType>
        draw(hops::RandomNumberGenerator &randomNumberGenerator, long thinning = 1) override {
            PYBIND11_OVERRIDE_PURE(
                    PYBIND11_TYPE(std::pair < double, VectorType > ),
                    MarkovChainBase,
                    draw
            );
        }

        VectorType getState() const override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    VectorType,
                    MarkovChainBase,
                    "state",
                    getState
            );
        }

        void setState(const VectorType &x) override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    void,
                    MarkovChainBase,
                    "state",
                    setState,
                    x
            );
        }

        double getStateNegativeLogLikelihood() override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    double,
                    MarkovChainBase,
                    "state_negative_log_likelihood",
                    getStateNegativeLogLikelihood
            );
        }

        std::any getParameter(const hops::ProposalParameter &parameter) const override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    std::any,
                    MarkovChainBase,
                    "_get_parameter",
                    getParameter,
                    parameter
            );
        }

        void setParameter(const hops::ProposalParameter &parameter, const std::any &value) override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    void,
                    MarkovChainBase,
                    "_set_parameter",
                    setParameter,
                    parameter,
                    value
            );
        }

    };

    MarkovChain createMarkovChain(const Proposal *proposal,
                                  const Problem &problem,
                                  std::optional <RandomNumberGenerator> parallelTemperingSyncRng = std::nullopt,
                                  double exchangeAttemptProbability = 0.1,
                                  bool proposalIsReversibleJump=false) {
        if (problem.A.rows() != problem.b.rows()) {
            throw std::runtime_error(
                    "Dimension mismatch between row dimension of right-hand side operator A and row dimension of left-hand side vector b.");
        }

        if (!proposalIsReversibleJump && (problem.startingPoint && problem.A.cols() != problem.startingPoint->rows()) ) {
            throw std::runtime_error(
                    "Dimension mismatch between column dimension of right-hand side operator A and row dimension of vector starting_point.");
        }
        else if (proposalIsReversibleJump && (problem.startingPoint && problem.A.cols()*2 != problem.startingPoint->rows()) ) {
            // With reversible jump the model is also tracking the model space. Therefore, the states are twice the size!
            throw std::runtime_error(
                    "Dimension mismatch between column dimension of right-hand side operator A and row dimension of vector starting_point.");
        }

        std::optional <LinearTransformation> transformation = std::nullopt;
        if (problem.transformation && problem.shift) {
            transformation = LinearTransformation(problem.transformation.value(), problem.shift.value());
        }

        return MarkovChain(proposal,
                           problem.model,
                           transformation,
                           parallelTemperingSyncRng,
                           exchangeAttemptProbability = exchangeAttemptProbability);
    }

    void addMarkovChain(py::module &m) {
        py::classh<MarkovChain>(m, "MarkovChain", doc::MarkovChain::base)
                .def(py::init(&createMarkovChain),
                     doc::MarkovChain::__init__,
                     py::arg("proposal"),
                     py::arg("problem"),
                     py::arg("parallelTemperingSyncRng") = std::nullopt,
                     py::arg("exchangeAttemptProbability") = 0.1,
                     py::arg("isReversibleJumpChain") = false)
        .def("draw", [](MarkovChain &self,
                                RandomNumberGenerator &rng,
                                long thinning = 1) -> std::pair<double, VectorType> {
                         return self.draw(rng.rng, thinning);
                     },
                     doc::MarkovChain::draw,
                     py::arg("rng"),
                     py::arg("thinning") = 1)
                .def_property("state", &MarkovChain::getState, &MarkovChain::setState, doc::MarkovChain::state)
                .def_property("model", &MarkovChain::getModel, &MarkovChain::setModel, doc::MarkovChain::model)
                .def_property("proposal", &MarkovChain::getProposal, &MarkovChain::setProposal, doc::MarkovChain::proposal)
                .def_property_readonly("problem", &MarkovChain::getProblem)
                .def_readwrite("parallelTemperingSyncRng", &MarkovChain::parallelTemperingSyncRng)
                .def_property("exchangeAttemptProbability", &MarkovChain::getExchangeAttemptProbability,
                              &MarkovChain::setExchangeAttemptProbability,
                              doc::MarkovChain::exchangeAttemptProbability)
                .def_property_readonly("state_negative_log_likelihood", &MarkovChain::getStateNegativeLogLikelihood,
                                       doc::MarkovChain::stateNegativeLogLikelihood)
                .def_property_readonly("state_log_density", [](MarkovChain &self) {
                                           return -self.getStateNegativeLogLikelihood();
                                       },
                                       doc::MarkovChain::stateLogDensity)
                .def(py::pickle([](const MarkovChain &self) {
                                    return py::make_tuple(self.proposal->copyProposal().release(), self.getProblem(), self.getState(),
                                                          self.parallelTemperingSyncRng, self.exchangeAttemptProbability);
                                },
                                [](py::tuple t) {
                                    if (t.size() != 5) throw std::runtime_error("Invalid state!");
                                    auto state = t[2].cast<VectorType>();
                                    auto proposal = t[0].cast<Proposal *>();
                                    bool isReversibleJumpChain = proposal->getProposalName().substr(0, 6) == "RJMCMC";
                                    auto markovChain = createMarkovChain(proposal,
                                                                         t[1].cast<Problem>(),
                                                                         t[3].cast<std::optional<RandomNumberGenerator>>(),
                                                                         t[4].cast<double>(),
                                                                         isReversibleJumpChain);

                                    markovChain.setState(t[2].cast<VectorType>());
                                    return markovChain;
                                }));
    }
} // namespace hopsy

#endif // HOPSY_MARKOVCHAIN_HPP
