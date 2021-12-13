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
    class MarkovChain : public hops::MarkovChain {
    public:
        MarkovChain() = default;

        MarkovChain(const std::shared_ptr<Proposal> proposal, 
                    const std::shared_ptr<Model> model = nullptr, 
                    const std::optional<LinearTransformation>& transformation = std::nullopt) : 
                proposal(nullptr), 
                model(nullptr), 
                transformation(transformation) {
            createMarkovChain(this, proposal, model, transformation);
        }

	   	std::pair<double, VectorType> draw(hops::RandomNumberGenerator &randomNumberGenerator, long thinning = 1) override {
            return markovChain->draw(randomNumberGenerator, thinning);
        }

	   	VectorType getState() const override {
            return markovChain->getState();
        }

	   	void setState(const VectorType& x) override {
            return markovChain->setState(x);
        }

	   	double getStateNegativeLogLikelihood() override {
            return markovChain->getStateNegativeLogLikelihood();
        }

	  	std::any getParameter(const hops::ProposalParameter& parameter) const override {
            return markovChain->getParameter(parameter);
        }

	    void setParameter(const hops::ProposalParameter &parameter, const std::any &value) override {
            return markovChain->setParameter(parameter, value);
        }


        std::shared_ptr<Proposal> getProposal() const {
            return proposal;
        }

        void setProposal(const std::shared_ptr<Proposal> proposal) {
            createMarkovChain(this, proposal, this->model, this->transformation);
        }

        Model* getModel() const {
            return model.get();
        }

        void setModel(const std::shared_ptr<Model> model) {
            createMarkovChain(this, this->proposal, model, this->transformation);
        }

        std::shared_ptr<hops::MarkovChain>& getMarkovChain() {
            return markovChain;
        }

    private:
        std::shared_ptr<hops::MarkovChain> markovChain;

        std::shared_ptr<Proposal> proposal;
        std::shared_ptr<Model> model;
        std::optional<LinearTransformation> transformation;

        static inline void createMarkovChain(MarkovChain* mc,
                                             const std::shared_ptr<Proposal> proposal, 
                                             const std::shared_ptr<Model> model, 
                                             const std::optional<LinearTransformation>& transformation) {
            if (model && transformation) {
                auto tmp = hops::MarkovChainAdapter(
                        hops::MetropolisHastingsFilter(
                            hops::ModelMixin(
                                hops::StateTransformation(
                                    hopsy::ProposalWrapper(proposal),
                                    *transformation
                                ),
                                hopsy::ModelWrapper(model)
                            )
                        )
                );

                mc->proposal = tmp.getProposalPtr();
                mc->model = tmp.getModelPtr();

                mc->markovChain = std::make_unique<decltype(tmp)>(tmp);

            } else if (model && !transformation) {
                auto tmp = hops::MarkovChainAdapter(
                        hops::MetropolisHastingsFilter(
                            hops::ModelMixin(
                                hopsy::ProposalWrapper(proposal),
                                hopsy::ModelWrapper(model)
                            )
                        )
                );

                mc->proposal = tmp.getProposalPtr();
                mc->model = tmp.getModelPtr();

                mc->markovChain = std::make_unique<decltype(tmp)>(tmp);

            } else if (!model && transformation) {
                auto name = proposal->getProposalName();
                if (symmetricProposals.find(name) != symmetricProposals.end()) {
                    auto tmp = hops::MarkovChainAdapter(
                            hops::NoOpDrawAdapter(
                                hops::StateTransformation(
                                    hopsy::ProposalWrapper(proposal),
                                    *transformation
                                )
                            )
                    );

                    mc->proposal = tmp.getProposalPtr();
                    mc->markovChain = std::make_unique<decltype(tmp)>(tmp);
                } else {
                    auto tmp = hops::MarkovChainAdapter(
                            hops::MetropolisHastingsFilter(
                                hops::StateTransformation(
                                    hopsy::ProposalWrapper(proposal),
                                    *transformation
                                )
                            )
                    );

                    mc->proposal = tmp.getProposalPtr();
                    mc->markovChain = std::make_unique<decltype(tmp)>(tmp);
                }

            } else {
                auto name = proposal->getProposalName();
                if (symmetricProposals.find(name) != symmetricProposals.end()) {
                    auto tmp = hops::MarkovChainAdapter(
                            hops::NoOpDrawAdapter(
                                    hopsy::ProposalWrapper(proposal)
                            )
                    );

                    mc->proposal = tmp.getProposalPtr();
                    mc->markovChain = std::make_unique<decltype(tmp)>(tmp);

                } else {
                    auto tmp = hops::MarkovChainAdapter(
                            hops::MetropolisHastingsFilter(
                                    hopsy::ProposalWrapper(proposal)
                            )
                    );

                    mc->proposal = tmp.getProposalPtr();
                    mc->markovChain = std::make_unique<decltype(tmp)>(tmp);
                }
            }
        }

        const static inline std::set<std::string> symmetricProposals{"BallWalk", "CoordinateHitAndRun", "Gaussian", "HitAndRun"};
    };
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::MarkovChain);

namespace hopsy {
    template<typename MarkovChainBase = MarkovChain>
	class MarkovChainTrampoline : public MarkovChainBase, public py::trampoline_self_life_support {
	   	std::pair<double, VectorType> draw(hops::RandomNumberGenerator &randomNumberGenerator, long thinning = 1) override {
			PYBIND11_OVERRIDE_PURE(
				PYBIND11_TYPE(std::pair<double, VectorType>),
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

	   	void setState(const VectorType& x) override {
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

	  	std::any getParameter(const hops::ProposalParameter& parameter) const override {
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

    MarkovChain createMarkovChain(const std::shared_ptr<Proposal> proposal, const Problem& problem) {
        // hacky proposal initialization if proposal is uninitalized
        //if (!proposal.isInitialized) {
        //py::dict local;
        //local["proposal"] = proposal.get();
        //local["problem"] = problem;

        //py::exec(R"(
        //    try: 
        //        name = proposal.name
        //    except:
        //        print(type(proposal))
        //        if type(proposal) == core.AdapriveMetropolisProposal:
        //            proposal = core.AdapriveMetropolisProposal(problem)
        //        elif type(proposal) == core.CSmMALAProposal:
        //            proposal = CSmMALAProposal(problem)
        //        elif type(proposal) == core.GaussianProposal:
        //            proposal = core.GaussianProposal(problem)
        //)", local);

        //*proposal = *local["proposal"].cast<Proposal*>();
        //}


        if (problem.A.rows() != problem.b.rows()) {
            throw std::runtime_error("Dimension mismatch between row dimension of right-hand side operator A and row dimension of left-hand side vector b.");
        }

        if (problem.startingPoint && problem.A.cols() != problem.startingPoint->rows()) {
            throw std::runtime_error("Dimension mismatch between column dimension of right-hand side operator A and row dimension of vector starting_point.");
        }

        MarkovChain mc;

        if (!problem.model && !problem.transformation && !problem.shift) {
            mc = MarkovChain(std::shared_ptr<Proposal>(proposal));

        } else if (!problem.model && problem.transformation) {
            VectorType shift = VectorType::Zero(problem.startingPoint->cols());
            if (!problem.shift) {
                shift = *problem.shift;
            }

            mc = MarkovChain(std::shared_ptr<Proposal>(proposal), 
                             nullptr, 
                             LinearTransformation(*problem.transformation, *problem.shift));

        } else if (problem.model && !problem.transformation) {
            mc = MarkovChain(std::shared_ptr<Proposal>(proposal), problem.model);

        } else if (problem.model && problem.transformation) {
            VectorType shift = VectorType::Zero(problem.startingPoint->cols());
            if (!problem.shift) {
                shift = *problem.shift;
            }

            mc = MarkovChain(std::shared_ptr<Proposal>(proposal), 
                             problem.model, 
                             LinearTransformation(*problem.transformation, *problem.shift));
        }

        return mc;
    }

    void addMarkovChain(py::module& m) {
        py::classh<MarkovChain>(m, "MarkovChain")
            .def(py::init(&createMarkovChain),
                    py::arg("proposal"), 
                    py::arg("problem"))
            //.def(py::init([] (const py::object& metaclass, const Problem& problem) {
            //            Proposal* proposal = (metaclass.attr("__call__")(problem).cast<Proposal*>());
            //            return createMarkovChain(proposal, problem);
            //        }),
            //        py::arg("proposal"), 
            //        py::arg("problem"))
            .def("draw", [] (MarkovChain& self, 
                             RandomNumberGenerator& rng, 
                             long thinning = 1) -> std::pair<double, VectorType> {
                        return self.draw(rng.rng, thinning);
                    }, py::arg("rng"), py::arg("thinning") = 1)
            .def_property("state", &MarkovChain::getState, &MarkovChain::setState)
            .def_property("model", &MarkovChain::getModel, &MarkovChain::setModel)
            .def_property("proposal", &MarkovChain::getProposal, &MarkovChain::setProposal)
            .def_property_readonly("state_negative_log_likelihood", &MarkovChain::getStateNegativeLogLikelihood)
            .def("_get_parameter", [] (const MarkovChain& self, 
                                       const ProposalParameter& parameter) {
                        return std::any_cast<double>(self.getParameter(parameter));
                    }, py::arg("param"))
            .def("_set_parameter", [] (MarkovChain& self, 
                                       const ProposalParameter& parameter,
                                       double value) {
                        return self.setParameter(parameter, std::any(value));
                    }, py::arg("param"), py::arg("value"))
        ;
    }
} // namespace hopsy

#endif // HOPSY_MARKOVCHAIN_HPP
