#ifndef HOPSY_TUNING_HPP
#define HOPSY_TUNING_HPP

#include <memory>
#include <tuple>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "../../extern/hops/src/hops/hops.hpp"

#include "doc.hpp"
#include "hops/MarkovChain/Tuning/ThompsonSamplingTuner.hpp"
#include "markovchain.hpp"
#include "misc.hpp"
#include "random.hpp"

namespace py = pybind11;

namespace hopsy {
    using TuningTarget = hops::TuningTarget;

    // currently deprecated. 
    // TODO: 
    // in order to have a tuning target, which 
    //     - can also be evaluated in "stand-alone" fashion,
    //       that is by calling 
    //       ```
    //           target = TuningTarget(...) # whatever arguments are needed
    //           target(x=0.5)              # evalute target at x=0.5
    //       ```
    //     - allows for accessing the underlying estimator (in our case the markov chains)
    // we require a wrapper, that complies with hops requirements for a tuning target, but
    // which also stores the `hopsy.MarkovChain`s to return the correct objects when accessing
    // `target.markov_chains`.
    //
    class TuningTargetWrapper {
    public:
        TuningTargetWrapper() = default;

		TuningTargetWrapper(TuningTarget* target) : target(target) {

        };

        std::tuple<double, double> operator() (const VectorType x, const std::vector<hops::RandomNumberGenerator*>& randomNumberGenerators) {
            return target->operator()(x, randomNumberGenerators);
        }

        std::string getName() const {
            return target->getName();
        }

        std::vector<std::shared_ptr<MarkovChain>> markovChain;

	private:
        TuningTarget* target;
    };

    class PyTuningTarget : public TuningTarget {
    public:
        PyTuningTarget() = default;

        PyTuningTarget(py::object pyObj) : pyObj(std::move(pyObj)) {};

        std::tuple<double, double> operator() (const VectorType x, const std::vector<hops::RandomNumberGenerator*>& randomNumberGenerators) {
            std::vector<RandomNumberGenerator*> _randomNumberGenerators(randomNumberGenerators.size());
            for (size_t i = 0; i < randomNumberGenerators.size(); ++i) {
                _randomNumberGenerators[i]->rng = *randomNumberGenerators[i];
            }

            auto returnValue = pyObj.attr("__call__")(x, _randomNumberGenerators).cast<std::tuple<double, double>>();

            // propagate changes to the rngs which were caused in the python call above back to the passed random generators.
            for (size_t i = 0; i < randomNumberGenerators.size(); ++i) {
                *randomNumberGenerators[i] = _randomNumberGenerators[i]->rng;
            }

            return returnValue;
        }

        std::string getName() {
            return pyObj.attr("name").cast<std::string>();
        }

    private:
        py::object pyObj;
    };

    template<typename TargetType, typename ...Args>
    TargetType createTarget(std::vector<MarkovChain*>& markovChain, 
                            unsigned long numberOfTestSamples,
                            Args... args
                            ) {
        std::vector<std::shared_ptr<hops::MarkovChain>> _markovChain(markovChain.size());
        for (size_t i = 0; i < markovChain.size(); ++i) {
            _markovChain[i] = markovChain[i]->getMarkovChain();
        }

        return TargetType{_markovChain, numberOfTestSamples, args...};
    }

    using AcceptanceRateTarget = hops::AcceptanceRateTarget;
    using ExpectedSquaredJumpDistanceTarget = hops::ExpectedSquaredJumpDistanceTarget;

    using ThompsonSampling = hops::ThompsonSamplingTuner::param_type;
    using GridSearch = hops::GridSearchTuner::param_type;

    template<typename MethodType>
    void tune(typename MethodType::param_type& methodParams, 
              TuningTarget* target, 
              std::vector<RandomNumberGenerator*>& randomNumberGenerator) {
        VectorType optimalParameters;
        double optimalTargetValue;

        std::vector<hops::RandomNumberGenerator*> _randomNumberGenerator(randomNumberGenerator.size());
        for (size_t i = 0; i < randomNumberGenerator.size(); ++i) {
            _randomNumberGenerator[i] = &randomNumberGenerator[i]->rng;
        }
        TuningTargetWrapper _target{target};
        MethodType::tune(optimalParameters, optimalTargetValue, _randomNumberGenerator, methodParams, _target);
    }

    void addTuning(py::module& m) {
        // tuning targets
        py::class_<TuningTarget>(m, "TuningTarget"/*, doc::TuningTarget::base*/);

        py::class_<AcceptanceRateTarget, TuningTarget>(m, "AcceptanceRateTarget", doc::AcceptanceRateTarget::base)
            .def(py::init([] (double acceptanceRate,
                              unsigned long numberOfTestSamples, 
                              std::vector<MarkovChain*>& markovChain) { 
                            return createTarget<AcceptanceRateTarget, double>(
                                    markovChain, numberOfTestSamples, acceptanceRate);
                        }), 
                doc::AcceptanceRateTarget::__init__,
                py::arg("markov_chains"), 
                py::arg("acceptance_rate") = 0.234,
                py::arg("n_test_samples") = 1000)
            .def_readwrite("acceptance_rate", &AcceptanceRateTarget::acceptanceRateTargetValue, 
                    doc::AcceptanceRateTarget::acceptanceRate)
            .def("__call__", [] (AcceptanceRateTarget& self, 
                                 const VectorType& x, 
                                 std::vector<RandomNumberGenerator*>& randomNumberGenerators) {
                        std::vector<hops::RandomNumberGenerator*> _randomNumberGenerators(randomNumberGenerators.size());
                        for (size_t i = 0; i < randomNumberGenerators.size(); ++i) {
                            _randomNumberGenerators[i] = &randomNumberGenerators[i]->rng;
                        }
                        return self(x, _randomNumberGenerators);
                    },
                    py::arg("x"), py::arg("rngs"), doc::AcceptanceRateTarget::__call__)
        ;

        py::class_<ExpectedSquaredJumpDistanceTarget, TuningTarget>(m, "ExpectedSquaredJumpDistanceTarget", doc::ExpectedSquaredJumpDistanceTarget::base)
            .def(py::init([] (std::vector<MarkovChain*>& markovChain,
                              unsigned long numberOfTestSamples, 
                              unsigned long lags, 
                              bool considerTimeCost) { 
                            std::vector<unsigned long> _lags;
                            for (unsigned long i = 0; i < lags; ++i) {
                                _lags.push_back(i+1);
                            }
                            return createTarget<ExpectedSquaredJumpDistanceTarget, std::vector<unsigned long>, bool>(
                                    markovChain, numberOfTestSamples, _lags, considerTimeCost);
                        }), 
                doc::ExpectedSquaredJumpDistanceTarget::__init__, 
                py::arg("markov_chains"), 
                py::arg("n_test_samples") = 1000,
                py::arg("lags") = 1,
                py::arg("consider_time_cost") = false)
            .def(py::init(&createTarget<ExpectedSquaredJumpDistanceTarget, std::vector<unsigned long>, bool>),
                py::arg("markov_chains"), 
                py::arg("n_test_samples") = 1000,
                py::arg("lags") = std::vector<unsigned long>{1},
                py::arg("consider_time_cost") = false)
            .def_readwrite("lags", &ExpectedSquaredJumpDistanceTarget::lags, doc::ExpectedSquaredJumpDistanceTarget::lags)
            .def_readwrite("consider_time_cost", &ExpectedSquaredJumpDistanceTarget::considerTimeCost, 
                    doc::ExpectedSquaredJumpDistanceTarget::considerTimeCost)
            .def("__call__", [] (ExpectedSquaredJumpDistanceTarget& self, 
                                 const VectorType& x, 
                                 std::vector<RandomNumberGenerator*>& randomNumberGenerators) {
                        std::vector<hops::RandomNumberGenerator*> _randomNumberGenerators(randomNumberGenerators.size());
                        for (size_t i = 0; i < randomNumberGenerators.size(); ++i) {
                            _randomNumberGenerators[i] = &randomNumberGenerators[i]->rng;
                        }
                        return self(x, _randomNumberGenerators);
                    },
                    py::arg("x"), py::arg("rngs"), doc::ExpectedSquaredJumpDistanceTarget::__call__)
        ;

        // tuning methods
        py::class_<ThompsonSampling>(m, "ThompsonSamplingTuning", doc::ThompsonSampling::base)
            .def(py::init<size_t, size_t, size_t, size_t, size_t, double, double, double, size_t, bool>(),
                   py::arg("iterationsToTestStepSize") = 1000,
                   py::arg("posteriorUpdateIterations") = 100,
                   py::arg("pureSamplingIterations") = 1,
                   py::arg("iterationsForConvergence") = 5,
                   py::arg("stepSizeGridSize") = 100,
                   py::arg("stepSizeLowerBound") = 1e-5,
                   py::arg("stepSizeUpperBound") = 1e5,
                   py::arg("smoothingLength") = .5,
                   py::arg("randomSeed") = 0,
                   py::arg("recordData") = false)
          .def_readwrite("n_posterior_updates", &ThompsonSampling::posteriorUpdateIterations, doc::ThompsonSampling::posteriorUpdateIterations);

        m.def("tune", &tune<hops::ThompsonSamplingTuner>, 
                py::arg("method"), py::arg("target"), py::arg("rngs"));
    }
}

#endif // HOPSY_TUNING_HPP
