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

		TuningTargetWrapper(const std::unique_ptr<TuningTarget>& target) : target(target->copyTuningTarget()) {};

        std::tuple<double, double> operator() (const VectorType x) {
            return target->operator()(x);
        }

        std::string getName() const {
            return target->getName();
        }

	private:
        std::unique_ptr<TuningTarget> target;
    };

    class PyTuningTarget : public TuningTarget {
    public:
        PyTuningTarget() = default;

        PyTuningTarget(py::object pyObj) : pyObj(std::move(pyObj)) {};

        std::tuple<double, double> operator() (const VectorType x) {
            return pyObj.attr("__call__")(x).cast<std::tuple<double, double>>();
        }

        std::string getName() {
            try {
                return pyObj.attr("get_name")().cast<std::string>();
            } catch (...) {
                return "";
            }
        }

        std::vector<std::shared_ptr<hops::MarkovChain>> markovChain;
        std::vector<hops::RandomNumberGenerator*> randomNumberGenerator;
        unsigned long numberOfTestSamples;

    private:
        py::object pyObj;
    };

    template<typename TargetType, typename ...Args>
    TargetType createTarget(std::vector<std::shared_ptr<MarkovChain>>& markovChain, 
                            std::vector<RandomNumberGenerator*>& randomNumberGenerator,
                            unsigned long numberOfTestSamples,
                            Args... args
                            ) {
        std::vector<std::shared_ptr<hops::MarkovChain>> _markovChain(markovChain.size());
        for (size_t i = 0; i < markovChain.size(); ++i) {
            _markovChain[i] = markovChain[i]->getMarkovChain();
        }

        std::vector<hops::RandomNumberGenerator*> _randomNumberGenerator;
        for (size_t i = 0; i < randomNumberGenerator.size(); ++i) {
            _randomNumberGenerator[i] = &randomNumberGenerator[i]->rng;
        }

        return TargetType{_markovChain, _randomNumberGenerator, numberOfTestSamples, args...};
    }

    using AcceptanceRateTarget = hops::AcceptanceRateTarget;
    using ExpectedSquaredJumpDistanceTarget = hops::ExpectedSquaredJumpDistanceTarget;

    using ThompsonSampling = hops::ThompsonSamplingTuner::param_type;
    using GridSearch = hops::GridSearchTuner::param_type;

    template<typename MethodType>
    void tune(typename MethodType::param_type& methodParams, 
              TuningTarget* target, 
              const std::vector<std::shared_ptr<MarkovChain>>& markovChain, 
              std::vector<std::shared_ptr<RandomNumberGenerator>>& randomNumberGenerator) {
        VectorType optimalParameters;
        double optimalTargetValue;

        std::vector<std::shared_ptr<hops::MarkovChain>> _markovChain(markovChain.size());
        for (size_t i = 0; i < markovChain.size(); ++i) {
            _markovChain[i] = markovChain[i]->getMarkovChain();
        }

        std::vector<hops::RandomNumberGenerator*> _randomNumberGenerator;
        for (size_t i = 0; i < randomNumberGenerator.size(); ++i) {
            _randomNumberGenerator[i] = &randomNumberGenerator[i]->rng;
        }

        MethodType::tune(optimalParameters, optimalTargetValue, _markovChain, _randomNumberGenerator, methodParams, *target);
    }

    void addTuning(py::module& m) {
        // tuning targets
        py::class_<AcceptanceRateTarget>(m, "AcceptanceRateTarget", doc::AcceptanceRateTarget::base)
            .def(py::init(
                        [] (double acceptanceRate) { 
                            AcceptanceRateTarget tmp; 
                            tmp.acceptanceRateTargetValue = acceptanceRate; 
                            return tmp;
                        }), 
                doc::AcceptanceRateTarget::__init__, 
                py::arg("acceptance_rate") = .234)
            .def_readwrite("acceptance_rate", &AcceptanceRateTarget::acceptanceRateTargetValue, doc::AcceptanceRateTarget::acceptanceRate);

        py::class_<ExpectedSquaredJumpDistanceTarget>(m, "ExpectedSquaredJumpDistanceTarget", doc::ExpectedSquaredJumpDistanceTarget::base)
            .def(py::init(
                        [] (unsigned long lags, 
                            bool considerTimeCost,
                            unsigned long numberOfTestSamples, 
                            std::vector<std::shared_ptr<MarkovChain>>& markovChain,
                            std::vector<RandomNumberGenerator*>& randomNumberGenerator) 
                        { 
                            std::vector<unsigned long> _lags;
                            for (unsigned long i = 0; i < lags; ++i) {
                                _lags.push_back(i+1);
                            }
                            return createTarget<ExpectedSquaredJumpDistanceTarget, std::vector<unsigned long>, bool>(
                                    markovChain, randomNumberGenerator, numberOfTestSamples, _lags, considerTimeCost);
                        }), 
                doc::ExpectedSquaredJumpDistanceTarget::__init__, 
                py::arg("lags") = 1,
                py::arg("consider_time_cost") = false,
                py::arg("n_test_samples") = 1000,
                py::arg("markov_chains") = std::vector<std::shared_ptr<MarkovChain>>(),
                py::arg("rngs") = std::vector<RandomNumberGenerator*>())
            .def(py::init(
                        [] (std::vector<unsigned long> lags,
                            bool considerTimeCost,
                            unsigned long numberOfTestSamples, 
                            std::vector<std::shared_ptr<MarkovChain>>& markovChain,
                            std::vector<RandomNumberGenerator*>& randomNumberGenerator) 
                        { 
                            return createTarget<ExpectedSquaredJumpDistanceTarget, std::vector<unsigned long>, bool>(
                                    markovChain, randomNumberGenerator, numberOfTestSamples, lags, considerTimeCost);
                        }), 
                py::arg("lags") = std::vector<unsigned long>{1},
                py::arg("consider_time_cost") = false,
                py::arg("n_test_samples") = 1000,
                py::arg("markov_chains") = std::vector<std::shared_ptr<MarkovChain>>(),
                py::arg("rngs") = std::vector<RandomNumberGenerator*>())
            .def_readwrite("lags", &ExpectedSquaredJumpDistanceTarget::lags, doc::ExpectedSquaredJumpDistanceTarget::lags)
            .def_readwrite("consider_time_cost", &ExpectedSquaredJumpDistanceTarget::considerTimeCost, doc::ExpectedSquaredJumpDistanceTarget::considerTimeCost);

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
    }


}

#endif // HOPSY_TUNING_HPP
