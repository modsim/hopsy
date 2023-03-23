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
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::TuningTarget);

namespace hopsy {

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

	private:
        TuningTarget* target;
    };

    class PyTuningTarget : public TuningTarget {
    public:
        PyTuningTarget() = default;

        PyTuningTarget(py::object pyObj) : pyObj(std::move(pyObj)) {};

        std::pair<double, double> operator()(const VectorType& x, const std::vector<hops::RandomNumberGenerator*>& randomNumberGenerators) override {
            std::vector<RandomNumberGenerator*> _randomNumberGenerators(randomNumberGenerators.size());
            for (size_t i = 0; i < randomNumberGenerators.size(); ++i) {
                _randomNumberGenerators[i]->rng = *randomNumberGenerators[i];
            }

            auto returnValue = pyObj.attr("__call__")(x, _randomNumberGenerators).cast<std::pair<double, double>>();

            // propagate changes to the rngs which were caused in the python call above back to the passed random generators.
            for (size_t i = 0; i < randomNumberGenerators.size(); ++i) {
                *randomNumberGenerators[i] = _randomNumberGenerators[i]->rng;
            }

            return returnValue;
        }

        std::string getName() const override {
            return pyObj.attr("name").cast<std::string>();
        }

        std::unique_ptr<TuningTarget> copyTuningTarget() const override {
            return pyObj.attr("deepcopy")().cast<std::unique_ptr<TuningTarget>>();
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
    std::pair<VectorType, MatrixType> tune(typename MethodType::param_type& methodParams,
                                           TuningTarget* target,
                                           std::vector<RandomNumberGenerator*>& randomNumberGenerator) {
        VectorType optimalParameters;
        double optimalTargetValue;
        MatrixType data;

        std::vector<hops::RandomNumberGenerator*> _randomNumberGenerator(randomNumberGenerator.size());
        for (size_t i = 0; i < randomNumberGenerator.size(); ++i) {
            _randomNumberGenerator[i] = &randomNumberGenerator[i]->rng;
        }

        TuningTargetWrapper _target{target};
        MethodType::tune(optimalParameters,
                         optimalTargetValue,
                         _randomNumberGenerator,
                         methodParams,
                         _target,
                         data);

        return {optimalParameters, data};
    }
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::PyTuningTarget);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::AcceptanceRateTarget);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::ExpectedSquaredJumpDistanceTarget);

namespace hopsy {
    void addTuning(py::module& m) {
        // tuning targets
        py::classh<TuningTarget>(m, "TuningTarget", doc::TuningTarget::base);

        py::classh<AcceptanceRateTarget, TuningTarget>(m, "AcceptanceRateTarget", doc::AcceptanceRateTarget::base)
            .def(py::init([] (std::vector<MarkovChain*>& markovChain,
                              unsigned long numberOfTestSamples,
                              double acceptanceRate,
                              unsigned long order) {
                            return createTarget<AcceptanceRateTarget, double, unsigned long>(
                                    markovChain, numberOfTestSamples, acceptanceRate, order);
                        }),
                doc::AcceptanceRateTarget::__init__,
                py::arg("markov_chains"),
                py::arg("n_test_samples") = 1000,
                py::arg("acceptance_rate") = 0.234,
                py::arg("order") = 1)
            .def_readwrite("n_test_samples", &AcceptanceRateTarget::numberOfTestSamples,
                    doc::AcceptanceRateTarget::numberOfTestSamples)
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

        py::classh<ExpectedSquaredJumpDistanceTarget, TuningTarget>(m, "ExpectedSquaredJumpDistanceTarget", doc::ExpectedSquaredJumpDistanceTarget::base)
            .def(py::init([] (std::vector<MarkovChain*>& markovChain,
                              unsigned long numberOfTestSamples,
                              unsigned long lags,
                              bool considerTimeCost,
                              bool estimateCovariance) {
                            std::vector<unsigned long> _lags;
                            for (unsigned long i = 0; i < lags; ++i) {
                                _lags.push_back(i+1);
                            }
                            return createTarget<ExpectedSquaredJumpDistanceTarget, std::vector<unsigned long>, bool, bool>(
                                    markovChain, numberOfTestSamples, _lags, considerTimeCost, estimateCovariance);
                        }),
                doc::ExpectedSquaredJumpDistanceTarget::__init__,
                py::arg("markov_chains"),
                py::arg("n_test_samples") = 1000,
                py::arg("lags") = 1,
                py::arg("consider_time_cost") = false,
                py::arg("estimate_covariance") = true)
            .def(py::init(&createTarget<ExpectedSquaredJumpDistanceTarget, std::vector<unsigned long>, bool, bool>),
                py::arg("markov_chains"),
                py::arg("n_test_samples") = 1000,
                py::arg("lags") = std::vector<unsigned long>{1},
                py::arg("consider_time_cost") = false,
                py::arg("estimate_covariance") = true)
            .def_readwrite("n_test_samples", &ExpectedSquaredJumpDistanceTarget::numberOfTestSamples,
                    doc::ExpectedSquaredJumpDistanceTarget::numberOfTestSamples)
            .def_readwrite("lags", &ExpectedSquaredJumpDistanceTarget::lags,
                    doc::ExpectedSquaredJumpDistanceTarget::lags)
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

        py::classh<PyTuningTarget, TuningTarget>(m, "PyTuningTarget", doc::PyTuningTarget::base)
            .def(py::init<py::object>(),
                doc::PyTuningTarget::__init__,
                py::arg("tuning_target"))
            .def("__call__", &PyTuningTarget::operator(),
                    py::arg("x"), py::arg("rngs"), doc::PyTuningTarget::__call__)
        ;

        // tuning methods
        py::class_<ThompsonSampling>(m, "ThompsonSamplingTuning", doc::ThompsonSampling::base)
            .def(py::init<size_t, size_t, size_t, size_t, double, double, double, size_t, bool>(),
                    doc::ThompsonSampling::__init__,
                    py::arg("n_posterior_update") = 100,
                    py::arg("n_pure_sampling") = 1,
                    py::arg("n_convergence") = 5,
                    py::arg("grid_size") = 101,
                    py::arg("lower_bound") = 1e-5,
                    py::arg("upper_bound") = 1e5,
                    py::arg("smoothing_length") = .5,
                    py::arg("random_seed") = 0,
                    py::arg("record_data") = false)
          .def_readonly("n_converged", &ThompsonSampling::posteriorUpdateIterationsNeeded,
				    doc::ThompsonSampling::posteriorUpdateIterationsNeeded)
          .def_readwrite("n_posterior_updates", &ThompsonSampling::posteriorUpdateIterations,
				    doc::ThompsonSampling::posteriorUpdateIterations)
          .def_readwrite("n_pure_sampling", &ThompsonSampling::pureSamplingIterations,
					doc::ThompsonSampling::pureSamplingIterations)
          .def_readwrite("n_convergence", &ThompsonSampling::iterationsForConvergence,
					doc::ThompsonSampling::iterationsForConvergence)
          .def_readwrite("grid_size", &ThompsonSampling::stepSizeGridSize,
					doc::ThompsonSampling::stepSizeGridSize)
          .def_readwrite("lower_bound", &ThompsonSampling::stepSizeLowerBound,
					doc::ThompsonSampling::stepSizeLowerBound)
          .def_readwrite("upper_bound", &ThompsonSampling::stepSizeUpperBound,
					doc::ThompsonSampling::stepSizeUpperBound)
          .def_readwrite("smoothing_length", &ThompsonSampling::smoothingLength,
					doc::ThompsonSampling::smoothingLength)
          .def_readwrite("random_seed", &ThompsonSampling::randomSeed, doc::ThompsonSampling::randomSeed)
          .def_readwrite("record_data", &ThompsonSampling::recordData, doc::ThompsonSampling::recordData);

        m.def("tune", &tune<hops::ThompsonSamplingTuner>,
                doc::tune,
                py::arg("method"), py::arg("target"), py::arg("rngs"));
    }
}

#endif // HOPSY_TUNING_HPP
