#ifndef HOPSY_MODEL_HPP
#define HOPSY_MODEL_HPP

#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "hops/hops.hpp"
#include "misc.hpp"

namespace py = pybind11;

namespace hopsy {
    using Model = hops::Model;
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Model);

namespace hopsy {
    template<typename ModelBase = Model>
    class ModelTrampoline : public ModelBase, public py::trampoline_self_life_support {
    public:
        /* Inherit the constructors */
        using ModelBase::ModelBase;

        double logDensity(const VectorType &x) {
            return -this->computeNegativeLogLikelihood(x);
        }

        std::optional<VectorType> logGradient(const VectorType &x) {
            return this->compute_log_likelihood_gradient(x);
        }

        std::optional<MatrixType> logCurvature(const VectorType &x) {
            return this->computeExpectedFisherInformation(x);
        }

        double computeNegativeLogLikelihood(const VectorType &x) override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    double,     /* Return type */
                    ModelBase,       /* Parent class */
                    "compute_negative_log_likelihood",
                    computeNegativeLogLikelihood,  /* Name of function in C++ (must match Python name) */
                    x
            );
        }

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &x) override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    std::optional<MatrixType>,     /* Return type */
                    ModelBase,       /* Parent class */
                    "compute_expected_fisher_information",
                    computeExpectedFisherInformation,  /* Name of function in C++ (must match Python name) */
                    x
            );
        }

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    std::optional<VectorType>,     /* Return type */
                    ModelBase,       /* Parent class */
                    "compute_log_likelihood_gradient",
                    computeLogLikelihoodGradient,  /* Name of function in C++ (must match Python name) */
                    x
            );
        }

        std::vector<std::string> getDimensionNames() const override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    std::vector<std::string>,   // Return type
                    ModelBase,                  // Parent class
                    "dimension_names",          // Python function name
                    copyModel                   // C++ function name
            );
        }

        std::unique_ptr<Model> copyModel() const override {
            PYBIND11_OVERRIDE_PURE_NAME(
                    std::unique_ptr<Model>,     // Return type
                    ModelBase,                  // Parent class
                    "__copy__",                 // Python function name
                    copyModel                    // C++ function name
            );
        }
    };

    class PyModel : public Model {
    public:
        PyModel(py::object pyObj) : pyObj(std::move(pyObj)) {};

        double computeNegativeLogLikelihood(const VectorType &x) override {
            if(hasattr(pyObj, "log_density")) {
                return -pyObj.attr("log_density")(x).cast<double>();
            }
            else if(hasattr(pyObj, "compute_negative_log_likelihood")) {
                return pyObj.attr("compute_negative_log_likelihood")(x).cast<double>();
            }

            throw std::runtime_error("Please implement the log_density function for your model");
        }

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override {
            if (hasattr(pyObj, "log_gradient")) {
                return pyObj.attr("log_gradient")(x).cast<std::optional<VectorType>>();
            }
            else if(hasattr(pyObj, "compute_negative_log_likelihood")) {
                return pyObj.attr("compute_log_likelihood_gradient")(x).cast<std::optional<VectorType>>();
            }

            return std::nullopt;
        }

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &x) override {
            if(hasattr(pyObj, "log_curvature")) {
               return pyObj.attr("log_curvature")(x).cast<std::optional<MatrixType>>();
            }
            else if(hasattr(pyObj, "compute_expected_fisher_information")) {
               return pyObj.attr("compute_expected_fisher_information")(x).cast<std::optional<MatrixType>>();
            }

            return std::nullopt;
        }


        std::vector<std::string> getDimensionNames() const override {
            try {
                return pyObj.attr("dimension_names").cast<std::vector<std::string>>();
            }
            catch (...) {
                return {};
            }
        }

        std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<PyModel>(PyModel(pyObj));
        }

        std::string __repr__() const {
            std::string repr{""};
            repr += "PyModel(";
            repr += "model=" + get__repr__(pyObj);
            repr += ")";
            return repr;
        }

        py::object pyObj;
    };

    class ModelWrapper : public Model {
    public:
        ModelWrapper(const std::shared_ptr<Model> model) {
            this->model = model->copyModel();
        }

        ModelWrapper(const ModelWrapper &other) {
            this->model = other.model->copyModel();
        }

        double computeNegativeLogLikelihood(const VectorType &x) override {
            return model->computeNegativeLogLikelihood(x);
        }

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override {
            return model->computeLogLikelihoodGradient(x);
        }

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &x) override {
            return model->computeExpectedFisherInformation(x);
        }

        std::vector<std::string> getDimensionNames() const override {
            return model->getDimensionNames();
        }

        std::unique_ptr<Model> copyModel() const override {
            return model->copyModel();
        }

        std::shared_ptr<Model> getModelPtr() {
            return model;
        }

        void setModelPtr(const std::shared_ptr<Model> &newModel) {
            ModelWrapper::model = newModel;
        }

    private:
        std::shared_ptr<Model> model;
    };

    class Gaussian : public hops::DegenerateGaussian {
    public:
        Gaussian(const VectorType &mean,
                 const MatrixType &covariance,
                 const std::vector<long> &inactive = std::vector<long>()) :
                hops::DegenerateGaussian(mean, covariance, inactive),
                fullMean(mean),
                fullCovariance(covariance) {
        }

        double computeNegativeLogLikelihood(const VectorType &x) override {
            return hops::DegenerateGaussian::computeNegativeLogLikelihood(x);
        }

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override {
            return hops::DegenerateGaussian::computeLogLikelihoodGradient(x);
        }

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &x) override {
            return hops::DegenerateGaussian::computeExpectedFisherInformation(x);
        }

        std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<Gaussian>(fullMean,
                                              fullCovariance, hops::DegenerateGaussian::getInactive());
        }

        const VectorType &getMean() const { return fullMean; }

        const MatrixType &getCovariance() const { return fullCovariance; }

        const std::vector<long> &getInactive() const { return hops::DegenerateGaussian::getInactive(); }

    private:
        VectorType fullMean;
        MatrixType fullCovariance;
    };

    using Mixture = hops::Mixture;
    using Rosenbrock = hops::Rosenbrock;
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Gaussian);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Mixture);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::PyModel);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Rosenbrock);

namespace hopsy {
    void addModels(py::module &m) {
        py::classh<Model, ModelTrampoline<>>(m, "Model",
                                             doc::Model::base)
                .def(py::init<>(), doc::Model::__init__)
                .def("log_density", [](Model &self, const VectorType& x)  {
                         return -self.computeNegativeLogLikelihood(x);
                     },
                     doc::Model::logDensity,
                     py::arg("x"))
                .def("log_gradient", &Model::computeLogLikelihoodGradient,
                     doc::Model::logGradient,
                     py::arg("x"))
                .def("log_curvature", &Model::computeExpectedFisherInformation,
                     doc::Model::logCurvature,
                     py::arg("x"))
                .def("compute_negative_log_likelihood", &Model::computeNegativeLogLikelihood,
                     doc::Model::computeNegativeLogLikelihood,
                     py::arg("x"))
                .def("compute_log_likelihood_gradient", &Model::computeLogLikelihoodGradient,
                     doc::Model::computeLogLikelihoodGradient,
                     py::arg("x"))
                .def("compute_expected_fisher_information", &Model::computeExpectedFisherInformation,
                     doc::Model::computeExpectedFisherInformation,
                     py::arg("x"))
                .def("__repr__", [](const Model &) -> std::string {
                    std::string repr = "hopsy.Model()";
                    return repr;
                });

        py::classh<Gaussian, Model, ModelTrampoline<Gaussian>>(m, "Gaussian",
                                                               doc::Gaussian::base)
                .def(py::init([](long dim) {
                         return Gaussian(VectorType::Zero(dim),
                                         MatrixType::Identity(dim, dim));
                     }),
                     doc::Gaussian::__init__,
                     py::arg("dim"))
                .def(py::init([](const VectorType &mean,
                                 const MatrixType &covariance,
                                 const std::vector<long> &inactives) {
                         if (!mean.size() && !covariance.size()) {
                             return Gaussian(VectorType::Zero(2),
                                             MatrixType::Identity(2, 2),
                                             inactives);
                         } else if (!mean.size() && covariance.size()) {
                             if (covariance.rows() != covariance.cols()) {
                                 throw std::invalid_argument("Covariance must be square matrix, but has shape (" +
                                                             std::to_string(covariance.rows()) + ", " +
                                                             std::to_string(covariance.cols()) + ").");
                             }
                             return Gaussian(VectorType::Zero(covariance.rows()),
                                             covariance,
                                             inactives);
                         } else if (mean.size() && !covariance.size()) {
                             return Gaussian(mean,
                                             MatrixType::Identity(mean.size(), mean.size()),
                                             inactives);
                         } else if (mean.size() && covariance.size()) {
                             return Gaussian(mean,
                                             covariance,
                                             inactives);
                         } else {
                             throw std::runtime_error("Invalid arguments when constructing hopsy.Gaussian.");
                         }
                     }),
                     doc::Gaussian::__init__,
                     py::arg("mean") = VectorType(),
                     py::arg("covariance") = MatrixType(),
                     py::arg("inactives") = std::vector<long>())
                .def_property("mean", &Gaussian::getMean, [](Gaussian &self,
                                                             const VectorType &mean) {
                    self = Gaussian(mean, self.getCovariance(), self.getInactive());
                }, doc::Gaussian::mean)
                .def_property("covariance", &Gaussian::getCovariance, [](Gaussian &self,
                                                                         const MatrixType &covariance) {
                    self = Gaussian(self.getMean(), covariance, self.getInactive());
                }, doc::Gaussian::covariance)
                .def_property("inactives", &Gaussian::getInactive, [](Gaussian &self,
                                                                      const std::vector<long> &inactives) {
                    self = Gaussian(self.getMean(), self.getCovariance(), inactives);
                }, doc::Gaussian::inactives)
                .def("log_density", [](Gaussian &self, const VectorType& x)  {
                         return -self.computeNegativeLogLikelihood(x);
                     },
                     doc::Gaussian::logDensity,
                     py::arg("x"))
                .def("log_gradient", &Model::computeLogLikelihoodGradient,
                     doc::Gaussian::logGradient,
                     py::arg("x"))
                .def("log_curvature", &Model::computeExpectedFisherInformation,
                     doc::Gaussian::logCurvature,
                     py::arg("x"))
                .def("compute_negative_log_likelihood", [](Gaussian &self, const VectorType &x) {
                         if (self.getMean().rows() != self.getCovariance().rows() ||
                             self.getMean().rows() != self.getCovariance().cols()) {
                             throw std::runtime_error("Dimension mismatch between mean with shape (" +
                                                      std::to_string(self.getMean().rows()) +
                                                      ",) and covariance with shape (" +
                                                      std::to_string(self.getCovariance().rows()) +
                                                      ", " + std::to_string(self.getCovariance().cols()) + ")");
                         }
                         return self.computeNegativeLogLikelihood(x);
                     },
                     doc::Gaussian::computeNegativeLogLikelihood,
                     py::arg("x"))
                .def("compute_log_likelihood_gradient", [](Gaussian &self, const VectorType &x) {
                         if (self.getMean().rows() != self.getCovariance().rows() ||
                             self.getMean().rows() != self.getCovariance().cols()) {
                             throw std::runtime_error("Dimension mismatch between mean with shape (" +
                                                      std::to_string(self.getMean().rows()) +
                                                      ",) and covariance with shape (" +
                                                      std::to_string(self.getCovariance().rows()) +
                                                      ", " + std::to_string(self.getCovariance().cols()) + ")");
                         }
                         return self.computeLogLikelihoodGradient(x);
                     },
                     doc::Gaussian::computeLogLikelihoodGradient,
                     py::arg("x"))
                .def("compute_expected_fisher_information", [](Gaussian &self, const VectorType &x) {
                         if (self.getMean().rows() != self.getCovariance().rows() ||
                             self.getMean().rows() != self.getCovariance().cols()) {
                             throw std::runtime_error("Dimension mismatch between mean with shape (" +
                                                      std::to_string(self.getMean().rows()) +
                                                      ",) and covariance with shape (" +
                                                      std::to_string(self.getCovariance().rows()) +
                                                      ", " + std::to_string(self.getCovariance().cols()) + ")");
                         }
                         return self.computeExpectedFisherInformation(x);
                     },
                     doc::Gaussian::computeExpectedFisherInformation,
                     py::arg("x"))
                .def("__repr__", [](const Gaussian &self) -> std::string {
                    std::string repr = "hopsy.Gaussian(";
                    repr += "mean=" + get__repr__(py::cast(self.getMean())) + ", ";
                    repr += "covariance=" + get__repr__(py::cast(self.getCovariance()));
                    if (self.getInactive().size()) {
                        repr += ", inactives=" + get__repr__(py::cast(self.getInactive()));
                    }
                    repr += ")";
                    return repr;
                })
                .def(py::pickle([](const Gaussian &self) { // __getstate__
                                    return py::make_tuple(self.getMean(),
                                                          self.getCovariance(),
                                                          self.getInactive());
                                },
                                [](py::tuple t) { // __setstate__
                                    if (t.size() != 3) throw std::runtime_error("Invalid state!");

                                    Gaussian p(t[0].cast<VectorType>(),
                                               t[1].cast<MatrixType>(),
                                               t[2].cast<std::vector<long>>());

                                    return p;
                                }
                ));

        py::classh<Mixture, Model, ModelTrampoline<Mixture>>(m, "Mixture",
                                                             doc::Mixture::base)
                .def(py::init([](const std::vector<Model *> &components) {
                         std::vector<std::shared_ptr<Model>> _components;
                         for (auto &component : components) {
                             _components.push_back(std::shared_ptr<Model>(component->copyModel()));
                         }
                         return Mixture(_components);
                     }),
                     doc::Mixture::__init__,
                     py::arg("components") = std::vector<Model *>())
                .def(py::init([](const std::vector<Model *> &components, const std::vector<double> &weights) {
                         std::vector<std::shared_ptr<Model>> _components;
                         for (auto &component : components) {
                             _components.push_back(std::shared_ptr<Model>(component->copyModel()));
                         }
                         return Mixture(_components, weights);
                     }),
                     py::arg("components"),
                     py::arg("weights"))
                .def_property("components", &Mixture::getComponents, [](Mixture &self,
                                                                        const std::vector<Model *> components) {
                    std::vector<std::shared_ptr<Model>> _components;
                    for (auto &component : components) {
                        _components.push_back(std::shared_ptr<Model>(component->copyModel()));
                    }
                    self = Mixture(_components, self.getWeights());
                }, doc::Mixture::weights)
                .def_property("weights", &Mixture::getWeights, [](Mixture &self,
                                                                  std::vector<double> weights) {
                    self = Mixture(self.getComponents(), weights);
                }, doc::Mixture::components)
                .def("log_density", [](Model &self, const VectorType& x)  {
                         return -self.computeNegativeLogLikelihood(x);
                     },
                     doc::Mixture::logDensity,
                     py::arg("x"))
                .def("log_gradient", &Model::computeLogLikelihoodGradient,
                     doc::Mixture::logGradient,
                     py::arg("x"))
                .def("log_curvature", &Model::computeExpectedFisherInformation,
                     doc::Mixture::logCurvature,
                     py::arg("x"))
                .def("compute_negative_log_likelihood", &Mixture::computeNegativeLogLikelihood,
                     doc::Mixture::computeNegativeLogLikelihood,
                     py::arg("x"))
                .def("compute_log_likelihood_gradient", &Mixture::computeLogLikelihoodGradient,
                     doc::Mixture::computeLogLikelihoodGradient,
                     py::arg("x"))
                .def("compute_expected_fisher_information", &Mixture::computeExpectedFisherInformation,
                     doc::Mixture::computeExpectedFisherInformation,
                     py::arg("x"))
                .def("__repr__", [](const Mixture &self) -> std::string {
                    std::string repr = "hopsy.Mixture(";

                    if (self.getComponents().size()) {
                        repr += "components=[";
                        for (auto &component : self.getComponents()) {
                            repr += get__repr__<Model>(component);
                            repr += (&component != &self.getComponents().back() ? ", " : "");
                        }
                        repr += "]";
                    }

                    bool allWeightsOne = true;
                    for (auto &weight : self.getWeights()) if (weight != 1) allWeightsOne = false;

                    if (self.getWeights().size() && !allWeightsOne) {
                        repr += ", weights=[";
                        for (auto &weight : self.getWeights()) {
                            repr += removeTrailingZeros(weight);
                            repr += (&weight != &self.getWeights().back() ? ", " : "");
                        }
                        repr += "]";
                    }

                    repr += ")";
                    return repr;
                })
                .def(py::pickle([](const Mixture &self) { // __getstate__
                                    /* Return a tuple that fully encodes the state of the object */
                                    std::vector<Model *> models;
                                    for (auto &model : self.getComponents()) models.push_back(model.get());
                                    return py::make_tuple(models,
                                                          self.getWeights());
                                },
                                [](py::tuple t) { // __setstate__
                                    if (t.size() != 2) throw std::runtime_error("Invalid state!");

                                    /* Create a new C++ instance */
                                    Mixture p(t[0].cast<std::vector<std::shared_ptr<Model>>>(),
                                              t[1].cast<std::vector<double>>());

                                    return p;
                                }
                ));

        py::classh<PyModel, Model, ModelTrampoline<PyModel>>(m, "PyModel",
                                                             doc::PyModel::base)
                .def(py::init<py::object>(),
                     doc::PyModel::__init__,
                     py::arg("model"))
                .def_readwrite("model", &PyModel::pyObj, doc::PyModel::model)
                .def("log_density", [](Model &self, const VectorType& x)  {
                         return -self.computeNegativeLogLikelihood(x);
                     },
                     doc::PyModel::logDensity,
                     py::arg("x"))
                .def("log_gradient", &Model::computeLogLikelihoodGradient,
                     doc::PyModel::logGradient,
                     py::arg("x"))
                .def("log_curvature", &Model::computeExpectedFisherInformation,
                     doc::PyModel::logCurvature,
                     py::arg("x"))
                .def("compute_negative_log_likelihood", &PyModel::computeNegativeLogLikelihood,
                     doc::PyModel::computeNegativeLogLikelihood,
                     py::arg("x"))
                .def("compute_log_likelihood_gradient", &PyModel::computeLogLikelihoodGradient,
                     doc::PyModel::computeLogLikelihoodGradient,
                     py::arg("x"))
                .def("compute_expected_fisher_information", &PyModel::computeExpectedFisherInformation,
                     doc::PyModel::computeExpectedFisherInformation,
                     py::arg("x"))
                .def("__repr__", &PyModel::__repr__)
                .def(py::pickle([](const PyModel &self) { // __getstate__
                                    /* Return a tuple that fully encodes the state of the object */
                                    return py::make_tuple(self.pyObj);
                                },
                                [](py::tuple t) { // __setstate__
                                    if (t.size() != 1) throw std::runtime_error("Invalid state!");

                                    /* Create a new C++ instance */
                                    PyModel p(t[0].cast<py::object>());

                                    return p;
                                }
                ));

        py::classh<Rosenbrock, Model, ModelTrampoline<Rosenbrock>>(m, "Rosenbrock",
                                                                   doc::Rosenbrock::base)
                .def(py::init([](long dim) {
                         return Rosenbrock(1, VectorType::Zero(static_cast<long>(dim / 2)));
                     }),
                     doc::Rosenbrock::__init__,
                     py::arg("dim"))
                .def(py::init<double, VectorType>(),
                     doc::Rosenbrock::__init__,
                     py::arg("scale") = 1,
                     py::arg("shift") = VectorType::Zero(1))
                .def_property("scale", &Rosenbrock::getScaleParameter, [](Rosenbrock &self,
                                                                          double scale) {
                    self = Rosenbrock(scale, self.getShiftParameter());
                }, doc::Rosenbrock::scale)
                .def_property("shift", &Rosenbrock::getShiftParameter, [](Rosenbrock &self,
                                                                          VectorType shift) {
                    self = Rosenbrock(self.getScaleParameter(), shift);
                }, doc::Rosenbrock::shift)
                .def("log_density", [](Model &self, const VectorType& x)  {
                         return -self.computeNegativeLogLikelihood(x);
                     },
                     doc::Rosenbrock::logDensity,
                     py::arg("x"))
                .def("log_gradient", &Model::computeLogLikelihoodGradient,
                     doc::Rosenbrock::logGradient,
                     py::arg("x"))
                .def("log_curvature", &Model::computeExpectedFisherInformation,
                     doc::Rosenbrock::logCurvature,
                     py::arg("x"))
                .def("compute_negative_log_likelihood", &Rosenbrock::computeNegativeLogLikelihood,
                     doc::Rosenbrock::computeNegativeLogLikelihood,
                     py::arg("x"))
                .def("compute_log_likelihood_gradient", &Rosenbrock::computeLogLikelihoodGradient,
                     doc::Rosenbrock::computeLogLikelihoodGradient,
                     py::arg("x"))
                .def("compute_expected_fisher_information", &Rosenbrock::computeExpectedFisherInformation,
                     doc::Rosenbrock::computeExpectedFisherInformation,
                     py::arg("x"))
                .def("__repr__", [](const Rosenbrock &self) -> std::string {
                    std::string repr = "hopsy.Rosenbrock(";
                    repr += "scale=" + removeTrailingZeros(self.getScaleParameter()) + ", ";
                    repr += "shift=" + get__repr__(py::cast(self.getShiftParameter()));
                    repr += ")";
                    return repr;
                })
                .def(py::pickle([](const Rosenbrock &self) { // __getstate__
                                    /* Return a tuple that fully encodes the state of the object */
                                    return py::make_tuple(self.getScaleParameter(), self.getShiftParameter());
                                },
                                [](py::tuple t) { // __setstate__
                                    if (t.size() != 2) throw std::runtime_error("Invalid state!");

                                    /* Create a new C++ instance */
                                    Rosenbrock p(t[0].cast<double>(),
                                                 t[1].cast<VectorType>());

                                    return p;
                                }
                ));
    }
}

#endif // HOPSY_MODEL_HPP
