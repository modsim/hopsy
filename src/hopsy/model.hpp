#ifndef HOPSY_MODEL_HPP
#define HOPSY_MODEL_HPP

#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

#include <Eigen/Core>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "../../extern/hops/src/hops/hops.hpp"

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

		double computeNegativeLogLikelihood(const VectorType& x) const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				double,     /* Return type */
				ModelBase,       /* Parent class */
                "compute_negative_log_likelihood",
				computeNegativeLogLikelihood,  /* Name of function in C++ (must match Python name) */
                x
			);
        }

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType& x) const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::optional<MatrixType>,     /* Return type */
				ModelBase,       /* Parent class */
                "compute_expected_fisher_information",
				computeExpectedFisherInformation,  /* Name of function in C++ (must match Python name) */
                x
			);
		}

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType& x) const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::optional<VectorType>,     /* Return type */
				ModelBase,       /* Parent class */
                "compute_log_likelihood_gradient",
				computeLogLikelihoodGradient,  /* Name of function in C++ (must match Python name) */
                x
			);
		}

        std::unique_ptr<Model> copyModel() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::unique_ptr<Model>,     // Return type 
				ModelBase,                  // Parent class
                "deepcopy",                 // Python function name
				copyModel                    // C++ function name
            );
		}
	};

    class PyModel : public Model {
	public:
		PyModel(py::object pyObj) : pyObj(std::move(pyObj)) {};

		double computeNegativeLogLikelihood(const VectorType& x) const override {
			return pyObj.attr("compute_negative_log_likelihood")(x).cast<double>();
		}

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType& x) const override {
			return pyObj.attr("compute_expected_fisher_information")(x).cast<std::optional<MatrixType>>();
		}

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType& x) const override {
			return pyObj.attr("compute_log_likelihood_gradient")(x).cast<std::optional<VectorType>>();
		}

        std::unique_ptr<Model> copyModel() const override {
			return std::make_unique<PyModel>(PyModel(pyObj));
		}

        std::string __repr__() const {
            std::string repr{""};
            repr += "PyModel(";
            repr += "udm=" + get__repr__(pyObj);
            repr += ")";
            return repr;
        }

		py::object pyObj;
	};

    class ModelWrapper : public Model {
    public:
        ModelWrapper(const std::shared_ptr<Model> model) {
            this->model = std::move(model->copyModel());
        }

        ModelWrapper(const ModelWrapper& other) {
            this->model = std::move(other.model->copyModel());
        }

        double computeNegativeLogLikelihood(const VectorType &x) const override {
            return model->computeNegativeLogLikelihood(x);
        }

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType& x) const override {
            return model->computeLogLikelihoodGradient(x);
        }

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType& x) const override {
            return model->computeExpectedFisherInformation(x);
        }

        std::optional<std::vector<std::string>> getDimensionNames() const override {
            return model->getDimensionNames();
        }

        std::unique_ptr<Model> copyModel() const override {
            return model->copyModel();
        }

        std::shared_ptr<Model> getModelPtr() {
            return model;
        }

    private:
        std::shared_ptr<Model> model;
    };

    using DegenerateGaussian = hops::DegenerateGaussian;
    using Mixture = hops::Mixture;
    using Rosenbrock =  hops::Rosenbrock;
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::DegenerateGaussian);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Mixture);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::PyModel);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Rosenbrock);

namespace hopsy {
    void addModels(py::module& m) {
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
            .def("__repr__", &hopsy::PyModel::__repr__)
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
    }
}

#endif // HOPSY_MODEL_HPP

