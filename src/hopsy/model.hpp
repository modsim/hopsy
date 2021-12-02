#ifndef HOPSY_HPP
#define HOPSY_HPP

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

        std::unique_ptr<Model> deepCopy() const override {
			PYBIND11_OVERRIDE_PURE_NAME(
				std::unique_ptr<Model>,     // Return type 
				ModelBase,                  // Parent class
                "deepcopy",                 // Python function name
				deepCopy                    // C++ function name
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

        std::unique_ptr<Model> deepCopy() const override {
			return std::make_unique<PyModel>(PyModel(pyObj));
		}

        std::string __repr__() const {
            std::string repr{""};
            repr += "PyModel(";
            repr += "udm=" + py::cast<std::string>(pyObj.attr("__repr__")());
            repr += ")";
            return repr;
        }

		py::object pyObj;
	};

    using DegenerateGaussian = hops::DegenerateGaussian;
    using Mixture = hops::Mixture;
    //typedef hops::MultivariateGaussianModel<MatrixType, VectorType> MultivariateGaussianModel;
    using Rosenbrock =  hops::Rosenbrock;
} // namespace hopsy

PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::DegenerateGaussian);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Mixture);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::PyModel);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Rosenbrock);

#endif // HOPSY_HPP

