#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include "../extern/hops/include/hops/hops.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace hopsy {
    typedef hops::DegenerateMultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd> DegenerateMultivariateGaussianModel;
    typedef hops::DynMultimodalModel<DegenerateMultivariateGaussianModel> MultimodalMultivariateGaussianModel;
    typedef hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd> MultivariateGaussianModel;
    typedef hops::RosenbrockModel<Eigen::MatrixXd, Eigen::VectorXd> RosenbrockModel;
    typedef hops::UniformDummyModel<Eigen::MatrixXd, Eigen::VectorXd> UniformModel;

    typedef hops::Problem<DegenerateMultivariateGaussianModel> DegenerateMultivariateGaussianProblem;
    typedef hops::Problem<MultimodalMultivariateGaussianModel> MultimodalMultivariateGaussianProblem;
    typedef hops::Problem<MultivariateGaussianModel> MultivariateGaussianProblem;
    typedef hops::Problem<RosenbrockModel> RosenbrockProblem;
    typedef hops::Problem<UniformModel> UniformProblem;

	typedef hops::Run<DegenerateMultivariateGaussianModel> DegenerateMultivariateGaussianRun;
    typedef hops::Run<MultimodalMultivariateGaussianModel> MultimodalMultivariateGaussianRun;
    typedef hops::Run<MultivariateGaussianModel> MultivariateGaussianRun;
    typedef hops::Run<RosenbrockModel> RosenbrockRun;
    typedef hops::Run<UniformModel> UniformRun;

	hops::Problem<UniformModel> Problem_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
		return hops::Problem<UniformModel>(A, b);
	}

	template<typename T>
	hops::Problem<T> Problem(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const T& t) {
		if constexpr(std::is_same<T, DegenerateMultivariateGaussianModel>::value) {
			return hops::Problem<DegenerateMultivariateGaussianModel>(A, b, t);
		}
		if constexpr(std::is_same<T, MultimodalMultivariateGaussianModel>::value) {
			return hops::Problem<MultimodalMultivariateGaussianModel>(A, b, t);
		}
		if constexpr(std::is_same<T, MultivariateGaussianModel>::value) {
			return hops::Problem<MultivariateGaussianModel>(A, b, t);
		}
		if constexpr(std::is_same<T, RosenbrockModel>::value) {
			return hops::Problem<RosenbrockModel>(A, b, t);
		}
		if constexpr(std::is_same<T, UniformModel>::value) {
			return hops::Problem<UniformModel>(A, b, t);
		}
	}

	template<typename T>
	hops::Run<T> Run(const hops::Problem<T>& t, 
					 std::string chainTypeString = "HitAndRun", 
					 unsigned long numberOfSamples = 1000, 
					 unsigned long numberOfChains = 1) {
		hops::MarkovChainType chainType;
		if (chainTypeString == "BallWalk") {
			chainType = hops::MarkovChainType::BallWalk;
		} else if (chainTypeString == "CoordinateHitAndRun") {
			chainType = hops::MarkovChainType::CoordinateHitAndRun;
		} else if (chainTypeString == "CSmMALA") {
			chainType = hops::MarkovChainType::CSmMALA;
		} else if (chainTypeString == "CSmMALANoGradient") {
			chainType = hops::MarkovChainType::CSmMALANoGradient;
		} else if (chainTypeString == "DikinWalk") {
			chainType = hops::MarkovChainType::DikinWalk;
		} else if (chainTypeString == "Gaussian") {
			chainType = hops::MarkovChainType::Gaussian;
		} else if (chainTypeString == "HitAndRun") {
			chainType = hops::MarkovChainType::HitAndRun;
		}

		if constexpr(std::is_same<T, DegenerateMultivariateGaussianModel>::value) {
			return hops::Run<DegenerateMultivariateGaussianModel>(t, chainType, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, MultimodalMultivariateGaussianModel>::value) {
			return hops::Run<MultimodalMultivariateGaussianModel>(t, chainType, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, MultivariateGaussianModel>::value) {
			return hops::Run<MultivariateGaussianModel>(t, chainType, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, RosenbrockModel>::value) {
			return hops::Run<RosenbrockModel>(t, chainType, numberOfSamples, numberOfChains);
		}
		if constexpr(std::is_same<T, UniformModel>::value) {
			return hops::Run<UniformModel>(t, chainType, numberOfSamples, numberOfChains);
		}
	}
}

namespace py = pybind11;

PYBIND11_MODULE(hopsy, m) {
    py::class_<hopsy::DegenerateMultivariateGaussianModel>(m, "DegenerateMultivariateGaussianModel")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>())
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd, std::vector<long>>());

    py::class_<hopsy::MultimodalMultivariateGaussianModel>(m, "MultimodalMultivariateGaussianModel")
        .def(py::init<std::vector<hopsy::DegenerateMultivariateGaussianModel>>());

    py::class_<hopsy::MultivariateGaussianModel>(m, "MultivariateGaussianModel")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>());

    py::class_<hopsy::RosenbrockModel>(m, "RosenbrockModel")
        .def(py::init<double, Eigen::VectorXd>());

    py::class_<hopsy::UniformModel>(m, "UniformModel")
        .def(py::init<>());


    py::class_<hopsy::DegenerateMultivariateGaussianProblem>(m, "DegenerateMultivariateGaussianProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::DegenerateMultivariateGaussianModel>());

    py::class_<hopsy::MultimodalMultivariateGaussianProblem>(m, "MultimodalMultivariateGaussianProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::MultimodalMultivariateGaussianModel>());

    py::class_<hopsy::MultivariateGaussianProblem>(m, "MultivariateGaussianProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::MultivariateGaussianModel>());

    py::class_<hopsy::RosenbrockProblem>(m, "RosenbrockProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::RosenbrockModel>());

    py::class_<hopsy::UniformProblem>(m, "UniformProblem")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd>())
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, hopsy::UniformModel>());


	m.def("Problem", &hopsy::Problem<hopsy::DegenerateMultivariateGaussianModel>);
	m.def("Problem", &hopsy::Problem<hopsy::MultimodalMultivariateGaussianModel>);
	m.def("Problem", &hopsy::Problem<hopsy::MultivariateGaussianModel>);
	m.def("Problem", &hopsy::Problem<hopsy::RosenbrockModel>);
	m.def("Problem", &hopsy::Problem<hopsy::UniformModel>);
	m.def("Problem", &hopsy::Problem_);


//    py::class_<hopsy::DegenerateMultivariateGaussianRun>(m, "DegenerateMultivariateGaussianRun")
//        .def(py::init<hopsy::DegenerateMultivariateGaussianRun>());
//
    py::class_<hopsy::MultimodalMultivariateGaussianRun>(m, "MultimodalMultivariateGaussianRun")
        .def(py::init<hopsy::MultimodalMultivariateGaussianProblem>())
        .def(py::init<hopsy::MultimodalMultivariateGaussianRun>())
        .def("get_data", &hopsy::MultimodalMultivariateGaussianRun::getData)
        .def("init", &hopsy::MultimodalMultivariateGaussianRun::init)
        .def("sample", &hopsy::MultimodalMultivariateGaussianRun::sample)
        .def("set_starting_points", &hopsy::MultimodalMultivariateGaussianRun::setStartingPoints);

    py::class_<hopsy::MultivariateGaussianRun>(m, "MultivariateGaussianRun")
        .def(py::init<hopsy::MultivariateGaussianProblem>())
        .def(py::init<hopsy::MultivariateGaussianRun>())
        .def("get_data", &hopsy::MultivariateGaussianRun::getData)
        .def("init", &hopsy::MultivariateGaussianRun::init)
        .def("sample", &hopsy::MultivariateGaussianRun::sample)
        .def("set_starting_points", &hopsy::MultivariateGaussianRun::setStartingPoints);

//    py::class_<hopsy::RosenbrockRun>(m, "RosenbrockRun")
//        .def(py::init<hopsy::RosenbrockRun>());
//
    py::class_<hopsy::UniformRun>(m, "UniformRun")
        .def(py::init<hopsy::UniformProblem>())
        .def(py::init<hopsy::UniformRun>())
        .def("get_data", &hopsy::UniformRun::getData)
        .def("init", &hopsy::UniformRun::init)
        .def("sample", &hopsy::UniformRun::sample)
        .def("set_starting_points", &hopsy::UniformRun::setStartingPoints);


	m.def("Run", &hopsy::Run<hopsy::DegenerateMultivariateGaussianModel>);
	m.def("Run", &hopsy::Run<hopsy::MultimodalMultivariateGaussianModel>);
	m.def("Run", &hopsy::Run<hopsy::MultivariateGaussianModel>);
	m.def("Run", &hopsy::Run<hopsy::RosenbrockModel>);
	m.def("Run", &hopsy::Run<hopsy::UniformModel>);
	//m.def("Run", &hopsy::Run_);


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
