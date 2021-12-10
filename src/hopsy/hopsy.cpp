#include <memory>
#include <string>

#include <Eigen/Core>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "doc.hpp"
#include "markovchain.hpp"
#include "misc.hpp"
#include "model.hpp"
#include "proposal.hpp"
#include "random.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    py::options options;
    options.disable_function_signatures();

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    hopsy::addModels(m);
    hopsy::addRandom(m);
    hopsy::addProblem(m);
    hopsy::addProposalParameters(m);
    hopsy::addProposals(m);
    hopsy::addMarkovChain(m);
}
