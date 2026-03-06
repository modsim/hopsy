#include <memory>
#include <string>

#include <Eigen/Core>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "doc.hpp"
#include "markovchain.hpp"
#include "misc.hpp"
#include "model.hpp"
#include "proposal.hpp"
#include "random.hpp"
#include "tuning.hpp"

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    py::options options;
    options.disable_function_signatures();

    m.doc() = hopsy::doc::base;

#ifdef IS_DEBUG
    m.attr("__is_debug__") = true;
#else
    m.attr("__is_debug__") = false;
#endif

    hopsy::addModels(m);
    hopsy::addRandom(m);
    hopsy::addProblem(m);
    hopsy::addProposalParameters(m);
    hopsy::addProposals(m);
    hopsy::addMarkovChain(m);
    hopsy::addTuning(m);
}
