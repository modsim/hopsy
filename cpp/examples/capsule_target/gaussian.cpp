// Independent producer: no hopsy C++ types, Eigen, or nanobind.
#include <Python.h>
#include <hopsy/target_abi_v1.h>
#include <atomic>
#include <chrono>
#include <thread>

namespace {
std::atomic<int> destroyed{0};
std::atomic<bool> entered{false}, proceed{false};
struct Context { bool block; int fail; };
int density(void *p, const double *x, double *out) noexcept {
    auto &c = *static_cast<Context *>(p);
    if (c.fail == 1) return 17;
    if (c.block) {
        entered = true;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!proceed && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        if (!proceed) return 18;
    }
    *out = -0.5 * (2*x[0]*x[0] + 2*x[0]*x[1] + 3*x[1]*x[1]);
    return 0;
}
int gradient(void *p, const double *x, double *out) noexcept {
    if (static_cast<Context *>(p)->fail == 2) return 19;
    out[0] = -2*x[0]-x[1]; out[1] = -x[0]-3*x[1]; return 0;
}
int hessian(void *p, const double *, double *out) noexcept {
    if (static_cast<Context *>(p)->fail == 3) return 20;
    out[0] = -2; out[1] = -1; out[2] = -1; out[3] = -3; return 0;
}
void cleanup(PyObject *capsule) {
    auto *d = static_cast<hopsy_target_v1 *>(PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule)));
    delete static_cast<Context *>(d->context);
    delete d;
    ++destroyed;
}
PyObject *make(PyObject *, PyObject *args) {
    int order=2, mode=0;
    if (!PyArg_ParseTuple(args, "|ii", &order, &mode)) return nullptr;
    auto *context = new Context{mode == 9, mode == 1 ? 1 : mode == 10 ? 2 : mode == 11 ? 3 : 0};
    auto *d = new hopsy_target_v1{1, sizeof(hopsy_target_v1), 2, context,
                                density, order >= 1 ? gradient : nullptr,
                                order >= 2 ? hessian : nullptr};
    if (mode == 6) d->dimension = size_t(-1);
    if (mode == 7) d->structure_size = 0;
    if (mode == 2) d->abi_version = 2;
    if (mode == 3) d->dimension = 0;
    if (mode == 4) d->log_gradient = nullptr;
    if (mode == 5) d->log_density = nullptr;
    PyObject *capsule = PyCapsule_New(d, mode == 8 ? "wrong.name" : HOPSY_TARGET_CAPSULE_NAME, cleanup);
    if (!capsule) { delete context; delete d; }
    return capsule;
}
PyObject *count(PyObject *, PyObject *) { return PyLong_FromLong(destroyed); }
PyObject *started(PyObject *, PyObject *) { return PyBool_FromLong(entered); }
PyObject *release(PyObject *, PyObject *) { proceed = true; Py_RETURN_NONE; }
PyObject *reset(PyObject *, PyObject *) { entered = false; proceed = false; Py_RETURN_NONE; }
PyMethodDef methods[] = {{"make", make, METH_VARARGS, "Gaussian capsule (order, test mode)."},
    {"destroyed", count, METH_NOARGS, nullptr}, {"started", started, METH_NOARGS, nullptr},
    {"release", release, METH_NOARGS, nullptr}, {"reset", reset, METH_NOARGS, nullptr}, {nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_capsule_gaussian", nullptr, -1, methods};
}
PyMODINIT_FUNC PyInit__capsule_gaussian() { return PyModule_Create(&module); }
