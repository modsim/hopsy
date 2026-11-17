#ifndef HOPSY_BINDING_TARGETS_H
#define HOPSY_BINDING_TARGETS_H
#include <limits>
#include <hopsy/target_abi_v1.h>
#include <hopsy/target/concepts.h>

namespace hopsy::bindings {
    // Public marker: optional methods are intentionally absent.
    struct PythonTargetBase {};
    struct CapsuleTarget {
        nb::object owner;
        const hopsy_target_v1 *descriptor;
        explicit CapsuleTarget(nb::object capsule) : owner(std::move(capsule)) {
            if (!PyCapsule_IsValid(owner.ptr(), HOPSY_TARGET_CAPSULE_NAME))
                throw nb::value_error("expected capsule named hopsy.target.v1");
            descriptor = static_cast<const hopsy_target_v1 *>(
                PyCapsule_GetPointer(owner.ptr(), HOPSY_TARGET_CAPSULE_NAME));
            if (descriptor->abi_version != 1 || descriptor->structure_size < sizeof(hopsy_target_v1))
                throw nb::value_error("invalid target ABI version or structure size");
            if (!descriptor->dimension || descriptor->dimension > size_t(std::numeric_limits<Index>::max()) ||
                !descriptor->log_density || (descriptor->log_hessian && !descriptor->log_gradient))
                throw nb::value_error("invalid target dimension or callbacks");
        }
    };
    struct TargetMethods {
        nb::object owner, density, gradient, hessian;
        const hopsy_target_v1 *native = nullptr;
        int order = 0;
        explicit TargetMethods(nb::handle target) : owner(nb::borrow<nb::object>(target)) {
            if (nb::isinstance<CapsuleTarget>(target)) {
                native = nb::cast<const CapsuleTarget &>(target).descriptor;
                order = native->log_hessian ? 2 : native->log_gradient ? 1 : 0;
                return;
            }
            auto method = [&](const char *name, bool required) {
                nb::object f = nb::getattr(target, name, nb::none());
                if (f.is_none() && !required) return nb::object();
                if (!PyCallable_Check(f.ptr()))
                    throw nb::type_error((std::string("target.") + name + " must be callable").c_str());
                return f;
            };
            density = method("log_density", true);
            gradient = method("log_gradient", false);
            hessian = method("log_hessian", false);
            if (hessian && !gradient) throw nb::type_error("log_hessian requires log_gradient");
            order = hessian ? 2 : gradient ? 1 : 0;
        }
        void check(const State &x) const {
            if (!x.size() || !x.allFinite() || (native && size_t(x.size()) != native->dimension))
                throw std::invalid_argument("target state dimension mismatch or nonfinite state");
        }
        void call(hopsy_target_callback_v1 f, const State &x, double *out) const {
            check(x);
            int status = f(native->context, x.data(), out);
            if (status) throw std::runtime_error("native target callback failed with status " + std::to_string(status));
        }
    };
    template<int Order> struct RuntimeTarget {
        std::shared_ptr<TargetMethods> methods;
        double log_density(const State &x) const {
            if (methods->native) {
                double result;
                methods->call(methods->native->log_density, x, &result);
                return result;
            }
            nb::gil_scoped_acquire gil;
            methods->check(x);
            nb::object value = methods->density(nb::cast(x, nb::rv_policy::copy));
            try { return nb::cast<double>(value); }
            catch (const nb::cast_error &) { throw nb::type_error("target.log_density must return a scalar"); }
        }
        void log_gradient(const State &x, State &result) const requires(Order >= 1) {
            if (result.size() != x.size()) throw std::invalid_argument("gradient output dimension mismatch");
            if (methods->native) {
                methods->call(methods->native->log_gradient, x, result.data());
                return;
            }
            nb::gil_scoped_acquire gil;
            methods->check(x);
            nb::object output = methods->gradient(nb::cast(x, nb::rv_policy::copy));
            State value;
            try { value = nb::cast<State>(output); }
            catch (const nb::cast_error &) { throw nb::type_error("target.log_gradient must return a vector"); }
            if (value.size() != x.size()) throw std::invalid_argument("target gradient dimension mismatch");
            result = value;
        }
        State log_gradient(const State &x) const requires(Order >= 1) {
            State result(x.size());
            log_gradient(x, result);
            return result;
        }
        DenseMatrix log_hessian(const State &x) const requires(Order >= 2) {
            if (methods->native) {
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(x.size(), x.size());
                methods->call(methods->native->log_hessian, x, result.data());
                return result;
            }
            nb::gil_scoped_acquire gil;
            methods->check(x);
            nb::object output = methods->hessian(nb::cast(x, nb::rv_policy::copy));
            DenseMatrix result;
            try { result = nb::cast<DenseMatrix>(output); }
            catch (const nb::cast_error &) { throw nb::type_error("target.log_hessian must return a matrix"); }
            if (result.rows() != x.size() || result.cols() != x.size())
                throw std::invalid_argument("target Hessian dimension mismatch");
            return result;
        }
    };
    template<int Order> auto capsule_evaluate(const CapsuleTarget &target, const State &x) {
        auto methods = std::make_shared<TargetMethods>(nb::cast(&target, nb::rv_policy::reference));
        if (Order > methods->order) throw nb::value_error("target does not provide requested derivative order");
        nb::gil_scoped_release release;
        RuntimeTarget<Order> adapter{methods};
        if constexpr (Order == 0) return adapter.log_density(x);
        if constexpr (Order == 1) return adapter.log_gradient(x);
        if constexpr (Order == 2) return adapter.log_hessian(x);
    }
    template<class T> constexpr bool runtime_target = false;
    template<int O> constexpr bool runtime_target<RuntimeTarget<O>> = true;
    static_assert(TargetDensity<RuntimeTarget<0>, State>);
    static_assert(!DifferentiableTargetDensity<RuntimeTarget<0>, State>);
    static_assert(!TwiceDifferentiableTargetDensity<RuntimeTarget<0>, State>);
    static_assert(TargetDensity<RuntimeTarget<1>, State>);
    static_assert(DifferentiableTargetDensity<RuntimeTarget<1>, State>);
    static_assert(!TwiceDifferentiableTargetDensity<RuntimeTarget<1>, State>);
    static_assert(TargetDensity<RuntimeTarget<2>, State>);
    static_assert(DifferentiableTargetDensity<RuntimeTarget<2>, State>);
    static_assert(TwiceDifferentiableTargetDensity<RuntimeTarget<2>, State>);
}
#endif
