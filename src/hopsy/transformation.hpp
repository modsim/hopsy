#ifndef HOPSY_TRANSFORMATION_HPP
#define HOPSY_TRANSFORMATION_HPP

#include <memory>
#include <utility>

#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/smart_holder.h>
#include <pybind11/stl.h>
#include <pybind11/trampoline_self_life_support.h>

#include "../../extern/hops/src/hops/hops.hpp"

#include "misc.hpp"

namespace py = pybind11;

namespace hopsy {
//    using Transformation = hops::Transformation;
    using LinearTransformation = hops::LinearTransformation;
} // namespace hopsy

//PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::Transformation);
//
//namespace hopsy {
//    template<typename TransformationBase = Transformation>
//	class TransformationTrampoline : public TransformationBase, public py::trampoline_self_life_support {
//	   	VectorType apply(const VectorType& x) const override {
//			PYBIND11_OVERRIDE_PURE(
//				VectorType,
//				TransformationBase,
//				apply,
//                x
//			);
//		}
//
//	   	VectorType revert(const VectorType& x) const override {
//			PYBIND11_OVERRIDE_PURE(
//				VectorType,
//				TransformationBase,
//				revert,
//                x
//			);
//		}
//    };
//
//    class TransformationWrapper : public Transformation {
//    public:
//        TransformationWrapper(const Transformation* transformation) {
//            transformation = dynamic_cast<Transformation*>(transformation->copyTransformation().release());
//        };
//
//        VectorType apply(const VectorType& x) const override {
//            return transformation->apply(x);
//        }
//
//        VectorType revert(const VectorType& x) const override {
//            return transformation->revert(x);
//        }
//
//        std::unique_ptr<Transformation> copyTransformation() const override {
//            return transformation->copyTransformation();
//        }
//
//        Transformation* getTransformationPtr() {
//            return transformation;
//        }
//
//    private:
//        Transformation* transformation;
//    };
//
//    using LinearTransformation = hops::LinearTransformation;
//} // namespace hopsy
//
//PYBIND11_SMART_HOLDER_TYPE_CASTERS(hopsy::LinearTransformation);

#endif // HOPSY_TRANSFORMATION_HPP
