# hopsy.gpu: CUDA extension, only built with -DHOPSY_BUILD_GPU=ON.
# Builds hopsy/gpu/_core, the compiled part of the hopsy.gpu package.

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$"))
    message(FATAL_ERROR "HOPSY_BUILD_GPU is only supported on Linux x86_64 "
                        "(got ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR})")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "80;90")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "hopsy.gpu: CUDA ${CUDAToolkit_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")

# Static library with CUDA-free headers (e.g. device.hpp)
# Host-only code (python bindings, C++ tests etc.) can use it.
add_library(hopsy_gpu STATIC
    device.cu
)
target_compile_features(hopsy_gpu PUBLIC cxx_std_17 cuda_std_17)
# Link libcudart statically
set_target_properties(hopsy_gpu PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
)
target_link_libraries(hopsy_gpu PRIVATE CUDA::cudart_static)

# Python extension hopsy.gpu._core; hopsy/gpu/__init__.py exports its public API.
pybind11_add_module(hopsy_gpu_core
    module.cpp
)
set_target_properties(hopsy_gpu_core PROPERTIES OUTPUT_NAME _core)
target_link_libraries(hopsy_gpu_core PRIVATE
    hopsy_gpu
    pybind11::module
)
install(TARGETS hopsy_gpu_core DESTINATION hopsy/gpu)
