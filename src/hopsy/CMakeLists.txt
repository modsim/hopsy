pybind11_add_module(core hopsy.cpp doc.cpp)

if(HOPSY_BUILD_GPU)
    add_subdirectory(gpu)
    target_link_libraries(core PRIVATE cuda_sampler)
    target_compile_definitions(core PRIVATE CUDA=1)
endif()