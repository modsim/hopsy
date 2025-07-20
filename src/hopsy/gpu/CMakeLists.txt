cmake_minimum_required(VERSION 3.10)
project(cuda_sampler LANGUAGES CXX CUDA)

# Find source files in this directory
file(GLOB_RECURSE GPU_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.cu")

# Create a static library
add_library(cuda_sampler STATIC ${GPU_SOURCES})

# C++ and CUDA standards
set_target_properties(cuda_sampler PROPERTIES
    CXX_STANDARD 20
    CUDA_STANDARD 14
    CUDA_ARCHITECTURES "80;90"
    POSITION_INDEPENDENT_CODE ON
)

# Disable Eigen download – Hopsy already provides Eigen
target_include_directories(cuda_sampler
    PUBLIC
        ${CMAKE_SOURCE_DIR}/extern/eigen3
        ${CMAKE_CURRENT_SOURCE_DIR}  # For local headers in gpu/
)

# CUDA-specific compile options
target_compile_options(cuda_sampler PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcudafe --diag_suppress=20012>
)

# Add debug options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_sampler PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-g> 
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endif()

# Find CUDA toolkit manually
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    message(STATUS "CUDA DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(FATAL_ERROR "CUDA not found. Set -DCUDA_TOOLKIT_ROOT_DIR")
endif()

# Optional: static vs dynamic linking
option(USE_STATIC_CUDA_LIBS "Link statically to CUDA libraries" OFF)

if(USE_STATIC_CUDA_LIBS)
    message(STATUS "Linking with static CUDA libraries")

    set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

    set(STATIC_CUDA_LIBS
        ${CUDA_LIB_DIR}/libcudart_static.a
        ${CUDA_LIB_DIR}/libcusolver_static.a
        ${CUDA_LIB_DIR}/libcusolver_lapack_static.a
        ${CUDA_LIB_DIR}/libcusolver_metis_static.a
        ${CUDA_LIB_DIR}/libcusparse_static.a
        ${CUDA_LIB_DIR}/libcublas_static.a
        ${CUDA_LIB_DIR}/libcublasLt_static.a
        ${CUDA_LIB_DIR}/libcurand_static.a
        ${CUDA_LIB_DIR}/libculibos.a  # must be last
    )

    set(SYSTEM_LIBS -lpthread -ldl -lrt -lm)

    target_link_libraries(cuda_sampler PRIVATE ${STATIC_CUDA_LIBS} ${SYSTEM_LIBS})

    set_target_properties(cuda_sampler PROPERTIES
        LINK_SEARCH_START_STATIC TRUE
        LINK_SEARCH_END_STATIC TRUE
    )
else()
    message(STATUS "Linking with dynamic CUDA libraries")

    find_package(CUDAToolkit REQUIRED)

    target_link_libraries(cuda_sampler PRIVATE
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
    )
endif()
