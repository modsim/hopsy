cmake_minimum_required(VERSION 3.15...4.3)
project(hopsy LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_SYSTEM_EIGEN "Use system Eigen" OFF)
option(HOPSY_NO_EIGEN_VECTORIZE "Disable Eigen vectorization" OFF)

if(HOPSY_NO_EIGEN_VECTORIZE)
    add_compile_definitions(EIGEN_DONT_VECTORIZE)
endif()

if(USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
    get_target_property(_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    set(EIGEN3_INCLUDE_DIR "${_eigen_inc}" CACHE PATH "Eigen3 include dir" FORCE)
else()
    FetchContent_Declare(
        eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE

        # Populate only. Do not add Eigen's own CMake project.
        SOURCE_SUBDIR __unused__
    )

    FetchContent_MakeAvailable(eigen3)

    # extern/hops still expects this cache variable via its legacy FindEigen3.cmake
    set(EIGEN3_INCLUDE_DIR "${eigen3_SOURCE_DIR}" CACHE PATH "Eigen3 include dir" FORCE)

    # Provide a modern imported target for this project
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    endif()
endif()

set(HOPS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/hops/include/hops")

set(HOPS_LIBRARY_TYPE STATIC CACHE STRING "use static library" FORCE)
set(HOPS_BUILD_NATIVE OFF CACHE BOOL "" FORCE)
set(HOPS_BINARIES OFF CACHE BOOL "" FORCE)
set(HOPS_TUNING ON CACHE BOOL "" FORCE)
set(HOPS_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HOPS_TESTS OFF CACHE BOOL "" FORCE)
set(HOPS_MPI OFF CACHE BOOL "" FORCE)
set(HOPS_OPENMP OFF CACHE BOOL "" FORCE)
set(HOPS_GUROBI OFF CACHE BOOL "" FORCE)
set(HOPS_DNEST4_SUPPORT OFF CACHE BOOL "" FORCE)
set(HOPS_NO_INSTALL ON CACHE BOOL "" FORCE)
set(EVENT__DISABLE_TESTS ON CACHE BOOL "" FORCE)

add_subdirectory(extern/hops)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(core
    src/hopsy/hopsy.cpp
    src/hopsy/doc.cpp
)

target_link_libraries(core PRIVATE
    hops
    pybind11::module
    Eigen3::Eigen
)

# Treat Eigen headers as system headers to reduce warning noise
target_include_directories(core
    SYSTEM PRIVATE
        "${EIGEN3_INCLUDE_DIR}"
    PRIVATE
        "${HOPS_INCLUDE_DIR}"
)

target_compile_definitions(core PRIVATE
    HOPS_WITHIN_HOPSY
    BUILD_INFO="${HOPSY_BUILD_INFO}"
    VERSION_INFO="${HOPSY_VERSION_INFO}"
    $<$<CONFIG:Debug>:IS_DEBUG>
)

if(MSVC)
    target_compile_options(core PRIVATE /W4)
else()
    target_compile_options(core PRIVATE -Wall -Wextra)
endif()

install(TARGETS core DESTINATION hopsy)
