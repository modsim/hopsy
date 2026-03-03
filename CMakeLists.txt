cmake_minimum_required(VERSION 3.15...3.26)
project(hopsy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_SYSTEM_EIGEN "Use system Eigen" OFF)

if(USE_SYSTEM_EIGEN)
    find_package(Eigen3 REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

set(HOPS_LIBRARY_TYPE STATIC CACHE STRING "use header only" FORCE)
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
set(HOPS_LIBRARY_TYPE STATIC CACHE STRING "")
set(HOPS_BUILD_NATIVE OFF CACHE BOOL "")
set(HOPS_BINARIES OFF CACHE BOOL "")

add_subdirectory(extern/hops)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(core src/hopsy/hopsy.cpp src/hopsy/doc.cpp)
target_link_libraries(core PRIVATE
    hops
    Eigen3::Eigen
    pybind11::module
)
# Silence compiler warnings in Eigen3 code
get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(core SYSTEM PRIVATE ${EIGEN_INC})

target_compile_definitions(core PRIVATE
    HOPS_WITHIN_HOPSY
    BUILD_INFO="${HOPSY_BUILD_INFO}"
    VERSION_INFO="${HOPSY_VERSION_INFO}"
    $<$<CONFIG:Debug>:IS_DEBUG>
)

if (MSVC)
    target_compile_options(core PRIVATE /W4)
else()
    target_compile_options(core PRIVATE -Wall -Wextra)
endif()

install(TARGETS core DESTINATION hopsy)
