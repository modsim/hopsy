cmake_minimum_required(VERSION 3.4...4.3)
project(hopsy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HOPSY_NO_EIGEN_VECTORIZE "Disable Eigen vectorization" OFF)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Compiling in debug mode.")
endif()

if (HOPSY_NO_EIGEN_VECTORIZE)
  add_compile_definitions(EIGEN_DONT_VECTORIZE)
endif()

add_compile_definitions($<$<CONFIG:Debug>:IS_DEBUG>)

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
set(EVENT__DISABLE_TESTS ON CACHE BOOL "" FORCE)

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen3")
set(HOPS_INCLUDE_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/extern/hops/include/hops")

include_directories(SYSTEM "${EIGEN3_INCLUDE_DIR}")

add_subdirectory(extern/hops)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(core src/hopsy/hopsy.cpp src/hopsy/doc.cpp)
target_link_libraries(core PRIVATE pybind11::module)

if (NOT TARGET core)
  message(FATAL_ERROR "Target 'core' not found. Ensure src/hopsy defines it.")
endif()

target_include_directories(core
  SYSTEM PUBLIC
    $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
  PUBLIC
    $<BUILD_INTERFACE:${HOPS_INCLUDE_DIR}>
)

target_compile_definitions(core PRIVATE HOPS_WITHIN_HOPSY)
set_property(TARGET core PROPERTY CXX_STANDARD 17)
set_property(TARGET core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET core PROPERTY CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(core PRIVATE -Wall -Wextra)
endif()

target_link_libraries(core PRIVATE hops)

target_compile_definitions(core PRIVATE
  BUILD_INFO="${HOPSY_BUILD_INFO}"
  VERSION_INFO="${HOPSY_VERSION_INFO}"
)

install(TARGETS core DESTINATION hopsy)
