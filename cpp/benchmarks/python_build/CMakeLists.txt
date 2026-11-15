cmake_minimum_required(VERSION 3.26)
project(hopsy_binding_benchmarks LANGUAGES CXX)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
set(HOPSY_BUILD_PYTHON ON CACHE BOOL "" FORCE)
set(HOPSY_BUILD_CPP_TESTS OFF CACHE BOOL "" FORCE)
set(HOPSY_INSTALL_CPP OFF CACHE BOOL "" FORCE)
# The installed nanobind package supplies its CMake functions but has no root
# CMakeLists.txt. Its source override avoids fetching another nanobind checkout.
add_subdirectory(../.. core)
