cmake_minimum_required(VERSION 3.4...3.18)
project(hopsy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif ()

set(HOPS_EXAMPLES OFF)
set(HOPS_TESTS OFF)

add_subdirectory(extern/hops)
add_subdirectory(extern/pybind11)

set_target_properties(hops PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

pybind11_add_module(hopsy src/hopsy.cpp)
target_link_libraries(hopsy PRIVATE hops)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(hopsy PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
