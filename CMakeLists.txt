cmake_minimum_required(VERSION 3.4...3.18)
project(hopsy)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(EIGEN3_INCLUDE_DIR "../../extern/eigen3")

set(HOPS_EXAMPLES OFF)
set(HOPS_TESTS OFF)
set(EVENT__DISABLE_TESTS ON)
add_subdirectory(extern/hops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall")
endif ()

option(HOPSY_TESTS "Enable hopsy tests" OFF)

#set(HOPS_HEADER_ONLY ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHOPS_HEADER_ONLY")

set(EIGEN3_INCLUDE_DIR "extern/eigen3")
set(HOPS_INCLUDE_DIR "extern/hops/include/hops")

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${HOPS_INCLUDE_DIR})

add_subdirectory(extern/pybind11)

pybind11_add_module(hopsy src/hopsy.cpp)

#target_compile_options(hopsy PUBLIC "-DHOPS_HEADER_ONLY")

target_include_directories(hopsy PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(hopsy PUBLIC ${HOPS_INCLUDE_DIR})

target_link_libraries(hopsy PRIVATE hops)
if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    target_link_libraries(hopsy PRIVATE stdc++fs)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif ()


# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(hopsy PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

if (HOPSY_TESTS)
    enable_testing()
    add_subdirectory(examples)
    add_custom_command(TARGET hopsy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hopsy> examples/${targetfile}
    )
endif()

