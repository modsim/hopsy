cmake_minimum_required(VERSION 3.4...3.18)
project(hopsy)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message("Compiling in debug mode.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

option(HOPSY_TESTS "Enable hopsy tests" OFF)
option(HOPSY_NO_EIGEN_VECTORIZE "Enable hopsy tests" ON)

if (HOPSY_NO_EIGEN_VECTORIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_VECTORIZE")
endif ()

set(VERSION_INFO "")

#set(HOPS_LIBRARY_TYPE SHARED CACHE STRING "use header only" FORCE)
set(HOPS_LIBRARY_TYPE STATIC CACHE STRING "use header only" FORCE)
#set(HOPS_LIBRARY_TYPE "HEADER_ONLY")
set(HOPS_BINARIES OFF)
set(HOPS_TESTS OFF)
set(HOPS_OPENMP OFF)
set(HOPS_NO_INSTALL ON)
set(EVENT__DISABLE_TESTS ON)
set(EIGEN3_INCLUDE_DIR "../../extern/eigen3")
add_subdirectory(extern/hops)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall")
endif ()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHOPS_HEADER_ONLY")

set(EIGEN3_INCLUDE_DIR "extern/eigen3")
set(HOPS_INCLUDE_DIR "extern/hops/include/hops")

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${HOPS_INCLUDE_DIR})

add_subdirectory(extern/pybind11)
add_subdirectory(src/hopsy)

#set_target_properties(_hopsy
#    PROPERTIES
#    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/hopsy"
#    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/hopsy"
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/hopsy"
#)

#pybind11_add_module(_hopsy src/hopsy/hopsy.cpp src/hopsy/doc.cpp)

#target_compile_options(hopsy PUBLIC "-DHOPS_HEADER_ONLY")

target_include_directories(core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${EIGEN3_INCLUDE_DIR})
target_include_directories(core PUBLIC ${HOPS_INCLUDE_DIR})

target_link_libraries(core PRIVATE hops)
if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    target_link_libraries(core PRIVATE stdc++fs)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif ()


# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(core PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

if (HOPSY_TESTS)
    enable_testing()
    add_subdirectory(examples)
    add_custom_command(TARGET core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_hopsy> examples/${targetfile}
    )
endif()

