cmake_minimum_required(VERSION 3.4...3.18)
project(hopsy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall")
endif ()

set(HOPS_EXAMPLES OFF)
set(HOPS_TESTS OFF)
#set(HOPS_HEADER_ONLY ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHOPS_HEADER_ONLY")

#find_package(Eigen3 REQUIRED)

enable_testing()
add_subdirectory(examples)

set(EVENT__DISABLE_TESTS ON)
add_subdirectory(extern/hops)
add_subdirectory(extern/pybind11)

#set_target_properties(hops PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

pybind11_add_module(hopsy src/hopsy.cpp)
#target_compile_options(hopsy PUBLIC "-DHOPS_HEADER_ONLY")
#target_include_directories(hopsy PUBLIC "${EIGEN3_INCLUDE_DIR}")
target_include_directories(hopsy PUBLIC "extern/hops/include/hops")
target_link_libraries(hopsy PUBLIC hops)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(hopsy PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

add_custom_command(TARGET hopsy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hopsy> tests/${targetfile}
        )
