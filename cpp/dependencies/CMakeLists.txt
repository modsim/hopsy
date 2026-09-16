include(FetchContent)


##########
# Eigen3 #
##########

set(HOPSY_EIGEN_MIN_VERSION "5.0")
set(HOPSY_EIGEN_FETCH_VERSION "5.0.1")

find_package(Eigen3 ${HOPSY_EIGEN_MIN_VERSION} QUIET NO_MODULE)


if(NOT Eigen3_FOUND)
    message(
            STATUS
            "Eigen3 not found; fetching Eigen ${HOPSY_EIGEN_VERSION}"
    )
    FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG ${HOPSY_EIGEN_FETCH_VERSION}
            GIT_SHALLOW    TRUE
    )

    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(Eigen3)
else()
    message(STATUS "Using installed Eigen3")
endif()


#######
# PCG #
#######

set(HOPSY_PCG_GIT_COMMIT 428802d1a5634f96bcd0705fab379ff0113bcf13 CACHE STRING)

FetchContent_Declare(
        pcg_cpp
        GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
        GIT_TAG ${HOPSY_PCG_GIT_COMMIT}
        GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(pcg_cpp)

add_library(hopsy_pcg INTERFACE)

target_include_directories(
        hopsy_pcg
        INTERFACE
        ${pcg_cpp_SOURCE_DIR}/include
)

add_library(
        hopsy::pcg
        ALIAS hopsy_pcg
)
