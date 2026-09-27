include(FetchContent)


##########
# Eigen3 #
##########

set(HOPSY_EIGEN_MIN_VERSION "5.0")
set(HOPSY_EIGEN_FETCH_VERSION "5.0.1")

find_package(
        Eigen3 ${HOPSY_EIGEN_MIN_VERSION}
        QUIET
        NO_MODULE
)

if(NOT Eigen3_FOUND)
    message(
            STATUS
            "Eigen3 not found; fetching Eigen ${HOPSY_EIGEN_FETCH_VERSION}"
    )

    FetchContent_Declare(
            eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG ${HOPSY_EIGEN_FETCH_VERSION}
            GIT_SHALLOW TRUE

            # Populate Eigen, but don't add Eigen's own CMake project.
            SOURCE_SUBDIR cmake
    )

    FetchContent_MakeAvailable(eigen3)

    add_library(
            Eigen3::Eigen
            INTERFACE
            IMPORTED
            GLOBAL
    )

    set_target_properties(
            Eigen3::Eigen
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${eigen3_SOURCE_DIR}"
    )
else()
    message(STATUS "Using installed Eigen3")
endif()

#######
# PCG #
#######

set(HOPSY_PCG_GIT_COMMIT "428802d1a5634f96bcd0705fab379ff0113bcf13" CACHE STRING "PCG commit used by hopsy")

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
        $<BUILD_INTERFACE:${pcg_cpp_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(
        hopsy_pcg
        PROPERTIES
        EXPORT_NAME pcg
)

add_library(
        hopsy::pcg
        ALIAS hopsy_pcg
)


##########
# Catch2 #
##########

if(HOPSY_BUILD_CPP_TESTS)

    set(HOPSY_CATCH2_VERSION "3.8.1")

    find_package(
            Catch2 ${HOPSY_CATCH2_VERSION}
            QUIET
    )

    if(NOT Catch2_FOUND)
        message(
                STATUS
                "Catch2 not found; fetching Catch2 ${HOPSY_CATCH2_VERSION}"
        )

        FetchContent_Declare(
                Catch2
                GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                GIT_TAG v${HOPSY_CATCH2_VERSION}
                GIT_SHALLOW TRUE
        )

        FetchContent_MakeAvailable(Catch2)

        set(
                HOPSY_CATCH2_EXTRAS_DIR
                "${catch2_SOURCE_DIR}/extras"
                PARENT_SCOPE
        )
    else()
        message(STATUS "Using installed Catch2")
    endif()

endif()
