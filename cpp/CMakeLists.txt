cmake_minimum_required(VERSION 3.26)

project(
        hopsy-core
        VERSION 2.0.0
        LANGUAGES CXX
)
set(HOPSY_CORE_VERSION "${PROJECT_VERSION}.dev0")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(
        HOPSY_BUILD_CPP_TESTS
        "Build the hopsy C++ tests"
        ${PROJECT_IS_TOP_LEVEL}
)

option(
        HOPSY_INSTALL_CPP
        "Install the hopsy C++ library and headers"
        ${PROJECT_IS_TOP_LEVEL}
)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

add_library(
        hopsy_core
        INTERFACE
)

add_library(
        hopsy::core
        ALIAS
        hopsy_core
)

set_target_properties(
        hopsy_core
        PROPERTIES
        EXPORT_NAME core
)

target_include_directories(
        hopsy_core
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(
        hopsy_core
        INTERFACE
        cxx_std_20
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

add_subdirectory(dependencies)

target_link_libraries(
        hopsy_core
        INTERFACE
        Eigen3::Eigen
        hopsy::pcg
)



# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if(HOPSY_BUILD_CPP_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

if(HOPSY_INSTALL_CPP)

    set(
            HOPSY_CMAKE_INSTALL_DIR
            ${CMAKE_INSTALL_LIBDIR}/cmake/hopsy
    )

    install(
            DIRECTORY "${pcg_cpp_SOURCE_DIR}/include/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    install(
            TARGETS
            hopsy_core
            hopsy_pcg
            EXPORT hopsyTargets
    )

    install(
            DIRECTORY include/hopsy
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(
            EXPORT hopsyTargets
            FILE hopsyTargets.cmake
            NAMESPACE hopsy::
            DESTINATION ${HOPSY_CMAKE_INSTALL_DIR}
    )

    configure_package_config_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hopsyConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/hopsyConfig.cmake
            INSTALL_DESTINATION ${HOPSY_CMAKE_INSTALL_DIR}
    )

    write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/hopsyConfigVersion.cmake
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY SameMajorVersion
            ARCH_INDEPENDENT
    )

    install(
            FILES
            ${CMAKE_CURRENT_BINARY_DIR}/hopsyConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/hopsyConfigVersion.cmake
            DESTINATION ${HOPSY_CMAKE_INSTALL_DIR}
    )

endif()
