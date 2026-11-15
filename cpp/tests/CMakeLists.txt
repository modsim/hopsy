add_executable(
        hopsy_core_tests
        direction_generator_test.cpp
        draw_test.cpp
        gaussian_kernel_test.cpp
        gaussian_family_test.cpp
        gaussian_cache_test.cpp
        problem_chain_test.cpp
        owned_chain_test.cpp
        bilinear_polytope_test.cpp
        hit_and_run_test.cpp
        linear_polytope_test.cpp
        line_kernel_test.cpp
        step_test.cpp
        uniform_target_test.cpp
)

target_link_libraries(
        hopsy_core_tests
        PRIVATE
        hopsy::core
        Catch2::Catch2WithMain
)

if(HOPSY_CATCH2_EXTRAS_DIR)
    list(
            APPEND
            CMAKE_MODULE_PATH
            "${HOPSY_CATCH2_EXTRAS_DIR}"
    )
endif()

include(Catch)
catch_discover_tests(hopsy_core_tests)

# Keep Eigen's allocation guard active even in optimized test builds.
target_compile_definitions(hopsy_core_tests PRIVATE EIGEN_RUNTIME_NO_MALLOC)
if(MSVC)
    target_compile_options(hopsy_core_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(hopsy_core_tests PRIVATE -UNDEBUG)
endif()

# Opt-in tests of the legacy reference; keep legacy dependencies off core users.
option(HOPSY_TEST_LEGACY_REFERENCE "Test legacy CSmMALA clipping for comparisons" OFF)
if(HOPSY_TEST_LEGACY_REFERENCE)
    set(HOPSY_LEGACY_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../extern/hops/src")
    add_executable(hopsy_legacy_csmala_tests legacy_csmala_test.cpp
        "${HOPSY_LEGACY_SOURCE}/hops/Utility/DefaultDimensionNames.cpp")
    target_include_directories(hopsy_legacy_csmala_tests PRIVATE "${HOPSY_LEGACY_SOURCE}")
    target_link_libraries(hopsy_legacy_csmala_tests PRIVATE hopsy::core Catch2::Catch2WithMain)
    catch_discover_tests(hopsy_legacy_csmala_tests)
endif()
