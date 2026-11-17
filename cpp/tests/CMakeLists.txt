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

# Standalone threaded ownership check, also suitable for sanitizer builds.
add_executable(hopsy_bilinear_smoke bilinear_sanitizer_smoke.cpp)
target_link_libraries(hopsy_bilinear_smoke PRIVATE hopsy::core)
add_test(NAME hopsy_bilinear_smoke COMMAND hopsy_bilinear_smoke)
