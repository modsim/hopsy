add_executable(
        hopsy_core_tests
        direction_generator_test.cpp
        draw_test.cpp
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
