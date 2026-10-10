add_executable(
        hopsy_core_tests
        direction_generator_test.cpp
        draw_test.cpp
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
