add_executable(
        hopsy_core_tests
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

include(Catch)
catch_discover_tests(hopsy_core_tests)
