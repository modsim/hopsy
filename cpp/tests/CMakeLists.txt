add_executable(
        hopsy_core_tests
        dirichlet_stationary_distribution_test.cpp
        parallel_tempering_test.cpp
        parallel_tempering_statistics_test.cpp
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
catch_discover_tests(hopsy_core_tests ADD_TAGS_AS_LABELS)
