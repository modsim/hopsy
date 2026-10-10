if(NOT COMMAND nanobind_add_module)
    message(FATAL_ERROR "Configure Python and nanobind before add_subdirectory(bindings).")
endif()

nanobind_add_module(_hopsy_core NOMINSIZE bindings.cpp)
target_link_libraries(_hopsy_core PRIVATE hopsy::core)

set(HOPSY_PYTHON_INSTALL_DIR "hopsy/experimental" CACHE STRING
    "Relative installation directory for the experimental Python extension")
install(TARGETS _hopsy_core
    LIBRARY DESTINATION "${HOPSY_PYTHON_INSTALL_DIR}"
    RUNTIME DESTINATION "${HOPSY_PYTHON_INSTALL_DIR}")

option(HOPSY_BUILD_PYTHON_TESTS "Register the Python binding tests (requires pytest and numpy)" OFF)
if(HOPSY_BUILD_PYTHON_TESTS)
    add_test(NAME hopsy_python_bindings
        COMMAND "${Python_EXECUTABLE}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    set_tests_properties(hopsy_python_bindings PROPERTIES
        ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:$<TARGET_FILE_DIR:_hopsy_core>")
endif()
