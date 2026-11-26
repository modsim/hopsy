if(NOT COMMAND nanobind_add_module)
    message(FATAL_ERROR "Configure Python and nanobind before add_subdirectory(bindings).")
endif()

nanobind_add_module(_hopsy_core NOMINSIZE bindings.cpp gaussian.cpp dense.cpp cycling.cpp
    random_coordinate.cpp tempering.cpp)
target_link_libraries(_hopsy_core PRIVATE hopsy::core)

set(HOPSY_PYTHON_INSTALL_DIR "hopsy" CACHE STRING
    "Relative installation directory for the Python 2.0 package")
install(TARGETS _hopsy_core
    LIBRARY DESTINATION "${HOPSY_PYTHON_INSTALL_DIR}"
    RUNTIME DESTINATION "${HOPSY_PYTHON_INSTALL_DIR}")

install(FILES python/hopsy/__init__.py DESTINATION "${HOPSY_PYTHON_INSTALL_DIR}")

option(HOPSY_BUILD_PYTHON_TESTS "Register the Python binding tests (requires pytest and numpy)" OFF)
if(HOPSY_BUILD_PYTHON_TESTS)
    add_test(NAME hopsy_python_bindings
        COMMAND "${Python_EXECUTABLE}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    set_tests_properties(hopsy_python_bindings PROPERTIES
        ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:$<TARGET_FILE_DIR:_hopsy_core>")
endif()

if(HOPSY_BUILD_PYTHON_TESTS)
    Python_add_library(_capsule_gaussian MODULE WITH_SOABI ../examples/capsule_target/gaussian.cpp)
    target_include_directories(_capsule_gaussian PRIVATE ../include)
    target_compile_features(_capsule_gaussian PRIVATE cxx_std_20)
    target_link_libraries(_capsule_gaussian PRIVATE Threads::Threads)
endif()
