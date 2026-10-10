    nanobind_add_module(
            _hopsy_core
            NOMINSIZE
            src/python/bindings.cpp
    )

    target_link_libraries(
            _hopsy_core
            PRIVATE
            hopsy::core
    )
