add_library(
    random_lib
    py_random.cpp
    numpy_random_state.cpp
    random_context.cpp
)

target_include_directories(
    random_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# The legacy NumPy kernels are CPU-dispatched native code. Build the
# dependency-free equivalent for the current deployment CPU as well; keep FMA
# contraction disabled because one-bit floating parity is part of the API.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        random_lib
        PRIVATE
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-ffp-contract=off>
            $<$<CONFIG:Release>:-fno-semantic-interposition>
            $<$<CONFIG:Release>:-fno-math-errno>
            $<$<CONFIG:Release>:-fno-strict-aliasing>
    )
endif()

add_executable(
    test_random
    test_random.cpp
)

target_link_libraries(
    test_random
    PRIVATE
        random_lib
)

# Developer-only executables are always discoverable targets but deliberately
# excluded from the default build.  Neither target links Python or NumPy.
add_executable(
    benchmark_random
    EXCLUDE_FROM_ALL
    benchmark_random.cpp
)

target_link_libraries(
    benchmark_random
    PRIVATE
        random_lib
)

add_executable(
    random_differential_harness
    EXCLUDE_FROM_ALL
    differential_harness.cpp
)

target_link_libraries(
    random_differential_harness
    PRIVATE
        random_lib
)

# Explicit developer target: the production library never links Python or
# NumPy. The command intentionally uses the repository-local pinned oracle
# environment and only runs when requested.
add_custom_target(
    random_differential
    COMMAND
        ${CMAKE_SOURCE_DIR}/.venv/bin/python
        ${CMAKE_CURRENT_SOURCE_DIR}/differential_test.py
        --cpp
        $<TARGET_FILE:random_differential_harness>
    DEPENDS
        random_differential_harness
    WORKING_DIRECTORY
        ${CMAKE_SOURCE_DIR}
    USES_TERMINAL
)
