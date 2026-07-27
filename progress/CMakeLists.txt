add_library(
    progress_lib
    progress.cpp
)

target_include_directories(
    progress_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    progress_lib
    PUBLIC
)

add_executable(
    progress_test
    test_progress.cpp
)

target_link_libraries(
    progress_test
    PRIVATE
        progress_lib
)

# Kept out of the default build; run explicitly when changing the rendering
# hot path: cmake --build <build-dir> --target progress_benchmark
add_executable(
    progress_benchmark
    EXCLUDE_FROM_ALL
    benchmark_progress.cpp
)

target_link_libraries(
    progress_benchmark
    PRIVATE
        progress_lib
)
