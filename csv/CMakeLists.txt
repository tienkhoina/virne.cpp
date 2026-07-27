add_library(csv_lib
    csv.cpp
)

target_include_directories(csv_lib
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(csv_lib
PUBLIC
    virne_tabulate
)

add_executable(
    csv_rfc4180_test
    test_csv.cpp
)

target_link_libraries(
    csv_rfc4180_test
    PRIVATE
        csv_lib
)

add_test(
    NAME csv_rfc4180
    COMMAND csv_rfc4180_test
)

add_executable(
    csv_benchmark
    EXCLUDE_FROM_ALL
    benchmark_csv.cpp
)

target_link_libraries(
    csv_benchmark
    PRIVATE
        csv_lib
)
