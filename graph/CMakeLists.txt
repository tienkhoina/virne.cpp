add_library(
    graph_lib

    graph.cpp

    sparse_matrix.cpp

    generators/waxman_generator.cpp
    generators/topology_generators.cpp

    generators/gml_loader.cpp
    

    nx/connectivity.cpp
    nx/shortest_paths.cpp
    nx/attributes.cpp
    nx/sparse.cpp
    nx/centrality.cpp
    nx/subgraph.cpp
    nx/relabel.cpp


    algorithms/floyd_warshall.cpp
    algorithms/bfs.cpp
    algorithms/bfs_stats.cpp
    algorithms/bfs_nx.cpp
    algorithms/dijkstra.cpp
    algorithms/k_shortest_paths.cpp
    algorithms/bidirectional_dijkstra.cpp
    algorithms/bidirectional_bfs.cpp

    io/graph_saver.cpp

    cache/weight_cache.cpp
)

target_include_directories(
    graph_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    graph_lib
    PUBLIC
        virne_boost
)

add_executable(
    graph_foundation_test
    tests/graph_foundation_test.cpp
)

target_link_libraries(
    graph_foundation_test
    PRIVATE
        graph_lib
        random_lib
)

add_test(
    NAME graph_foundation_completion
    COMMAND graph_foundation_test
)

add_executable(
    graph_edge_order_parity_test
    tests/edge_order_parity_test.cpp
)

target_link_libraries(
    graph_edge_order_parity_test
    PRIVATE
        graph_lib
)

add_test(
    NAME graph_edge_order_parity
    COMMAND graph_edge_order_parity_test
)

add_executable(
    graph_view_bulk_subgraph_test
    tests/view_bulk_subgraph_test.cpp
)

target_link_libraries(
    graph_view_bulk_subgraph_test
    PRIVATE
        graph_lib
)

add_test(
    NAME graph_view_bulk_subgraph
    COMMAND graph_view_bulk_subgraph_test
)

add_executable(
    graph_generator_parity_harness
    tests/generator_parity_harness.cpp
)

target_link_libraries(
    graph_generator_parity_harness
    PRIVATE
        graph_lib
        random_lib
)

add_executable(
    graph_completion_harness
    ../benchmarks/graph_completion_harness.cpp
)

target_link_libraries(
    graph_completion_harness
    PRIVATE
        graph_lib
        random_lib
)

add_executable(
    graph_view_overhead_benchmark
    ../benchmarks/view_overhead.cpp
)

target_link_libraries(
    graph_view_overhead_benchmark
    PRIVATE
        graph_lib
)

add_executable(
    graph_gml_compat_test
    test_gml.cpp
)

target_link_libraries(
    graph_gml_compat_test
    PRIVATE
        graph_lib
)

target_compile_definitions(
    graph_gml_compat_test
    PRIVATE
        VIRNE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_test(
    NAME graph_gml_compat
    COMMAND graph_gml_compat_test
)

add_executable(
    graph_sparse_compat_test
    tests/sparse_compat_test.cpp
)

target_link_libraries(
    graph_sparse_compat_test
    PRIVATE
        graph_lib
)

add_test(
    NAME graph_sparse_compat
    COMMAND graph_sparse_compat_test
)
