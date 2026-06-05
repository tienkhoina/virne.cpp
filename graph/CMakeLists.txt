add_library(
    graph_lib

    graph.cpp

    sparse_matrix.cpp

    generators/waxman_generator.cpp

    generators/gml_loader.cpp
    

    nx/connectivity.cpp
    nx/shortest_paths.cpp
    nx/attributes.cpp
    nx/sparse.cpp
    nx/centrality.cpp


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
    ${CMAKE_SOURCE_DIR}/libs/boost
)

