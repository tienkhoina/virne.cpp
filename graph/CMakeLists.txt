add_library(
    graph_lib

    graph.cpp

    generators/waxman_generator.cpp

    generators/gml_loader.cpp

    nx/connectivity.cpp
    nx/shortest_paths.cpp

    algorithms/floyd_warshall.cpp
    algorithms/bfs.cpp
    algorithms/dijkstra.cpp
    algorithms/k_shortest_paths.cpp
    algorithms/bidirectional_dijkstra.cpp
    algorithms/bidirectional_bfs.cpp

    cache/weight_cache.cpp
)

target_include_directories(
    graph_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/libs/boost
)

