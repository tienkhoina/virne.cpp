add_library(
    graph_lib

    graph.cpp

    generators/waxman_generator.cpp

    generators/gml_loader.cpp
)

target_include_directories(
    graph_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/libs/boost
)

