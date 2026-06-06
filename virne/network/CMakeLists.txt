add_library(vne_network)

target_sources(vne_network PRIVATE

    base_network.cpp
    physical_network.cpp
    virtual_network.cpp

    dataset_generator.cpp
    virtual_network_request_simulator.cpp

    attribute/base_attribute.cpp
    attribute/node_attribute.cpp
    attribute/link_attribute.cpp
    attribute/graph_attribute.cpp

    attribute/attribute_method.cpp
    attribute/attribute_benchmark_manager.cpp

    topology/topology_generator.cpp
    topology/topological_metric_calculator.cpp
)

target_include_directories(
    vne_network
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_network
    PUBLIC
    vne_core
)