find_package(Threads REQUIRED)

add_library(
    vne_network_topology_generator
    topology/topology_generator.cpp
)

add_library(
    vne_topological_metric_calculator
    topology/topological_metric_calculator.cpp
)

add_library(
    vne_attribute_method
    attribute/attribute_method.cpp
)

add_library(
    vne_base_attribute
    attribute/base_attribute.cpp
)

add_library(
    vne_node_attribute
    attribute/node_attribute.cpp
)

add_library(
    vne_link_attribute
    attribute/link_attribute.cpp
)

add_library(
    vne_graph_attribute
    attribute/graph_attribute.cpp
)

add_library(
    vne_attribute_benchmark_manager
    attribute/attribute_benchmark_manager.cpp
)

add_library(
    vne_attribute_factory
    attribute/attribute_factory.cpp
)

add_library(
    vne_base_network
    base_network.cpp
)

add_library(
    vne_virtual_network
    virtual_network.cpp
)

add_library(
    vne_physical_network
    physical_network.cpp
)

add_library(
    vne_virtual_network_event
    virtual_network_event.cpp
)

add_library(
    vne_virtual_network_request_simulator
    virtual_network_request_simulator.cpp
    virtual_network_request_simulator_io.cpp
)

add_library(
    vne_dataset_generator
    dataset_generator.cpp
)

target_include_directories(
    vne_network_topology_generator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_topological_metric_calculator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_attribute_method
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_base_attribute
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_node_attribute
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_link_attribute
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_graph_attribute
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_attribute_benchmark_manager
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_attribute_factory
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_base_network
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_virtual_network
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_physical_network
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_virtual_network_event
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_virtual_network_request_simulator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_dataset_generator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_attribute_method
    PUBLIC
        Threads::Threads
)

target_link_libraries(
    vne_base_attribute
    PUBLIC
        vne_attribute_method
        vne_utils_dataset_rng
)

target_link_libraries(
    vne_node_attribute
    PUBLIC
        vne_base_attribute
        graph_lib
        Threads::Threads
)

target_link_libraries(
    vne_link_attribute
    PUBLIC
        vne_node_attribute
        graph_lib
        Threads::Threads
)

target_link_libraries(
    vne_graph_attribute
    PUBLIC
        vne_link_attribute
        graph_lib
        Threads::Threads
)

target_link_libraries(
    vne_attribute_benchmark_manager
    PUBLIC
        vne_attribute_method
        Threads::Threads
)

target_link_libraries(
    vne_attribute_factory
    PUBLIC
        vne_graph_attribute
        vne_utils_setting
        Threads::Threads
)

target_link_libraries(
    vne_base_network
    PUBLIC
        vne_attribute_factory
        vne_attribute_benchmark_manager
        vne_network_topology_generator
        vne_utils_network
        vne_utils_setting
        graph_lib
        random_lib
        Threads::Threads
)

target_link_libraries(
    vne_virtual_network
    PUBLIC
        vne_base_network
)

target_link_libraries(
    vne_physical_network
    PUBLIC
        vne_base_network
)

target_link_libraries(
    vne_virtual_network_event
    PUBLIC
        vne_utils_dataset_core
        Threads::Threads
)

target_link_libraries(
    vne_virtual_network_request_simulator
    PUBLIC
        vne_virtual_network
        vne_virtual_network_event
        vne_utils_dataset_rng
        vne_utils_setting
        random_lib
        Threads::Threads
)

target_link_libraries(
    vne_dataset_generator
    PUBLIC
        config_lib
        vne_physical_network
        vne_virtual_network_request_simulator
        vne_utils_dataset_core
        vne_utils_setting
        random_lib
)

target_link_libraries(
    vne_network_topology_generator
    PUBLIC
        graph_lib
        random_lib
        Threads::Threads
)

target_link_libraries(
    vne_topological_metric_calculator
    PUBLIC
        graph_lib
        Threads::Threads
)

add_library(vne_network INTERFACE)

target_include_directories(
    vne_network
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_network
    INTERFACE
        vne_core
        vne_network_topology_generator
        vne_topological_metric_calculator
        vne_attribute_method
        vne_base_attribute
        vne_node_attribute
        vne_link_attribute
        vne_graph_attribute
        vne_attribute_benchmark_manager
        vne_attribute_factory
        vne_base_network
        vne_virtual_network
        vne_physical_network
        vne_virtual_network_event
        vne_virtual_network_request_simulator
        vne_dataset_generator
)
