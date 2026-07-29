find_package(Threads REQUIRED)

add_library(
    vne_solution
    solution.cpp
)

target_include_directories(
    vne_solution
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_solution
    PUBLIC
        vne_virtual_network
        vne_attribute_method
        Threads::Threads
)

add_library(
    vne_constraint_checker
    controller/constraint_checker.cpp
)

target_include_directories(
    vne_constraint_checker
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/controller
)

target_link_libraries(
    vne_constraint_checker
    PUBLIC
        vne_solution
        vne_physical_network
        vne_graph_attribute
        vne_utils_network
        Threads::Threads
)

add_library(
    vne_resource_updator
    controller/resource_updator.cpp
)

target_include_directories(
    vne_resource_updator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/controller
)

target_link_libraries(
    vne_resource_updator
    PUBLIC
        vne_constraint_checker
        vne_node_attribute
        vne_link_attribute
        vne_utils_network
        Threads::Threads
)

add_library(
    vne_topology_analyzer
    controller/topology_analyzer.cpp
)

target_include_directories(
    vne_topology_analyzer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/controller
)

target_link_libraries(
    vne_topology_analyzer
    PUBLIC
        vne_constraint_checker
        vne_link_attribute
        graph_lib
        Threads::Threads
)

add_library(
    vne_node_mapper
    controller/node_mapper.cpp
)

target_include_directories(
    vne_node_mapper
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/controller
)

target_link_libraries(
    vne_node_mapper
    PUBLIC
        vne_constraint_checker
        vne_resource_updator
        vne_physical_network
        Threads::Threads
)

add_library(
    vne_link_mapper
    controller/link_mapper.cpp
)

target_include_directories(
    vne_link_mapper
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/controller
)

target_link_libraries(
    vne_link_mapper
    PUBLIC
        vne_constraint_checker
        vne_resource_updator
        vne_topology_analyzer
        vne_physical_network
        vne_utils_network
        Threads::Threads
)

add_library(
    vne_controller
    controller/controller.cpp
)

target_include_directories(
    vne_controller
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/controller
)

target_link_libraries(
    vne_controller
    PUBLIC
        vne_constraint_checker
        vne_resource_updator
        vne_topology_analyzer
        vne_node_mapper
        vne_link_mapper
        vne_physical_network
        Threads::Threads
)

add_library(
    vne_counter
    counter.cpp
)

target_include_directories(
    vne_counter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_counter
    PUBLIC
        vne_solution
        vne_base_network
        vne_virtual_network
        vne_virtual_network_event
        csv_lib
        Threads::Threads
)

add_library(
    vne_recorder
    recorder.cpp
)

target_include_directories(
    vne_recorder
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_recorder
    PUBLIC
        vne_counter
        vne_physical_network
        vne_utils_config
        vne_utils_class_dict
        csv_lib
        Threads::Threads
)

add_library(
    vne_logger
    logger.cpp
)

target_include_directories(
    vne_logger
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_logger
    PUBLIC
        vne_utils_config
        csv_lib
        Threads::Threads
)

add_library(
    vne_environment
    environment.cpp
)

target_include_directories(
    vne_environment
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/controller
)

target_link_libraries(
    vne_environment
    PUBLIC
        vne_controller
        vne_counter
        vne_recorder
        vne_physical_network
        vne_virtual_network_request_simulator
        random_lib
        Threads::Threads
)

# Aggregate the completed non-ML core. Solver/system/RL wrappers stay separate.
add_library(vne_core INTERFACE)

target_link_libraries(
    vne_core
    INTERFACE
        vne_solution
        vne_constraint_checker
        vne_resource_updator
        vne_topology_analyzer
        vne_node_mapper
        vne_link_mapper
        vne_controller
        vne_counter
        vne_recorder
        vne_logger
        vne_environment
)
