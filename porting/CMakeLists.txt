find_package(Threads REQUIRED)

add_executable(
    vne_utils_network_unit
    utils_network_unit.cpp
)

target_link_libraries(
    vne_utils_network_unit
    PRIVATE
        vne_utils_network
)

add_test(
    NAME vne_utils_network_unit
    COMMAND vne_utils_network_unit
)

add_test(
    NAME frozen_component_integrity
    COMMAND
        ${CMAKE_COMMAND}
        -DSOURCE_ROOT=${CMAKE_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_frozen.cmake
)

add_executable(
    vne_utils_network_harness
    utils_network_harness.cpp
)

target_link_libraries(
    vne_utils_network_harness
    PRIVATE
        vne_utils_network
)

add_executable(
    vne_topology_generator_unit
    topology_generator_unit.cpp
)

target_link_libraries(
    vne_topology_generator_unit
    PRIVATE
        vne_network_topology_generator
)

add_test(
    NAME vne_topology_generator_unit
    COMMAND vne_topology_generator_unit
)

add_executable(
    vne_topology_generator_harness
    topology_generator_harness.cpp
)

target_link_libraries(
    vne_topology_generator_harness
    PRIVATE
        vne_network_topology_generator
)

add_executable(
    vne_topological_metric_calculator_harness
    topological_metric_calculator_harness.cpp
)

target_link_libraries(
    vne_topological_metric_calculator_harness
    PRIVATE
        vne_topological_metric_calculator
)

add_executable(
    vne_topological_metric_calculator_unit
    topological_metric_calculator_unit.cpp
)

target_link_libraries(
    vne_topological_metric_calculator_unit
    PRIVATE
        vne_topological_metric_calculator
)

add_test(
    NAME vne_topological_metric_calculator_unit
    COMMAND vne_topological_metric_calculator_unit
)

add_executable(
    vne_class_dict_unit
    class_dict_unit.cpp
)

target_link_libraries(
    vne_class_dict_unit
    PRIVATE
        vne_utils_class_dict
)

add_test(
    NAME vne_class_dict_unit
    COMMAND vne_class_dict_unit
)

add_executable(
    vne_class_dict_harness
    class_dict_harness.cpp
)

target_link_libraries(
    vne_class_dict_harness
    PRIVATE
        vne_utils_class_dict
)

add_executable(
    vne_setting_unit
    setting_unit.cpp
)

target_link_libraries(
    vne_setting_unit
    PRIVATE
        vne_utils_setting
)

add_test(
    NAME vne_setting_unit
    COMMAND vne_setting_unit
)

add_executable(
    vne_setting_harness
    setting_harness.cpp
)

target_link_libraries(
    vne_setting_harness
    PRIVATE
        vne_utils_setting
)

add_executable(
    vne_stats_unit
    stats_unit.cpp
)

target_link_libraries(
    vne_stats_unit
    PRIVATE
        vne_utils_stats
)

add_test(
    NAME vne_stats_unit
    COMMAND vne_stats_unit
)

add_executable(
    vne_stats_harness
    stats_harness.cpp
)

target_link_libraries(
    vne_stats_harness
    PRIVATE
        vne_utils_stats
        Threads::Threads
)

add_executable(
    vne_manager_unit
    manager_unit.cpp
)

target_link_libraries(
    vne_manager_unit
    PRIVATE
        vne_utils_manager
)

add_test(
    NAME vne_manager_unit
    COMMAND vne_manager_unit
)

add_executable(
    vne_manager_harness
    manager_harness.cpp
)

target_link_libraries(
    vne_manager_harness
    PRIVATE
        vne_utils_manager
)

add_executable(
    vne_manager_benchmark
    manager_benchmark.cpp
)

target_link_libraries(
    vne_manager_benchmark
    PRIVATE
        vne_utils_manager
)

add_executable(
    vne_dataset_core_unit
    dataset_core_unit.cpp
)

target_link_libraries(
    vne_dataset_core_unit
    PRIVATE
        vne_utils_dataset_core
)

add_test(
    NAME vne_dataset_core_unit
    COMMAND vne_dataset_core_unit
)

add_executable(
    vne_dataset_core_harness
    dataset_core_harness.cpp
)

target_link_libraries(
    vne_dataset_core_harness
    PRIVATE
        vne_utils_dataset_core
)

add_executable(
    vne_dataset_rng_unit
    dataset_rng_unit.cpp
)

target_link_libraries(
    vne_dataset_rng_unit
    PRIVATE
        vne_utils_dataset_rng
)

add_test(
    NAME vne_dataset_rng_unit
    COMMAND vne_dataset_rng_unit
)

add_executable(
    vne_dataset_rng_harness
    dataset_rng_harness.cpp
)

target_link_libraries(
    vne_dataset_rng_harness
    PRIVATE
        vne_utils_dataset_rng
)

add_executable(
    vne_dataset_xml_unit
    dataset_xml_unit.cpp
)

target_link_libraries(
    vne_dataset_xml_unit
    PRIVATE
        vne_utils_dataset_xml
)

add_test(
    NAME vne_dataset_xml_unit
    COMMAND vne_dataset_xml_unit
)

add_executable(
    vne_dataset_xml_harness
    dataset_xml_harness.cpp
)

target_link_libraries(
    vne_dataset_xml_harness
    PRIVATE
        vne_utils_dataset_xml
)

add_executable(
    vne_attribute_method_unit
    attribute_method_unit.cpp
)

target_link_libraries(
    vne_attribute_method_unit
    PRIVATE
        vne_attribute_method
)

add_test(
    NAME vne_attribute_method_unit
    COMMAND vne_attribute_method_unit
)

add_executable(
    vne_attribute_method_harness
    attribute_method_harness.cpp
)

target_link_libraries(
    vne_attribute_method_harness
    PRIVATE
        vne_attribute_method
        Threads::Threads
)

add_executable(
    vne_base_attribute_unit
    base_attribute_unit.cpp
)

target_link_libraries(
    vne_base_attribute_unit
    PRIVATE
        vne_base_attribute
        Threads::Threads
)

add_test(
    NAME vne_base_attribute_unit
    COMMAND vne_base_attribute_unit
)

add_executable(
    vne_base_attribute_harness
    base_attribute_harness.cpp
)

target_link_libraries(
    vne_base_attribute_harness
    PRIVATE
        vne_base_attribute
)

add_executable(
    vne_base_attribute_benchmark
    base_attribute_benchmark.cpp
)

target_link_libraries(
    vne_base_attribute_benchmark
    PRIVATE
        vne_base_attribute
)

add_executable(
    vne_node_attribute_unit
    node_attribute_unit.cpp
)

target_link_libraries(
    vne_node_attribute_unit
    PRIVATE
        vne_node_attribute
        Threads::Threads
)

add_test(
    NAME vne_node_attribute_unit
    COMMAND vne_node_attribute_unit
)

add_executable(
    vne_node_attribute_harness
    node_attribute_harness.cpp
)

target_link_libraries(
    vne_node_attribute_harness
    PRIVATE
        vne_node_attribute
)

add_executable(
    vne_node_attribute_benchmark
    node_attribute_benchmark.cpp
)

target_link_libraries(
    vne_node_attribute_benchmark
    PRIVATE
        vne_node_attribute
)

add_executable(
    vne_link_attribute_unit
    link_attribute_unit.cpp
)

target_link_libraries(
    vne_link_attribute_unit
    PRIVATE
        vne_link_attribute
        Threads::Threads
)

add_test(
    NAME vne_link_attribute_unit
    COMMAND vne_link_attribute_unit
)

add_executable(
    vne_link_attribute_harness
    link_attribute_harness.cpp
)

target_link_libraries(
    vne_link_attribute_harness
    PRIVATE
        vne_link_attribute
)

add_executable(
    vne_link_attribute_benchmark
    link_attribute_benchmark.cpp
)

target_link_libraries(
    vne_link_attribute_benchmark
    PRIVATE
        vne_link_attribute
)

add_executable(
    vne_graph_attribute_unit
    graph_attribute_unit.cpp
)

target_link_libraries(
    vne_graph_attribute_unit
    PRIVATE
        vne_graph_attribute
        Threads::Threads
)

add_test(
    NAME vne_graph_attribute_unit
    COMMAND vne_graph_attribute_unit
)

add_executable(
    vne_graph_attribute_harness
    graph_attribute_harness.cpp
)

target_link_libraries(
    vne_graph_attribute_harness
    PRIVATE
        vne_graph_attribute
)

add_executable(
    vne_graph_attribute_benchmark
    graph_attribute_benchmark.cpp
)

target_link_libraries(
    vne_graph_attribute_benchmark
    PRIVATE
        vne_graph_attribute
)

add_executable(
    vne_attribute_benchmark_manager_unit
    attribute_benchmark_manager_unit.cpp
)

target_link_libraries(
    vne_attribute_benchmark_manager_unit
    PRIVATE
        vne_attribute_benchmark_manager
        Threads::Threads
)

add_test(
    NAME vne_attribute_benchmark_manager_unit
    COMMAND vne_attribute_benchmark_manager_unit
)

add_executable(
    vne_attribute_benchmark_manager_harness
    attribute_benchmark_manager_harness.cpp
)

target_link_libraries(
    vne_attribute_benchmark_manager_harness
    PRIVATE
        vne_attribute_benchmark_manager
)

add_executable(
    vne_attribute_benchmark_manager_benchmark
    attribute_benchmark_manager_benchmark.cpp
)

target_link_libraries(
    vne_attribute_benchmark_manager_benchmark
    PRIVATE
        vne_attribute_benchmark_manager
)

add_executable(
    vne_attribute_factory_unit
    attribute_factory_unit.cpp
)

target_link_libraries(
    vne_attribute_factory_unit
    PRIVATE
        vne_attribute_factory
        Threads::Threads
)

add_test(
    NAME vne_attribute_factory_unit
    COMMAND vne_attribute_factory_unit
)

add_executable(
    vne_attribute_factory_harness
    attribute_factory_harness.cpp
)

target_link_libraries(
    vne_attribute_factory_harness
    PRIVATE
        vne_attribute_factory
)

add_executable(
    vne_attribute_factory_benchmark
    attribute_factory_benchmark.cpp
)

target_link_libraries(
    vne_attribute_factory_benchmark
    PRIVATE
        vne_attribute_factory
)

add_executable(
    vne_base_network_unit
    base_network_unit.cpp
)

target_link_libraries(
    vne_base_network_unit
    PRIVATE
        vne_base_network
        Threads::Threads
)

add_test(
    NAME vne_base_network_unit
    COMMAND vne_base_network_unit
)

add_executable(
    vne_virtual_network_unit
    virtual_network_unit.cpp
)

target_link_libraries(
    vne_virtual_network_unit
    PRIVATE
        vne_virtual_network
        Threads::Threads
)

add_test(
    NAME vne_virtual_network_unit
    COMMAND vne_virtual_network_unit
)

add_executable(
    vne_physical_network_unit
    physical_network_unit.cpp
)

target_link_libraries(
    vne_physical_network_unit
    PRIVATE
        vne_physical_network
        Threads::Threads
)

add_test(
    NAME vne_physical_network_unit
    COMMAND vne_physical_network_unit
)

add_executable(
    vne_virtual_network_event_unit
    virtual_network_event_unit.cpp
)

target_link_libraries(
    vne_virtual_network_event_unit
    PRIVATE
        vne_virtual_network_event
        Threads::Threads
)

add_test(
    NAME vne_virtual_network_event_unit
    COMMAND vne_virtual_network_event_unit
)

add_executable(
    vne_virtual_network_request_simulator_unit
    virtual_network_request_simulator_unit.cpp
)

target_link_libraries(
    vne_virtual_network_request_simulator_unit
    PRIVATE
        vne_virtual_network_request_simulator
        Threads::Threads
)

add_test(
    NAME vne_virtual_network_request_simulator_unit
    COMMAND vne_virtual_network_request_simulator_unit
)

add_executable(
    vne_virtual_network_request_simulator_io_unit
    virtual_network_request_simulator_io_unit.cpp
)

target_link_libraries(
    vne_virtual_network_request_simulator_io_unit
    PRIVATE
        vne_virtual_network_request_simulator
        Threads::Threads
)

add_test(
    NAME vne_virtual_network_request_simulator_io_unit
    COMMAND vne_virtual_network_request_simulator_io_unit
)

add_executable(
    vne_dataset_generator_unit
    dataset_generator_unit.cpp
)

target_link_libraries(
    vne_dataset_generator_unit
    PRIVATE
        vne_dataset_generator
        Threads::Threads
)

target_compile_definitions(
    vne_dataset_generator_unit
    PRIVATE
        VIRNE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_test(
    NAME vne_dataset_generator_unit
    COMMAND vne_dataset_generator_unit
)

add_executable(
    vne_solution_unit
    solution_unit.cpp
)

target_link_libraries(
    vne_solution_unit
    PRIVATE
        vne_solution
        Threads::Threads
)

add_test(
    NAME vne_solution_unit
    COMMAND vne_solution_unit
)

add_executable(
    vne_constraint_checker_unit
    constraint_checker_unit.cpp
)

target_link_libraries(
    vne_constraint_checker_unit
    PRIVATE
        vne_constraint_checker
        Threads::Threads
)

add_test(
    NAME vne_constraint_checker_unit
    COMMAND vne_constraint_checker_unit
)

add_executable(
    vne_resource_updator_unit
    resource_updator_unit.cpp
)

target_link_libraries(
    vne_resource_updator_unit
    PRIVATE
        vne_resource_updator
        Threads::Threads
)

add_test(
    NAME vne_resource_updator_unit
    COMMAND vne_resource_updator_unit
)

add_executable(
    vne_topology_analyzer_unit
    topology_analyzer_unit.cpp
)

target_link_libraries(
    vne_topology_analyzer_unit
    PRIVATE
        vne_topology_analyzer
        Threads::Threads
)

add_test(
    NAME vne_topology_analyzer_unit
    COMMAND vne_topology_analyzer_unit
)

add_executable(
    vne_node_mapper_unit
    node_mapper_unit.cpp
)

target_link_libraries(
    vne_node_mapper_unit
    PRIVATE
        vne_node_mapper
        Threads::Threads
)

add_test(
    NAME vne_node_mapper_unit
    COMMAND vne_node_mapper_unit
)

add_executable(
    vne_node_mapper_harness
    node_mapper_harness.cpp
)

target_link_libraries(
    vne_node_mapper_harness
    PRIVATE
        vne_node_mapper
        Threads::Threads
)

add_executable(
    vne_node_mapper_benchmark
    node_mapper_benchmark.cpp
)

target_link_libraries(
    vne_node_mapper_benchmark
    PRIVATE
        vne_node_mapper
        Threads::Threads
)

add_executable(
    vne_topology_analyzer_harness
    topology_analyzer_harness.cpp
)

target_link_libraries(
    vne_topology_analyzer_harness
    PRIVATE
        vne_topology_analyzer
        Threads::Threads
)

add_executable(
    vne_topology_analyzer_benchmark
    topology_analyzer_benchmark.cpp
)

target_link_libraries(
    vne_topology_analyzer_benchmark
    PRIVATE
        vne_topology_analyzer
        Threads::Threads
)

add_executable(
    vne_resource_updator_harness
    resource_updator_harness.cpp
)

target_link_libraries(
    vne_resource_updator_harness
    PRIVATE
        vne_resource_updator
        Threads::Threads
)

add_executable(
    vne_constraint_checker_harness
    constraint_checker_harness.cpp
)

target_link_libraries(
    vne_constraint_checker_harness
    PRIVATE
        vne_constraint_checker
        Threads::Threads
)

add_executable(
    vne_utils_config_unit
    utils_config_unit.cpp
)

target_link_libraries(
    vne_utils_config_unit
    PRIVATE
        vne_utils_config
        Threads::Threads
)

add_test(
    NAME vne_utils_config_unit
    COMMAND vne_utils_config_unit
)

add_executable(
    vne_solution_harness
    solution_harness.cpp
)

target_link_libraries(
    vne_solution_harness
    PRIVATE
        vne_solution
)

add_executable(
    vne_utils_config_harness
    utils_config_harness.cpp
)

target_link_libraries(
    vne_utils_config_harness
    PRIVATE
        vne_utils_config
)

add_executable(
    vne_dataset_generator_harness
    dataset_generator_harness.cpp
)

target_link_libraries(
    vne_dataset_generator_harness
    PRIVATE
        vne_dataset_generator
)

add_executable(
    vne_virtual_network_request_simulator_harness
    virtual_network_request_simulator_harness.cpp
)

target_link_libraries(
    vne_virtual_network_request_simulator_harness
    PRIVATE
        vne_virtual_network_request_simulator
)

add_executable(
    vne_virtual_network_request_simulator_io_harness
    virtual_network_request_simulator_io_harness.cpp
)

target_link_libraries(
    vne_virtual_network_request_simulator_io_harness
    PRIVATE
        vne_virtual_network_request_simulator
)

add_executable(
    vne_virtual_network_request_simulator_benchmark
    virtual_network_request_simulator_benchmark.cpp
)

target_link_libraries(
    vne_virtual_network_request_simulator_benchmark
    PRIVATE
        vne_virtual_network_request_simulator
)

add_executable(
    vne_virtual_network_event_harness
    virtual_network_event_harness.cpp
)

target_link_libraries(
    vne_virtual_network_event_harness
    PRIVATE
        vne_virtual_network_event
)

add_executable(
    vne_virtual_network_event_benchmark
    virtual_network_event_benchmark.cpp
)

target_link_libraries(
    vne_virtual_network_event_benchmark
    PRIVATE
        vne_virtual_network_event
)

add_executable(
    vne_physical_network_harness
    physical_network_harness.cpp
)

target_link_libraries(
    vne_physical_network_harness
    PRIVATE
        vne_physical_network
)

add_executable(
    vne_physical_network_benchmark
    physical_network_benchmark.cpp
)

target_link_libraries(
    vne_physical_network_benchmark
    PRIVATE
        vne_physical_network
)

add_executable(
    vne_virtual_network_harness
    virtual_network_harness.cpp
)

target_link_libraries(
    vne_virtual_network_harness
    PRIVATE
        vne_virtual_network
)

add_executable(
    vne_virtual_network_benchmark
    virtual_network_benchmark.cpp
)

target_link_libraries(
    vne_virtual_network_benchmark
    PRIVATE
        vne_virtual_network
)

add_executable(
    vne_base_network_harness
    base_network_harness.cpp
)

target_link_libraries(
    vne_base_network_harness
    PRIVATE
        vne_base_network
)

add_executable(
    vne_base_network_benchmark
    base_network_benchmark.cpp
)

target_link_libraries(
    vne_base_network_benchmark
    PRIVATE
        vne_base_network
)

add_executable(
    vne_link_mapper_unit
    link_mapper_unit.cpp
)

target_link_libraries(
    vne_link_mapper_unit
    PRIVATE
        vne_link_mapper
        Threads::Threads
)

add_test(
    NAME vne_link_mapper_unit
    COMMAND vne_link_mapper_unit
)

add_executable(
    vne_link_mapper_harness
    link_mapper_harness.cpp
)

target_link_libraries(
    vne_link_mapper_harness
    PRIVATE
        vne_link_mapper
        Threads::Threads
)

add_executable(
    vne_link_mapper_benchmark
    link_mapper_benchmark.cpp
)

target_link_libraries(
    vne_link_mapper_benchmark
    PRIVATE
        vne_link_mapper
        Threads::Threads
)

add_executable(
    vne_counter_unit
    counter_unit.cpp
)

target_link_libraries(
    vne_counter_unit
    PRIVATE
        vne_counter
        Threads::Threads
)

add_test(
    NAME vne_counter_unit
    COMMAND vne_counter_unit
)

add_executable(
    vne_counter_harness
    counter_harness.cpp
)

target_link_libraries(
    vne_counter_harness
    PRIVATE
        vne_counter
        Threads::Threads
)

add_executable(
    vne_counter_benchmark
    counter_benchmark.cpp
)

target_link_libraries(
    vne_counter_benchmark
    PRIVATE
        vne_counter
        Threads::Threads
)

add_executable(
    vne_recorder_unit
    recorder_unit.cpp
)

target_link_libraries(
    vne_recorder_unit
    PRIVATE
        vne_recorder
        Threads::Threads
)

add_test(
    NAME vne_recorder_unit
    COMMAND vne_recorder_unit
)

add_executable(
    vne_recorder_benchmark
    recorder_benchmark.cpp
)

target_link_libraries(
    vne_recorder_benchmark
    PRIVATE
        vne_recorder
        Threads::Threads
)

add_executable(
    vne_recorder_harness
    recorder_harness.cpp
)

target_link_libraries(
    vne_recorder_harness
    PRIVATE
        vne_recorder
        Threads::Threads
)

add_executable(
    vne_logger_unit
    logger_unit.cpp
)

target_link_libraries(
    vne_logger_unit
    PRIVATE
        vne_logger
        Threads::Threads
)

add_test(
    NAME vne_logger_unit
    COMMAND vne_logger_unit
)

add_executable(
    vne_logger_harness
    logger_harness.cpp
)

target_link_libraries(
    vne_logger_harness
    PRIVATE
        vne_logger
        Threads::Threads
)

add_executable(
    vne_logger_benchmark
    logger_benchmark.cpp
)

target_link_libraries(
    vne_logger_benchmark
    PRIVATE
        vne_logger
        Threads::Threads
)

add_executable(
    vne_controller_unit
    controller_unit.cpp
)

target_link_libraries(
    vne_controller_unit
    PRIVATE
        vne_controller
        Threads::Threads
)

add_test(
    NAME vne_controller_unit
    COMMAND vne_controller_unit
)

add_executable(
    vne_controller_harness
    controller_harness.cpp
)

target_link_libraries(
    vne_controller_harness
    PRIVATE
        vne_controller
        Threads::Threads
)

add_executable(
    vne_controller_benchmark
    controller_benchmark.cpp
)

target_link_libraries(
    vne_controller_benchmark
    PRIVATE
        vne_controller
        Threads::Threads
)

add_executable(
    vne_environment_unit
    environment_unit.cpp
)

target_link_libraries(
    vne_environment_unit
    PRIVATE
        vne_environment
        Threads::Threads
)

add_test(
    NAME vne_environment_unit
    COMMAND vne_environment_unit
)

add_executable(
    vne_environment_harness
    environment_harness.cpp
)

target_link_libraries(
    vne_environment_harness
    PRIVATE
        vne_environment
        Threads::Threads
)

add_executable(
    vne_environment_benchmark
    environment_benchmark.cpp
)

target_link_libraries(
    vne_environment_benchmark
    PRIVATE
        vne_environment
        Threads::Threads
)

add_executable(
    vne_link_rank_unit
    link_rank_unit.cpp
)

target_link_libraries(
    vne_link_rank_unit
    PRIVATE
        vne_link_rank
        Threads::Threads
)

add_test(
    NAME vne_link_rank_unit
    COMMAND vne_link_rank_unit
)

add_executable(
    vne_link_rank_harness
    link_rank_harness.cpp
)

target_link_libraries(
    vne_link_rank_harness
    PRIVATE
        vne_link_rank
        Threads::Threads
)

add_executable(
    vne_link_rank_benchmark
    link_rank_benchmark.cpp
)

target_link_libraries(
    vne_link_rank_benchmark
    PRIVATE
        vne_link_rank
        Threads::Threads
)

add_executable(
    vne_node_rank_unit
    node_rank_unit.cpp
)

target_link_libraries(
    vne_node_rank_unit
    PRIVATE
        vne_node_rank
        Threads::Threads
)

add_test(
    NAME vne_node_rank_unit
    COMMAND vne_node_rank_unit
)

add_executable(
    vne_python310_generic_timsort_probe
    python310_generic_timsort_probe.cpp
)

add_test(
    NAME vne_python310_generic_timsort_probe
    COMMAND vne_python310_generic_timsort_probe
)

add_executable(
    vne_node_rank_harness
    node_rank_harness.cpp
)

target_link_libraries(
    vne_node_rank_harness
    PRIVATE
        vne_node_rank
        Threads::Threads
)

add_executable(
    vne_node_rank_benchmark
    node_rank_benchmark.cpp
)

target_link_libraries(
    vne_node_rank_benchmark
    PRIVATE
        vne_node_rank
        Threads::Threads
)
