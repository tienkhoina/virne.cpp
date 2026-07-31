cmake_minimum_required(
    VERSION 3.20
)

project(
    VirneCpp
)

enable_testing()

#
# Compiler
#

set(
    CMAKE_CXX_STANDARD 17
)

set(
    CMAKE_CXX_STANDARD_REQUIRED ON
)

set(
    CMAKE_EXPORT_COMPILE_COMMANDS ON
)

#
# Third-party libraries
#

set(
    VIRNE_LIB_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

# All third-party dependencies are deliberately resolved from this repository.
# Do not use find_package() here: a clean machine must be able to configure and
# build VirneCpp without installing packages into the system or Conda prefix.

add_library(
    virne_boost
    INTERFACE
)

target_include_directories(
    virne_boost
    INTERFACE
        ${VIRNE_LIB_DIR}/boost
)

add_library(
    virne_tabulate
    INTERFACE
)

target_include_directories(
    virne_tabulate
    INTERFACE
        ${VIRNE_LIB_DIR}/tabulate/include
)

# yaml-cpp is built as a private static dependency from its vendored source.
# Disable optional tests/tools/contrib so configuring this project never pulls
# another dependency or creates unrelated build targets.
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_FORMAT_SOURCE OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(
    libs/yaml-cpp
)


#
# Project modules
#

add_subdirectory(
    config
)

add_subdirectory(
    graph
)

add_subdirectory(
    random
)

add_subdirectory(
    progress
)

add_subdirectory(
    csv
)

#
# Shared dependencies
#

add_library(
    virne_common
    INTERFACE
)

target_include_directories(
    virne_common
    INTERFACE
        ${VIRNE_LIB_DIR}/boost
)

target_link_libraries(
    virne_common
    INTERFACE
        config_lib
        graph_lib
        random_lib
        progress_lib
        csv_lib
        virne_boost
)

add_subdirectory(virne/core)
add_subdirectory(virne/network)
add_subdirectory(virne/solver)
add_subdirectory(virne/system)
add_subdirectory(virne/utils)

# Cross-language port verification targets live outside the frozen graph,
# config/YAML and CSV modules.
add_subdirectory(porting)

#
# Helper
#

function(
    virne_executable
    TARGET
    SOURCE
)

    add_executable(
        ${TARGET}
        ${SOURCE}
    )

    target_link_libraries(
        ${TARGET}
        PRIVATE
            virne_common
    )

endfunction()

#
# Executables
#

virne_executable(
    main
    main.cpp
)

target_link_libraries(
    main
    PRIVATE
        vne_system
)

target_compile_definitions(
    main
    PRIVATE
        VIRNE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME vne_main_multi_resource_smoke
    COMMAND main
        --config ${CMAKE_CURRENT_SOURCE_DIR}/setting/main.yaml
        solver.solver_name=ffd_rank
        +p_net_setting=p_net_setting_multi_resource
        +v_sim_setting=v_sim_setting_multi_resource
        p_net_setting.topology.num_nodes=20
        v_sim_setting.num_v_nets=8
        v_sim_setting.v_net_size.low=2
        v_sim_setting.v_net_size.high=4
        experiment.seed=0
        experiment.run_id=multi-resource-smoke
        experiment.if_save_config=false
        experiment.if_save_p_net=false
        experiment.if_save_v_nets=false
        recorder.if_save_records=false
        recorder.if_temp_save_records=false
        logger.backends=[]
        ++native.capture_solutions=false
        ++native.output.report=summary
        ++native.progress.enabled=false
)

set_tests_properties(
    vne_main_multi_resource_smoke
    PROPERTIES
        PASS_REGULAR_EXPRESSION "\"accepted\":4,\"rejected\":4"
)

add_test(
    NAME vne_main_config_log_smoke
    COMMAND main
        --config ${CMAKE_CURRENT_SOURCE_DIR}/setting/main.yaml
        solver.solver_name=order_rank
        +p_net_setting=p_net_setting_multi_resource
        +v_sim_setting=v_sim_setting_multi_resource
        p_net_setting.topology.num_nodes=8
        v_sim_setting.num_v_nets=1
        v_sim_setting.v_net_size.low=2
        v_sim_setting.v_net_size.high=2
        experiment.seed=0
        experiment.run_id=config-log-smoke
        experiment.if_save_config=false
        experiment.if_save_p_net=false
        experiment.if_save_v_nets=false
        recorder.if_save_records=false
        recorder.if_temp_save_records=false
        logger.level=INFO
        logger.backends=[console]
        ++native.output.report=none
        ++native.progress.enabled=false
)

set_tests_properties(
    vne_main_config_log_smoke
    PROPERTIES
        PASS_REGULAR_EXPRESSION
            "reciprocal: true;simulation:;Native config:"
)

virne_executable(
    graph_test
    graph_test.cpp
)

virne_executable(
    test_api
    test_api.cpp
)

virne_executable(
    test_bfs_full
    test_bfs_full.cpp
)

virne_executable(
    test_digraph
    test_digraph.cpp
)

virne_executable(
    test_config
    test_config.cpp
)

target_compile_definitions(
    test_config
    PRIVATE
        VIRNE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(
    NAME digraph_public_api
    COMMAND test_digraph
)

add_test(
    NAME config_setting_matrix
    COMMAND test_config
)

add_test(
    NAME random_python_compat
    COMMAND test_random
)

add_test(
    NAME progress_behavior
    COMMAND progress_test
)

# Deterministic DiGraph parity/benchmark driver.  The Python NetworkX oracle
# lives in benchmarks/ and is intentionally not installed or linked into the
# production targets.
virne_executable(
    graph_nx_harness
    benchmarks/graph_nx_harness.cpp
)

virne_executable(
    gml_harness
    benchmarks/gml_harness.cpp
)

add_executable(
    csv_test
    csv_test.cpp
)

target_link_libraries(
    csv_test
    PRIVATE
        csv_lib
)
