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

# Windows SDK headers expose min/max macros unless NOMINMAX is set. They
# corrupt standard-library calls in otherwise portable frozen leaves when the
# native OR-Tools ABI is built with clang-cl/MSVC.
if(WIN32)
    add_compile_definitions(NOMINMAX BOOST_ALL_NO_LIB)
endif()

#
# Third-party libraries
#

set(
    VIRNE_LIB_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

# LibTorch is an opt-in binary dependency.  The frozen non-RL leaves do not
# link it, so a normal build remains independent of Torch and keeps the
# existing ABI.  Enabling this option builds the isolated probe and exposes
# the local `virne_libtorch` interface for future ML/RL leaves.  The Torch
# package config is the one shipped by the vendored archive; no system/Conda
# Torch is accepted.
option(
    VIRNE_ENABLE_LIBTORCH
    "Build the local LibTorch probe and expose virne_libtorch"
    OFF
)

set(
    VIRNE_LIBTORCH_ROOT
    ${VIRNE_LIB_DIR}/libtorch
    CACHE PATH
    "Vendored LibTorch installation root"
)

if(VIRNE_ENABLE_LIBTORCH)
    if(NOT EXISTS
        "${VIRNE_LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
        message(FATAL_ERROR
            "Missing vendored LibTorch package config: "
            "${VIRNE_LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
    endif()

    # LibTorch supplies its own Caffe2/Torch imported targets and ABI flags.
    # Restrict the search to the workspace so a host Torch cannot silently
    # change the binary ABI or numerical backend.
    find_package(
        Torch
        CONFIG
        REQUIRED
        PATHS "${VIRNE_LIBTORCH_ROOT}"
        NO_DEFAULT_PATH
    )

    if(NOT TARGET torch)
        message(FATAL_ERROR "Vendored LibTorch did not define target torch")
    endif()

    add_library(
        virne_libtorch
        INTERFACE
    )

    target_link_libraries(
        virne_libtorch
        INTERFACE
            torch
    )

    # TorchConfig already carries the old/new libstdc++ ABI flag.  Keep the
    # include and library directories explicit as well so the runtime loader
    # finds the sibling LibTorch DSOs without LD_LIBRARY_PATH.
    target_include_directories(
        virne_libtorch
        INTERFACE
            ${TORCH_INCLUDE_DIRS}
    )
    target_link_directories(
        virne_libtorch
        INTERFACE
            ${VIRNE_LIBTORCH_ROOT}/lib
    )

    file(
        STRINGS
        "${VIRNE_LIBTORCH_ROOT}/build-version"
        VIRNE_LIBTORCH_VERSION
        LIMIT_COUNT 1
    )
    if(NOT VIRNE_LIBTORCH_VERSION)
        set(VIRNE_LIBTORCH_VERSION "unknown")
    endif()

    add_executable(
        vne_libtorch_probe
        porting/libtorch_probe.cpp
    )
    target_link_libraries(
        vne_libtorch_probe
        PRIVATE
            virne_libtorch
    )
    target_compile_definitions(
        vne_libtorch_probe
        PRIVATE
            "VIRNE_LIBTORCH_VERSION=\"${VIRNE_LIBTORCH_VERSION}\""
    )
    if(UNIX)
        set_property(
            TARGET vne_libtorch_probe
            APPEND
            PROPERTY BUILD_RPATH "${VIRNE_LIBTORCH_ROOT}/lib"
        )
        set_property(
            TARGET vne_libtorch_probe
            APPEND
            PROPERTY INSTALL_RPATH "${VIRNE_LIBTORCH_ROOT}/lib"
        )
    endif()

    add_test(
        NAME vne_libtorch_probe
        COMMAND vne_libtorch_probe --threads 1 --device cpu
    )
endif()

# OR-Tools is consumed through the official prebuilt C++ distribution pinned
# in DEPENDENCIES.md. Keep Linux (Docker/GCC) and Windows (MSVC-compatible
# clang-cl) payloads inside the workspace; no system/Conda solver is accepted.
if(WIN32)
    set(VIRNE_ORTOOLS_ROOT ${VIRNE_LIB_DIR}/ortools-win)
    set(VIRNE_ORTOOLS_IMPLIB ${VIRNE_ORTOOLS_ROOT}/lib/ortools.lib)
    set(VIRNE_ORTOOLS_RUNTIME ${VIRNE_ORTOOLS_ROOT}/bin/ortools.dll)
    if(NOT EXISTS ${VIRNE_ORTOOLS_IMPLIB})
        message(FATAL_ERROR
            "Missing vendored OR-Tools Windows C++ payload: ${VIRNE_ORTOOLS_IMPLIB}")
    endif()
    add_library(virne_ortools SHARED IMPORTED GLOBAL)
    set_target_properties(virne_ortools PROPERTIES
        IMPORTED_IMPLIB "${VIRNE_ORTOOLS_IMPLIB}"
        IMPORTED_LOCATION "${VIRNE_ORTOOLS_RUNTIME}"
        INTERFACE_INCLUDE_DIRECTORIES "${VIRNE_ORTOOLS_ROOT}/include"
        INTERFACE_COMPILE_DEFINITIONS
            "OR_BUILD_DLL;OR_PROTO_DLL=__declspec(dllimport);USE_MATH_OPT;USE_BOP;USE_CBC;USE_CLP;USE_GLOP;USE_HIGHS;USE_PDLP;USE_SCIP;__WIN32__;PROTOBUF_USE_DLLS;ABSL_CONSUME_DLL")
    file(GLOB VIRNE_ORTOOLS_WINDOWS_LIBS
        "${VIRNE_ORTOOLS_ROOT}/lib/*.lib")
    list(REMOVE_ITEM VIRNE_ORTOOLS_WINDOWS_LIBS "${VIRNE_ORTOOLS_IMPLIB}")
    target_link_libraries(virne_ortools INTERFACE ${VIRNE_ORTOOLS_WINDOWS_LIBS})
else()
    set(VIRNE_ORTOOLS_ROOT ${VIRNE_LIB_DIR}/ortools)
    set(VIRNE_ORTOOLS_LIBRARY
        ${VIRNE_ORTOOLS_ROOT}/lib/libortools.so.9.15.6755)
    if(NOT EXISTS ${VIRNE_ORTOOLS_LIBRARY})
        message(FATAL_ERROR
            "Missing vendored OR-Tools Linux C++ payload: ${VIRNE_ORTOOLS_LIBRARY}")
    endif()
    add_library(virne_ortools SHARED IMPORTED GLOBAL)
    set_target_properties(virne_ortools PROPERTIES
        IMPORTED_LOCATION "${VIRNE_ORTOOLS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${VIRNE_ORTOOLS_ROOT}/include"
        INTERFACE_COMPILE_DEFINITIONS
            "OR_PROTO_DLL=;USE_MATH_OPT;USE_BOP;USE_CBC;USE_CLP;USE_GLOP;USE_HIGHS;USE_PDLP;USE_SCIP"
        INTERFACE_LINK_DIRECTORIES "${VIRNE_ORTOOLS_ROOT}/lib"
        BUILD_RPATH "${VIRNE_ORTOOLS_ROOT}/lib"
        INSTALL_RPATH "${VIRNE_ORTOOLS_ROOT}/lib")
    file(GLOB VIRNE_ORTOOLS_LINUX_LIBS
        "${VIRNE_ORTOOLS_ROOT}/lib/*.so.*")
    list(REMOVE_ITEM VIRNE_ORTOOLS_LINUX_LIBS "${VIRNE_ORTOOLS_LIBRARY}")
    target_link_libraries(virne_ortools INTERFACE ${VIRNE_ORTOOLS_LINUX_LIBS})
endif()

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

if(WIN32)
    add_custom_command(
        TARGET main
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${VIRNE_ORTOOLS_ROOT}/bin"
            "$<TARGET_FILE_DIR:main>"
        VERBATIM
    )
else()
    set_property(
        TARGET main
        APPEND PROPERTY BUILD_RPATH "${VIRNE_ORTOOLS_ROOT}/lib"
    )
endif()

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
    NAME vne_main_exact_with_risk_multi_resource_smoke
    COMMAND main
        --config ${CMAKE_CURRENT_SOURCE_DIR}/setting/main.yaml
        solver.solver_name=exact_with_risk
        +p_net_setting=p_net_setting_multi_resource
        +v_sim_setting=v_sim_setting_multi_resource
        p_net_setting.topology.num_nodes=12
        v_sim_setting.num_v_nets=3
        v_sim_setting.v_net_size.low=2
        v_sim_setting.v_net_size.high=3
        experiment.seed=0
        experiment.run_id=exact-risk-multi-resource-smoke
        experiment.if_save_config=false
        experiment.if_save_p_net=false
        experiment.if_save_v_nets=false
        recorder.if_save_records=false
        recorder.if_temp_save_records=false
        logger.backends=[]
        ++native.capture_solutions=false
        ++native.output.report=summary
        ++native.progress.enabled=false
        ++native.exact.time_limit_ms=5000
        ++native.exact.search_node_limit=100000
        ++native.workers.exact=2
        ++native.exact_risk.scarcity_weight=1
        ++native.exact_risk.balance_weight=1
        ++native.exact_risk.criticality_weight=1
)

set_tests_properties(
    vne_main_exact_with_risk_multi_resource_smoke
    PROPERTIES
        PASS_REGULAR_EXPRESSION
            "\"solver\":\"exact_with_risk\".*\"processed_requests\":3.*\"accepted\":3"
        TIMEOUT 30
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
