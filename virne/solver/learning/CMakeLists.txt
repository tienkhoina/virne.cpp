add_library(
    vne_learning_obs_handler
    obs_handler.cpp
)

target_include_directories(
    vne_learning_obs_handler
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_learning_obs_handler
    PUBLIC
        vne_core
        vne_physical_network
        vne_virtual_network
        vne_topological_metric_calculator
)

target_compile_features(
    vne_learning_obs_handler
    PUBLIC
        cxx_std_17
)

add_subdirectory(pyg_lite)
add_subdirectory(neural_network)
add_subdirectory(rl_core)
add_subdirectory(rl_policy)

add_library(vne_learning_solver_registry solver_registry.cpp online_learning.cpp)
target_link_libraries(vne_learning_solver_registry PUBLIC vne_base_solver
    PRIVATE vne_learning_session vne_learning_instance_environment vne_learning_online_environment vne_node_rank vne_controller vne_counter)
target_compile_features(vne_learning_solver_registry PUBLIC cxx_std_17)
target_include_directories(vne_learning_solver_registry SYSTEM PRIVATE ${VIRNE_LIB_DIR}/boost)
target_compile_options(vne_learning_solver_registry PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic;-Werror;-ffp-contract=off>)
