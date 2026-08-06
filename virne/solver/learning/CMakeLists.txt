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
