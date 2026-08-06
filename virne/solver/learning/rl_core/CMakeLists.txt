add_library(
    vne_learning_tensor_convertor
    tensor_convertor.cpp
)

add_library(
    vne_learning_feature_constructor
    feature_constructor.cpp
)

add_library(
    vne_learning_reward_calculator
    reward_calculator.cpp
)

add_library(
    vne_learning_rollout_returns
    rollout_returns.cpp
)

add_library(
    vne_learning_rollout_buffer
    rollout_buffer.cpp
)

add_library(
    vne_learning_action_mask
    action_mask.cpp
)

add_library(
    vne_learning_running_mean_std
    running_mean_std.cpp
)

add_library(
    vne_learning_observation_sampler
    observation_sampler.cpp
)

add_library(
    vne_learning_graph_observation_sampler
    graph_observation_sampler.cpp
)

add_library(
    vne_learning_policy_action
    policy_action.cpp
)

add_library(
    vne_learning_policy_config
    policy_config.cpp
)

add_library(
    vne_learning_mlp_policy_builder
    mlp_policy_builder.cpp
)

add_library(
    vne_learning_mlp_inference
    mlp_inference.cpp
)

target_include_directories(
    vne_learning_tensor_convertor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_feature_constructor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_reward_calculator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_rollout_returns
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_rollout_buffer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_action_mask
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_running_mean_std
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_observation_sampler
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_graph_observation_sampler
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_policy_action
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_policy_config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_mlp_policy_builder
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_mlp_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_learning_tensor_convertor
    PUBLIC
        virne_libtorch
)

target_link_libraries(
    vne_learning_feature_constructor
    PUBLIC
        vne_learning_obs_handler
        vne_core
        vne_physical_network
        vne_virtual_network
        vne_topological_metric_calculator
)

target_link_libraries(
    vne_learning_reward_calculator
    PUBLIC
        vne_solution
        vne_physical_network
        vne_virtual_network
)

target_compile_features(
    vne_learning_tensor_convertor
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_feature_constructor
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_reward_calculator
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_rollout_returns
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_rollout_buffer
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_action_mask
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_running_mean_std
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_observation_sampler
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_graph_observation_sampler
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_policy_action
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_policy_config
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_mlp_policy_builder
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_mlp_inference
    PUBLIC
        cxx_std_17
)

target_link_libraries(
    vne_learning_rollout_buffer
    PUBLIC
        vne_learning_rollout_returns
)

target_link_libraries(
    vne_learning_action_mask
    PUBLIC
        virne_libtorch
)

target_link_libraries(
    vne_learning_observation_sampler
    PUBLIC
        vne_learning_tensor_convertor
        vne_learning_pyg_lite
)

target_link_libraries(
    vne_learning_graph_observation_sampler
    PUBLIC
        vne_learning_tensor_convertor
        vne_learning_pyg_lite
)

target_link_libraries(
    vne_learning_policy_action
    PUBLIC
        vne_learning_action_mask
        virne_libtorch
)

target_link_libraries(
    vne_learning_mlp_policy_builder
    PUBLIC
        vne_learning_policy_config
        vne_learning_mlp_policy
)

target_link_libraries(
    vne_learning_mlp_inference
    PUBLIC
        vne_learning_mlp_policy_builder
        vne_learning_tensor_convertor
        vne_learning_policy_action
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vne_learning_rollout_returns PRIVATE -ffp-contract=off)
    target_compile_options(vne_learning_rollout_buffer PRIVATE -ffp-contract=off)
    target_compile_options(vne_learning_running_mean_std PRIVATE -ffp-contract=off)
endif()
