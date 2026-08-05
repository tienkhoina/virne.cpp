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

target_link_libraries(
    vne_learning_rollout_buffer
    PUBLIC
        vne_learning_rollout_returns
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vne_learning_rollout_returns PRIVATE -ffp-contract=off)
    target_compile_options(vne_learning_rollout_buffer PRIVATE -ffp-contract=off)
endif()
