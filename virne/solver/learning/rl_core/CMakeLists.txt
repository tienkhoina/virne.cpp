add_library(
    vne_learning_tensor_convertor
    tensor_convertor.cpp
)

add_library(
    vne_learning_feature_constructor
    feature_constructor.cpp
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
