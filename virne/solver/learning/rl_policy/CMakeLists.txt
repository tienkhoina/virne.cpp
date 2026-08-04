add_library(
    vne_learning_mlp_policy
    mlp_policy.cpp
)

target_include_directories(
    vne_learning_mlp_policy
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_learning_mlp_policy
    PUBLIC
        vne_learning_mlp
)

target_compile_features(
    vne_learning_mlp_policy
    PUBLIC
        cxx_std_17
)
