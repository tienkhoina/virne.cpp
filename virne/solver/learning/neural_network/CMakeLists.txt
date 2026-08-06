add_library(
    vne_learning_mlp
    mlp.cpp
)

add_library(
    vne_learning_attention
    attention.cpp
)

target_include_directories(
    vne_learning_mlp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_attention
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_learning_mlp
    PUBLIC
        virne_libtorch
)

target_link_libraries(
    vne_learning_attention
    PUBLIC
        virne_libtorch
)

target_compile_features(
    vne_learning_mlp
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_attention
    PUBLIC
        cxx_std_17
)
