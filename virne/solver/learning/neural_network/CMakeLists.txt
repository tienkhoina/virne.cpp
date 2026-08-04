add_library(
    vne_learning_mlp
    mlp.cpp
)

target_include_directories(
    vne_learning_mlp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_learning_mlp
    PUBLIC
        virne_libtorch
)

target_compile_features(
    vne_learning_mlp
    PUBLIC
        cxx_std_17
)
