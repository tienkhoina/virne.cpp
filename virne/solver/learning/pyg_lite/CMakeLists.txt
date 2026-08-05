add_library(
    vne_learning_pyg_lite
    pyg_lite.cpp
    message_passing.cpp
    hetero_data.cpp
    bipartite_gat.cpp
    hetero_conv.cpp
)

target_include_directories(
    vne_learning_pyg_lite
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_learning_pyg_lite
    PUBLIC
        virne_libtorch
)

target_compile_features(
    vne_learning_pyg_lite
    PUBLIC
        cxx_std_17
)
