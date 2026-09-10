add_library(
    vne_learning_mlp
    mlp.cpp
)

add_library(
    vne_learning_attention
    attention.cpp
)

add_library(
    vne_learning_neural_tensor
    neural_tensor.cpp
)

add_library(
    vne_learning_gnn
    gnn.cpp
)

add_library(
    vne_learning_res
    res.cpp
)

add_library(
    vne_learning_sinkhorn
    sinkhorn.cpp
)

add_library(
    vne_learning_multipler
    multipler.cpp
)

add_library(
    vne_learning_loss
    loss.cpp
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

target_include_directories(
    vne_learning_neural_tensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_gnn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_learning_res
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

foreach(VIRNE_NEURAL_TARGET IN ITEMS
        vne_learning_sinkhorn
        vne_learning_multipler
        vne_learning_loss)
    target_include_directories(
        ${VIRNE_NEURAL_TARGET}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
endforeach()

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

target_link_libraries(
    vne_learning_neural_tensor
    PUBLIC
        virne_libtorch
)

target_link_libraries(
    vne_learning_gnn
    PUBLIC
        vne_learning_pyg_lite
        virne_libtorch
)

target_link_libraries(
    vne_learning_res
    PUBLIC
        virne_libtorch
)

foreach(VIRNE_NEURAL_TARGET IN ITEMS
        vne_learning_sinkhorn
        vne_learning_multipler
        vne_learning_loss)
    target_link_libraries(
        ${VIRNE_NEURAL_TARGET}
        PUBLIC
            virne_libtorch
    )
endforeach()

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

target_compile_features(
    vne_learning_neural_tensor
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_gnn
    PUBLIC
        cxx_std_17
)

target_compile_features(
    vne_learning_res
    PUBLIC
        cxx_std_17
)

foreach(VIRNE_NEURAL_TARGET IN ITEMS
        vne_learning_sinkhorn
        vne_learning_multipler
        vne_learning_loss)
    target_compile_features(
        ${VIRNE_NEURAL_TARGET}
        PUBLIC
            cxx_std_17
    )
endforeach()

add_library(vne_learning_neural_network_remaining INTERFACE)
target_link_libraries(
    vne_learning_neural_network_remaining
    INTERFACE
        vne_learning_res
        vne_learning_sinkhorn
        vne_learning_multipler
        vne_learning_loss
        vne_learning_gnn
)
