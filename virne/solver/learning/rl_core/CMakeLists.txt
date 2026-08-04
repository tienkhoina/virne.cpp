add_library(
    vne_learning_tensor_convertor
    tensor_convertor.cpp
)

target_include_directories(
    vne_learning_tensor_convertor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_learning_tensor_convertor
    PUBLIC
        virne_libtorch
)

target_compile_features(
    vne_learning_tensor_convertor
    PUBLIC
        cxx_std_17
)
