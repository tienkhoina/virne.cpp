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

add_library(vne_learning_cnn_policy cnn_policy.cpp)
target_link_libraries(vne_learning_cnn_policy PUBLIC virne_libtorch)
target_compile_features(vne_learning_cnn_policy PUBLIC cxx_std_17)

add_library(vne_learning_gnn_mlp_policy gnn_mlp_policy.cpp)
target_link_libraries(vne_learning_gnn_mlp_policy PUBLIC vne_learning_gnn vne_learning_mlp)
target_compile_features(vne_learning_gnn_mlp_policy PUBLIC cxx_std_17)

foreach(target IN ITEMS vne_learning_cnn_policy vne_learning_gnn_mlp_policy)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic;-Werror>)
endforeach()

add_library(vne_learning_policy_registry policy_registry.cpp)
target_link_libraries(vne_learning_policy_registry PUBLIC vne_learning_mlp_policy
    vne_learning_cnn_policy vne_learning_gnn_mlp_policy vne_learning_att_policy
    vne_learning_dual_policy vne_learning_seq2seq_policy)
target_compile_features(vne_learning_policy_registry PUBLIC cxx_std_17)
target_compile_options(vne_learning_policy_registry PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic;-Werror>)

add_library(vne_learning_graph_module graph_module.cpp)
target_link_libraries(vne_learning_graph_module PUBLIC vne_learning_gnn)
add_library(vne_learning_att_policy att_policy.cpp)
target_link_libraries(vne_learning_att_policy PUBLIC vne_learning_mlp)
add_library(vne_learning_dual_policy dual_gnn_policy.cpp)
target_link_libraries(vne_learning_dual_policy PUBLIC vne_learning_graph_module)
add_library(vne_learning_seq2seq_policy seq2seq_policy.cpp)
target_link_libraries(vne_learning_seq2seq_policy PUBLIC vne_learning_graph_module)
foreach(target IN ITEMS vne_learning_graph_module vne_learning_att_policy
        vne_learning_dual_policy vne_learning_seq2seq_policy)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic;-Werror>)
endforeach()
