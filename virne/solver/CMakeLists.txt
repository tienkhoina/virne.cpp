find_package(Threads REQUIRED)

add_library(vne_link_rank
    rank/link_rank.cpp
)

target_include_directories(
    vne_link_rank
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_link_rank
    PUBLIC
    vne_base_network
    Threads::Threads
)

add_library(vne_node_rank
    rank/node_rank.cpp
)

target_include_directories(
    vne_node_rank
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_node_rank
    PUBLIC
    vne_base_network
    random_lib
    Threads::Threads
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vne_node_rank PRIVATE -ffp-contract=off)
endif()

add_library(vne_base_solver
    base_solver.cpp
)

target_include_directories(
    vne_base_solver
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_base_solver
    PUBLIC
    vne_solution
    vne_utils_config
    Threads::Threads
)

add_library(vne_exact_solver
    exact/exact_solver.cpp
    exact/exact_with_risk.cpp
)

target_include_directories(
    vne_exact_solver
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_exact_solver
    PUBLIC
    vne_base_solver
    vne_controller
    vne_physical_network
    virne_ortools
    random_lib
    Threads::Threads
)

# The official OR-Tools Windows payload uses C++20 for both MSVC and MinGW;
# Linux/GCC retains the C++17 contract. Keep this conditional local to the
# exact leaf so the frozen modules do not inherit a language-level change.
if(WIN32)
    target_compile_features(vne_exact_solver PRIVATE cxx_std_20)
else()
    target_compile_features(vne_exact_solver PRIVATE cxx_std_17)
endif()

add_library(vne_heuristic_node_rank
    heuristic/node_rank.cpp
)

target_include_directories(
    vne_heuristic_node_rank
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_heuristic_node_rank
    PUBLIC
    vne_base_solver
    vne_controller
    vne_node_rank
    vne_physical_network
    Threads::Threads
)

add_library(vne_heuristic_ffd_rank
    heuristic/ffd_rank.cpp
)

target_include_directories(
    vne_heuristic_ffd_rank
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_heuristic_ffd_rank
    PUBLIC
    vne_heuristic_node_rank
)

add_library(vne_heuristic_standard_rank_variants
    heuristic/standard_rank_variants.cpp
)

target_include_directories(
    vne_heuristic_standard_rank_variants
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_heuristic_standard_rank_variants
    PUBLIC
    vne_heuristic_node_rank
)

add_library(vne_heuristic_random_rank
    heuristic/random_rank.cpp
)

target_include_directories(
    vne_heuristic_random_rank
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_heuristic_random_rank
    PUBLIC
    vne_heuristic_node_rank
)

add_library(vne_heuristic_custom_rank_variants
    heuristic/custom_rank_variants.cpp
    heuristic/python_int_set_order.cpp
)

target_include_directories(
    vne_heuristic_custom_rank_variants
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_heuristic_custom_rank_variants
    PUBLIC
    vne_heuristic_node_rank
    vne_constraint_checker
)

add_library(vne_meta_heuristic
    meta_heuristic/meta_heuristic.cpp
)

target_include_directories(
    vne_meta_heuristic
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_meta_heuristic
    PUBLIC
        vne_base_solver
        vne_controller
        vne_physical_network
        vne_virtual_network
        random_lib
        Threads::Threads
)

if(MSVC)
    target_compile_features(vne_meta_heuristic PRIVATE cxx_std_20)
else()
    target_compile_features(vne_meta_heuristic PRIVATE cxx_std_17)
endif()

add_library(vne_solver)

target_sources(vne_solver PRIVATE

    debug_obs.cpp

    heuristic/bfs_trials.cpp
    heuristic/ego_network.cpp
    heuristic/fit.cpp
    heuristic/joint_pr.cpp
    heuristic/registry.cpp

    heuristic/nea_bc.cpp
    heuristic/nea_par.cpp

    heuristic/nrm_bc_env.cpp
    heuristic/nrm_par_solver.cpp

)

target_include_directories(
    vne_solver
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_solver
    PUBLIC
        vne_network
        vne_base_solver
        vne_exact_solver
        vne_meta_heuristic
        vne_heuristic_custom_rank_variants
    vne_heuristic_ffd_rank
    vne_heuristic_node_rank
    vne_heuristic_random_rank
    vne_heuristic_standard_rank_variants
    vne_link_rank
    vne_node_rank
)
