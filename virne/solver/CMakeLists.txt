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

add_library(vne_solver)

target_sources(vne_solver PRIVATE

    base_solver.cpp
    debug_obs.cpp

    heuristic/bfs_trials.cpp
    heuristic/ego_network.cpp
    heuristic/fit.cpp
    heuristic/joint_pr.cpp

    heuristic/nea_bc.cpp
    heuristic/nea_par.cpp

    heuristic/node_rank.cpp

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
    vne_link_rank
    vne_node_rank
)
