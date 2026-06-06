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

    rank/node_rank.cpp
    rank/link_rank.cpp
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
)