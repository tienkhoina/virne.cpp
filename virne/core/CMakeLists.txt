add_library(vne_core)

target_sources(vne_core PRIVATE
    counter.cpp
    environment.cpp
    logger.cpp
    recorder.cpp
    solution.cpp

    controller/controller.cpp
    controller/node_mapper.cpp
    controller/link_mapper.cpp
    controller/constraint_checker.cpp
    controller/resource_updator.cpp
    controller/topology_analyzer.cpp
)

target_include_directories(
    vne_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_core
    PUBLIC
    virne_common
)