add_library(vne_utils)

target_sources(vne_utils PRIVATE

    class_dict.cpp
    config.cpp
    dataset.cpp

    manager.cpp
    network.cpp

    setting.cpp
    stats.cpp

    virtualize.cpp
)

target_include_directories(
    vne_utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_utils
    PUBLIC
    virne_common
)