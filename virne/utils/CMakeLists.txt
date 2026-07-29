find_package(Threads REQUIRED)

# Leaf component target.  Keep it independent from the remaining Virne stubs
# so its unit/differential/performance gates prove the port in isolation.
add_library(vne_utils_network
    network.cpp
)

add_library(vne_utils_class_dict
    class_dict.cpp
)

add_library(vne_utils_setting
    setting.cpp
)

add_library(vne_utils_stats
    stats.cpp
)

add_library(vne_utils_manager
    manager.cpp
)

add_library(vne_utils_dataset_core
    dataset.cpp
)

add_library(vne_utils_dataset_rng
    dataset_rng.cpp
)

add_library(vne_utils_dataset_xml
    dataset_xml.cpp
)

add_library(vne_utils_config
    utils_config.cpp
)

target_include_directories(
    vne_utils_network
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_utils_class_dict
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_utils_setting
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_utils_stats
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_utils_manager
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_utils_dataset_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_utils_dataset_core
    PUBLIC
        Threads::Threads
)

target_include_directories(
    vne_utils_dataset_rng
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_utils_dataset_rng
    PUBLIC
        vne_utils_dataset_core
        random_lib
        Threads::Threads
)

target_include_directories(
    vne_utils_dataset_xml
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(
    vne_utils_config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_utils_dataset_xml
    PUBLIC
        vne_utils_dataset_core
        graph_lib
        Threads::Threads
    PRIVATE
        virne_boost
)

target_link_libraries(
    vne_utils_config
    PUBLIC
        config_lib
        random_lib
        vne_attribute_method
        vne_utils_dataset_core
        Threads::Threads
)

add_library(vne_utils_dataset INTERFACE)

target_link_libraries(
    vne_utils_dataset
    INTERFACE
        vne_utils_dataset_core
        vne_utils_dataset_rng
        vne_utils_dataset_xml
)

target_link_libraries(
    vne_utils_network
    PUBLIC
        graph_lib
        Threads::Threads
)

target_link_libraries(
    vne_utils_class_dict
    PUBLIC
        Threads::Threads
)

target_link_libraries(
    vne_utils_setting
    PUBLIC
        virne_boost
        yaml-cpp::yaml-cpp
        Threads::Threads
)

add_library(vne_utils)

target_sources(vne_utils PRIVATE

    config.cpp

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
        vne_utils_class_dict
        vne_utils_network
        vne_utils_setting
        vne_utils_stats
        vne_utils_manager
        vne_utils_dataset
        vne_utils_config
)
