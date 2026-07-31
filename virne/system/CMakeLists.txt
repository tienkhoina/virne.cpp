add_library(vne_system)

target_sources(vne_system PRIVATE

    base_system.cpp
    online_system.cpp
    offline_system.cpp
    changeable_system.cpp
    time_window_system.cpp
    main_config.cpp
    main_runtime.cpp
)

target_include_directories(
    vne_system
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    vne_system
    PUBLIC
    vne_environment
    vne_solver
    config_lib
    vne_utils_setting
)
