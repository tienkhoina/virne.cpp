file(
    GLOB CONFIG_SRCS
    CONFIGURE_DEPENDS
    *.cpp
)

add_library(
    config_lib
    ${CONFIG_SRCS}
)

target_include_directories(
    config_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(
    config_lib
    PUBLIC
    yaml-cpp
)