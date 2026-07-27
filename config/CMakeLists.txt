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

add_executable(
    config_hydra_test
    tests/config_hydra_test.cpp
)

target_link_libraries(
    config_hydra_test
    PRIVATE
    config_lib
)

target_compile_definitions(
    config_hydra_test
    PRIVATE
    VIRNE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_test(
    NAME config_hydra_compat
    COMMAND config_hydra_test
)

set_tests_properties(
    config_hydra_compat
    PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
