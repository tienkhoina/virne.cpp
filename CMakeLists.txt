cmake_minimum_required(VERSION 3.20)

project(VirneCpp)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(libs/yaml-cpp)
add_subdirectory(config)

#
# Boost headers
#
include_directories(
    ${CMAKE_SOURCE_DIR}/libs/boost
)

add_executable(
    main
    main.cpp
)

target_link_libraries(
    main
    PRIVATE
    config_lib
)

add_executable(
    graph_test
    graph_test.cpp
)

target_link_libraries(
    graph_test
    PRIVATE
    config_lib
)