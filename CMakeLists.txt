cmake_minimum_required(
    VERSION 3.20
)

project(
    VirneCpp
)

#
# Compiler
#

set(
    CMAKE_CXX_STANDARD 17
)

set(
    CMAKE_CXX_STANDARD_REQUIRED ON
)

set(
    CMAKE_EXPORT_COMPILE_COMMANDS ON
)

#
# Third-party libraries
#

add_subdirectory(
    libs/yaml-cpp
)


#
# Project modules
#

add_subdirectory(
    config
)

add_subdirectory(
    graph
)

add_subdirectory(
    random
)

add_subdirectory(
    progress
)

#
# Shared dependencies
#

add_library(
    virne_common
    INTERFACE
)

target_include_directories(
    virne_common
    INTERFACE
        ${CMAKE_SOURCE_DIR}/libs/boost
)

target_link_libraries(
    virne_common
    INTERFACE
        config_lib
        graph_lib
        random_lib
        progress_lib
)

#
# Helper
#

function(
    virne_executable
    TARGET
    SOURCE
)

    add_executable(
        ${TARGET}
        ${SOURCE}
    )

    target_link_libraries(
        ${TARGET}
        PRIVATE
            virne_common
    )

endfunction()

#
# Executables
#

virne_executable(
    main
    main.cpp
)

virne_executable(
    graph_test
    graph_test.cpp
)

virne_executable(
    test_api
    test_api.cpp
)