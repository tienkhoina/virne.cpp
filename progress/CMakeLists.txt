add_library(
    progress_lib
    progress.cpp
)

target_include_directories(
    progress_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    progress_lib
    PUBLIC
)