add_library(
    random_lib
    py_random.cpp
)

target_include_directories(
    random_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)