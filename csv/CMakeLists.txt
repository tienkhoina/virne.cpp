add_library(csv_lib
    csv.cpp
)

target_include_directories(csv_lib
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/libs/tabulate/include
)