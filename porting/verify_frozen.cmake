cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

function(verify_frozen_directory relative_directory expected_count expected_sha256)
    set(root "${SOURCE_ROOT}/${relative_directory}")
    if(NOT IS_DIRECTORY "${root}")
        message(FATAL_ERROR "Frozen directory is missing: ${root}")
    endif()

    file(GLOB_RECURSE files LIST_DIRECTORIES false "${root}/*")
    list(FILTER files EXCLUDE REGEX "/\\.git/")
    list(FILTER files EXCLUDE REGEX "/__pycache__/")
    list(SORT files)

    set(records "")
    foreach(path IN LISTS files)
        file(RELATIVE_PATH relative_path "${root}" "${path}")
        string(REPLACE "\\" "/" relative_path "${relative_path}")
        file(SHA256 "${path}" file_sha256)
        string(TOLOWER "${file_sha256}" file_sha256)
        string(APPEND records "${relative_path}\t${file_sha256}\n")
    endforeach()

    list(LENGTH files actual_count)
    string(SHA256 actual_sha256 "${records}")
    if(NOT actual_count EQUAL expected_count OR
       NOT actual_sha256 STREQUAL expected_sha256)
        if(REPORT_ONLY)
            message(WARNING
                "Frozen component ${relative_directory} differs. "
                "Expected count/hash ${expected_count}/${expected_sha256}; "
                "actual ${actual_count}/${actual_sha256}")
        else()
            message(FATAL_ERROR
                "Frozen component ${relative_directory} changed. "
                "Expected count/hash ${expected_count}/${expected_sha256}; "
                "actual ${actual_count}/${actual_sha256}")
        endif()
    endif()
    message(STATUS
        "Frozen component ${relative_directory}: ${actual_count} files, "
        "sha256=${actual_sha256}")
endfunction()

verify_frozen_directory(
    graph
    69
    32e9ab7d4a6e89c9443f433469c374c933bb3568a348c0a284448f7651248459)
verify_frozen_directory(
    csv
    8
    170a68f055aece99b45a41ee4039ca282ef22ddbf5376cabbc34ab754d67a530)
verify_frozen_directory(
    config
    12
    d9881b1c59a9e9a32c3599e12d8630e373079d667552d93efe25f3fe0c7a72a3)
verify_frozen_directory(
    libs/yaml-cpp
    397
    8ef1f48c64160474b818ac4644f19bc6b2b204a5f74c07c13389448d951bda8a)
