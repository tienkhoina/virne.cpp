#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <system_error>

namespace virne::utils
{

enum class ManagerErrorCode : std::uint8_t
{
    legacy_temp_join_type_error,
    enumeration_failed,
    not_directory,
    unsafe_path_escape,
    remove_failed,
    output_failed,
};

enum class ManagerOperation : std::uint8_t
{
    list_temp_root,
    list_save_root,
    list_algorithm,
    list_records,
    remove_run_tree,
    remove_empty_directory,
    emit_delete_line,
};

class ManagerException : public std::runtime_error
{
public:
    ManagerException(
        ManagerErrorCode code,
        ManagerOperation operation,
        std::filesystem::path path,
        std::error_code system_error,
        std::string message);

    ManagerErrorCode code() const noexcept;
    ManagerOperation operation() const noexcept;
    const std::filesystem::path& path() const noexcept;
    const std::error_code& system_error() const noexcept;

private:
    ManagerErrorCode code_;
    ManagerOperation operation_;
    std::filesystem::path path_;
    std::error_code system_error_;
};

struct EmptyDirectoryConfig
{
    std::filesystem::path record_dir;
    std::filesystem::path log_dir;
    std::filesystem::path save_dir;
};

void delete_temp_files(const std::filesystem::path& directory);

void clean_save_dir(const std::filesystem::path& directory);
void clean_save_dir(
    const std::filesystem::path& directory,
    std::ostream& output);

void delete_empty_dir(const EmptyDirectoryConfig& config);

} // namespace virne::utils
