#include "manager.h"

#include <array>
#include <cerrno>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace virne::utils
{
namespace
{

using Path = std::filesystem::path;

#if defined(__unix__) || defined(__APPLE__)

class NativeDirectoryStream
{
public:
    explicit NativeDirectoryStream(const Path& directory)
        : stream_(::opendir(directory.c_str()))
    {
    }

    NativeDirectoryStream(const NativeDirectoryStream&) = delete;
    NativeDirectoryStream& operator=(const NativeDirectoryStream&) = delete;

    ~NativeDirectoryStream()
    {
        if (stream_ != nullptr)
        {
            static_cast<void>(::closedir(stream_));
        }
    }

    explicit operator bool() const noexcept
    {
        return stream_ != nullptr;
    }

    dirent* next(std::error_code& error) noexcept
    {
        errno = 0;
        dirent* const entry = ::readdir(stream_);
        if (entry == nullptr && errno != 0)
        {
            error = std::error_code(errno, std::generic_category());
        }
        return entry;
    }

private:
    DIR* stream_ = nullptr;
};

bool is_dot_entry(const char* name) noexcept
{
    return name[0] == '.' &&
        (name[1] == '\0' ||
         (name[1] == '.' && name[2] == '\0'));
}

#endif

ManagerErrorCode enumeration_code(const std::error_code& error) noexcept
{
    return error == std::errc::not_a_directory
        ? ManagerErrorCode::not_directory
        : ManagerErrorCode::enumeration_failed;
}

[[noreturn]] void throw_enumeration(
    ManagerOperation operation,
    const Path& path,
    const std::error_code& error)
{
    throw ManagerException(
        enumeration_code(error),
        operation,
        path,
        error,
        "unable to enumerate directory: " + path.string());
}

std::vector<Path> list_directory(
    const Path& directory,
    ManagerOperation operation)
{
#if defined(__unix__) || defined(__APPLE__)
    NativeDirectoryStream stream(directory);
    if (!stream)
    {
        throw_enumeration(
            operation,
            directory,
            std::error_code(errno, std::generic_category()));
    }

    std::vector<Path> entries;
    std::error_code error;
    while (dirent* const entry = stream.next(error))
    {
        if (!is_dot_entry(entry->d_name))
        {
            entries.emplace_back(directory / entry->d_name);
        }
    }
    if (error)
    {
        throw_enumeration(operation, directory, error);
    }
    return entries;
#else
    std::error_code error;
    std::filesystem::directory_iterator iterator(directory, error);
    if (error)
    {
        throw_enumeration(operation, directory, error);
    }

    std::vector<Path> entries;
    const std::filesystem::directory_iterator end;
    while (iterator != end)
    {
        entries.push_back(iterator->path());
        iterator.increment(error);
        if (error)
        {
            throw_enumeration(operation, directory, error);
        }
    }
    return entries;
#endif
}

bool directory_is_empty(
    const Path& directory,
    ManagerOperation operation)
{
#if defined(__unix__) || defined(__APPLE__)
    NativeDirectoryStream stream(directory);
    if (!stream)
    {
        throw_enumeration(
            operation,
            directory,
            std::error_code(errno, std::generic_category()));
    }

    bool empty = true;
    std::error_code error;
    while (dirent* const entry = stream.next(error))
    {
        if (!is_dot_entry(entry->d_name))
        {
            empty = false;
        }
    }
    if (error)
    {
        throw_enumeration(operation, directory, error);
    }
    return empty;
#else
    std::error_code error;
    std::filesystem::directory_iterator iterator(directory, error);
    if (error)
    {
        throw_enumeration(operation, directory, error);
    }

    bool empty = true;
    const std::filesystem::directory_iterator end;
    while (iterator != end)
    {
        empty = false;
        iterator.increment(error);
        if (error)
        {
            throw_enumeration(operation, directory, error);
        }
    }
    return empty;
#endif
}

bool is_directory_compat(const Path& path) noexcept
{
#if defined(__unix__) || defined(__APPLE__)
    struct stat status
    {
    };
    return ::stat(path.c_str(), &status) == 0 && S_ISDIR(status.st_mode);
#else
    std::error_code error;
    const std::filesystem::file_status status =
        std::filesystem::status(path, error);
    return !error && std::filesystem::is_directory(status);
#endif
}

bool exists_compat(const Path& path) noexcept
{
#if defined(__unix__) || defined(__APPLE__)
    struct stat status
    {
    };
    return ::stat(path.c_str(), &status) == 0;
#else
    std::error_code error;
    const std::filesystem::file_status status =
        std::filesystem::status(path, error);
    return !error && std::filesystem::exists(status);
#endif
}

bool has_path_prefix(const Path& path, const Path& prefix) noexcept
{
    auto path_part = path.begin();
    for (auto prefix_part = prefix.begin();
         prefix_part != prefix.end();
         ++prefix_part, ++path_part)
    {
        if (path_part == path.end() || *path_part != *prefix_part)
        {
            return false;
        }
    }
    return true;
}

Path canonical_for_safety(
    const Path& path,
    const Path& reported_path)
{
    std::error_code error;
    Path canonical = std::filesystem::canonical(path, error);
    if (error)
    {
        throw ManagerException(
            ManagerErrorCode::unsafe_path_escape,
            ManagerOperation::remove_run_tree,
            reported_path,
            error,
            "unable to verify recursive-delete containment: " +
                reported_path.string());
    }
    return canonical;
}

void verify_recursive_delete_target(
    const Path& root,
    const Path& candidate,
    std::optional<Path>& canonical_root)
{
    std::error_code error;
    const std::filesystem::file_status candidate_status =
        std::filesystem::symlink_status(candidate, error);
    if (error || !std::filesystem::is_directory(candidate_status))
    {
        if (!error)
        {
            error = std::make_error_code(std::errc::not_a_directory);
        }
        throw ManagerException(
            ManagerErrorCode::remove_failed,
            ManagerOperation::remove_run_tree,
            candidate,
            error,
            "recursive delete requires a non-symlink directory: " +
                candidate.string());
    }

    if (!canonical_root)
    {
        canonical_root = canonical_for_safety(root, candidate);
    }
    const Path canonical_candidate =
        canonical_for_safety(candidate, candidate);
    if (!has_path_prefix(canonical_candidate, *canonical_root))
    {
        throw ManagerException(
            ManagerErrorCode::unsafe_path_escape,
            ManagerOperation::remove_run_tree,
            candidate,
            {},
            "recursive-delete target escapes the supplied root: " +
                candidate.string());
    }
}

void remove_run_tree(
    const Path& root,
    const Path& candidate,
    std::optional<Path>& canonical_root)
{
    verify_recursive_delete_target(root, candidate, canonical_root);

    std::error_code error;
    const std::uintmax_t removed =
        std::filesystem::remove_all(candidate, error);
    if (error || removed == 0)
    {
        if (!error)
        {
            error = std::make_error_code(
                std::errc::no_such_file_or_directory);
        }
        throw ManagerException(
            ManagerErrorCode::remove_failed,
            ManagerOperation::remove_run_tree,
            candidate,
            error,
            "unable to remove run tree: " + candidate.string());
    }
}

void emit_delete_line(std::ostream& output, const Path& run_path)
{
    const std::string native_path = run_path.string();
    output.write("Delate ", 7);
    output.write(
        native_path.data(),
        static_cast<std::streamsize>(native_path.size()));
    output.put('\n');
    if (!output)
    {
        throw ManagerException(
            ManagerErrorCode::output_failed,
            ManagerOperation::emit_delete_line,
            run_path,
            {},
            "unable to emit deletion line");
    }
}

void remove_empty_directory(const Path& path)
{
#if defined(__unix__) || defined(__APPLE__)
    if (::rmdir(path.c_str()) != 0)
    {
        const std::error_code error(errno, std::generic_category());
        throw ManagerException(
            ManagerErrorCode::remove_failed,
            ManagerOperation::remove_empty_directory,
            path,
            error,
            "unable to remove empty directory: " + path.string());
    }
#else
    std::error_code error;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(status))
    {
        if (!error)
        {
            error = std::make_error_code(std::errc::not_a_directory);
        }
        throw ManagerException(
            ManagerErrorCode::remove_failed,
            ManagerOperation::remove_empty_directory,
            path,
            error,
            "unable to remove empty directory: " + path.string());
    }

    const bool removed = std::filesystem::remove(path, error);
    if (error || !removed)
    {
        if (!error)
        {
            error = std::make_error_code(
                std::errc::no_such_file_or_directory);
        }
        throw ManagerException(
            ManagerErrorCode::remove_failed,
            ManagerOperation::remove_empty_directory,
            path,
            error,
            "unable to remove empty directory: " + path.string());
    }
#endif
}

} // namespace

ManagerException::ManagerException(
    ManagerErrorCode code,
    ManagerOperation operation,
    std::filesystem::path path,
    std::error_code system_error,
    std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      operation_(operation),
      path_(std::move(path)),
      system_error_(system_error)
{
}

ManagerErrorCode ManagerException::code() const noexcept
{
    return code_;
}

ManagerOperation ManagerException::operation() const noexcept
{
    return operation_;
}

const std::filesystem::path& ManagerException::path() const noexcept
{
    return path_;
}

const std::error_code& ManagerException::system_error() const noexcept
{
    return system_error_;
}

void delete_temp_files(const std::filesystem::path& directory)
{
    if (!directory_is_empty(
            directory, ManagerOperation::list_temp_root))
    {
        throw ManagerException(
            ManagerErrorCode::legacy_temp_join_type_error,
            ManagerOperation::list_temp_root,
            directory,
            {},
            "legacy delete_temp_files joins a list as a path operand");
    }
}

void clean_save_dir(const std::filesystem::path& directory)
{
    clean_save_dir(directory, std::cout);
}

void clean_save_dir(
    const std::filesystem::path& directory,
    std::ostream& output)
{
    const std::vector<Path> root_entries = list_directory(
        directory, ManagerOperation::list_save_root);

    std::vector<Path> algorithm_directories;
    algorithm_directories.reserve(root_entries.size());
    for (const Path& entry : root_entries)
    {
        if (is_directory_compat(entry))
        {
            algorithm_directories.push_back(entry);
        }
    }

    std::optional<Path> canonical_root;
    for (const Path& algorithm_directory : algorithm_directories)
    {
        const std::vector<Path> run_entries = list_directory(
            algorithm_directory, ManagerOperation::list_algorithm);
        for (const Path& run_path : run_entries)
        {
            const Path records_path = run_path / "records";
            bool remove_run = !exists_compat(records_path);
            if (!remove_run)
            {
                remove_run = directory_is_empty(
                    records_path,
                    ManagerOperation::list_records);
            }
            if (!remove_run)
            {
                continue;
            }

            remove_run_tree(directory, run_path, canonical_root);
            emit_delete_line(output, run_path);
        }
    }
}

void delete_empty_dir(const EmptyDirectoryConfig& config)
{
    const std::array<Path, 3> paths{
        config.record_dir,
        config.log_dir,
        config.save_dir};
    for (const Path& path : paths)
    {
        if (!exists_compat(path))
        {
            continue;
        }
        if (!directory_is_empty(
                path,
                ManagerOperation::remove_empty_directory))
        {
            continue;
        }
        remove_empty_directory(path);
    }
}

} // namespace virne::utils
