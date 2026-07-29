#include "manager.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

namespace fs = std::filesystem;
using virne::utils::EmptyDirectoryConfig;
using virne::utils::ManagerException;

std::string bytes_to_hex(std::string_view bytes)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(bytes.size() * 2U);
    for (const char character : bytes)
    {
        const auto byte = static_cast<unsigned char>(character);
        encoded.push_back(digits[byte >> 4U]);
        encoded.push_back(digits[byte & 0x0FU]);
    }
    return encoded;
}

bool has_path_prefix(const fs::path& path, const fs::path& prefix) noexcept
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

fs::path checked_sandbox(std::string_view text)
{
    const fs::path sandbox =
        fs::absolute(fs::path(std::string(text))).lexically_normal();
    const fs::path temporary =
        fs::absolute(fs::temp_directory_path()).lexically_normal();
    const std::string name = sandbox.filename().string();
    if (sandbox.parent_path() != temporary ||
        (name.rfind("virne_manager_diff_", 0) != 0 &&
         name.rfind("virne_manager_bench_", 0) != 0))
    {
        throw std::invalid_argument(
            "sandbox must be a direct temporary child with a manager prefix");
    }
    return sandbox;
}

fs::path checked_input(
    const fs::path& sandbox,
    std::string_view text)
{
    const fs::path input = fs::path(std::string(text));
    const fs::path absolute = fs::absolute(input).lexically_normal();
    if (!has_path_prefix(absolute, sandbox))
    {
        throw std::invalid_argument("operation path escapes harness sandbox");
    }
    return input;
}

void print_result(
    const ManagerException* error,
    std::string_view output)
{
    std::cout << "status=" << (error == nullptr ? "ok" : "error") << '\n';
    if (error == nullptr)
    {
        std::cout << "code=-1\noperation=-1\npath_hex=\nsystem_value=0\n"
                  << "what_hex=\n";
    }
    else
    {
        std::cout << "code=" << static_cast<unsigned int>(error->code())
                  << '\n'
                  << "operation="
                  << static_cast<unsigned int>(error->operation()) << '\n'
                  << "path_hex=" << bytes_to_hex(error->path().string())
                  << '\n'
                  << "system_value=" << error->system_error().value() << '\n'
                  << "what_hex=" << bytes_to_hex(error->what()) << '\n';
    }
    std::cout << "output_hex=" << bytes_to_hex(output) << '\n';
}

int run(int argc, char** argv)
{
    if (argc < 4)
    {
        throw std::invalid_argument(
            "usage: vne_manager_harness <sandbox> <operation> <paths...>");
    }
    const fs::path sandbox = checked_sandbox(argv[1]);
    const std::string_view operation(argv[2]);
    std::ostringstream captured;
    try
    {
        if (operation == "delete_temp")
        {
            if (argc != 4)
            {
                throw std::invalid_argument("delete_temp expects one path");
            }
            virne::utils::delete_temp_files(
                checked_input(sandbox, argv[3]));
        }
        else if (operation == "clean_save")
        {
            if (argc != 4)
            {
                throw std::invalid_argument("clean_save expects one path");
            }
            virne::utils::clean_save_dir(
                checked_input(sandbox, argv[3]), captured);
        }
        else if (operation == "delete_empty")
        {
            if (argc != 6)
            {
                throw std::invalid_argument("delete_empty expects three paths");
            }
            virne::utils::delete_empty_dir(EmptyDirectoryConfig{
                checked_input(sandbox, argv[3]),
                checked_input(sandbox, argv[4]),
                checked_input(sandbox, argv[5])});
        }
        else
        {
            throw std::invalid_argument("unknown manager operation");
        }
    }
    catch (const ManagerException& error)
    {
        print_result(&error, captured.str());
        return 0;
    }

    print_result(nullptr, captured.str());
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "manager harness misuse: " << error.what() << '\n';
        return 64;
    }
}
