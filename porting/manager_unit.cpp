#include "manager.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <utility>

namespace
{

namespace fs = std::filesystem;
using virne::utils::EmptyDirectoryConfig;
using virne::utils::ManagerErrorCode;
using virne::utils::ManagerException;
using virne::utils::ManagerOperation;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

class TemporaryTree
{
public:
    TemporaryTree()
    {
        static std::atomic<std::uint64_t> counter{0};
        const fs::path temporary_root = fs::temp_directory_path();
        for (int attempt = 0; attempt < 100; ++attempt)
        {
            const auto tick = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            root_ = temporary_root /
                ("virne_manager_unit_" + std::to_string(tick) + "_" +
                 std::to_string(counter.fetch_add(1)));
            std::error_code error;
            if (fs::create_directory(root_, error))
            {
                return;
            }
            if (error && error != std::errc::file_exists)
            {
                throw std::runtime_error(
                    "unable to create manager unit temporary directory");
            }
        }
        throw std::runtime_error("unable to allocate unique temporary tree");
    }

    TemporaryTree(const TemporaryTree&) = delete;
    TemporaryTree& operator=(const TemporaryTree&) = delete;

    ~TemporaryTree()
    {
        const fs::path temporary_root =
            fs::absolute(fs::temp_directory_path()).lexically_normal();
        const fs::path absolute_root =
            fs::absolute(root_).lexically_normal();
        const std::string filename = absolute_root.filename().string();
        if (absolute_root.parent_path() == temporary_root &&
            filename.rfind("virne_manager_unit_", 0) == 0)
        {
            std::error_code ignored;
            fs::remove_all(absolute_root, ignored);
        }
    }

    const fs::path& root() const noexcept
    {
        return root_;
    }

private:
    fs::path root_;
};

void write_file(const fs::path& path, std::string_view bytes = "x")
{
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output)
    {
        throw std::runtime_error("unable to create unit fixture file");
    }
}

template <typename Callable>
ManagerException expect_manager_exception(
    Callable&& callable,
    ManagerErrorCode code,
    ManagerOperation operation)
{
    try
    {
        callable();
    }
    catch (const ManagerException& error)
    {
        expect(error.code() == code, "manager error code mismatch");
        expect(error.operation() == operation, "manager operation mismatch");
        return error;
    }
    throw std::runtime_error("expected ManagerException");
}

class RejectingBuffer final : public std::streambuf
{
protected:
    std::streamsize xsputn(const char*, std::streamsize) override
    {
        return 0;
    }

    int_type overflow(int_type) override
    {
        return traits_type::eof();
    }
};

void test_exception_accessors()
{
    const fs::path path("diagnostic path");
    const std::error_code system_error =
        std::make_error_code(std::errc::permission_denied);
    const ManagerException error(
        ManagerErrorCode::remove_failed,
        ManagerOperation::remove_run_tree,
        path,
        system_error,
        "diagnostic");
    expect(error.code() == ManagerErrorCode::remove_failed, "code accessor");
    expect(
        error.operation() == ManagerOperation::remove_run_tree,
        "operation accessor");
    expect(error.path() == path, "path accessor");
    expect(error.system_error() == system_error, "system error accessor");
    expect(std::string(error.what()) == "diagnostic", "what accessor");
}

void test_delete_temp_files()
{
    TemporaryTree tree;
    const fs::path missing = tree.root() / "missing";
    expect_manager_exception(
        [&] { virne::utils::delete_temp_files(missing); },
        ManagerErrorCode::enumeration_failed,
        ManagerOperation::list_temp_root);

    const fs::path regular = tree.root() / "regular";
    write_file(regular);
    expect_manager_exception(
        [&] { virne::utils::delete_temp_files(regular); },
        ManagerErrorCode::not_directory,
        ManagerOperation::list_temp_root);

    const fs::path empty = tree.root() / "empty";
    fs::create_directory(empty);
    virne::utils::delete_temp_files(empty);
    expect(fs::is_empty(empty), "empty temp directory changed");

    const fs::path nonempty = tree.root() / "nonempty";
    fs::create_directory(nonempty);
    const fs::path temp_file = nonempty / "keep_temp.bin";
    write_file(temp_file, "payload");
    expect_manager_exception(
        [&] { virne::utils::delete_temp_files(nonempty); },
        ManagerErrorCode::legacy_temp_join_type_error,
        ManagerOperation::list_temp_root);
    expect(fs::exists(temp_file), "legacy temp bug unexpectedly deleted file");
}

void test_clean_save_dir_core()
{
    TemporaryTree tree;
    const fs::path missing = tree.root() / "missing";
    std::ostringstream output;
    expect_manager_exception(
        [&] { virne::utils::clean_save_dir(missing, output); },
        ManagerErrorCode::enumeration_failed,
        ManagerOperation::list_save_root);

    const fs::path root_file = tree.root() / "root_file";
    write_file(root_file);
    expect_manager_exception(
        [&] { virne::utils::clean_save_dir(root_file, output); },
        ManagerErrorCode::not_directory,
        ManagerOperation::list_save_root);

    const fs::path save = tree.root() / "save";
    fs::create_directory(save);
    write_file(save / "ignored.txt");
    const fs::path empty_algorithm = save / "empty_algorithm";
    fs::create_directory(empty_algorithm);

    const fs::path missing_records = save / "algo_missing" / "run missing";
    fs::create_directories(missing_records / "models");
    write_file(missing_records / "models" / "weights.bin");

    const fs::path empty_records = save / "algo_empty" / "run_empty";
    fs::create_directories(empty_records / "records");

    const fs::path retained = save / "algo_keep" / "run_keep";
    fs::create_directories(retained / "records");
    write_file(retained / "records" / ".hidden", "keep");

    std::ostringstream captured;
    virne::utils::clean_save_dir(save, captured);
    expect(fs::exists(empty_algorithm), "empty algorithm should remain");
    expect(!fs::exists(missing_records), "missing-record run should be deleted");
    expect(!fs::exists(empty_records), "empty-record run should be deleted");
    expect(fs::exists(retained), "nonempty-record run should remain");
    const std::string text = captured.str();
    expect(
        text.find("Delate " + missing_records.string() + "\n") !=
            std::string::npos,
        "missing-record deletion line mismatch");
    expect(
        text.find("Delate " + empty_records.string() + "\n") !=
            std::string::npos,
        "empty-record deletion line mismatch");
}

void test_clean_save_dir_errors_and_output()
{
    {
        TemporaryTree tree;
        const fs::path save = tree.root() / "save";
        const fs::path run = save / "algo" / "run";
        fs::create_directories(run);
        write_file(run / "records");
        std::ostringstream output;
        expect_manager_exception(
            [&] { virne::utils::clean_save_dir(save, output); },
            ManagerErrorCode::not_directory,
            ManagerOperation::list_records);
        expect(fs::exists(run), "records-file failure deleted run");
    }
    {
        TemporaryTree tree;
        const fs::path save = tree.root() / "save";
        const fs::path algorithm = save / "algo";
        fs::create_directories(algorithm);
        const fs::path run_file = algorithm / "run_file";
        write_file(run_file);
        std::ostringstream output;
        expect_manager_exception(
            [&] { virne::utils::clean_save_dir(save, output); },
            ManagerErrorCode::remove_failed,
            ManagerOperation::remove_run_tree);
        expect(fs::exists(run_file), "regular run entry was deleted");
    }
    {
        TemporaryTree tree;
        const fs::path save = tree.root() / "save";
        const fs::path run = save / "algo" / "run";
        fs::create_directories(run);
        RejectingBuffer buffer;
        std::ostream output(&buffer);
        expect_manager_exception(
            [&] { virne::utils::clean_save_dir(save, output); },
            ManagerErrorCode::output_failed,
            ManagerOperation::emit_delete_line);
        expect(!fs::exists(run), "sink failure must happen after deletion");
    }
}

void test_clean_save_dir_symlink_safety()
{
    TemporaryTree tree;
    const fs::path save = tree.root() / "save";
    const fs::path outside = tree.root() / "outside";
    const fs::path outside_run = outside / "run";
    fs::create_directories(outside_run);
    fs::create_directories(save);

    std::error_code error;
    fs::create_directory_symlink(outside, save / "algo_link", error);
    expect(!error, "unable to create algorithm symlink fixture");
    std::ostringstream output;
    expect_manager_exception(
        [&] { virne::utils::clean_save_dir(save, output); },
        ManagerErrorCode::unsafe_path_escape,
        ManagerOperation::remove_run_tree);
    expect(fs::exists(outside_run), "escaped symlink target was deleted");
    expect(output.str().empty(), "unsafe target emitted deletion output");

    TemporaryTree final_link_tree;
    const fs::path final_save = final_link_tree.root() / "save";
    const fs::path final_outside = final_link_tree.root() / "outside";
    fs::create_directories(final_save / "algo");
    fs::create_directories(final_outside);
    error.clear();
    fs::create_directory_symlink(
        final_outside,
        final_save / "algo" / "run_link",
        error);
    expect(!error, "unable to create final run symlink fixture");
    expect_manager_exception(
        [&] { virne::utils::clean_save_dir(final_save, output); },
        ManagerErrorCode::remove_failed,
        ManagerOperation::remove_run_tree);
    expect(fs::exists(final_outside), "final symlink target was deleted");
}

void test_clean_save_dir_relative_unicode()
{
    TemporaryTree tree;
    const fs::path previous = fs::current_path();
    struct CurrentPathGuard
    {
        fs::path path;
        ~CurrentPathGuard()
        {
            std::error_code ignored;
            fs::current_path(path, ignored);
        }
    } guard{previous};

    fs::current_path(tree.root());
    const fs::path relative_save = fs::path("save space") / "đồ-thị";
    const fs::path relative_run = relative_save / "thuật toán" / "run one";
    fs::create_directories(relative_run);
    std::ostringstream output;
    virne::utils::clean_save_dir(relative_save, output);
    expect(
        output.str() == "Delate " + relative_run.string() + "\n",
        "relative/native deletion line mismatch");
}

void test_delete_empty_dir()
{
    {
        TemporaryTree tree;
        virne::utils::delete_empty_dir({
            tree.root() / "missing_record",
            tree.root() / "missing_log",
            tree.root() / "missing_save"});
    }
    {
        TemporaryTree tree;
        const fs::path save = tree.root() / "save";
        const fs::path record = save / "record";
        const fs::path log = save / "log";
        fs::create_directories(record);
        fs::create_directories(log);
        virne::utils::delete_empty_dir({record, log, save});
        expect(!fs::exists(save), "nested empty-directory cascade failed");
    }
    {
        TemporaryTree tree;
        const fs::path duplicate = tree.root() / "duplicate";
        fs::create_directory(duplicate);
        virne::utils::delete_empty_dir({duplicate, duplicate, duplicate});
        expect(!fs::exists(duplicate), "duplicate empty path was not removed");
    }
    {
        TemporaryTree tree;
        const fs::path nonempty = tree.root() / "nonempty";
        fs::create_directory(nonempty);
        write_file(nonempty / "keep");
        virne::utils::delete_empty_dir({nonempty, {}, {}});
        expect(fs::exists(nonempty / "keep"), "nonempty directory changed");
    }
    {
        TemporaryTree tree;
        const fs::path removed_first = tree.root() / "first";
        const fs::path file_second = tree.root() / "second";
        const fs::path untouched_later = tree.root() / "later";
        fs::create_directory(removed_first);
        write_file(file_second);
        fs::create_directory(untouched_later);
        expect_manager_exception(
            [&]
            {
                virne::utils::delete_empty_dir(
                    {removed_first, file_second, untouched_later});
            },
            ManagerErrorCode::not_directory,
            ManagerOperation::remove_empty_directory);
        expect(!fs::exists(removed_first), "earlier removal did not persist");
        expect(fs::exists(file_second), "regular file was removed");
        expect(fs::exists(untouched_later), "later path was visited after error");
    }
    {
        TemporaryTree tree;
        const fs::path target = tree.root() / "target";
        const fs::path link = tree.root() / "link";
        fs::create_directory(target);
        std::error_code error;
        fs::create_directory_symlink(target, link, error);
        expect(!error, "unable to create empty-dir symlink fixture");
        expect_manager_exception(
            [&] { virne::utils::delete_empty_dir({link, {}, {}}); },
            ManagerErrorCode::remove_failed,
            ManagerOperation::remove_empty_directory);
        expect(fs::exists(link), "directory symlink was removed");
        expect(fs::exists(target), "directory symlink target was removed");
    }
}

void test_empty_directory_config_value_semantics_and_long_paths()
{
    TemporaryTree tree;
    fs::path deep = tree.root();
    for (int index = 0; index < 8; ++index)
    {
        deep /=
            "long_manager_component_" + std::to_string(index) +
            "_0123456789abcdef";
    }

    const fs::path save = deep / "save";
    const fs::path record = save / "record";
    const fs::path log = save / "log";
    fs::create_directories(record);
    fs::create_directory(log);
    expect(
        fs::absolute(log).native().size() > 260U,
        "long-path fixture did not exceed the legacy path limit");

    const EmptyDirectoryConfig original{record, log, save};
    EmptyDirectoryConfig copied = original;
    expect(copied.record_dir == record, "copied record_dir changed");
    expect(copied.log_dir == log, "copied log_dir changed");
    expect(copied.save_dir == save, "copied save_dir changed");

    EmptyDirectoryConfig moved = std::move(copied);
    expect(moved.record_dir == record, "moved record_dir changed");
    expect(moved.log_dir == log, "moved log_dir changed");
    expect(moved.save_dir == save, "moved save_dir changed");

    virne::utils::delete_empty_dir(moved);
    expect(!fs::exists(save), "long-path empty-directory cascade failed");
    expect(
        original.record_dir == record && original.log_dir == log &&
            original.save_dir == save,
        "copy/move test mutated the original fixed fields");
}

} // namespace

int main()
{
    try
    {
        test_exception_accessors();
        test_delete_temp_files();
        test_clean_save_dir_core();
        test_clean_save_dir_errors_and_output();
        test_clean_save_dir_symlink_safety();
        test_clean_save_dir_relative_unicode();
        test_delete_empty_dir();
        test_empty_directory_config_value_semantics_and_long_paths();
        std::cout << "manager unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "manager unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
