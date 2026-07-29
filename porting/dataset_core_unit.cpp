#include "dataset.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace fs = std::filesystem;
using virne::utils::DatasetAttributeSpec;
using virne::utils::DatasetErrorCode;
using virne::utils::DatasetException;
using virne::utils::DatasetFileNameConfig;
using virne::utils::DatasetFileNameRequest;
using virne::utils::DatasetOperation;
using virne::utils::DatasetScalar;
using virne::utils::DatasetTopologyKind;
using virne::utils::DatasetValueKind;
using virne::utils::DistributionKind;
using virne::utils::DistributionSpec;
using virne::utils::OrderedFileNameItem;
using virne::utils::PhysicalDatasetPathRequest;
using virne::utils::PhysicalDatasetSetting;
using virne::utils::VirtualDatasetPathRequest;
using virne::utils::VirtualDatasetSetting;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
DatasetException expect_dataset_exception(
    Callable&& callable,
    DatasetErrorCode code,
    DatasetOperation operation)
{
    try
    {
        callable();
    }
    catch (const DatasetException& error)
    {
        expect(error.code() == code, "dataset error code mismatch");
        expect(error.operation() == operation, "dataset operation mismatch");
        return error;
    }
    throw std::runtime_error("expected DatasetException");
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
                ("virne_dataset_core_unit_" + std::to_string(tick) + "_" +
                 std::to_string(counter.fetch_add(1)));
            std::error_code error;
            if (fs::create_directory(root_, error))
            {
                return;
            }
            if (error && error != std::errc::file_exists)
            {
                throw std::runtime_error("unable to create dataset unit root");
            }
        }
        throw std::runtime_error("unable to allocate dataset unit root");
    }

    TemporaryTree(const TemporaryTree&) = delete;
    TemporaryTree& operator=(const TemporaryTree&) = delete;

    ~TemporaryTree()
    {
        const fs::path temporary_root =
            fs::absolute(fs::temp_directory_path()).lexically_normal();
        const fs::path absolute_root = fs::absolute(root_).lexically_normal();
        if (absolute_root.parent_path() == temporary_root &&
            absolute_root.filename().string().rfind(
                "virne_dataset_core_unit_", 0) == 0)
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

DatasetScalar integer(std::int64_t value)
{
    return DatasetScalar{value};
}

DistributionSpec uniform(std::int64_t low, std::int64_t high)
{
    DistributionSpec result;
    result.kind = DistributionKind::uniform;
    result.low = integer(low);
    result.high = integer(high);
    return result;
}

DatasetAttributeSpec attribute(
    std::uint32_t id,
    std::string name,
    DistributionSpec distribution)
{
    return {id, std::move(name), std::move(distribution)};
}

PhysicalDatasetSetting physical_setting(fs::path save_dir)
{
    PhysicalDatasetSetting setting;
    setting.save_dir = std::move(save_dir);
    setting.topology.num_nodes = 100;
    setting.topology.topology_type = DatasetTopologyKind::waxman;
    setting.topology.wm_alpha = DatasetScalar{0.5};
    setting.topology.wm_beta = DatasetScalar{0.2};
    setting.node_attributes = {
        attribute(1, "cpu", uniform(50, 100)),
        attribute(2, "max_cpu", {})};
    setting.link_attributes = {attribute(3, "bw", uniform(50, 100))};
    return setting;
}

VirtualDatasetSetting virtual_setting(fs::path save_dir)
{
    VirtualDatasetSetting setting;
    setting.save_dir = std::move(save_dir);
    setting.num_virtual_networks = 1000;
    setting.size_low = 2;
    setting.size_high = 10;
    setting.topology_type = DatasetTopologyKind::random;
    setting.lifetime.kind = DistributionKind::exponential;
    setting.lifetime.scale = integer(500);
    setting.arrival_lambda = DatasetScalar{0.04};
    setting.node_attributes = {attribute(1, "cpu", uniform(0, 20))};
    setting.link_attributes = {attribute(2, "bw", uniform(0, 50))};
    return setting;
}

void test_enum_boundaries_and_exception_accessors()
{
    expect(
        virne::utils::distribution_kind_from_string("uniform") ==
            DistributionKind::uniform,
        "uniform enum boundary mismatch");
    expect(
        virne::utils::distribution_kind_from_string("") ==
            DistributionKind::none,
        "empty distribution boundary mismatch");
    expect_dataset_exception(
        [] { (void)virne::utils::distribution_kind_from_string("bad"); },
        DatasetErrorCode::invalid_distribution,
        DatasetOperation::resolve_distribution);

    for (const auto& item : std::vector<std::pair<std::string_view, DatasetTopologyKind>>{
             {"path", DatasetTopologyKind::path},
             {"star", DatasetTopologyKind::star},
             {"grid_2d", DatasetTopologyKind::grid_2d},
             {"waxman", DatasetTopologyKind::waxman},
             {"random", DatasetTopologyKind::random}})
    {
        const DatasetTopologyKind kind =
            virne::utils::dataset_topology_kind_from_string(item.first);
        expect(kind == item.second, "topology enum boundary mismatch");
        expect(
            virne::utils::dataset_topology_kind_name(kind) == item.first,
            "topology enum spelling mismatch");
    }
    expect_dataset_exception(
        [] { (void)virne::utils::dataset_topology_kind_from_string("bad"); },
        DatasetErrorCode::invalid_topology,
        DatasetOperation::resolve_topology);
    expect_dataset_exception(
        []
        {
            (void)virne::utils::dataset_topology_kind_name(
                static_cast<DatasetTopologyKind>(255));
        },
        DatasetErrorCode::invalid_topology,
        DatasetOperation::resolve_topology);

    const DatasetException error(
        DatasetErrorCode::invalid_parameter,
        DatasetOperation::format_parameters,
        "diagnostic",
        7,
        fs::path("fixture"));
    expect(error.code() == DatasetErrorCode::invalid_parameter, "code accessor");
    expect(
        error.operation() == DatasetOperation::format_parameters,
        "operation accessor");
    expect(error.input_index() == 7, "input index accessor");
    expect(error.path() == fs::path("fixture"), "path accessor");
}

void test_scalar_formatting()
{
    const auto check = [](DatasetScalar value, std::string_view expected)
    {
        expect(
            virne::utils::format_dataset_scalar(value) == expected,
            "dataset scalar formatting mismatch");
    };
    check(std::monostate{}, "None");
    check(integer(std::numeric_limits<std::int64_t>::min()),
          "-9223372036854775808");
    check(integer(std::numeric_limits<std::int64_t>::max()),
          "9223372036854775807");
    check(DatasetScalar{true}, "True");
    check(DatasetScalar{false}, "False");
    check(DatasetScalar{std::string("a-b=c\n")}, "a-b=c\n");
    check(DatasetScalar{1.0}, "1.0");
    check(DatasetScalar{-0.0}, "-0.0");
    check(DatasetScalar{1.0e6}, "1000000.0");
    check(DatasetScalar{1.0e15}, "1000000000000000.0");
    check(DatasetScalar{1.0e16}, "1e+16");
    check(DatasetScalar{1.0e-4}, "0.0001");
    check(DatasetScalar{1.0e-5}, "1e-05");
    check(DatasetScalar{std::numeric_limits<double>::infinity()}, "inf");
    check(DatasetScalar{-std::numeric_limits<double>::infinity()}, "-inf");
    check(DatasetScalar{std::numeric_limits<double>::quiet_NaN()}, "nan");
}

void test_parameter_helpers_and_average_stub()
{
    DistributionSpec exponential;
    exponential.kind = DistributionKind::exponential;
    exponential.scale = integer(500);
    expect(
        virne::utils::get_parameters_string(
            virne::utils::get_distribution_parameters(exponential)) == "500",
        "exponential parameter mismatch");

    DistributionSpec poisson;
    poisson.kind = DistributionKind::poisson;
    poisson.lambda = DatasetScalar{0.04};
    expect(
        virne::utils::get_parameters_string(
            virne::utils::get_distribution_parameters(poisson)) == "0.04",
        "poisson parameter mismatch");
    expect(
        virne::utils::get_parameters_string(
            virne::utils::get_distribution_parameters(uniform(-2, 7))) ==
            "[-2-7]",
        "uniform parameter mismatch");

    DistributionSpec customized;
    customized.kind = DistributionKind::customized;
    customized.minimum = DatasetScalar{std::string("a")};
    customized.maximum = DatasetScalar{false};
    expect(
        virne::utils::get_parameters_string(
            virne::utils::get_distribution_parameters(customized)) ==
            "[a-False]",
        "customized parameter mismatch");
    expect(
        virne::utils::get_parameters_string({}) == "None",
        "empty parameter spelling mismatch");
    expect(
        virne::utils::get_parameters_string({DatasetScalar{true}}) == "True",
        "single parameter spelling mismatch");

    DistributionSpec missing;
    missing.kind = DistributionKind::uniform;
    missing.low = integer(1);
    expect_dataset_exception(
        [&] { (void)virne::utils::get_distribution_parameters(missing); },
        DatasetErrorCode::missing_parameter,
        DatasetOperation::resolve_distribution);
    DistributionSpec normal;
    normal.kind = DistributionKind::normal;
    expect_dataset_exception(
        [&] { (void)virne::utils::get_distribution_parameters(normal); },
        DatasetErrorCode::unsupported_parameter_distribution,
        DatasetOperation::resolve_distribution);
    expect(
        !virne::utils::get_distribution_average(
             normal, DatasetValueKind::floating),
        "average stub invented a value");
}

void test_file_names()
{
    const DatasetFileNameConfig config{"solver"};
    expect(
        virne::utils::generate_file_name(config, 0, {}) ==
            "solver-records-0-.csv",
        "empty filename mismatch");
    const std::vector<OrderedFileNameItem> items = {
        {"alpha", integer(1)},
        {"flag", DatasetScalar{true}},
        {"raw-key", DatasetScalar{std::string("x=y/z")}},
        {"none", DatasetScalar{std::monostate{}}}};
    expect(
        virne::utils::generate_file_name(config, -3, items) ==
            "solver-records--3-alpha=1-flag=True-raw-key=x=y/z-none=None.csv",
        "ordered filename mismatch");
}

void test_physical_and_virtual_paths()
{
    const PhysicalDatasetSetting physical = physical_setting("dataset/p_net");
    const fs::path expected_physical =
        fs::path("dataset/p_net") /
        "100-waxman_[0.5-0.2]-cpu_[50-100]-max_cpu_None-bw_[50-100]-seed_0";
    expect(
        virne::utils::get_p_net_dataset_dir_from_setting(
            physical, DatasetScalar{std::int64_t{0}}) == expected_physical,
        "generated physical path mismatch");

    const VirtualDatasetSetting virtual_value = virtual_setting("dataset/v_nets");
    const fs::path expected_virtual =
        fs::path("dataset/v_nets") /
        "1000-[2-10]-random-500-0.04-cpu_[0-20]-bw_[0-50]-seed_False";
    expect(
        virne::utils::get_v_nets_dataset_dir_from_setting(
            virtual_value, DatasetScalar{false}) == expected_virtual,
        "virtual path mismatch");

    TemporaryTree tree;
    const fs::path topology_file = tree.root() / "Geant.v2.gml";
    std::ofstream(topology_file).put('x');
    PhysicalDatasetSetting from_file = physical_setting(tree.root() / "out");
    from_file.topology.file_path = topology_file;
    expect(
        virne::utils::get_p_net_dataset_dir_from_setting(from_file)
                .filename()
                .string()
                .rfind("Geant-", 0) == 0,
        "physical first-dot filename mismatch");

    const fs::path hidden_file = tree.root() / ".hidden.gml";
    std::ofstream(hidden_file).put('x');
    from_file.topology.file_path = hidden_file;
    expect(
        virne::utils::get_p_net_dataset_dir_from_setting(from_file)
                .filename()
                .string()
                .rfind("-cpu_", 0) == 0,
        "physical hidden filename mismatch");

    from_file.topology.file_path = tree.root() / "missing.gml";
    expect(
        virne::utils::get_p_net_dataset_dir_from_setting(from_file)
                .filename()
                .string()
                .rfind("100-waxman_", 0) == 0,
        "missing topology did not use generated branch");

    VirtualDatasetSetting invalid = virtual_value;
    invalid.lifetime = {};
    invalid.lifetime.kind = DistributionKind::normal;
    expect_dataset_exception(
        [&]
        {
            (void)virne::utils::get_v_nets_dataset_dir_from_setting(invalid);
        },
        DatasetErrorCode::unsupported_parameter_distribution,
        DatasetOperation::resolve_distribution);
}

void test_copy_move_and_batches()
{
    const PhysicalDatasetSetting original = physical_setting("dataset/p_net");
    PhysicalDatasetSetting copied = original;
    PhysicalDatasetSetting moved = std::move(copied);
    expect(
        moved.node_attributes.size() == 2 &&
            moved.node_attributes[0].id == 1 &&
            original.node_attributes[0].name == "cpu",
        "typed physical setting copy/move mismatch");

    std::vector<DatasetFileNameRequest> names;
    names.reserve(4096);
    for (std::size_t index = 0; index < 4096; ++index)
    {
        names.push_back({
            {"solver"},
            static_cast<std::int64_t>(index),
            {{"index", DatasetScalar{static_cast<std::int64_t>(index)}},
             {"even", DatasetScalar{index % 2 == 0}}}});
    }
    const std::vector<std::string> baseline =
        virne::utils::generate_file_names_batch(names, 1);
    for (const std::size_t workers : {0U, 2U, 4U, 8U})
    {
        expect(
            virne::utils::generate_file_names_batch(names, workers) == baseline,
            "filename batch worker output drift");
    }

    std::vector<PhysicalDatasetPathRequest> physical_requests(512);
    std::vector<VirtualDatasetPathRequest> virtual_requests(512);
    for (std::size_t index = 0; index < physical_requests.size(); ++index)
    {
        physical_requests[index].setting = original;
        physical_requests[index].seed =
            DatasetScalar{static_cast<std::int64_t>(index)};
        virtual_requests[index].setting = virtual_setting("dataset/v_nets");
        virtual_requests[index].seed = DatasetScalar{index % 2 == 0};
    }
    const auto physical_baseline =
        virne::utils::get_p_net_dataset_dirs_batch(physical_requests, 1);
    const auto virtual_baseline =
        virne::utils::get_v_nets_dataset_dirs_batch(virtual_requests, 1);
    for (const std::size_t workers : {2U, 4U, 8U})
    {
        expect(
            virne::utils::get_p_net_dataset_dirs_batch(
                physical_requests, workers) == physical_baseline,
            "physical batch worker output drift");
        expect(
            virne::utils::get_v_nets_dataset_dirs_batch(
                virtual_requests, workers) == virtual_baseline,
            "virtual batch worker output drift");
    }

    std::vector<VirtualDatasetPathRequest> invalid(10);
    for (VirtualDatasetPathRequest& request : invalid)
    {
        request.setting = virtual_setting("dataset/v_nets");
    }
    invalid[5].setting.lifetime.kind = DistributionKind::normal;
    invalid[8].setting.lifetime.kind = DistributionKind::normal;
    for (const std::size_t workers : {1U, 2U, 4U, 8U})
    {
        const DatasetException error = expect_dataset_exception(
            [&]
            {
                (void)virne::utils::get_v_nets_dataset_dirs_batch(
                    invalid, workers);
            },
            DatasetErrorCode::unsupported_parameter_distribution,
            DatasetOperation::resolve_distribution);
        expect(error.input_index() == 5, "batch did not report lowest index");
    }
}

} // namespace

int main()
{
    try
    {
        test_enum_boundaries_and_exception_accessors();
        test_scalar_formatting();
        test_parameter_helpers_and_average_stub();
        test_file_names();
        test_physical_and_virtual_paths();
        test_copy_move_and_batches();
        std::cout << "dataset core unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "dataset core unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
