#include "utils_config.h"

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace fs = std::filesystem;
namespace attribute = virne::network::attribute;
namespace utils = virne::utils;

using attribute::AttributeKind;
using utils::DatasetAttributeSpec;
using utils::DatasetScalar;
using utils::DatasetTopologyKind;
using utils::DistributionKind;
using utils::DistributionSpec;
using utils::RunDirectoryInput;
using utils::RunIdTimestamp;
using utils::SimulationConfigInput;
using utils::SimulationConfigSummary;
using utils::UtilsConfigErrorCode;
using utils::UtilsConfigException;
using utils::UtilsConfigOperation;

void expect(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Callable>
UtilsConfigException expect_utils_config_exception(
    Callable&& callable,
    UtilsConfigErrorCode code,
    UtilsConfigOperation operation)
{
    try
    {
        std::forward<Callable>(callable)();
    }
    catch (const UtilsConfigException& error)
    {
        expect(error.code() == code, "utils-config error code mismatch");
        expect(
            error.operation() == operation,
            "utils-config operation mismatch");
        return error;
    }
    throw std::runtime_error("expected UtilsConfigException");
}

DistributionSpec uniform_integer(std::int64_t low, std::int64_t high)
{
    DistributionSpec result;
    result.kind = DistributionKind::uniform;
    result.low = DatasetScalar{low};
    result.high = DatasetScalar{high};
    return result;
}

DatasetAttributeSpec dataset_attribute(
    std::uint32_t id,
    std::string name,
    DistributionSpec distribution = {})
{
    DatasetAttributeSpec result;
    result.id = id;
    result.name = std::move(name);
    result.distribution = std::move(distribution);
    return result;
}

SimulationConfigInput make_simulation_input(std::int64_t variant = 0)
{
    SimulationConfigInput input;

    input.physical_dataset.save_dir =
        fs::path("datasets") / ("p" + std::to_string(variant));
    input.physical_dataset.topology.num_nodes = 6 + variant;
    input.physical_dataset.topology.topology_type =
        DatasetTopologyKind::waxman;
    input.physical_dataset.topology.wm_alpha = DatasetScalar{0.5};
    input.physical_dataset.topology.wm_beta = DatasetScalar{0.2};
    input.physical_dataset.node_attributes = {
        dataset_attribute(0U, "cpu", uniform_integer(1, 9)),
        dataset_attribute(1U, "max_cpu"),
        dataset_attribute(2U, "tag")};
    input.physical_dataset.link_attributes = {
        dataset_attribute(0U, "bw", uniform_integer(2, 10)),
        dataset_attribute(1U, "max_bw")};

    input.virtual_dataset.save_dir =
        fs::path("datasets") / ("v" + std::to_string(variant));
    input.virtual_dataset.num_virtual_networks = 12 + variant;
    input.virtual_dataset.size_low = 2;
    input.virtual_dataset.size_high = 5;
    input.virtual_dataset.topology_type = DatasetTopologyKind::random;
    input.virtual_dataset.lifetime.kind = DistributionKind::exponential;
    input.virtual_dataset.lifetime.scale = DatasetScalar{std::int64_t{50}};
    input.virtual_dataset.arrival_lambda = DatasetScalar{0.25};
    input.virtual_dataset.node_attributes = {
        dataset_attribute(0U, "cpu", uniform_integer(0, 4)),
        dataset_attribute(1U, "status"),
        dataset_attribute(2U, "pos"),
        dataset_attribute(3U, "gpu", uniform_integer(1, 3))};
    input.virtual_dataset.link_attributes = {
        dataset_attribute(0U, "bw", uniform_integer(1, 8)),
        dataset_attribute(1U, "status"),
        dataset_attribute(2U, "ltc")};

    input.physical_node_attributes.kinds = {
        AttributeKind::resource,
        AttributeKind::extrema,
        AttributeKind::status};
    input.physical_link_attributes.kinds = {
        AttributeKind::resource,
        AttributeKind::extrema};
    input.virtual_node_attributes.kinds = {
        AttributeKind::resource,
        AttributeKind::status,
        AttributeKind::position,
        AttributeKind::resource};
    input.virtual_link_attributes.kinds = {
        AttributeKind::resource,
        AttributeKind::status,
        AttributeKind::latency};
    input.extracted_attribute_kinds.included = {
        true, true, false, false, true};
    input.seed = DatasetScalar{std::int64_t{7 + variant}};
    return input;
}

void expect_summary_equal(
    const SimulationConfigSummary& actual,
    const SimulationConfigSummary& expected,
    std::string_view context)
{
    const bool equal =
        actual.p_net_dataset_dir == expected.p_net_dataset_dir &&
        actual.v_nets_dataset_dir == expected.v_nets_dataset_dir &&
        actual.p_net_setting_num_nodes ==
            expected.p_net_setting_num_nodes &&
        actual.p_net_setting_num_node_attrs ==
            expected.p_net_setting_num_node_attrs &&
        actual.p_net_setting_num_link_attrs ==
            expected.p_net_setting_num_link_attrs &&
        actual.p_net_setting_num_node_resource_attrs ==
            expected.p_net_setting_num_node_resource_attrs &&
        actual.p_net_setting_num_link_resource_attrs ==
            expected.p_net_setting_num_link_resource_attrs &&
        actual.p_net_setting_num_node_extrema_attrs ==
            expected.p_net_setting_num_node_extrema_attrs &&
        actual.p_net_setting_num_link_extrema_attrs ==
            expected.p_net_setting_num_link_extrema_attrs &&
        actual.v_sim_setting_num_node_attrs ==
            expected.v_sim_setting_num_node_attrs &&
        actual.v_sim_setting_num_link_attrs ==
            expected.v_sim_setting_num_link_attrs &&
        actual.v_sim_setting_num_node_resource_attrs ==
            expected.v_sim_setting_num_node_resource_attrs &&
        actual.v_sim_setting_num_link_resource_attrs ==
            expected.v_sim_setting_num_link_resource_attrs &&
        actual.v_sim_setting_num_node_non_status_attrs ==
            expected.v_sim_setting_num_node_non_status_attrs &&
        actual.v_sim_setting_num_link_non_status_attrs ==
            expected.v_sim_setting_num_link_non_status_attrs &&
        actual.feature_constructor.num_extracted_p_node_attrs ==
            expected.feature_constructor.num_extracted_p_node_attrs &&
        actual.feature_constructor.num_extracted_p_link_attrs ==
            expected.feature_constructor.num_extracted_p_link_attrs &&
        actual.feature_constructor.num_extracted_v_node_attrs ==
            expected.feature_constructor.num_extracted_v_node_attrs &&
        actual.feature_constructor.num_extracted_v_link_attrs ==
            expected.feature_constructor.num_extracted_v_link_attrs &&
        actual.feature_constructor.p_num_nodes ==
            expected.feature_constructor.p_num_nodes;
    expect(equal, context);
}

void test_injected_run_id_and_continuation()
{
    PyRandom random(42U);
    const RunIdTimestamp timestamp{2024, 2, 29, 3, 4, 5};
    expect(
        utils::generate_run_id(timestamp, "worker-a", random) ==
            "worker-a-20240229T030405-1824",
        "injected run ID differs from the CPython oracle");
    expect(
        random.randint(0, 9999) == 409,
        "run ID consumed the wrong PyRandom continuation state");
}

void test_system_run_id_shape_and_continuation()
{
    PyRandom random(91U);
    PyRandom control(91U);
    static_cast<void>(control.randint(0, 9999));

    const std::string run_id = utils::generate_run_id(random);
    expect(run_id.size() >= 22U, "system run ID is too short");
    const std::size_t suffix = run_id.size() - 21U;
    expect(run_id[suffix] == '-', "system run ID timestamp separator");
    expect(run_id[suffix + 9U] == 'T', "system run ID T separator");
    expect(run_id[suffix + 16U] == '-', "system run ID random separator");
    for (std::size_t index = 1U; index < 21U; ++index)
    {
        if (index == 9U || index == 16U)
        {
            continue;
        }
        expect(
            std::isdigit(static_cast<unsigned char>(run_id[suffix + index])) != 0,
            "system run ID contains a non-digit field");
    }
    expect(
        random.randint(0, 9999) == control.randint(0, 9999),
        "system run ID consumed the wrong PyRandom state");
}

Config make_resolved_config()
{
    return Config(YAML::Load(R"yaml(
base_dir: runs
fixed_run_id: run-17
copy_value: 17
resolved_copy: ${copy_value}
resolved_text: value-${copy_value}
experiment:
  save_root_dir: ${base_dir}
  run_id: ${fixed_run_id}
solver:
  solver_name: typed-solver
)yaml"));
}

void test_config_resolution_and_mapping_validation()
{
    const Config config = make_resolved_config();
    const YAML::Node resolved = utils::resolve_config_to_node(config);
    expect(resolved.IsMap(), "resolved Config root is not a mapping");
    expect(
        resolved["resolved_copy"].as<int>() == 17,
        "whole-value Config interpolation did not retain its type");
    expect(
        resolved["resolved_text"].as<std::string>() == "value-17",
        "embedded Config interpolation was not resolved");

    YAML::Node raw_mapping = YAML::Load("{value: 3}");
    YAML::Node raw_result = utils::resolve_config_to_node(raw_mapping);
    raw_result["value"] = 8;
    expect(
        raw_mapping["value"].as<int>() == 8,
        "raw YAML mapping did not preserve shared-node identity");

    expect_utils_config_exception(
        []
        {
            static_cast<void>(
                utils::resolve_config_to_node(YAML::Load("[1, 2]")));
        },
        UtilsConfigErrorCode::invalid_mapping_root,
        UtilsConfigOperation::resolve_config);

    const Config invalid_config(YAML::Load("scalar-root"));
    expect_utils_config_exception(
        [&]
        {
            static_cast<void>(utils::resolve_config_to_node(invalid_config));
        },
        UtilsConfigErrorCode::invalid_mapping_root,
        UtilsConfigOperation::resolve_config);
}

void test_complete_typed_summary()
{
    const SimulationConfigInput input = make_simulation_input();
    const SimulationConfigSummary summary =
        utils::derive_simulation_config(input);

    const fs::path expected_physical = fs::path("datasets") / "p0" /
        "6-waxman_[0.5-0.2]-cpu_[1-9]-max_cpu_None-tag_None-"
        "bw_[2-10]-max_bw_None-seed_7";
    const fs::path expected_virtual = fs::path("datasets") / "v0" /
        "12-[2-5]-random-50-0.25-cpu_[0-4]-status_None-pos_None-"
        "gpu_[1-3]-bw_[1-8]-status_None-ltc_None-seed_7";
    expect(
        summary.p_net_dataset_dir == expected_physical,
        "physical dataset path summary mismatch");
    expect(
        summary.v_nets_dataset_dir == expected_virtual,
        "virtual dataset path summary mismatch");

    expect(summary.p_net_setting_num_nodes == 6, "physical node count mismatch");
    expect(summary.p_net_setting_num_node_attrs == 3U, "physical node attrs mismatch");
    expect(summary.p_net_setting_num_link_attrs == 2U, "physical link attrs mismatch");
    expect(summary.p_net_setting_num_node_resource_attrs == 1U, "physical node resources mismatch");
    expect(summary.p_net_setting_num_link_resource_attrs == 1U, "physical link resources mismatch");
    expect(summary.p_net_setting_num_node_extrema_attrs == 1U, "physical node extrema mismatch");
    expect(summary.p_net_setting_num_link_extrema_attrs == 1U, "physical link extrema mismatch");

    expect(summary.v_sim_setting_num_node_attrs == 4U, "virtual node attrs mismatch");
    expect(summary.v_sim_setting_num_link_attrs == 3U, "virtual link attrs mismatch");
    expect(summary.v_sim_setting_num_node_resource_attrs == 2U, "virtual node resources mismatch");
    expect(summary.v_sim_setting_num_link_resource_attrs == 1U, "virtual link resources mismatch");
    expect(summary.v_sim_setting_num_node_non_status_attrs == 3U, "virtual node non-status mismatch");
    expect(summary.v_sim_setting_num_link_non_status_attrs == 2U, "virtual link non-status mismatch");

    expect(summary.feature_constructor.num_extracted_p_node_attrs == 2U, "extracted physical node count mismatch");
    expect(summary.feature_constructor.num_extracted_p_link_attrs == 2U, "extracted physical link count mismatch");
    expect(summary.feature_constructor.num_extracted_v_node_attrs == 2U, "extracted virtual node count mismatch");
    expect(summary.feature_constructor.num_extracted_v_link_attrs == 2U, "extracted virtual link count mismatch");
    expect(summary.feature_constructor.p_num_nodes == 6, "feature constructor physical size mismatch");
}

void test_typed_summary_errors()
{
    SimulationConfigInput mismatch = make_simulation_input();
    mismatch.physical_node_attributes.kinds.pop_back();
    expect_utils_config_exception(
        [&]
        {
            static_cast<void>(utils::derive_simulation_config(mismatch));
        },
        UtilsConfigErrorCode::attribute_kind_count_mismatch,
        UtilsConfigOperation::derive_simulation_config);

    SimulationConfigInput invalid = make_simulation_input();
    invalid.virtual_link_attributes.kinds[1] =
        static_cast<AttributeKind>(255U);
    expect_utils_config_exception(
        [&]
        {
            static_cast<void>(utils::derive_simulation_config(invalid));
        },
        UtilsConfigErrorCode::invalid_attribute_kind,
        UtilsConfigOperation::derive_simulation_config);

}

void test_batch_workers_order_and_failure_selection()
{
    std::vector<SimulationConfigInput> inputs;
    inputs.reserve(7U);
    for (std::int64_t index = 0; index < 7; ++index)
    {
        inputs.push_back(make_simulation_input(index));
    }

    const std::vector<SimulationConfigSummary> baseline =
        utils::derive_simulation_configs_batch(inputs, 1U);
    expect(baseline.size() == inputs.size(), "batch result size mismatch");
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        const std::vector<SimulationConfigSummary> actual =
            utils::derive_simulation_configs_batch(inputs, workers);
        expect(actual.size() == baseline.size(), "worker batch size mismatch");
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            expect_summary_equal(
                actual[index],
                baseline[index],
                "worker batch changed result order or content");
        }
    }

    const std::vector<SimulationConfigInput> empty;
    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        expect(
            utils::derive_simulation_configs_batch(empty, workers).empty(),
            "empty worker batch must remain empty");
    }

    std::vector<SimulationConfigInput> failures;
    failures.reserve(4U);
    for (std::int64_t index = 0; index < 4; ++index)
    {
        failures.push_back(make_simulation_input(index));
    }
    failures[1].physical_node_attributes.kinds[0] =
        static_cast<AttributeKind>(255U);
    failures[3].virtual_link_attributes.kinds.pop_back();

    for (const std::size_t workers : {0U, 1U, 2U, 8U})
    {
        // Index 1 is invalid-kind; index 3 is count-mismatch. The distinct
        // codes make the documented lowest-index selection observable.
        expect_utils_config_exception(
            [&]
            {
                static_cast<void>(
                    utils::derive_simulation_configs_batch(failures, workers));
            },
            UtilsConfigErrorCode::invalid_attribute_kind,
            UtilsConfigOperation::derive_simulation_config);
    }
}

void test_run_directory_direct_and_config_adapter()
{
    const RunDirectoryInput direct{
        fs::path("root"), "solver-a", "run-a"};
    expect(
        utils::get_run_id_dir(direct) ==
            fs::path("root") / "solver-a" / "run-a",
        "direct run directory join mismatch");

    const Config config = make_resolved_config();
    const RunDirectoryInput adapted =
        utils::run_directory_input_from_config(config);
    expect(adapted.save_root_dir == fs::path("runs"), "adapted save root mismatch");
    expect(adapted.solver_name == "typed-solver", "adapted solver mismatch");
    expect(adapted.run_id == "run-17", "adapted run ID mismatch");
    expect(
        utils::get_run_id_dir(adapted) ==
            fs::path("runs") / "typed-solver" / "run-17",
        "adapted run directory mismatch");
}

} // namespace

int main()
{
    try
    {
        test_injected_run_id_and_continuation();
        test_system_run_id_shape_and_continuation();
        test_config_resolution_and_mapping_validation();
        test_complete_typed_summary();
        test_typed_summary_errors();
        test_batch_workers_order_and_failure_selection();
        test_run_directory_direct_and_config_adapter();
        std::cout << "utils config unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "utils config unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
