#pragma once
#include "feature_constructor_fixture.h"
#include "../virne/solver/learning/rl_core/online_environment.h"

namespace online_cluster_fixture {
namespace rl = virne::solver::learning::rl_core;
namespace net = virne::network;
namespace core = virne::core;
inline net::VirtualNetworkRequestSimulator simulator() {
    net::VirtualNetworkSimulationConfig config;
    config.num_virtual_networks = 3;
    const auto parameter = [](const double low,const double high) {
        net::SimulationDistribution result;
        result.value_kind = virne::utils::DatasetValueKind::floating;
        result.distribution.kind = virne::utils::DistributionKind::uniform;
        result.distribution.low = low; result.distribution.high = high; return result;
    };
    config.virtual_network_size.value_kind = virne::utils::DatasetValueKind::integer;
    config.virtual_network_size.distribution.kind = virne::utils::DistributionKind::uniform;
    config.virtual_network_size.distribution.low = std::int64_t{4};
    config.virtual_network_size.distribution.high = std::int64_t{5};
    config.lifetime = parameter(5,6); config.arrival_rate = parameter(1,2);
    config.topology_type = net::TopologyType::Path;
    using namespace feature_constructor_fixture;
    for (const auto name : {"cpu","gpu","ram"}) config.node_attribute_specs.push_back(resource_spec(name,attribute::AttributeOwner::node));
    for (const auto name : {"bw","latency"}) config.link_attribute_specs.push_back(resource_spec(name,attribute::AttributeOwner::link));
    std::vector<net::VirtualNetwork> requests;
    for (int id = 0; id < 3; ++id) {
        auto v = virtual_network(); v.set_request_id(id); v.set_arrival_time(id == 2 ? 6. : static_cast<double>(id));
        v.set_lifetime(5.); requests.push_back(std::move(v));
    }
    using Type = net::VirtualEventType;
    std::vector<net::VirtualNetworkEvent> events{{0,Type::arrival,0,0.},{1,Type::arrival,1,1.},
        {2,Type::leave,0,5.},{3,Type::leave,1,6.},{4,Type::arrival,2,6.},{5,Type::leave,2,11.}};
    return net::VirtualNetworkRequestSimulator::from_state(std::move(config),std::move(requests),std::move(events));
}
inline core::EnvironmentConfig lifecycle_config(const std::string& name) {
    core::EnvironmentConfig config;
    config.controller.node_resources = {0,1,2}; config.controller.link_resources = {0,1};
    config.counter.node_resources = std::vector<core::CounterResourceId>{0,1,2};
    config.counter.link_resources = std::vector<core::CounterResourceId>{0,1};
    config.recorder.save_root_dir = "/tmp/virne-online-cluster"; config.recorder.solver_name = name;
    config.recorder.run_id = "run"; config.recorder.temporary_records = false; return config;
}
inline rl::InstanceEnvironmentOptions options(const std::string& name) {
    rl::InstanceEnvironmentOptions options;
    if (name == "placement") options.mode = rl::InstanceStepMode::placement;
    if (name == "pair") { options.mode = rl::InstanceStepMode::node_pair; options.features = rl::FeatureConstructorKind::p_net_v_net; }
    if (name == "slots") options.mode = rl::InstanceStepMode::node_slots;
    if (name == "solution") options.mode = rl::InstanceStepMode::solution;
    if (name == "reject") options.allow_rejection = true;
    return options;
}
inline std::unique_ptr<rl::OnlineEnvironment> environment(const std::string& name) {
    return std::make_unique<rl::OnlineEnvironment>(feature_constructor_fixture::physical_network(),simulator(),lifecycle_config(name),options(name));
}
}  // namespace online_cluster_fixture
