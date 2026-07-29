#pragma once

#include "base_network.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

class RandomContext;

namespace virne::network {

enum class PhysicalTopologyOrigin : std::uint8_t {
    generated,
    loaded_gml,
    generated_after_gml_error,
};

struct PhysicalNetworkBuildOptions {
    std::optional<std::uint32_t> seed;
    std::size_t factory_workers = 1U;
    std::size_t attribute_workers = 1U;
};

struct PhysicalNetworkBuildReport {
    PhysicalTopologyOrigin origin = PhysicalTopologyOrigin::generated;
    std::optional<std::string> requested_file;
    std::optional<std::string> gml_error;
};

class PhysicalNetwork final : public BaseNetwork {
public:
    PhysicalNetwork();
    explicit PhysicalNetwork(BaseNetworkConstruction construction);
    explicit PhysicalNetwork(BaseNetwork&& network);

    PhysicalNetwork(PhysicalNetwork&& other) noexcept;
    PhysicalNetwork& operator=(PhysicalNetwork&& other) noexcept;
    PhysicalNetwork(const PhysicalNetwork&) = delete;
    PhysicalNetwork& operator=(const PhysicalNetwork&) = delete;

    static PhysicalNetwork from_setting(
        const virne::utils::SettingDocument& config,
        const PhysicalNetworkBuildOptions& options = {});

    static PhysicalNetwork from_setting(
        const virne::utils::SettingDocument& config,
        RandomContext& random,
        const PhysicalNetworkBuildOptions& options = {});

    const PhysicalNetworkBuildReport& build_report() const noexcept;

    void to_gml(const std::string& path) const;
    void save_dataset(
        const std::string& directory,
        const std::string& file_name = "p_net.gml") const;

    static PhysicalNetwork load_dataset(
        const std::string& directory,
        const std::string& file_name = "p_net.gml");

    PhysicalNetwork clone() const;

private:
    PhysicalNetworkBuildReport build_report_;
};

}  // namespace virne::network
