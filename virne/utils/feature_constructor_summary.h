#pragma once

#include <cstddef>
#include <cstdint>

namespace virne::utils {

struct FeatureConstructorSummary {
    std::size_t num_extracted_p_node_attrs = 0;
    std::size_t num_extracted_p_link_attrs = 0;
    std::size_t num_extracted_v_node_attrs = 0;
    std::size_t num_extracted_v_link_attrs = 0;
    std::int64_t p_num_nodes = 0;
};

}  // namespace virne::utils
