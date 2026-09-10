#pragma once

#include <torch/nn/module.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace virne::porting {

inline void emit_tensor_bytes(const std::string& name, const torch::Tensor& input) {
    std::cout << name << '|';
    if (!input.defined()) {
        std::cout << "undefined\n";
        return;
    }
    const auto tensor = input.detach().cpu().contiguous();
    std::cout << c10::toString(tensor.scalar_type()) << '|';
    for (const auto size : tensor.sizes()) std::cout << size << ',';
    std::cout << '|';
    const auto* bytes = static_cast<const std::uint8_t*>(tensor.data_ptr());
    constexpr char hex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < tensor.nbytes(); ++i) {
        std::cout << hex[bytes[i] >> 4] << hex[bytes[i] & 15];
    }
    std::cout << '\n';
}

inline void emit_module_bytes(torch::nn::Module& model, const bool gradients) {
    std::vector<std::pair<std::string, torch::Tensor>> tensors;
    std::unordered_set<const c10::TensorImpl*> seen;
    for (const auto& item : model.named_parameters(true)) {
        tensors.emplace_back("state." + item.key(), item.value());
        if (gradients && seen.insert(item.value().unsafeGetTensorImpl()).second) {
            tensors.emplace_back("grad." + item.key(), item.value().grad());
        }
    }
    for (const auto& item : model.named_buffers(true)) {
        tensors.emplace_back("state." + item.key(), item.value());
    }
    std::sort(tensors.begin(), tensors.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    for (const auto& item : tensors) emit_tensor_bytes(item.first, item.second);
}

inline void fill_fixture_parameters(torch::nn::Module& model) {
    torch::NoGradGuard no_grad;
    std::int64_t index = 1;
    std::unordered_set<const c10::TensorImpl*> seen;
    for (const auto& item : model.named_parameters(true)) {
        if (!seen.insert(item.value().unsafeGetTensorImpl()).second) continue;
        item.value().copy_(torch::arange(item.value().numel(), item.value().options())
            .reshape(item.value().sizes()).add(static_cast<double>(index++)).mul(0.01));
    }
}

}  // namespace virne::porting
