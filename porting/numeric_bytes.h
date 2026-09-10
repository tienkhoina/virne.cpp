#pragma once
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace virne::porting {
template <class T>
void emit_numeric_bytes(const std::string& name, const char* dtype,
    const std::vector<T>& values, const std::vector<std::size_t>& shape) {
    std::cout << name << '|' << dtype << '|';
    for (auto extent : shape) std::cout << extent << ',';
    std::cout << '|';
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
    constexpr char hex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < values.size() * sizeof(T); ++i) std::cout << hex[bytes[i] >> 4] << hex[bytes[i] & 15];
    std::cout << '\n';
}
template <class T>
void emit_numeric_scalar(const std::string& name, const char* dtype, const T value) {
    emit_numeric_bytes(name, dtype, std::vector<T>{value}, {});
}
}  // namespace virne::porting
