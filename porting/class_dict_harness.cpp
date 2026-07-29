#include "class_dict.h"

#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;
using virne::utils::ClassAnyList;
using virne::utils::ClassAnyListPtr;
using virne::utils::ClassDict;
using virne::utils::ClassDictItem;
using virne::utils::ClassDictSnapshot;
using virne::utils::ClassFieldId;
using virne::utils::ClassMapping;
using virne::utils::ClassMappingItem;
using virne::utils::ClassMappingPtr;

struct DifferentialRecord
{
    std::string name;
    std::string payload;
    std::vector<std::pair<std::string, std::string>> facts;
};

enum class BenchmarkKind : std::uint8_t
{
    StringGet,
    IdGet,
    ResolvedReferenceGet,
    ResolvedReferenceSet,
    FromDict,
    ToDict,
    BatchFromDict,
    BatchToDict,
};

struct BenchmarkCase
{
    const char* name;
    BenchmarkKind kind;
    std::size_t items;
    std::size_t fields;
};

constexpr BenchmarkCase kBenchmarkCases[] = {
    {"string_get", BenchmarkKind::StringGet, 200000, 256},
    {"id_get", BenchmarkKind::IdGet, 200000, 256},
    {"resolved_reference_get", BenchmarkKind::ResolvedReferenceGet, 200000, 256},
    {"resolved_reference_set", BenchmarkKind::ResolvedReferenceSet, 200000, 256},
    {"from_dict", BenchmarkKind::FromDict, 256, 128},
    {"to_dict", BenchmarkKind::ToDict, 128, 128},
    {"batch_from_dict", BenchmarkKind::BatchFromDict, 512, 64},
    {"batch_to_dict", BenchmarkKind::BatchToDict, 512, 64},
};

std::string hex_encode(std::string_view bytes)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.resize(bytes.size() * 2);
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
        const auto byte = static_cast<unsigned char>(bytes[index]);
        output[index * 2] = digits[byte >> 4U];
        output[index * 2 + 1] = digits[byte & 0x0fU];
    }
    return output;
}

std::uint64_t double_bits(double value) noexcept
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

class CanonicalSerializer
{
public:
    std::string serialize(const std::any& value)
    {
        if (!value.has_value())
        {
            return "n";
        }
        if (value.type() == typeid(bool))
        {
            return std::any_cast<bool>(value) ? "b1" : "b0";
        }
        if (value.type() == typeid(std::int64_t))
        {
            return "i" + std::to_string(std::any_cast<std::int64_t>(value));
        }
        if (value.type() == typeid(double))
        {
            std::ostringstream output;
            output << 'f' << std::hex << std::setw(16) << std::setfill('0')
                   << double_bits(std::any_cast<double>(value));
            return output.str();
        }
        if (value.type() == typeid(std::string))
        {
            const auto& text = std::any_cast<const std::string&>(value);
            return "s" + std::to_string(text.size()) + ":" + text;
        }
        if (value.type() == typeid(ClassAnyList))
        {
            return serialize_list(
                std::any_cast<const ClassAnyList&>(value), "l");
        }
        if (value.type() == typeid(ClassAnyListPtr))
        {
            return serialize_list_pointer(
                std::any_cast<const ClassAnyListPtr&>(value));
        }
        if (value.type() == typeid(ClassMapping))
        {
            return serialize_mapping(
                std::any_cast<const ClassMapping&>(value), "m");
        }
        if (value.type() == typeid(ClassMappingPtr))
        {
            return serialize_mapping_pointer(
                std::any_cast<const ClassMappingPtr&>(value));
        }
        throw std::invalid_argument("unsupported canonical std::any type");
    }

    std::string serialize(const ClassDict& values)
    {
        std::string output = "d" + std::to_string(values.size()) + "{";
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            const ClassFieldId id{static_cast<std::uint32_t>(index)};
            const std::string_view key = values.field_name(id);
            output += "k" + std::to_string(key.size()) + ":";
            output.append(key.data(), key.size());
            output += '=' + serialize(values.at(id)) + ';';
        }
        output += '}';
        return output;
    }

    std::string serialize(const ClassDictSnapshot& values)
    {
        std::string output = "d" + std::to_string(values.size()) + "{";
        for (const ClassDictItem& item : values)
        {
            output += "k" + std::to_string(item.key.size()) + ":" + item.key;
            output += '=' + serialize(item.value) + ';';
        }
        output += '}';
        return output;
    }

private:
    std::string serialize_list(
        const ClassAnyList& values,
        std::string_view tag)
    {
        std::string output(tag);
        output += std::to_string(values.size()) + '[';
        for (const auto& value : values)
        {
            output += serialize(value) + ';';
        }
        output += ']';
        return output;
    }

    std::string serialize_list_pointer(const ClassAnyListPtr& values)
    {
        if (!values)
        {
            return "Lnull";
        }
        const auto found = identities_.find(values.get());
        if (found != identities_.end())
        {
            return "r" + std::to_string(found->second);
        }
        const std::size_t identity = identities_.size();
        identities_.emplace(values.get(), identity);
        return "L" + std::to_string(identity) +
               serialize_list(*values, "");
    }

    std::string serialize_mapping(
        const ClassMapping& mapping,
        std::string_view tag)
    {
        std::string output(tag);
        output += mapping.ordered ? "o" : "p";
        output += std::to_string(mapping.items.size()) + '{';
        for (const ClassMappingItem& item : mapping.items)
        {
            output += serialize(item.key) + '=' + serialize(item.value) + ';';
        }
        output += '}';
        return output;
    }

    std::string serialize_mapping_pointer(const ClassMappingPtr& mapping)
    {
        if (!mapping)
        {
            return "Mnull";
        }
        const auto found = identities_.find(mapping.get());
        if (found != identities_.end())
        {
            return "r" + std::to_string(found->second);
        }
        const std::size_t identity = identities_.size();
        identities_.emplace(mapping.get(), identity);
        return "M" + std::to_string(identity) +
               serialize_mapping(*mapping, "");
    }

    std::unordered_map<const void*, std::size_t> identities_;
};

std::string canonical(const ClassDict& values)
{
    return CanonicalSerializer{}.serialize(values);
}

std::string canonical(const ClassDictSnapshot& values)
{
    return CanonicalSerializer{}.serialize(values);
}

std::string canonical(const std::any& value)
{
    return CanonicalSerializer{}.serialize(value);
}

ClassMappingPtr make_ordered_mapping()
{
    auto mapping = std::make_shared<ClassMapping>();
    mapping->ordered = true;
    mapping->items.push_back({std::int64_t{1}, std::string("one")});
    mapping->items.push_back({std::string("two"), std::int64_t{2}});
    return mapping;
}

ClassDictSnapshot make_snapshot(
    std::size_t object_index,
    std::size_t field_count)
{
    ClassDictSnapshot snapshot;
    snapshot.reserve(field_count);
    for (std::size_t field = 0; field < field_count; ++field)
    {
        snapshot.push_back(
            {"field_" + std::to_string(field),
             std::int64_t(object_index * 1000 + field)});
    }
    return snapshot;
}

std::vector<DifferentialRecord> differential_records()
{
    std::vector<DifferentialRecord> records;

    {
        ClassDict values;
        std::any fallback = std::int64_t{42};
        records.push_back(
            {"empty",
             canonical(values),
             {{"missing_is_none", values.find("missing") ? "0" : "1"},
              {"default_identity",
               &values.get_or("missing", fallback) == &fallback ? "1" : "0"}}});
    }

    {
        ClassDict values = ClassDict::from_dict(
            {{"integer", std::int64_t{-7}},
             {"boolean", true},
             {"float", -0.0},
             {"text", std::string("h\xc3\xa9llo\n")},
             {"none", std::any{}}});
        records.push_back(
            {"from_primitives",
             canonical(values),
             {{"index_0", canonical(values.at_index(0))},
              {"index_minus_1", canonical(values.at_index(-1))}}});
    }

    {
        ClassDict values;
        const ClassFieldId first = values.set_value("first", std::int64_t{1});
        const ClassFieldId second = values.set_value("second", std::int64_t{2});
        const ClassFieldId first_again =
            values.set_value("first", std::int64_t{9});
        const ClassFieldId empty_name =
            values.set_value("", std::int64_t{3});
        records.push_back(
            {"overwrite_order",
             canonical(values),
             {{"ids",
               std::to_string(first.value) + "," +
                   std::to_string(second.value) + "," +
                   std::to_string(first_again.value) + "," +
                   std::to_string(empty_name.value)}}});
    }

    {
        ClassDict values;
        const ClassFieldId existing =
            values.set_value("existing", std::int64_t{11});
        const ClassFieldId resolved_existing =
            values.resolve_or_create("existing");
        const ClassFieldId created = values.resolve_or_create("created");
        records.push_back(
            {"resolve_or_create",
             canonical(values),
             {{"existing_id_same",
               existing == resolved_existing ? "1" : "0"},
              {"created_none", values.at(created).has_value() ? "0" : "1"}}});
    }

    {
        ClassDict values = ClassDict::from_dict(
            {{"a", std::int64_t{1}}, {"b", std::int64_t{2}}});
        values.update(
            ClassDictSnapshot{
                {"b", std::int64_t{20}}, {"c", std::int64_t{3}}});
        ClassDict other = ClassDict::from_dict(
            {{"c", std::int64_t{30}}, {"d", std::int64_t{4}}});
        values.update(other);
        values.update(
            ClassDictSnapshot{
                {"d", std::int64_t{40}}, {"e", std::int64_t{5}}});
        records.push_back({"update_sequence", canonical(values), {}});
    }

    {
        auto shared = std::make_shared<ClassAnyList>();
        shared->push_back(std::int64_t{1});
        ClassDict values = ClassDict::from_dict({{"items", shared}});
        shared->push_back(std::int64_t{2});
        const auto& stored = values.at_as<ClassAnyListPtr>(
            *values.find_field_id("items"));
        records.push_back(
            {"shallow_input",
             canonical(values),
             {{"same_identity", stored.get() == shared.get() ? "1" : "0"}}});
    }

    {
        auto shared = std::make_shared<ClassAnyList>();
        shared->push_back(std::int64_t{1});
        shared->push_back(std::string("x"));
        ClassDict values = ClassDict::from_dict(
            {{"left", shared}, {"right", shared}});
        ClassDictSnapshot snapshot = values.to_dict();
        const auto& left = std::any_cast<const ClassAnyListPtr&>(
            snapshot[0].value);
        const auto& right = std::any_cast<const ClassAnyListPtr&>(
            snapshot[1].value);
        const bool alias_preserved = left.get() == right.get();
        const bool detached = left.get() != shared.get();
        left->push_back(std::int64_t{3});
        records.push_back(
            {"deepcopy_alias",
             canonical(snapshot),
             {{"alias_preserved", alias_preserved ? "1" : "0"},
              {"detached", detached ? "1" : "0"},
              {"source_size", std::to_string(shared->size())}}});
    }

    {
        auto shared = std::make_shared<ClassAnyList>();
        shared->push_back(std::int64_t{10});
        ClassDict nested = ClassDict::from_dict({{"inner", shared}});
        ClassDict values = ClassDict::from_dict(
            {{"outer", shared}, {"nested", nested}});
        const ClassDictSnapshot snapshot = values.to_dict();
        const auto& outer = std::any_cast<const ClassAnyListPtr&>(
            snapshot[0].value);
        const auto& nested_copy = std::any_cast<const ClassDict&>(
            snapshot[1].value);
        const auto& inner = nested_copy.at_as<ClassAnyListPtr>(
            *nested_copy.find_field_id("inner"));
        const bool alias_preserved = outer.get() == inner.get();
        const bool detached = outer.get() != shared.get();
        outer->push_back(std::int64_t{11});
        records.push_back(
            {"nested_class_dict_deepcopy",
             canonical(std::any(inner)),
             {{"alias_preserved", alias_preserved ? "1" : "0"},
              {"detached", detached ? "1" : "0"},
              {"source_size", std::to_string(shared->size())}}});
    }

    {
        auto cycle = std::make_shared<ClassAnyList>();
        cycle->push_back(cycle);
        ClassDict values = ClassDict::from_dict({{"cycle", cycle}});
        const ClassDictSnapshot snapshot = values.to_dict();
        const std::string payload = canonical(snapshot);
        const auto& cycle_copy = std::any_cast<const ClassAnyListPtr&>(
            snapshot[0].value);
        // The fixture owns these strong-reference cycles.  Preserve and
        // serialize them first, then release them explicitly because C++
        // shared_ptr has no cyclic garbage collector.
        cycle_copy->clear();
        cycle->clear();
        records.push_back({"deepcopy_cycle", payload, {}});
    }

    {
        const ClassMappingPtr mapping = make_ordered_mapping();
        ClassDict values = ClassDict::from_dict(
            {{"node_slots", mapping},
             {"ordinary", mapping},
             {"link_paths", mapping},
             {"node_slots_info", mapping},
             {"link_paths_info", mapping}});
        const ClassDictSnapshot snapshot = values.to_dict();
        const auto& node_slots = std::any_cast<const ClassMappingPtr&>(
            snapshot[0].value);
        const auto& ordinary = std::any_cast<const ClassMappingPtr&>(
            snapshot[1].value);
        const auto& link_paths = std::any_cast<const ClassMappingPtr&>(
            snapshot[2].value);
        records.push_back(
            {"ordered_mapping_conversion",
             canonical(snapshot),
             {{"special_distinct",
               node_slots.get() != link_paths.get() ? "1" : "0"},
              {"ordinary_detached",
               ordinary.get() != mapping.get() ? "1" : "0"}}});
    }

    {
        ClassDict values = ClassDict::from_dict(
            {{"a", std::int64_t{1}},
             {"b", std::int64_t{2}},
             {"c", std::int64_t{3}}});
        std::string positive_error;
        std::string negative_error;
        try
        {
            (void)values.at_index(3);
        }
        catch (const std::out_of_range&)
        {
            positive_error = "index_error";
        }
        try
        {
            (void)values.at_index(-4);
        }
        catch (const std::out_of_range&)
        {
            negative_error = "index_error";
        }
        records.push_back(
            {"integer_indexing",
             canonical(values),
             {{"zero", canonical(values.at_index(0))},
              {"minus_one", canonical(values.at_index(-1))},
              {"bool_true", canonical(values.at_index(1))},
              {"positive_error", positive_error},
              {"negative_error", negative_error}}});
    }

    {
        ClassDict values;
        values.set("explicit_none", std::any{});
        const std::any fallback = std::int64_t{99};
        const std::any* explicit_none = values.find("explicit_none");
        records.push_back(
            {"missing_and_explicit_none",
             canonical(values),
             {{"missing_default", canonical(values.get_or("missing", fallback))},
              {"explicit_found", explicit_none ? "1" : "0"},
              {"explicit_value",
               explicit_none ? canonical(*explicit_none) : "missing"}}});
    }

    {
        ClassDict source = ClassDict::from_dict(
            {{"x", std::int64_t{1}}, {"y", std::int64_t{2}}});
        ClassDict target = ClassDict::from_dict(
            {{"prefix", std::int64_t{0}}, {"x", std::int64_t{-1}}});
        target.update(source);
        target.update(target);
        records.push_back({"update_class_dict", canonical(target), {}});
    }

    {
        ClassDict values = ClassDict::from_dict(
            {{"kept", std::int64_t{1}}});
        // Python silently ignores positional arguments that are neither dict
        // nor ClassDict. The strongly typed C++ API simply has no such input.
        records.push_back({"unsupported_update_ignored", canonical(values), {}});
    }

    {
        ClassDict values = ClassDict::from_dict(
            {{"", std::int64_t{1}},
             {"caf\xc3\xa9", std::string("\xe2\x98\x83")},
             {"line\nkey", std::string("nul\0byte", 8)}});
        records.push_back({"string_keys_and_bytes", canonical(values), {}});
    }

    {
        std::vector<ClassDictSnapshot> input;
        for (std::size_t index = 0; index < 32; ++index)
        {
            input.push_back(make_snapshot(index, 12));
        }
        const auto sequential = ClassDict::from_dict_batch(input, 1);
        const auto parallel = ClassDict::from_dict_batch(input, 8);
        const auto sequential_output = ClassDict::to_dict_batch(sequential, 1);
        const auto parallel_output = ClassDict::to_dict_batch(parallel, 8);
        records.push_back(
            {"batch_order_workers",
             canonical(parallel_output.front()) +
                 canonical(parallel_output.back()),
             {{"same_first",
               canonical(sequential_output.front()) ==
                       canonical(parallel_output.front())
                   ? "1"
                   : "0"},
              {"same_last",
               canonical(sequential_output.back()) ==
                       canonical(parallel_output.back())
                   ? "1"
                   : "0"}}});
    }

    return records;
}

void emit_differential()
{
    const auto records = differential_records();
    std::cout << "diff_count=" << records.size() << '\n';
    for (std::size_t index = 0; index < records.size(); ++index)
    {
        const auto& record = records[index];
        const std::string prefix = "diff[" + std::to_string(index) + "]";
        std::cout << prefix << ".name=" << record.name << '\n';
        std::cout << prefix << ".payload_hex=" << hex_encode(record.payload)
                  << '\n';
        std::cout << prefix << ".fact_count=" << record.facts.size() << '\n';
        for (std::size_t fact = 0; fact < record.facts.size(); ++fact)
        {
            std::cout << prefix << ".fact[" << fact << "].name="
                      << record.facts[fact].first << '\n';
            std::cout << prefix << ".fact[" << fact << "].value_hex="
                      << hex_encode(record.facts[fact].second) << '\n';
        }
    }
}

std::uint64_t mix_checksum(
    std::uint64_t checksum,
    std::uint64_t value) noexcept
{
    checksum ^= value + 0x9e3779b97f4a7c15ULL +
                (checksum << 6U) + (checksum >> 2U);
    return checksum;
}

struct BenchmarkFixtures
{
    explicit BenchmarkFixtures(std::size_t batch_item_count)
        : lookup_snapshot(make_snapshot(0, 256)),
          lookup_dict(ClassDict::from_dict(lookup_snapshot)),
          lookup_id(*lookup_dict.find_field_id("field_127")),
          medium_snapshot(make_snapshot(7, 128)),
          medium_dict(ClassDict::from_dict(medium_snapshot))
    {
        batch_snapshots.reserve(batch_item_count);
        for (std::size_t index = 0; index < batch_item_count; ++index)
        {
            batch_snapshots.push_back(make_snapshot(index, 64));
        }
        batch_dicts = ClassDict::from_dict_batch(batch_snapshots, 1);

        auto list = std::make_shared<ClassAnyList>();
        for (std::int64_t value = 0; value < 32; ++value)
        {
            list->push_back(value);
        }
        medium_dict.set("nested", list);
        medium_dict.set("node_slots", make_ordered_mapping());
    }

    ClassDictSnapshot lookup_snapshot;
    ClassDict lookup_dict;
    ClassFieldId lookup_id;
    ClassDictSnapshot medium_snapshot;
    ClassDict medium_dict;
    std::vector<ClassDictSnapshot> batch_snapshots;
    std::vector<ClassDict> batch_dicts;
};

std::uint64_t run_benchmark_operation(
    const BenchmarkCase& item,
    BenchmarkFixtures& fixtures,
    std::size_t workers)
{
    std::uint64_t checksum = 0;
    switch (item.kind)
    {
    case BenchmarkKind::StringGet:
        for (std::size_t iteration = 0; iteration < item.items; ++iteration)
        {
            const std::any* value = fixtures.lookup_dict.find("field_127");
            checksum = mix_checksum(
                checksum,
                static_cast<std::uint64_t>(
                    std::any_cast<std::int64_t>(*value)) + iteration);
        }
        break;

    case BenchmarkKind::IdGet:
        for (std::size_t iteration = 0; iteration < item.items; ++iteration)
        {
            checksum = mix_checksum(
                checksum,
                static_cast<std::uint64_t>(
                    fixtures.lookup_dict.at_as<std::int64_t>(
                        fixtures.lookup_id)) + iteration);
        }
        break;

    case BenchmarkKind::ResolvedReferenceGet:
    {
        const std::int64_t& value =
            fixtures.lookup_dict.at_as<std::int64_t>(fixtures.lookup_id);
        for (std::size_t iteration = 0; iteration < item.items; ++iteration)
        {
            checksum = mix_checksum(
                checksum,
                static_cast<std::uint64_t>(value) + iteration);
        }
        break;
    }

    case BenchmarkKind::ResolvedReferenceSet:
    {
        std::int64_t& value =
            fixtures.lookup_dict.at_as<std::int64_t>(fixtures.lookup_id);
        value = 127;
        for (std::size_t iteration = 0; iteration < item.items; ++iteration)
        {
            value += static_cast<std::int64_t>((iteration & 1U) != 0);
            checksum = mix_checksum(
                checksum,
                static_cast<std::uint64_t>(value));
        }
        break;
    }

    case BenchmarkKind::FromDict:
        for (std::size_t iteration = 0; iteration < item.items; ++iteration)
        {
            const ClassDict value = ClassDict::from_dict(fixtures.medium_snapshot);
            checksum = mix_checksum(
                checksum,
                static_cast<std::uint64_t>(
                    value.at_as<std::int64_t>(ClassFieldId{127})));
        }
        break;

    case BenchmarkKind::ToDict:
        for (std::size_t iteration = 0; iteration < item.items; ++iteration)
        {
            const ClassDictSnapshot value = fixtures.medium_dict.to_dict();
            checksum = mix_checksum(
                checksum,
                value.size() +
                    static_cast<std::uint64_t>(
                        std::any_cast<std::int64_t>(value[127].value)));
        }
        break;

    case BenchmarkKind::BatchFromDict:
    {
        const auto values = ClassDict::from_dict_batch(
            fixtures.batch_snapshots, workers);
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            checksum = mix_checksum(
                checksum,
                static_cast<std::uint64_t>(
                    values[index].at_as<std::int64_t>(ClassFieldId{63})));
        }
        break;
    }

    case BenchmarkKind::BatchToDict:
    {
        const auto values = ClassDict::to_dict_batch(
            fixtures.batch_dicts, workers);
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            checksum = mix_checksum(
                checksum,
                static_cast<std::uint64_t>(
                    std::any_cast<std::int64_t>(values[index][63].value)));
        }
        break;
    }
    }
    return checksum;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return (values.size() & 1U) != 0
        ? values[middle]
        : (values[middle - 1] + values[middle]) * 0.5;
}

double percentile95(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
    return values[index];
}

void emit_benchmark(
    std::size_t workers,
    std::size_t warmups,
    std::size_t repetitions,
    std::size_t batch_items)
{
    BenchmarkFixtures fixtures(batch_items);
    std::cout << "bench_count="
              << sizeof(kBenchmarkCases) / sizeof(kBenchmarkCases[0]) << '\n';
    std::cout << "bench_workers=" << workers << '\n';
    std::cout << "bench_warmups=" << warmups << '\n';
    std::cout << "bench_repetitions=" << repetitions << '\n';

    for (std::size_t index = 0;
         index < sizeof(kBenchmarkCases) / sizeof(kBenchmarkCases[0]);
         ++index)
    {
        const BenchmarkCase& item = kBenchmarkCases[index];
        for (std::size_t warmup = 0; warmup < warmups; ++warmup)
        {
            (void)run_benchmark_operation(item, fixtures, workers);
        }

        std::vector<double> samples;
        samples.reserve(repetitions);
        std::uint64_t checksum = 0;
        for (std::size_t repetition = 0;
             repetition < repetitions;
             ++repetition)
        {
            const auto start = Clock::now();
            const std::uint64_t current =
                run_benchmark_operation(item, fixtures, workers);
            const auto stop = Clock::now();
            samples.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());
            if (repetition == 0)
            {
                checksum = current;
            }
            else if (checksum != current)
            {
                throw std::runtime_error("benchmark checksum changed");
            }
        }

        const double sample_median = median(samples);
        std::vector<double> deviations;
        deviations.reserve(samples.size());
        for (double sample : samples)
        {
            deviations.push_back(std::abs(sample - sample_median));
        }

        const std::string prefix = "bench[" + std::to_string(index) + "]";
        const bool is_batch =
            item.kind == BenchmarkKind::BatchFromDict ||
            item.kind == BenchmarkKind::BatchToDict;
        std::cout << prefix << ".name=" << item.name << '\n';
        std::cout << prefix << ".items="
                  << (is_batch ? batch_items : item.items) << '\n';
        std::cout << prefix << ".fields=" << item.fields << '\n';
        std::cout << std::setprecision(17);
        std::cout << prefix << ".cpp_median_ms=" << sample_median << '\n';
        std::cout << prefix << ".cpp_mad_ms=" << median(deviations) << '\n';
        std::cout << prefix << ".cpp_p95_ms=" << percentile95(samples) << '\n';
        std::cout << prefix << ".checksum=" << checksum << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        bool benchmark = false;
        std::size_t workers = 0;
        std::size_t warmups = 3;
        std::size_t repetitions = 11;
        std::size_t batch_items = 512;

        for (int index = 1; index < argc; ++index)
        {
            const std::string_view argument(argv[index]);
            if (argument == "--differential")
            {
                benchmark = false;
            }
            else if (argument == "--benchmark")
            {
                benchmark = true;
            }
            else if (argument == "--workers" && index + 1 < argc)
            {
                workers = static_cast<std::size_t>(
                    std::stoull(argv[++index]));
            }
            else if (argument == "--warmups" && index + 1 < argc)
            {
                warmups = static_cast<std::size_t>(
                    std::stoull(argv[++index]));
            }
            else if (argument == "--repetitions" && index + 1 < argc)
            {
                repetitions = static_cast<std::size_t>(
                    std::stoull(argv[++index]));
            }
            else if (argument == "--batch-items" && index + 1 < argc)
            {
                batch_items = static_cast<std::size_t>(
                    std::stoull(argv[++index]));
            }
            else if (argument == "--help")
            {
                std::cout
                    << "usage: vne_class_dict_harness "
                       "[--differential|--benchmark] [--workers N] "
                       "[--warmups N] [--repetitions N] "
                       "[--batch-items N]\n";
                return 0;
            }
            else
            {
                throw std::invalid_argument(
                    "unknown or incomplete argument: " + std::string(argument));
            }
        }

        if (repetitions == 0)
        {
            throw std::invalid_argument("--repetitions must be positive");
        }
        if (batch_items == 0)
        {
            throw std::invalid_argument("--batch-items must be positive");
        }
        if (benchmark)
        {
            emit_benchmark(workers, warmups, repetitions, batch_items);
        }
        else
        {
            emit_differential();
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "vne_class_dict_harness: " << error.what() << '\n';
        return 1;
    }
}
