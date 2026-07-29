#include "class_dict.h"

#include <any>
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using virne::utils::ClassAnyList;
using virne::utils::ClassAnyListPtr;
using virne::utils::ClassDict;
using virne::utils::ClassDictItem;
using virne::utils::ClassDictSnapshot;
using virne::utils::ClassFieldId;
using virne::utils::ClassMapping;
using virne::utils::ClassMappingItem;
using virne::utils::ClassMappingPtr;

static_assert(
    sizeof(ClassFieldId) == sizeof(std::uint32_t),
    "hot-loop field IDs must stay compact");
static_assert(
    std::is_same_v<decltype(ClassDictItem::key), std::string>,
    "dynamic names belong only at the snapshot/boundary layer");

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& message)
{
    try
    {
        function();
    }
    catch (const Exception&)
    {
        return;
    }
    catch (...)
    {
        throw std::runtime_error(message + " (wrong exception type)");
    }
    throw std::runtime_error(message + " (no exception)");
}

template <typename T>
const T& any_as(const std::any& value, const std::string& message)
{
    const T* typed = std::any_cast<T>(&value);
    if (!typed)
    {
        throw std::runtime_error(message);
    }
    return *typed;
}

const std::any& snapshot_value(
    const ClassDictSnapshot& snapshot,
    std::size_t index,
    std::string_view key)
{
    require(index < snapshot.size(), "snapshot index out of range");
    require(snapshot[index].key == key, "snapshot insertion order mismatch");
    return snapshot[index].value;
}

ClassMappingPtr make_mapping(bool ordered)
{
    auto mapping = std::make_shared<ClassMapping>();
    mapping->ordered = ordered;
    mapping->items.push_back(
        ClassMappingItem{std::int64_t{1}, std::string("one")});
    mapping->items.push_back(
        ClassMappingItem{std::string("two"), std::int64_t{2}});
    return mapping;
}

struct ThrowOnCopy
{
    explicit ThrowOnCopy(int value = 0)
        : id(value)
    {
    }

    ThrowOnCopy(const ThrowOnCopy& other)
        : id(other.id)
    {
        if (throw_now.load(std::memory_order_relaxed))
        {
            throw std::runtime_error(
                "copy blocked " + std::to_string(id));
        }
    }

    ThrowOnCopy& operator=(const ThrowOnCopy& other)
    {
        if (throw_now.load(std::memory_order_relaxed))
        {
            throw std::runtime_error(
                "copy blocked " + std::to_string(other.id));
        }
        id = other.id;
        return *this;
    }

    int id = 0;
    static std::atomic<bool> throw_now;
};

std::atomic<bool> ThrowOnCopy::throw_now{false};

struct ReentrantOnCopy
{
    explicit ReentrantOnCopy(int value = 0)
        : id(value)
    {
    }

    ReentrantOnCopy(const ReentrantOnCopy& other)
        : id(other.id)
    {
        bool expected = true;
        if (!armed.compare_exchange_strong(
                expected, false, std::memory_order_relaxed))
        {
            return;
        }

        std::vector<ClassDictSnapshot> nested_input(64);
        for (std::size_t index = 0; index < nested_input.size(); ++index)
        {
            nested_input[index].push_back(
                {"value", std::int64_t{static_cast<std::int64_t>(index)}});
        }
        const auto nested_output = ClassDict::from_dict_batch(nested_input, 8);
        if (nested_output.size() != nested_input.size())
        {
            throw std::runtime_error("reentrant batch size mismatch");
        }
        nested_calls.fetch_add(1, std::memory_order_relaxed);
    }

    ReentrantOnCopy& operator=(const ReentrantOnCopy&) = default;

    int id = 0;
    static std::atomic<bool> armed;
    static std::atomic<std::size_t> nested_calls;
};

std::atomic<bool> ReentrantOnCopy::armed{false};
std::atomic<std::size_t> ReentrantOnCopy::nested_calls{0};

ClassDictSnapshot make_batch_snapshot(std::size_t object_index)
{
    ClassDictSnapshot snapshot;
    snapshot.reserve(32);
    for (std::size_t field = 0; field < 32; ++field)
    {
        snapshot.push_back(
            {"field_" + std::to_string(field),
             std::int64_t(object_index * 1000 + field)});
    }
    auto list = std::make_shared<ClassAnyList>();
    list->push_back(std::int64_t{static_cast<std::int64_t>(object_index)});
    list->push_back(std::string("payload"));
    snapshot.push_back({"payload", list});
    return snapshot;
}

void require_batch_equal(
    const std::vector<ClassDictSnapshot>& left,
    const std::vector<ClassDictSnapshot>& right)
{
    require(left.size() == right.size(), "batch size mismatch");
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        require(
            left[index].size() == right[index].size(),
            "batch snapshot field count mismatch");
        for (std::size_t field = 0; field < left[index].size(); ++field)
        {
            require(
                left[index][field].key == right[index][field].key,
                "batch snapshot key order mismatch");
            if (left[index][field].key == "payload")
            {
                const auto& lhs = any_as<ClassAnyListPtr>(
                    left[index][field].value,
                    "left payload type mismatch");
                const auto& rhs = any_as<ClassAnyListPtr>(
                    right[index][field].value,
                    "right payload type mismatch");
                require(lhs && rhs, "batch payload is null");
                require(
                    any_as<std::int64_t>((*lhs)[0], "left payload value") ==
                        any_as<std::int64_t>((*rhs)[0], "right payload value"),
                    "batch payload mismatch");
            }
            else
            {
                require(
                    any_as<std::int64_t>(
                        left[index][field].value,
                        "left scalar type mismatch") ==
                        any_as<std::int64_t>(
                            right[index][field].value,
                            "right scalar type mismatch"),
                    "batch scalar mismatch");
            }
        }
    }
}

} // namespace

int main()
{
    try
    {
        ClassDict empty;
        require(empty.empty() && empty.size() == 0, "empty state mismatch");
        require(!empty.find_field_id("missing"), "missing ID should be absent");
        require(empty.find("missing") == nullptr, "missing lookup should be null");
        const std::any default_value = std::string("fallback");
        require(
            &empty.get_or("missing", default_value) == &default_value,
            "get_or must preserve default identity");
        require_throws<std::out_of_range>(
            [&] { (void)empty.at_index(0); },
            "empty integer lookup must fail");

        const ClassFieldId alpha = empty.set_value("alpha", std::int64_t{7});
        const ClassFieldId beta = empty.set_value("beta", std::string("b"));
        const ClassFieldId none = empty.set("none", std::any{});
        require(alpha.value == 0 && beta.value == 1 && none.value == 2,
                "field IDs do not follow insertion order");
        require(!empty.at(none).has_value(), "explicit None representation mismatch");
        require(empty.find("none") != nullptr, "explicit None appears missing");
        require(
            any_as<std::int64_t>(empty.at(alpha), "alpha type mismatch") == 7,
            "alpha value mismatch");
        require(
            any_as<std::string>(empty.at_index(-2), "negative index type") == "b",
            "negative integer index mismatch");
        require(
            !empty.at_index(-1).has_value(),
            "last negative integer index mismatch");
        require_throws<std::out_of_range>(
            [&] { (void)empty.at_index(3); },
            "positive out-of-range index accepted");
        require_throws<std::out_of_range>(
            [&] { (void)empty.at_index(-4); },
            "negative out-of-range index accepted");
        require_throws<std::out_of_range>(
            [&]
            {
                (void)empty.at_index(std::numeric_limits<std::int64_t>::min());
            },
            "INT64_MIN index accepted");

        const ClassFieldId alpha_again = empty.set_value("alpha", std::int64_t{9});
        require(alpha_again == alpha, "overwrite changed stable field ID");
        require(empty.size() == 3, "overwrite changed field count");
        require(
            any_as<std::int64_t>(empty.at_index(0), "overwrite type") == 9,
            "overwrite value mismatch");
        const ClassFieldId beta_resolved = empty.resolve_or_create("beta");
        require(beta_resolved == beta, "resolve changed existing ID");
        require(
            any_as<std::string>(empty.at(beta), "resolve destroyed value") == "b",
            "resolve_or_create overwrote an existing value");
        const ClassFieldId gamma = empty.resolve_or_create("gamma");
        require(!empty.at(gamma).has_value(), "new resolved field is not None");
        empty.set(gamma, std::int64_t{11});
        require(
            any_as<std::int64_t>(empty.at(gamma), "ID set type") == 11,
            "ID set failed");
        require_throws<std::out_of_range>(
            [&] { (void)empty.at(ClassFieldId{999}); },
            "invalid field ID accepted");

        const ClassDictSnapshot initial{
            {"first", std::int64_t{1}},
            {"second", std::int64_t{2}},
            {"third", std::int64_t{3}}};
        ClassDict updated = ClassDict::from_dict(initial);
        updated.update(
            ClassDictSnapshot{
                {"second", std::int64_t{20}},
                {"fourth", std::int64_t{4}}});
        require(updated.size() == 4, "update field count mismatch");
        const ClassDictSnapshot updated_snapshot = updated.to_dict();
        require(
            any_as<std::int64_t>(
                snapshot_value(updated_snapshot, 1, "second"),
                "updated second type") == 20,
            "update did not preserve order/overwrite value");
        (void)snapshot_value(updated_snapshot, 3, "fourth");

        ClassDict copied(updated);
        require(
            copied.find_field_id("third") == updated.find_field_id("third"),
            "copy did not rebuild ID index");
        copied.set_value("second", std::int64_t{200});
        require(
            any_as<std::int64_t>(
                updated.at(*updated.find_field_id("second")),
                "source after copy type") == 20,
            "copy aliases source field storage");
        ClassDict moved(std::move(copied));
        require(moved.size() == 4, "move lost fields");
        ClassDict assigned;
        assigned = moved;
        ClassDict move_assigned;
        move_assigned = std::move(assigned);
        require(move_assigned.size() == 4, "assignment lost fields");
        move_assigned.update(move_assigned);
        require(move_assigned.size() == 4, "self-update changed fields");
        ClassDict swapped;
        swapped.set_value("only", std::int64_t{1});
        swap(swapped, move_assigned);
        require(swapped.size() == 4 && move_assigned.size() == 1,
                "swap broke field/index ownership");
        require(
            swapped.find_field_id("fourth").has_value(),
            "swap index was not valid");

        auto shared_list = std::make_shared<ClassAnyList>();
        shared_list->push_back(std::int64_t{1});
        ClassDict shallow = ClassDict::from_dict(
            {{"left", shared_list}, {"right", shared_list}});
        shared_list->push_back(std::int64_t{2});
        const auto& stored_left = any_as<ClassAnyListPtr>(
            shallow.at(*shallow.find_field_id("left")),
            "stored shared list type");
        require(stored_left.get() == shared_list.get(),
                "from_dict must be shallow like Python");
        require(stored_left->size() == 2, "shallow mutation was not observed");

        const ClassDictSnapshot deep = shallow.to_dict();
        const auto& deep_left = any_as<ClassAnyListPtr>(
            snapshot_value(deep, 0, "left"),
            "deep left type");
        const auto& deep_right = any_as<ClassAnyListPtr>(
            snapshot_value(deep, 1, "right"),
            "deep right type");
        require(deep_left && deep_right, "deep copied list is null");
        require(deep_left.get() != shared_list.get(),
                "to_dict retained original mutable identity");
        require(deep_left.get() == deep_right.get(),
                "to_dict did not preserve deepcopy alias memo");
        deep_left->push_back(std::int64_t{3});
        require(shared_list->size() == 2,
                "mutating to_dict output changed ClassDict source");

        auto nested_shared = std::make_shared<ClassAnyList>();
        nested_shared->push_back(std::int64_t{10});
        ClassDict nested = ClassDict::from_dict({{"inner", nested_shared}});
        ClassDict outer = ClassDict::from_dict(
            {{"outer", nested_shared}, {"nested", nested}});
        const ClassDictSnapshot outer_snapshot = outer.to_dict();
        const auto& outer_copy = any_as<ClassAnyListPtr>(
            outer_snapshot[0].value,
            "outer shared list type");
        const auto& nested_copy = any_as<ClassDict>(
            outer_snapshot[1].value,
            "nested ClassDict type");
        const auto& nested_inner_copy = any_as<ClassAnyListPtr>(
            nested_copy.at(*nested_copy.find_field_id("inner")),
            "nested shared list type");
        require(outer_copy.get() == nested_inner_copy.get(),
                "deepcopy memo did not span an outer/nested ClassDict");
        require(outer_copy.get() != nested_shared.get(),
                "nested ClassDict deepcopy retained its source list");
        outer_copy->push_back(std::int64_t{11});
        require(nested_inner_copy->size() == 2,
                "nested deepcopy alias was not preserved");
        require(nested_shared->size() == 1,
                "nested deepcopy mutation reached the source graph");

        auto cyclic = std::make_shared<ClassAnyList>();
        cyclic->push_back(cyclic);
        ClassDict cyclic_dict = ClassDict::from_dict({{"cycle", cyclic}});
        const auto cyclic_snapshot = cyclic_dict.to_dict();
        const auto& cycle_copy = any_as<ClassAnyListPtr>(
            cyclic_snapshot[0].value,
            "cycle copy type");
        const auto& cycle_child = any_as<ClassAnyListPtr>(
            (*cycle_copy)[0],
            "cycle child type");
        require(cycle_copy.get() == cycle_child.get(),
                "deepcopy cycle was not preserved");
        require(cycle_copy.get() != cyclic.get(),
                "deepcopy cycle retained original identity");
        // shared_ptr deliberately models Python graph identity for this
        // compatibility surface, but unlike Python's GC it cannot collect a
        // strong reference cycle.  The fixture owns both graphs and breaks
        // them after the cycle/alias contract has been checked.
        cycle_copy->clear();
        cyclic->clear();

        const ClassMappingPtr ordered_mapping = make_mapping(true);
        ClassDict mapping_dict = ClassDict::from_dict(
            {{"node_slots", ordered_mapping},
             {"ordinary", ordered_mapping},
             {"link_paths", ordered_mapping}});
        const ClassDictSnapshot mappings = mapping_dict.to_dict();
        const auto& node_slots = any_as<ClassMappingPtr>(
            mappings[0].value,
            "node_slots mapping type");
        const auto& ordinary = any_as<ClassMappingPtr>(
            mappings[1].value,
            "ordinary mapping type");
        const auto& link_paths = any_as<ClassMappingPtr>(
            mappings[2].value,
            "link_paths mapping type");
        require(node_slots && ordinary && link_paths, "mapping copy is null");
        require(!node_slots->ordered && ordinary->ordered && !link_paths->ordered,
                "special ordered mapping conversion mismatch");
        require(node_slots.get() != link_paths.get(),
                "two special dict conversions should create distinct dicts");
        require(ordinary.get() != ordered_mapping.get(),
                "ordinary mapping was not deep copied");
        require(node_slots->items.size() == 2,
                "mapping items were not preserved");

        ClassMapping mapping_value;
        mapping_value.ordered = true;
        mapping_value.items.push_back({std::int64_t{1}, std::int64_t{2}});
        ClassDict mapping_by_value = ClassDict::from_dict(
            {{"node_slots_info", mapping_value}});
        const auto mapping_value_snapshot = mapping_by_value.to_dict();
        require(
            !any_as<ClassMapping>(
                 mapping_value_snapshot[0].value,
                 "mapping value type").ordered,
            "by-value special mapping was not converted");

        constexpr std::size_t batch_size = 256;
        std::vector<ClassDictSnapshot> batch_input;
        batch_input.reserve(batch_size);
        for (std::size_t index = 0; index < batch_size; ++index)
        {
            batch_input.push_back(make_batch_snapshot(index));
        }
        const std::vector<ClassDict> sequential_objects =
            ClassDict::from_dict_batch(batch_input, 1);
        const std::vector<ClassDictSnapshot> sequential_output =
            ClassDict::to_dict_batch(sequential_objects, 1);
        for (std::size_t workers :
             std::array<std::size_t, 6>{2, 4, 8, 0, 1, 1000000})
        {
            const std::vector<ClassDict> objects =
                ClassDict::from_dict_batch(batch_input, workers);
            const std::vector<ClassDictSnapshot> output =
                ClassDict::to_dict_batch(objects, workers);
            require_batch_equal(output, sequential_output);
        }

        std::atomic<bool> concurrent_ok{true};
        std::vector<std::thread> callers;
        for (std::size_t caller = 0; caller < 6; ++caller)
        {
            callers.emplace_back([&, caller]
            {
                try
                {
                    const auto objects = ClassDict::from_dict_batch(
                        batch_input, caller % 2 == 0 ? 4 : 8);
                    const auto output = ClassDict::to_dict_batch(objects, 8);
                    require_batch_equal(output, sequential_output);
                }
                catch (...)
                {
                    concurrent_ok.store(false, std::memory_order_relaxed);
                }
            });
        }
        for (auto& caller : callers)
        {
            caller.join();
        }
        require(concurrent_ok.load(), "concurrent batch callers diverged");

        std::vector<ClassDict> throwing_batch;
        throwing_batch.reserve(64);
        ThrowOnCopy::throw_now.store(false, std::memory_order_relaxed);
        for (std::size_t index = 0; index < 64; ++index)
        {
            ClassDict item;
            if (index == 8 || index == 17)
            {
                item.set("bad", std::any(ThrowOnCopy(
                    static_cast<int>(index))));
            }
            else
            {
                item.set_value("good", std::int64_t{
                    static_cast<std::int64_t>(index)});
            }
            throwing_batch.push_back(std::move(item));
        }
        ThrowOnCopy::throw_now.store(true, std::memory_order_relaxed);
        bool caught_background_failure = false;
        try
        {
            (void)ClassDict::to_dict_batch(throwing_batch, 8);
        }
        catch (const std::runtime_error& error)
        {
            // With 64 items/8 contiguous blocks, indices 8 and 17 belong to
            // background workers 1 and 2.
            // The lower input index must win regardless of completion order.
            caught_background_failure =
                std::string(error.what()) == "copy blocked 8";
        }
        require(caught_background_failure,
                "batch background/error-index selection mismatch");
        ThrowOnCopy::throw_now.store(false, std::memory_order_relaxed);

        std::vector<ClassDict> reentrant_batch(64);
        reentrant_batch[8].set(
            "reentrant", std::any(ReentrantOnCopy{8}));
        ReentrantOnCopy::nested_calls.store(0, std::memory_order_relaxed);
        ReentrantOnCopy::armed.store(true, std::memory_order_relaxed);
        const auto reentrant_output = ClassDict::to_dict_batch(
            reentrant_batch, 8);
        require(reentrant_output.size() == reentrant_batch.size(),
                "reentrant outer batch size mismatch");
        require(
            ReentrantOnCopy::nested_calls.load(std::memory_order_relaxed) == 1,
            "custom copy did not complete one reentrant batch");

        std::cout << "vne_class_dict_unit: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        ThrowOnCopy::throw_now.store(false, std::memory_order_relaxed);
        std::cerr << "vne_class_dict_unit: FAIL: " << error.what() << '\n';
        return 1;
    }
}
