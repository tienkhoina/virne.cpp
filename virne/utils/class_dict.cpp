#include "class_dict.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace
{

using virne::utils::ClassAnyList;
using virne::utils::ClassAnyListPtr;
using virne::utils::ClassDict;
using virne::utils::ClassDictSnapshot;
using virne::utils::ClassMapping;
using virne::utils::ClassMappingItem;
using virne::utils::ClassMappingPtr;

using DeepCopyMemo = std::unordered_map<const void*, std::any>;

thread_local bool inside_class_dict_parallel_task = false;

class ParallelTaskScope
{
public:
    ParallelTaskScope() noexcept
        : previous_(inside_class_dict_parallel_task)
    {
        inside_class_dict_parallel_task = true;
    }

    ~ParallelTaskScope()
    {
        inside_class_dict_parallel_task = previous_;
    }

    ParallelTaskScope(const ParallelTaskScope&) = delete;
    ParallelTaskScope& operator=(const ParallelTaskScope&) = delete;

private:
    bool previous_;
};

class DeterministicExecutor
{
public:
    DeterministicExecutor() = default;
    DeterministicExecutor(const DeterministicExecutor&) = delete;
    DeterministicExecutor& operator=(const DeterministicExecutor&) = delete;

    ~DeterministicExecutor()
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    template <typename Function>
    void run(std::size_t worker_count, Function&& function)
    {
        if (worker_count <= 1)
        {
            function(0);
            return;
        }

        std::lock_guard<std::mutex> execution_lock(execution_mutex_);
        ensure_workers(worker_count - 1);
        std::vector<std::exception_ptr> failures(worker_count);
        {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            task_ = [callable = std::forward<Function>(function), &failures](
                        std::size_t worker_index) mutable
            {
                try
                {
                    callable(worker_index);
                }
                catch (...)
                {
                    failures[worker_index] = std::current_exception();
                }
            };
            active_background_workers_ = worker_count - 1;
            completed_background_workers_ = 0;
            ++generation_;
        }
        ready_.notify_all();
        task_(0);

        std::unique_lock<std::mutex> state_lock(state_mutex_);
        finished_.wait(state_lock, [this]
        {
            return completed_background_workers_ ==
                   active_background_workers_;
        });
        task_ = {};
        state_lock.unlock();

        for (const auto& failure : failures)
        {
            if (failure)
            {
                std::rethrow_exception(failure);
            }
        }
    }

private:
    void ensure_workers(std::size_t count)
    {
        while (workers_.size() < count)
        {
            const std::size_t worker_index = workers_.size() + 1;
            std::size_t initial_generation = 0;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                initial_generation = generation_;
            }
            workers_.emplace_back(
                [this, worker_index, initial_generation]
                {
                    worker_loop(worker_index, initial_generation);
                });
        }
    }

    void worker_loop(
        std::size_t worker_index,
        std::size_t seen_generation)
    {
        std::unique_lock<std::mutex> lock(state_mutex_);
        for (;;)
        {
            ready_.wait(lock, [this, &seen_generation]
            {
                return generation_ != seen_generation || stopping_;
            });
            if (stopping_)
            {
                return;
            }
            seen_generation = generation_;
            if (worker_index > active_background_workers_)
            {
                continue;
            }
            const auto* task = &task_;
            lock.unlock();
            (*task)(worker_index);
            lock.lock();
            ++completed_background_workers_;
            if (completed_background_workers_ ==
                active_background_workers_)
            {
                finished_.notify_one();
            }
        }
    }

    std::mutex execution_mutex_;
    std::mutex state_mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::function<void(std::size_t)> task_;
    std::size_t generation_ = 0;
    std::size_t active_background_workers_ = 0;
    std::size_t completed_background_workers_ = 0;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

DeterministicExecutor& class_dict_executor()
{
    static DeterministicExecutor executor;
    return executor;
}

std::size_t available_worker_width()
{
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    {
        const int available = CPU_COUNT(&affinity);
        if (available > 0)
        {
            return static_cast<std::size_t>(available);
        }
    }
#endif
    return std::max<std::size_t>(
        1, std::thread::hardware_concurrency());
}

std::size_t resolved_worker_count(
    std::size_t requested,
    std::size_t item_count,
    std::size_t work_units)
{
    if (item_count == 0)
    {
        return 1;
    }
    if (requested == 1)
    {
        return 1;
    }
    std::size_t available = 0;
    if (requested == 0)
    {
        // Repeated 8-CPU sweeps selected a field-count threshold rather than
        // an object-count threshold so heterogeneous batches do not fan out
        // merely because they contain many tiny dictionaries.
        if (work_units < 8192)
        {
            return 1;
        }
        available = available_worker_width();
        requested = std::min<std::size_t>(8, available);
    }
    else
    {
        available = available_worker_width();
    }
    return std::max<std::size_t>(
        1,
        std::min(
            std::min(requested, item_count),
            available));
}

template <typename Values>
std::size_t batch_work_units(const Values& values) noexcept
{
    std::size_t total = 0;
    for (const auto& value : values)
    {
        const std::size_t current = value.size();
        if (current > std::numeric_limits<std::size_t>::max() - total)
        {
            return std::numeric_limits<std::size_t>::max();
        }
        total += current;
    }
    return total;
}

template <typename Function>
void deterministic_parallel_for(
    std::size_t item_count,
    std::size_t work_units,
    std::size_t requested_workers,
    Function&& function)
{
    const std::size_t worker_count = resolved_worker_count(
        requested_workers, item_count, work_units);
    // A custom std::any copy hook may invoke a ClassDict batch recursively.
    // The shared executor serializes top-level callers, so a nested attempt to
    // acquire it would deadlock. Nested work stays deterministic/sequential.
    if (worker_count == 1 || inside_class_dict_parallel_task)
    {
        for (std::size_t index = 0; index < item_count; ++index)
        {
            function(index);
        }
        return;
    }

    std::vector<std::exception_ptr> failures(item_count);
    class_dict_executor().run(worker_count, [&](std::size_t worker_index)
    {
        ParallelTaskScope task_scope;
        // Contiguous static blocks remove the atomic hot spot and prevent
        // adjacent vector result metadata from bouncing between worker cache
        // lines. Output/error slots remain in input order regardless of
        // completion order.
        const std::size_t base = item_count / worker_count;
        const std::size_t remainder = item_count % worker_count;
        const std::size_t begin = worker_index * base +
                                  std::min(worker_index, remainder);
        const std::size_t end = begin + base +
                                (worker_index < remainder ? 1 : 0);
        for (std::size_t index = begin; index < end; ++index)
        {
            try
            {
                function(index);
            }
            catch (...)
            {
                failures[index] = std::current_exception();
            }
        }
    });

    for (const auto& failure : failures)
    {
        if (failure)
        {
            std::rethrow_exception(failure);
        }
    }
}

std::any deep_copy_value(
    const std::any& value,
    DeepCopyMemo& memo);

ClassDict deep_copy_class_dict(
    const ClassDict& source,
    DeepCopyMemo& memo)
{
    ClassDict copy;
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        const virne::utils::ClassFieldId id{
            static_cast<std::uint32_t>(index)};
        copy.set(source.field_name(id), deep_copy_value(source.at(id), memo));
    }
    return copy;
}

std::any deep_copy_value(
    const std::any& value,
    DeepCopyMemo& memo)
{
    if (!value.has_value())
    {
        return {};
    }
    if (value.type() == typeid(ClassAnyList))
    {
        const auto& source = std::any_cast<const ClassAnyList&>(value);
        ClassAnyList copy;
        copy.reserve(source.size());
        for (const auto& item : source)
        {
            copy.push_back(deep_copy_value(item, memo));
        }
        return copy;
    }
    if (value.type() == typeid(ClassAnyListPtr))
    {
        const auto& source = std::any_cast<const ClassAnyListPtr&>(value);
        if (!source)
        {
            return ClassAnyListPtr{};
        }
        const auto found = memo.find(source.get());
        if (found != memo.end())
        {
            return found->second;
        }

        auto copy = std::make_shared<ClassAnyList>();
        memo.emplace(source.get(), copy);
        copy->reserve(source->size());
        for (const auto& item : *source)
        {
            copy->push_back(deep_copy_value(item, memo));
        }
        return copy;
    }
    if (value.type() == typeid(ClassMapping))
    {
        const auto& source = std::any_cast<const ClassMapping&>(value);
        ClassMapping copy;
        copy.ordered = source.ordered;
        copy.items.reserve(source.items.size());
        for (const ClassMappingItem& item : source.items)
        {
            copy.items.push_back(
                {deep_copy_value(item.key, memo),
                 deep_copy_value(item.value, memo)});
        }
        return copy;
    }
    if (value.type() == typeid(ClassMappingPtr))
    {
        const auto& source = std::any_cast<const ClassMappingPtr&>(value);
        if (!source)
        {
            return ClassMappingPtr{};
        }
        const auto found = memo.find(source.get());
        if (found != memo.end())
        {
            return found->second;
        }

        auto copy = std::make_shared<ClassMapping>();
        copy->ordered = source->ordered;
        memo.emplace(source.get(), copy);
        copy->items.reserve(source->items.size());
        for (const ClassMappingItem& item : source->items)
        {
            copy->items.push_back(
                {deep_copy_value(item.key, memo),
                 deep_copy_value(item.value, memo)});
        }
        return copy;
    }
    if (value.type() == typeid(ClassDict))
    {
        return deep_copy_class_dict(
            std::any_cast<const ClassDict&>(value), memo);
    }

    // std::any copies its held value. Standard value containers therefore
    // retain their normal deep value semantics; pointer-like custom types keep
    // their C++ copy semantics and may supply a value wrapper if needed.
    return value;
}

} // namespace

namespace virne::utils
{

ClassDict::ClassDict(const ClassDict& other)
    : fields_(other.fields_)
{
    rebuild_index();
}

ClassDict::ClassDict(ClassDict&& other) noexcept
    : fields_(std::move(other.fields_)),
      field_index_(std::move(other.field_index_))
{
    // Standard-container move construction preserves references to moved
    // elements.  The string_view keys therefore continue to address the deque
    // strings now owned by *this; no second hash/index build is needed.
    other.fields_.clear();
    other.field_index_.clear();
}

ClassDict& ClassDict::operator=(const ClassDict& other)
{
    if (this != &other)
    {
        ClassDict copy(other);
        swap(copy);
    }
    return *this;
}

ClassDict& ClassDict::operator=(ClassDict&& other) noexcept
{
    if (this != &other)
    {
        ClassDict moved(std::move(other));
        swap(moved);
    }
    return *this;
}

bool ClassDict::is_plain_mapping_snapshot_field(
    std::string_view key) noexcept
{
    return key == "node_slots" ||
           key == "link_paths" ||
           key == "node_slots_info" ||
           key == "link_paths_info";
}

std::optional<ClassFieldId> ClassDict::find_field_id(
    std::string_view key) const
{
    const auto found = field_index_.find(key);
    if (found == field_index_.end())
    {
        return std::nullopt;
    }
    return found->second;
}

ClassFieldId ClassDict::resolve_or_create(
    std::string_view key)
{
    if (fields_.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::overflow_error("ClassDict field ID space exhausted");
    }

    const ClassFieldId candidate{
        static_cast<std::uint32_t>(fields_.size())};
    fields_.push_back(
        {std::string(key),
         std::any{},
         false});
    try
    {
        const auto inserted = field_index_.emplace(
            std::string_view(fields_.back().key), candidate);
        if (inserted.second)
        {
            fields_.back().plain_mapping_on_snapshot =
                is_plain_mapping_snapshot_field(key);
            return candidate;
        }
        fields_.pop_back();
        return inserted.first->second;
    }
    catch (...)
    {
        fields_.pop_back();
        throw;
    }
}

ClassFieldId ClassDict::set(
    std::string_view key,
    std::any value)
{
    if (fields_.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::overflow_error("ClassDict field ID space exhausted");
    }

    const ClassFieldId candidate{
        static_cast<std::uint32_t>(fields_.size())};
    fields_.push_back(
        {std::string(key),
         std::move(value),
         false});
    const auto inserted = [&]
    {
        try
        {
            return field_index_.emplace(
                std::string_view(fields_.back().key), candidate);
        }
        catch (...)
        {
            fields_.pop_back();
            throw;
        }
    }();
    if (inserted.second)
    {
        fields_.back().plain_mapping_on_snapshot =
            is_plain_mapping_snapshot_field(key);
        return candidate;
    }

    std::any replacement = std::move(fields_.back().value);
    fields_.pop_back();
    const ClassFieldId existing = inserted.first->second;
    fields_[existing.value].value = std::move(replacement);
    return existing;
}

void ClassDict::set(
    ClassFieldId id,
    std::any value)
{
    fields_[checked_index(id)].value = std::move(value);
}

std::size_t ClassDict::checked_index(ClassFieldId id) const
{
    const std::size_t index = id.value;
    if (index >= fields_.size())
    {
        throw std::out_of_range("ClassDict field ID is out of range");
    }
    return index;
}

std::size_t ClassDict::checked_integer_index(std::int64_t index) const
{
    const std::uint64_t field_count = fields_.size();
    if (index >= 0)
    {
        const auto positive = static_cast<std::uint64_t>(index);
        if (positive < field_count)
        {
            return static_cast<std::size_t>(positive);
        }
    }
    else
    {
        // Avoid negating INT64_MIN. The magnitude of -(index + 1), plus one,
        // is representable as uint64_t.
        const std::uint64_t magnitude =
            static_cast<std::uint64_t>(-(index + 1)) + 1;
        if (magnitude <= field_count)
        {
            return static_cast<std::size_t>(field_count - magnitude);
        }
    }
    throw std::out_of_range("ClassDict integer index is out of range");
}

std::any& ClassDict::at(ClassFieldId id)
{
    return fields_[checked_index(id)].value;
}

const std::any& ClassDict::at(ClassFieldId id) const
{
    return fields_[checked_index(id)].value;
}

std::string_view ClassDict::field_name(ClassFieldId id) const
{
    return fields_[checked_index(id)].key;
}

std::any& ClassDict::at_index(std::int64_t index)
{
    return fields_[checked_integer_index(index)].value;
}

const std::any& ClassDict::at_index(std::int64_t index) const
{
    return fields_[checked_integer_index(index)].value;
}

std::any* ClassDict::find(std::string_view key)
{
    const auto id = find_field_id(key);
    return id ? &fields_[id->value].value : nullptr;
}

const std::any* ClassDict::find(std::string_view key) const
{
    const auto id = find_field_id(key);
    return id ? &fields_[id->value].value : nullptr;
}

std::any& ClassDict::get_or(
    std::string_view key,
    std::any& default_value)
{
    std::any* value = find(key);
    return value ? *value : default_value;
}

const std::any& ClassDict::get_or(
    std::string_view key,
    const std::any& default_value) const
{
    const std::any* value = find(key);
    return value ? *value : default_value;
}

void ClassDict::update(const ClassDictSnapshot& values)
{
    for (const ClassDictItem& item : values)
    {
        set(item.key, item.value);
    }
}

void ClassDict::update(const ClassDict& values)
{
    if (this == &values)
    {
        return;
    }
    for (const Field& field : values.fields_)
    {
        set(field.key, field.value);
    }
}

ClassDict ClassDict::from_dict(
    const ClassDictSnapshot& values)
{
    ClassDict result;
    result.field_index_.reserve(values.size());
    result.update(values);
    return result;
}

ClassDictSnapshot ClassDict::to_dict() const
{
    ClassDictSnapshot result;
    result.reserve(fields_.size());
    DeepCopyMemo memo;
    for (const Field& field : fields_)
    {
        std::any value = deep_copy_value(field.value, memo);
        if (field.plain_mapping_on_snapshot &&
            value.type() == typeid(ClassMapping))
        {
            std::any_cast<ClassMapping&>(value).ordered = false;
        }
        else if (field.plain_mapping_on_snapshot &&
                 value.type() == typeid(ClassMappingPtr))
        {
            const auto& mapping = std::any_cast<const ClassMappingPtr&>(value);
            if (mapping)
            {
                auto plain = std::make_shared<ClassMapping>(*mapping);
                plain->ordered = false;
                value = std::move(plain);
            }
        }
        result.push_back({field.key, std::move(value)});
    }
    return result;
}

std::vector<ClassDict> ClassDict::from_dict_batch(
    const std::vector<ClassDictSnapshot>& values,
    std::size_t worker_count)
{
    std::vector<ClassDict> results(values.size());
    deterministic_parallel_for(
        values.size(),
        worker_count == 0 ? batch_work_units(values) : 0,
        worker_count,
        [&](std::size_t index)
        {
            results[index] = from_dict(values[index]);
        });
    return results;
}

std::vector<ClassDictSnapshot> ClassDict::to_dict_batch(
    const std::vector<ClassDict>& values,
    std::size_t worker_count)
{
    std::vector<ClassDictSnapshot> results(values.size());
    deterministic_parallel_for(
        values.size(),
        worker_count == 0 ? batch_work_units(values) : 0,
        worker_count,
        [&](std::size_t index)
        {
            results[index] = values[index].to_dict();
        });
    return results;
}

void ClassDict::rebuild_index()
{
    field_index_.clear();
    field_index_.reserve(fields_.size());
    for (std::size_t index = 0; index < fields_.size(); ++index)
    {
        field_index_.emplace(
            std::string_view(fields_[index].key),
            ClassFieldId{static_cast<std::uint32_t>(index)});
    }
}

void ClassDict::swap(ClassDict& other) noexcept
{
    using std::swap;
    swap(fields_, other.fields_);
    swap(field_index_, other.field_index_);
}

void swap(ClassDict& left, ClassDict& right) noexcept
{
    left.swap(right);
}

} // namespace virne::utils
