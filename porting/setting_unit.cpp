#include "setting.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace
{

using virne::utils::ReturnedSettingError;
using virne::utils::SettingDocument;
using virne::utils::SettingErrorCode;
using virne::utils::SettingException;
using virne::utils::SettingFormat;
using virne::utils::SettingInteger;
using virne::utils::SettingKeyId;
using virne::utils::SettingMode;
using virne::utils::SettingObject;
using virne::utils::SettingValue;
using virne::utils::SettingValueKind;

int failures = 0;

void fail(std::string_view message, int line)
{
    std::cerr << "line " << line << ": " << message << '\n';
    ++failures;
}

#define CHECK(condition)                                                        \
    do                                                                          \
    {                                                                           \
        if (!(condition))                                                       \
        {                                                                       \
            fail(#condition, __LINE__);                                         \
        }                                                                       \
    } while (false)

#define CHECK_EQ(left, right)                                                   \
    do                                                                          \
    {                                                                           \
        const auto& check_left = (left);                                         \
        const auto& check_right = (right);                                       \
        if (!(check_left == check_right))                                        \
        {                                                                       \
            fail(std::string(#left) + " == " + #right, __LINE__);              \
        }                                                                       \
    } while (false)

template <typename Function>
void check_exception(
    Function&& function,
    SettingErrorCode expected,
    int line)
{
    try
    {
        std::invoke(std::forward<Function>(function));
        fail("expected SettingException", line);
    }
    catch (const SettingException& error)
    {
        if (error.code() != expected)
        {
            fail("unexpected SettingException code", line);
        }
    }
    catch (...)
    {
        fail("expected SettingException, received another exception", line);
    }
}

#define CHECK_SETTING_EXCEPTION(expression, code)                               \
    check_exception([&]() { static_cast<void>(expression); }, code, __LINE__)

std::string read_bytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

void write_bytes(const std::filesystem::path& path, std::string_view bytes)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error("unable to create unit-test fixture");
    }
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        const auto base = std::filesystem::temp_directory_path();
        for (std::uint64_t index = 0; index < 10000U; ++index)
        {
            path_ = base / ("virne_setting_unit_" + std::to_string(index));
            std::error_code error;
            if (std::filesystem::create_directory(path_, error))
            {
                return;
            }
        }
        throw std::runtime_error("unable to create unique unit-test directory");
    }

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

SettingDocument make_mapping_document()
{
    SettingDocument document{SettingValue::make_object()};
    SettingObject& root = document.root.as_object();

    auto b = SettingValue::make_object();
    b.as_object().set("z", SettingValue(2.5));
    b.as_object().set("y", SettingValue(nullptr));
    root.set("b", std::move(b));

    auto a = SettingValue::make_list();
    a.as_list().emplace_back(std::int64_t{1});
    a.as_list().emplace_back(true);
    root.set("a", std::move(a));
    root.set("empty_map", SettingValue::make_object());
    root.set("empty_list", SettingValue::make_list());
    return document;
}

void test_fixed_discriminants_and_modes()
{
    CHECK(SettingFormat::json != SettingFormat::yaml);
    CHECK(SettingMode::read != SettingMode::write);
    CHECK(SettingValueKind::integer != SettingValueKind::real);

    using virne::utils::parse_setting_mode;
    CHECK_EQ(parse_setting_mode("r"), SettingMode::read);
    CHECK_EQ(parse_setting_mode("r+"), SettingMode::read_update);
    CHECK_EQ(parse_setting_mode("w"), SettingMode::write);
    CHECK_EQ(parse_setting_mode("w+"), SettingMode::write_update);
    CHECK_EQ(parse_setting_mode("a"), SettingMode::append);
    CHECK_EQ(parse_setting_mode("a+"), SettingMode::append_update);
    CHECK_EQ(parse_setting_mode("x"), SettingMode::exclusive_create);
    CHECK_EQ(parse_setting_mode("x+"), SettingMode::exclusive_create_update);
    CHECK_SETTING_EXCEPTION(
        parse_setting_mode("rb"), SettingErrorCode::invalid_mode);

    using virne::utils::setting_format_from_path;
    CHECK_EQ(setting_format_from_path("settings.json"), SettingFormat::json);
    CHECK_EQ(setting_format_from_path("settingsjson"), SettingFormat::json);
    CHECK_EQ(setting_format_from_path("settings.yaml"), SettingFormat::yaml);
    CHECK_EQ(setting_format_from_path("settingsyaml"), SettingFormat::yaml);
    CHECK(!setting_format_from_path("settings.yml").has_value());
    CHECK(!setting_format_from_path("settings.JSON").has_value());
    CHECK(!setting_format_from_path("son").has_value());
}

void test_key_ids_and_value_access()
{
    SettingObject object;
    const SettingKeyId alpha = object.resolve_or_create("alpha");
    CHECK_EQ(alpha.value, std::uint32_t{0});
    object.set(alpha, SettingValue(std::int64_t{3}));

    const SettingKeyId beta = object.set("beta", SettingValue("text"));
    CHECK_EQ(beta.value, std::uint32_t{1});
    CHECK_EQ(object.find_key_id("alpha"), std::optional<SettingKeyId>(alpha));
    CHECK(!object.find_key_id("missing").has_value());
    CHECK_EQ(object.key_name(alpha), std::string_view("alpha"));
    CHECK_EQ(object.at(alpha).as_integer(), SettingInteger{3});

    // This is the hot-path contract: no string reaches the repeated loop.
    for (std::int64_t value = 0; value < 10000; ++value)
    {
        object.set(alpha, SettingValue(value));
        CHECK_EQ(object.at(alpha).as_integer(), SettingInteger{value});
    }
    CHECK_EQ(object.set("alpha", SettingValue(std::int64_t{12})), alpha);
    CHECK_EQ(object.size(), std::size_t{2});

    SettingObject copied(object);
    const auto copied_alpha = copied.find_key_id("alpha");
    CHECK(copied_alpha.has_value());
    copied.set(*copied_alpha, SettingValue(std::int64_t{99}));
    CHECK_EQ(copied.at(*copied_alpha).as_integer(), SettingInteger{99});
    CHECK_EQ(object.at(alpha).as_integer(), SettingInteger{12});

    SettingObject moved(std::move(copied));
    const auto moved_alpha = moved.find_key_id("alpha");
    CHECK(moved_alpha.has_value());
    CHECK_EQ(moved.at(*moved_alpha).as_integer(), SettingInteger{99});

    SettingObject assigned;
    assigned.set("old", SettingValue(false));
    assigned = object;
    // Assignment replaces the schema; callers must resolve IDs again.
    const auto assigned_alpha = assigned.find_key_id("alpha");
    CHECK(assigned_alpha.has_value());
    CHECK(!assigned.find_key_id("old").has_value());

    SettingObject move_assigned;
    move_assigned.set("discarded", SettingValue(nullptr));
    move_assigned = std::move(assigned);
    const auto move_assigned_alpha = move_assigned.find_key_id("alpha");
    CHECK(move_assigned_alpha.has_value());
    CHECK_EQ(
        move_assigned.at(*move_assigned_alpha).as_integer(),
        SettingInteger{12});

    SettingObject other;
    other.set("other", SettingValue(true));
    swap(move_assigned, other);
    CHECK(move_assigned.find_key_id("other").has_value());
    CHECK(other.find_key_id("alpha").has_value());

    try
    {
        static_cast<void>(other.at(SettingKeyId{999U}));
        fail("invalid object-local ID must be rejected", __LINE__);
    }
    catch (const std::exception&)
    {
        // The public contract guarantees rejection, but does not prescribe a
        // SettingErrorCode for programmer misuse of an object-local ID.
    }
}

void test_json_parse_and_dump()
{
    using virne::utils::dump_setting;
    using virne::utils::parse_setting;

    const std::string input =
        "{\"b\": 1, \"a\": \"h\\u00e9\\ud83d\\ude00\", "
        "\"b\": 3, \"n\": null, \"t\": true, \"f\": false, "
        "\"huge\": 123456789012345678901234567890, "
        "\"nan\": NaN, \"pos\": Infinity, \"neg\": -Infinity}";
    const SettingDocument document = parse_setting(input, SettingFormat::json);
    const SettingObject& root = document.root.as_object();
    CHECK_EQ(root.size(), std::size_t{9});

    const auto b = root.find_key_id("b");
    const auto a = root.find_key_id("a");
    const auto huge = root.find_key_id("huge");
    const auto nan = root.find_key_id("nan");
    const auto pos = root.find_key_id("pos");
    const auto neg = root.find_key_id("neg");
    CHECK(b && a && huge && nan && pos && neg);
    CHECK_EQ(root.at(*b).as_integer(), SettingInteger{3});
    CHECK_EQ(root.key_name(*b), std::string_view("b"));
    CHECK_EQ(root.key_name(*a), std::string_view("a"));
    CHECK_EQ(root.at(*a).as_string(), std::string("h\xC3\xA9\xF0\x9F\x98\x80"));
    CHECK_EQ(
        root.at(*huge).as_integer(),
        SettingInteger("123456789012345678901234567890"));
    CHECK(std::isnan(root.at(*nan).as_real()));
    CHECK(std::isinf(root.at(*pos).as_real()));
    CHECK(root.at(*pos).as_real() > 0.0);
    CHECK(std::isinf(root.at(*neg).as_real()));
    CHECK(root.at(*neg).as_real() < 0.0);

    CHECK_EQ(
        dump_setting(document, SettingFormat::json),
        std::string(
            "{\"b\": 3, \"a\": \"h\\u00e9\\ud83d\\ude00\", "
            "\"n\": null, \"t\": true, \"f\": false, "
            "\"huge\": 123456789012345678901234567890, "
            "\"nan\": NaN, \"pos\": Infinity, \"neg\": -Infinity}"));

    SettingDocument negative_zero{SettingValue(-0.0)};
    CHECK_EQ(dump_setting(negative_zero, SettingFormat::json), "-0.0");
    SettingDocument unicode{SettingValue("h\xC3\xA9llo")};
    CHECK_EQ(dump_setting(unicode, SettingFormat::json), "\"h\\u00e9llo\"");

    CHECK_SETTING_EXCEPTION(
        parse_setting("{\"a\":}", SettingFormat::json),
        SettingErrorCode::parse_error);
    CHECK_SETTING_EXCEPTION(
        parse_setting("\xEF\xBB\xBF{}", SettingFormat::json),
        SettingErrorCode::parse_error);
    CHECK_SETTING_EXCEPTION(
        parse_setting("[1, 2] trailing", SettingFormat::json),
        SettingErrorCode::parse_error);
}

void test_yaml_parse_and_dump()
{
    using virne::utils::dump_setting;
    using virne::utils::parse_setting;

    const SettingDocument parsed = parse_setting(
        "flag: yes\n"
        "octal: 012\n"
        "nothing: null\n"
        "real: 1.5\n"
        "exponent_text: 1e-5\n"
        "items:\n"
        "- one\n"
        "- false\n"
        "empty_list: []\n"
        "empty_map: {}\n",
        SettingFormat::yaml);
    const SettingObject& root = parsed.root.as_object();
    CHECK(root.at(*root.find_key_id("flag")).as_bool());
    CHECK_EQ(
        root.at(*root.find_key_id("octal")).as_integer(),
        SettingInteger{10});
    CHECK(root.at(*root.find_key_id("nothing")).is_null());
    CHECK_EQ(root.at(*root.find_key_id("real")).as_real(), 1.5);
    CHECK_EQ(
        root.at(*root.find_key_id("exponent_text")).as_string(),
        std::string("1e-5"));
    CHECK_EQ(root.at(*root.find_key_id("items")).as_list().size(), std::size_t{2});
    CHECK(root.at(*root.find_key_id("empty_list")).as_list().empty());
    CHECK(root.at(*root.find_key_id("empty_map")).as_object().empty());

    const SettingDocument document = make_mapping_document();
    CHECK_EQ(
        dump_setting(document, SettingFormat::yaml),
        std::string(
            "a:\n"
            "- 1\n"
            "- true\n"
            "b:\n"
            "  y: null\n"
            "  z: 2.5\n"
            "empty_list: []\n"
            "empty_map: {}\n"));

    CHECK_EQ(
        dump_setting(SettingDocument{SettingValue(nullptr)}, SettingFormat::yaml),
        "null\n...\n");
    CHECK_EQ(
        dump_setting(SettingDocument{SettingValue(true)}, SettingFormat::yaml),
        "true\n...\n");
    CHECK_EQ(
        dump_setting(
            SettingDocument{SettingValue(std::int64_t{1})},
            SettingFormat::yaml),
        "1\n...\n");
    CHECK_EQ(
        dump_setting(SettingDocument{SettingValue(1.0)}, SettingFormat::yaml),
        "1.0\n...\n");
    CHECK_EQ(
        dump_setting(SettingDocument{SettingValue("yes")}, SettingFormat::yaml),
        "'yes'\n");
    CHECK_EQ(
        dump_setting(
            SettingDocument{SettingValue("h\xC3\xA9llo")},
            SettingFormat::yaml),
        "\"h\\xE9llo\"\n");

    const SettingDocument aliases = parse_setting(
        "a: &shared\n- 1\nb: *shared\n",
        SettingFormat::yaml);
    const SettingObject& alias_root = aliases.root.as_object();
    const SettingValue& alias_a = alias_root.at(*alias_root.find_key_id("a"));
    const SettingValue& alias_b = alias_root.at(*alias_root.find_key_id("b"));
    CHECK_EQ(alias_a.list_ptr().get(), alias_b.list_ptr().get());
    CHECK_EQ(
        dump_setting(aliases, SettingFormat::yaml),
        "a: &id001\n- 1\nb: *id001\n");

    CHECK_SETTING_EXCEPTION(
        parse_setting(
            "!!python/object/apply:os.system ['echo unsafe']\n",
            SettingFormat::yaml),
        SettingErrorCode::unsupported_yaml_feature);
    CHECK_SETTING_EXCEPTION(
        parse_setting("? [a, b]\n: 1\n", SettingFormat::yaml),
        SettingErrorCode::unsupported_yaml_feature);
    CHECK_SETTING_EXCEPTION(
        parse_setting("date: 2024-01-02\n", SettingFormat::yaml),
        SettingErrorCode::unsupported_yaml_feature);
    CHECK_SETTING_EXCEPTION(
        parse_setting(
            "base: &base {a: 1}\nout:\n  <<: *base\n",
            SettingFormat::yaml),
        SettingErrorCode::unsupported_yaml_feature);
    CHECK_SETTING_EXCEPTION(
        parse_setting("value: !!binary YQ==\n", SettingFormat::yaml),
        SettingErrorCode::unsupported_yaml_feature);
    CHECK_SETTING_EXCEPTION(
        parse_setting("a: 1\n---\nb: 2\n", SettingFormat::yaml),
        SettingErrorCode::parse_error);
}

void test_file_compatibility()
{
    using virne::utils::conver_format;
    using virne::utils::read_setting;
    using virne::utils::write_setting;
    using virne::utils::write_setting_strict;

    TemporaryDirectory directory;
    const auto missing_wrong = directory.path() / "missing.txt";
    CHECK_SETTING_EXCEPTION(
        read_setting(missing_wrong.string()), SettingErrorCode::open_failed);

    const auto existing_wrong = directory.path() / "existing.txt";
    write_bytes(existing_wrong, "unchanged");
    CHECK_SETTING_EXCEPTION(
        read_setting(existing_wrong.string()),
        SettingErrorCode::unsupported_format);

    const SettingDocument document = make_mapping_document();
    const auto wrong_output = directory.path() / "output.txt";
    write_bytes(wrong_output, "must be truncated");
    const auto result = write_setting(document, wrong_output.string());
    CHECK(std::holds_alternative<ReturnedSettingError>(result));
    CHECK_EQ(
        std::get<ReturnedSettingError>(result).code,
        SettingErrorCode::unsupported_format);
    CHECK_EQ(read_bytes(wrong_output), std::string());

    const auto strict_wrong = directory.path() / "strict.txt";
    CHECK_SETTING_EXCEPTION(
        write_setting_strict(document, strict_wrong.string()),
        SettingErrorCode::unsupported_format);
    CHECK(std::filesystem::exists(strict_wrong));

    const auto json_path = directory.path() / "document.json";
    const auto json_result = write_setting(document, json_path.string());
    CHECK(std::holds_alternative<
          std::reference_wrapper<const SettingDocument>>(json_result));
    CHECK_EQ(
        &std::get<std::reference_wrapper<const SettingDocument>>(json_result).get(),
        &document);
    CHECK_EQ(
        read_bytes(json_path),
        virne::utils::dump_setting(document, SettingFormat::json));
    CHECK_EQ(
        virne::utils::dump_setting(
            read_setting(json_path.string()), SettingFormat::json),
        read_bytes(json_path));

    const SettingDocument one{SettingValue(std::int64_t{1})};
    const auto append_path = directory.path() / "append.json";
    write_setting_strict(one, append_path.string(), SettingMode::write);
    write_setting_strict(one, append_path.string(), SettingMode::append);
    CHECK_EQ(read_bytes(append_path), "11");

    const auto exclusive_path = directory.path() / "exclusive.json";
    write_setting_strict(one, exclusive_path.string(), SettingMode::exclusive_create);
    CHECK_SETTING_EXCEPTION(
        write_setting_strict(
            one, exclusive_path.string(), SettingMode::exclusive_create),
        SettingErrorCode::open_failed);

    const auto source = directory.path() / "source.json";
    const auto destination = directory.path() / "destination.yaml";
    write_bytes(source, "{\"b\": 1, \"a\": [true, null]}");
    conver_format(source.string(), destination.string());
    CHECK_EQ(read_bytes(destination), "a:\n- true\n- null\nb: 1\n");

    const auto converter_wrong = directory.path() / "destination.bad";
    conver_format(source.string(), converter_wrong.string());
    CHECK(std::filesystem::exists(converter_wrong));
    CHECK_EQ(read_bytes(converter_wrong), std::string());
}

void test_batch_determinism_and_error_order()
{
    using virne::utils::dump_setting;
    using virne::utils::dump_setting_batch;
    using virne::utils::parse_setting_batch;

    const std::vector<std::string> json_inputs{
        "{\"id\": 0, \"v\": [1, 2, 3]}",
        "{\"id\": 1, \"v\": null}",
        "{\"id\": 2, \"v\": \"h\\u00e9\"}",
        "{\"id\": 3, \"v\": true}",
        "{\"id\": 4, \"v\": 123456789012345678901234567890}"};

    std::vector<std::string> baseline;
    for (const SettingDocument& document :
         parse_setting_batch(json_inputs, SettingFormat::json, 1U))
    {
        baseline.push_back(dump_setting(document, SettingFormat::json));
    }

    for (const std::size_t workers : {std::size_t{0}, std::size_t{2},
                                      std::size_t{8}, std::size_t{64}})
    {
        const auto documents =
            parse_setting_batch(json_inputs, SettingFormat::json, workers);
        CHECK_EQ(
            dump_setting_batch(documents, SettingFormat::json, workers),
            baseline);
    }

    CHECK(parse_setting_batch({}, SettingFormat::json, 8U).empty());
    CHECK(dump_setting_batch({}, SettingFormat::json, 8U).empty());

    // Copies intentionally share recursive container identity. Concurrent
    // first-use YAML sorting must initialize the ID-order cache exactly once
    // without a race or output drift.
    SettingDocument cold_shared{SettingValue::make_object()};
    cold_shared.root.as_object().set(
        "z", SettingValue(std::int64_t{2}));
    cold_shared.root.as_object().set(
        "a", SettingValue(std::int64_t{1}));
    const std::vector<SettingDocument> shared_documents(
        128, cold_shared);
    const auto shared_outputs = dump_setting_batch(
        shared_documents, SettingFormat::yaml, 8U);
    CHECK_EQ(shared_outputs.size(), std::size_t{128});
    for (const std::string& output : shared_outputs)
    {
        CHECK_EQ(output, std::string("a: 1\nz: 2\n"));
    }

    // The earliest input failure wins even when a later worker fails sooner.
    const std::vector<std::string> invalid_yaml{
        "!!python/object/apply:os.system ['x']\n",
        "[unterminated\n",
        "valid: true\n"};
    for (const std::size_t workers : {std::size_t{1}, std::size_t{2},
                                      std::size_t{8}, std::size_t{0}})
    {
        CHECK_SETTING_EXCEPTION(
            parse_setting_batch(invalid_yaml, SettingFormat::yaml, workers),
            SettingErrorCode::unsupported_yaml_feature);
    }
}

} // namespace

int main()
{
    test_fixed_discriminants_and_modes();
    test_key_ids_and_value_access();
    test_json_parse_and_dump();
    test_yaml_parse_and_dump();
    test_file_compatibility();
    test_batch_determinism_and_error_order();

    if (failures != 0)
    {
        std::cerr << failures << " setting unit assertion(s) failed\n";
        return 1;
    }

    std::cout << "setting unit tests passed\n";
    return 0;
}
