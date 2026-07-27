#include "progress.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace
{
    class StreamCapture
    {
    public:
        StreamCapture()
            : old_out_(std::cout.rdbuf(out_.rdbuf()))
            , old_log_(std::clog.rdbuf(log_.rdbuf()))
        {
        }

        ~StreamCapture()
        {
            std::cout.rdbuf(old_out_);
            std::clog.rdbuf(old_log_);
        }

        const std::string output() const { return out_.str(); }
        const std::string log() const { return log_.str(); }

    private:
        std::ostringstream out_;
        std::ostringstream log_;
        std::streambuf* old_out_;
        std::streambuf* old_log_;
    };

    [[noreturn]] void fail(const std::string& message)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }

    void require(bool condition, const std::string& message)
    {
        if (!condition) {
            fail(message);
        }
    }

    std::size_t count_occurrences(const std::string& text, const std::string& needle)
    {
        std::size_t count = 0;
        std::size_t position = 0;
        while ((position = text.find(needle, position)) != std::string::npos) {
            ++count;
            position += needle.size();
        }
        return count;
    }

    bool stdout_is_terminal()
    {
#ifdef _WIN32
        return ::_isatty(::_fileno(stdout)) != 0;
#else
        return ::isatty(::fileno(stdout)) != 0;
#endif
    }

    void test_postfix_is_deferred_and_deterministic()
    {
        StreamCapture capture;
        Progress progress(10, "postfix");
        progress.update(1);
        const std::size_t before_postfix = capture.output().size();

        ProgressDict values;
        values.emplace("zeta", std::string("done"));
        values.emplace("alpha", 7);
        values.emplace("middle", 1.25);
        progress.set_postfix(values);
        require(capture.output().size() == before_postfix,
                "set_postfix must defer rendering to the rate-limited update");

        progress.finish();
        const std::string output = capture.output();
        require(output.find("alpha=7, middle=1.2500, zeta=done") != std::string::npos,
                "postfix keys or values are not formatted deterministically");
        require(output.find('\x1b') == std::string::npos,
                "progress output must not contain ANSI cursor movement");
    }

    void test_clamp_and_lifecycle()
    {
        StreamCapture capture;
        Progress progress(3, "clamp");
        progress.update(999);
        progress.finish();
        const std::string finished_output = capture.output();

        require(finished_output.find("100%") != std::string::npos,
                "an oversized update must clamp to 100 percent");
        require(finished_output.find("3/3") != std::string::npos,
                "an oversized update must clamp the counter to total");
        require(finished_output.find("999/3") == std::string::npos,
                "an oversized update leaked into the rendered counter");
        require(count_occurrences(finished_output, "clamp:") == 1,
                "update(total) followed by finish() rendered twice");

        progress.update(1);
        progress.set_postfix({{"ignored", 1}});
        progress.finish();
        require(capture.output() == finished_output,
                "updates after finish() must be harmless");
    }

    void test_safe_updates_and_zero_total()
    {
        {
            StreamCapture capture;
            Progress progress(10, "safe");
            require(progress.update_safe(5), "valid update_safe failed");
            require(!progress.update_safe(4), "decreasing update_safe must fail");
            require(capture.log().find("< previous") != std::string::npos,
                    "decreasing update_safe did not report its error");
            progress.finish();
        }

        {
            StreamCapture capture;
            Progress progress(0, "empty");
            require(progress.update_safe(0), "zero-total completion must be valid");
            progress.finish();
            const std::string output = capture.output();
            require(output.find("100%") != std::string::npos,
                    "zero-total progress must render as complete");
            require(output.find("0/0") != std::string::npos,
                    "zero-total progress counter is incorrect");
            require(output.find("nan") == std::string::npos &&
                        output.find("inf") == std::string::npos,
                    "zero-total progress produced a non-finite value");
        }
    }

    void test_repeated_values_do_not_redraw()
    {
        StreamCapture capture;
        Progress progress(1, "stable");
        const ProgressDict values{{"loss", 0.5}};
        progress.set_postfix(values);
        progress.update(1);
        progress.set_postfix(values);
        progress.update(1);
        progress.finish();
        require(count_occurrences(capture.output(), "stable:") == 1,
                "unchanged progress state was rendered more than once");
    }

    void test_redirected_output_has_no_terminal_controls()
    {
        if (stdout_is_terminal()) {
            return;
        }

        StreamCapture capture;
        Progress progress(2, "redirected");
        progress.update(2);
        progress.finish();
        const std::string output = capture.output();
        require(output.find('\r') == std::string::npos,
                "redirected output contains a carriage return");
        require(output.find('\x1b') == std::string::npos,
                "redirected output contains an ANSI escape");
        require(!output.empty() && output.back() == '\n',
                "redirected output is not newline terminated");
    }

    void test_absolute_and_incremental_contracts()
    {
        StreamCapture capture;
        Progress progress(10, "mixed");
        progress.update(4);
        require(progress.current() == 4,
                "absolute update no longer sets the completed count");
        progress.advance();
        progress.advance(3);
        require(progress.current() == 8,
                "advance(delta) did not increment from the absolute count");
        progress.advance(std::numeric_limits<std::size_t>::max());
        require(progress.current() == 10,
                "advance(delta) did not saturate without overflow");
        require(progress.total() == 10, "total accessor is incorrect");
        progress.close();
        require(progress.finished(), "close() did not finish Progress");
        require(capture.output().find("10/10") != std::string::npos,
                "incremental completion rendered an incorrect count");
    }

    void test_tqdm_adapter()
    {
        StreamCapture capture;
        TqdmProgress progress(3, "tqdm");
        progress.update();
        progress.update(2);
        require(progress.current() == 3,
                "TqdmProgress::update(n) is not delta based");
        progress.set_postfix({{"accepted", 2}});
        progress.close();
        require(progress.finished(), "TqdmProgress::close() did not finish");
        require(progress.total() == 3, "TqdmProgress total is incorrect");
        const auto output = capture.output();
        require(output.find("3/3") != std::string::npos,
                "TqdmProgress rendered an incorrect count");
        require(output.find("accepted=2") != std::string::npos,
                "TqdmProgress did not forward set_postfix");

        const auto finished_output = capture.output();
        progress.update(1);
        progress.finish();
        require(capture.output() == finished_output,
                "TqdmProgress lifecycle is not idempotent");
    }
}

int main()
{
    test_postfix_is_deferred_and_deterministic();
    test_clamp_and_lifecycle();
    test_safe_updates_and_zero_total();
    test_repeated_values_do_not_redraw();
    test_redirected_output_has_no_terminal_controls();
    test_absolute_and_incremental_contracts();
    test_tqdm_adapter();
    std::cout << "ALL PROGRESS TESTS PASSED\n";
    return 0;
}
