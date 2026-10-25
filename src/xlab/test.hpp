// salikorn / xlab / test — minimal self-registering test framework.
// No external dependency; CTest-compatible (non-zero exit on failure), JSONL results into
// runs/ via xlab::Run, per-test wall-clock timing, SKIP_IF support.
//
//   XLAB_TEST("fp16 roundtrip") { CHECK(f16_to_f32(f32_to_f16(1.f)) == 1.f); }
//   int main() { return xlab::test::run_all("fp16"); }
/// \file
/// \brief Minimal self-registering test framework (no external dependency).
///
/// Tests are declared with XLAB_TEST(name) in any translation unit and self-register
/// into a static registry via a file-static Registrar object; main() calls
/// run_all(suite). Results stream to stdout in a CTest-friendly form (per-test
/// ok/FAIL/SKIP lines, summary line) and as JSONL \c "test"/\c "suite_done" events
/// into \c runs/\<ts\>-test-\<suite\>/events.jsonl via xlab::Run. The process exit code
/// is 0 iff every registered test passed (or was skipped), which is what CTest needs.
///
/// Assertion macros: CHECK/CHECK_MSG (record, continue), CHECK_EQ/CHECK_CLOSE
/// (record + value printing), REQUIRE (abort the test on failure), CHECK_THROWS,
/// and SKIP_IF (abort the test, counted as skipped — used for hardware-dependent
/// tests, e.g. tests/test_gpu.cpp skips everything without a working adapter).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "xlab/log.hpp"

namespace xlab::test {

/// \brief One registered test: display name + callable body.
struct TestCase {
    std::string name;
    std::function<void()> fn;
};

/// \brief Process-wide test registry (function-local static; safe across TUs because
///        Registrar constructors all append here before main() runs).
/// \return Reference to the shared vector of registered tests, in registration order.
inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
/// \brief Failure flag for the currently running test; reset by run_all() before each
///        test and set by note_fail(). Read by run_all() to classify the outcome.
/// \return Reference to the shared per-test failure flag.
inline bool& current_failed() {
    static bool f = false;
    return f;
}

/// \brief File-static registration helper created by the XLAB_TEST macro; its
///        constructor appends the test to the registry before main() runs.
struct Registrar {
    /// \brief Registers one test case.
    /// \param name Display name (string literal from XLAB_TEST).
    /// \param fn   Test body (the function declared by XLAB_TEST).
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

/// Thrown by REQUIRE on failure and by SKIP_IF (skipped) to end the test early
/// \brief Control-flow exception that ends a test body early.
/// \note \p skipped distinguishes the two exits: true (SKIP_IF) counts the test as
///       skipped; false (REQUIRE failure) counts it as failed. Any other exception
///       escaping a test body is caught by run_all() and recorded as a failure.
struct TestSignal {
    bool skipped = false;
};

/// \brief Records a failure: sets the per-test flag and prints one
///        \c "[kind] expr -- detail" line to stdout.
/// \param kind   Assertion family, e.g. \c "check", \c "require", \c "exception",
///               \c "check_throws".
/// \param expr   Stringified expression text.
/// \param detail Optional human-readable values (diffs, exception text).
inline void note_fail(const char* kind, const std::string& expr, const std::string& detail) {
    current_failed() = true;
    std::printf("    [%s] %s%s%s\n", kind, expr.c_str(), detail.empty() ? "" : "  --  ",
                detail.c_str());
}

/// \brief Formats a double with %.9g for assertion diagnostics.
/// \param v Value to print.
/// \return Short decimal string used in CHECK_EQ/CHECK_CLOSE failure details.
inline std::string describe(double v) {
    char b[48];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

/// Run all registered tests; returns a process exit code (0 = all passed).
/// \brief Runs every registered test in registration order and logs results.
/// \param suite Suite label; names the run directory (\c test-\<suite\>) and the
///              final summary line.
/// \return 0 if all tests passed or were skipped, 1 if any failed — suitable as the
///         return value of main() and for CTest pass/fail detection.
/// \note Per-test exceptions: TestSignal (REQUIRE/SKIP_IF) is control flow;
///       std::exception is recorded as a failure with e.what(); anything else would
///       propagate and terminate the process. Each test also emits a JSONL \c "test"
///       event with name/status/ms, and a \c "suite_done" event with the tallies.
inline int run_all(const char* suite) {
    xlab::Run run(std::string("test-") + suite);
    int failed = 0, skipped = 0, passed = 0;
    for (auto& t : registry()) {
        current_failed() = false;
        bool was_skipped = false;
        auto t0 = std::chrono::steady_clock::now();
        try {
            t.fn();
        } catch (const TestSignal& s) {
            was_skipped = s.skipped;
            if (!s.skipped) current_failed() = true;
        } catch (const std::exception& e) {
            note_fail("exception", t.name, e.what());
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                        .count();
        if (was_skipped) {
            ++skipped;
            std::printf("  SKIP  %s (%.0f ms)\n", t.name.c_str(), ms);
        } else if (current_failed()) {
            ++failed;
            std::printf("  FAIL  %s (%.0f ms)\n", t.name.c_str(), ms);
        } else {
            ++passed;
            std::printf("  ok    %s (%.0f ms)\n", t.name.c_str(), ms);
        }
        run.event("test", std::string("\"name\":\"") + t.name + "\",\"status\":" +
                               (was_skipped ? "\"skipped\""
                                            : (current_failed() ? "\"failed\"" : "\"passed\"")) +
                               ",\"ms\":" + std::to_string(ms));
    }
    char summary[160];
    std::snprintf(summary, sizeof summary, "\"passed\":%d,\"failed\":%d,\"skipped\":%d", passed,
                  failed, skipped);
    run.event("suite_done", summary);
    std::printf("%s: %d passed, %d failed, %d skipped\n", suite, passed, failed, skipped);
    return failed == 0 ? 0 : 1;
}

}  // namespace xlab::test

#define XLAB_CAT2(a, b) a##b
#define XLAB_CAT(a, b) XLAB_CAT2(a, b)
/// \brief Declares and self-registers a test function. Expands to a static function
///        definition followed by a static Registrar whose constructor runs before
///        main(); the unique symbol comes from __LINE__, so one test per line.
/// \param name Quoted display name, e.g. XLAB_TEST("fp16 known bit patterns").
#define XLAB_TEST(name)                                                                \
    static void XLAB_CAT(xlab_test_fn_, __LINE__)();                                   \
    static ::xlab::test::Registrar XLAB_CAT(xlab_test_reg_, __LINE__)(                 \
        name, XLAB_CAT(xlab_test_fn_, __LINE__));                                      \
    static void XLAB_CAT(xlab_test_fn_, __LINE__)()

/// \brief Records pass/fail of \p cond and CONTINUES the test (use REQUIRE to stop).
#define CHECK(cond) \
    ::xlab::test::note_check(static_cast<bool>(cond), #cond)
/// \brief CHECK with an extra message printed on failure.
#define CHECK_MSG(cond, msg) ::xlab::test::note_check(static_cast<bool>(cond), #cond, (msg))

/// \brief Failure assertion: records the failure and ABORTS the test body via
///        TestSignal{skipped=false}; the test is counted as failed.
#define REQUIRE(cond)                                                     \
    do {                                                                  \
        if (!static_cast<bool>(cond)) {                                   \
            ::xlab::test::note_fail("require", #cond, "");                \
            throw ::xlab::test::TestSignal{};                             \
        }                                                                 \
    } while (0)

/// \brief Failure assertion on a statement: the test fails unless \p stmt throws
///        (any exception type counts as throwing).
#define CHECK_THROWS(stmt)                                                do {                                                                      bool threw_ = false;                                                  try {                                                                     stmt;                                                             } catch (const std::exception&) {                                         threw_ = true;                                                    } catch (...) {                                                           threw_ = true;                                                    }                                                                     if (!threw_) ::xlab::test::note_fail("check_throws", #stmt, "did not throw");     } while (0)

/// \brief Skips the rest of the test when \p cond holds: throws TestSignal{skipped=true};
///        the test is counted as SKIP, not pass or fail. Used to gate hardware- or
///        file-dependent tests (GPU adapter present, model shard on disk, ...).
#define SKIP_IF(cond)                                                       \
    do {                                                                    \
        if (static_cast<bool>(cond)) {                                      \
            throw ::xlab::test::TestSignal{/*.skipped =*/ true};            \
        }                                                                   \
    } while (0)

namespace xlab::test {
/// \brief Macro back-end for CHECK/CHECK_MSG: records a "check" failure when !ok.
/// \param ok   Evaluated condition.
/// \param expr Stringified condition text.
/// \param msg  Optional failure detail.
inline void note_check(bool ok, const char* expr, const std::string& msg = {}) {
    if (!ok) note_fail("check", expr, msg);
}
}  // namespace xlab::test

/// \brief Equality check with both operand values printed on failure. Operands are
///        compared with == after a single evaluation, then cast to double for
///        diagnostics (lossless for ints/floats; use CHECK_CLOSE for tolerance).
#define CHECK_EQ(a, b)                                                              \
    do {                                                                            \
        auto va_ = (a);                                                             \
        auto vb_ = (b);                                                             \
        if (!(va_ == vb_))                                                          \
            ::xlab::test::note_fail("check_eq", #a " == " #b,                       \
                                    ::xlab::test::describe(double(va_)) + " != " +  \
                                        ::xlab::test::describe(double(vb_)));        \
    } while (0)

/// \brief Absolute-tolerance float check: passes iff |a-b| <= tol (double math).
#define CHECK_CLOSE(a, b, tol)                                                      \
    do {                                                                            \
        double va_ = double(a);                                                     \
        double vb_ = double(b);                                                     \
        if (!(std::fabs(va_ - vb_) <= (tol)))                                       \
            ::xlab::test::note_fail("check_close", #a " ~ " #b " (tol " #tol ")",   \
                                    ::xlab::test::describe(va_) + " vs " +          \
                                        ::xlab::test::describe(vb_));               \
    } while (0)
