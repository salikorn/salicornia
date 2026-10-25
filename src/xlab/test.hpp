// salikorn / xlab / test — minimal self-registering test framework.
// No external dependency; CTest-compatible (non-zero exit on failure), JSONL results into
// runs/ via xlab::Run, per-test wall-clock timing, SKIP_IF support.
//
//   XLAB_TEST("fp16 roundtrip") { CHECK(f16_to_f32(f32_to_f16(1.f)) == 1.f); }
//   int main() { return xlab::test::run_all("fp16"); }
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

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
inline bool& current_failed() {
    static bool f = false;
    return f;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

// thrown by REQUIRE on failure and by SKIP_IF (skipped) to end the test early
struct TestSignal {
    bool skipped = false;
};

inline void note_fail(const char* kind, const std::string& expr, const std::string& detail) {
    current_failed() = true;
    std::printf("    [%s] %s%s%s\n", kind, expr.c_str(), detail.empty() ? "" : "  --  ",
                detail.c_str());
}

inline std::string describe(double v) {
    char b[48];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

// Run all registered tests; returns a process exit code (0 = all passed).
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
#define XLAB_TEST(name)                                                                \
    static void XLAB_CAT(xlab_test_fn_, __LINE__)();                                   \
    static ::xlab::test::Registrar XLAB_CAT(xlab_test_reg_, __LINE__)(                 \
        name, XLAB_CAT(xlab_test_fn_, __LINE__));                                      \
    static void XLAB_CAT(xlab_test_fn_, __LINE__)()

#define CHECK(cond) \
    ::xlab::test::note_check(static_cast<bool>(cond), #cond)
#define CHECK_MSG(cond, msg) ::xlab::test::note_check(static_cast<bool>(cond), #cond, (msg))

#define REQUIRE(cond)                                                     \
    do {                                                                  \
        if (!static_cast<bool>(cond)) {                                   \
            ::xlab::test::note_fail("require", #cond, "");                \
            throw ::xlab::test::TestSignal{};                             \
        }                                                                 \
    } while (0)

#define CHECK_THROWS(stmt)                                                do {                                                                      bool threw_ = false;                                                  try {                                                                     stmt;                                                             } catch (const std::exception&) {                                         threw_ = true;                                                    } catch (...) {                                                           threw_ = true;                                                    }                                                                     if (!threw_) ::xlab::test::note_fail("check_throws", #stmt, "did not throw");     } while (0)

#define SKIP_IF(cond)                                                       \
    do {                                                                    \
        if (static_cast<bool>(cond)) {                                      \
            throw ::xlab::test::TestSignal{/*.skipped =*/ true};            \
        }                                                                   \
    } while (0)

namespace xlab::test {
inline void note_check(bool ok, const char* expr, const std::string& msg = {}) {
    if (!ok) note_fail("check", expr, msg);
}
}  // namespace xlab::test

// numeric compares with value printing
#define CHECK_EQ(a, b)                                                              \
    do {                                                                            \
        auto va_ = (a);                                                             \
        auto vb_ = (b);                                                             \
        if (!(va_ == vb_))                                                          \
            ::xlab::test::note_fail("check_eq", #a " == " #b,                       \
                                    ::xlab::test::describe(double(va_)) + " != " +  \
                                        ::xlab::test::describe(double(vb_)));        \
    } while (0)

#define CHECK_CLOSE(a, b, tol)                                                      \
    do {                                                                            \
        double va_ = double(a);                                                     \
        double vb_ = double(b);                                                     \
        if (!(std::fabs(va_ - vb_) <= (tol)))                                       \
            ::xlab::test::note_fail("check_close", #a " ~ " #b " (tol " #tol ")",   \
                                    ::xlab::test::describe(va_) + " vs " +          \
                                        ::xlab::test::describe(vb_));               \
    } while (0)
