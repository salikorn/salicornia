// salikorn / xlab — minimal experiment-run logger.
// Every binary logs structured JSONL events into runs/\<timestamp\>-\<slug\>/events.jsonl
// (the on-disk form of the experiment registry until the SQLite layer lands in M2).
/// \file
/// \brief Minimal experiment-run logger: JSONL event records under runs/\<timestamp\>-\<slug\>/.
///
/// Every spike, tool and test binary writes structured one-line JSON events to
/// \c runs/\<timestamp\>-\<slug\>/events.jsonl. Until the SQLite experiment registry lands
/// (milestone M2) this file-per-run layout is the on-disk form of the registry, so
/// event lines are free-form JSON objects with only two reserved keys: \c "t"
/// (epoch milliseconds) and \c "type" (event kind); every other field is supplied by
/// the caller as a raw JSON fragment.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace xlab {

/// \brief Local-time directory slug in the form \c YYYYmmdd-HHMMSS.
/// \return Timestamp string used as the prefix of a run directory name. Uses
///         \c localtime_s, so it reflects the machine's local timezone and is not
///         monotonic across DST changes (acceptable for directory naming).
inline std::string now_slug() {
    using namespace std::chrono;
    std::time_t t = system_clock::to_time_t(system_clock::now());
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", &tm);
    return buf;
}

/// \brief Current wall-clock time as milliseconds since the Unix epoch.
/// \return Fractional milliseconds (double, sub-ms precision from the system clock).
///         Used as the \c "t" field of every logged event.
inline double epoch_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(system_clock::now().time_since_epoch()).count();
}

/// \brief One experiment run: a timestamped directory plus an append-only events file.
///
/// Constructing a Run immediately creates \c runs/\<now_slug()\>-\<slug\>/ and opens
/// \c events.jsonl in binary append mode. Every event() call flushes, so a crashed
/// process still leaves all events up to the last completed call — that is the point
/// of the JSONL-per-event design on an iGPU where drivers occasionally hang the process.
class Run {
  public:
    /// \brief Creates the run directory and opens events.jsonl for appending.
    /// \param slug Short label for the run, e.g. \c "s8-d3d12" or \c "test-fp16";
    ///             becomes the directory-name suffix after the timestamp.
    /// \note Creates parent directories as needed (std::filesystem::create_directories).
    ///       If the stream fails to open, later event() writes silently no-op.
    explicit Run(std::string_view slug) {
        dir_ = std::filesystem::path("runs") / (now_slug() + "-" + std::string(slug));
        std::filesystem::create_directories(dir_);
        events_.open(dir_ / "events.jsonl", std::ios::binary | std::ios::app);
    }

    /// \brief Directory holding this run's artifacts (\c events.jsonl and anything
    ///        else the caller drops in, e.g. profiles or dumps).
    /// \return Path \c runs/\<timestamp\>-\<slug\>/, valid after construction.
    const std::filesystem::path& dir() const { return dir_; }

    /// \brief Appends one JSON event line to events.jsonl, flushes, and echoes to stdout.
    /// \param type        Event kind, stored as the \c "type" value (e.g. \c "env",
    ///                    \c "cand_timed", \c "fatal").
    /// \param fields_json Raw JSON fragment WITHOUT surrounding braces, e.g.
    ///                    \c "\"gpu\":\"intel\",\"ms\":42" — inserted verbatim after
    ///                    \c "t"/\c "type". Passing it un-parsed is deliberate: callers
    ///                    log printf-formatted numbers without JSON machinery. Any
    ///                    embedded quotes/escapes are therefore the caller's duty.
    /// \note The line shape is \c {"t":\<epoch_ms\>,"type":\<type\>[,\<fields_json\>]}.
    ///       There is no escaping layer — a malformed fragment corrupts exactly one line.
    void event(std::string_view type, std::string_view fields_json = {}) {
        std::string line = "{\"t\":" + std::to_string(epoch_ms()) + ",\"type\":\"" +
                           std::string(type) + "\"";
        if (!fields_json.empty()) line += "," + std::string(fields_json);
        line += "}\n";
        events_ << line;
        events_.flush();
        std::cout << "[" << type << "] "
                  << (fields_json.empty() ? std::string("{}") : std::string(fields_json))
                  << std::endl;
    }

  private:
    std::filesystem::path dir_;
    std::ofstream events_;
};

}  // namespace xlab
