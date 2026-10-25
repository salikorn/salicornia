// salikorn / xlab — minimal experiment-run logger.
// Every binary logs structured JSONL events into runs/<timestamp>-<slug>/events.jsonl
// (the on-disk form of the experiment registry until the SQLite layer lands in M2).
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace xlab {

inline std::string now_slug() {
    using namespace std::chrono;
    std::time_t t = system_clock::to_time_t(system_clock::now());
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", &tm);
    return buf;
}

inline double epoch_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(system_clock::now().time_since_epoch()).count();
}

class Run {
  public:
    explicit Run(std::string_view slug) {
        dir_ = std::filesystem::path("runs") / (now_slug() + "-" + std::string(slug));
        std::filesystem::create_directories(dir_);
        events_.open(dir_ / "events.jsonl", std::ios::binary | std::ios::app);
    }

    const std::filesystem::path& dir() const { return dir_; }

    // fields_json: raw JSON fragment without surrounding braces, e.g. "\"gpu\":\"intel\",..."
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
