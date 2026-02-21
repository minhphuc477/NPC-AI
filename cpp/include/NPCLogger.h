#pragma once

#include <iostream>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <ctime>

namespace NPCInference {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERR
};

class NPCLogger {
public:
    static NPCLogger& Instance() {
        static NPCLogger instance;
        return instance;
    }

    void Log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::ostream& out = (level == LogLevel::ERR) ? std::cerr : std::cout;

        out << "[" << std::put_time(std::localtime(&time_t_now), "%H:%M:%S")
            << "." << std::setfill('0') << std::setw(3) << ms.count() << "] "
            << "[" << LevelToString(level) << "] "
            << message << std::endl;
    }

    static void Info(const std::string& msg) { Instance().Log(LogLevel::INFO, msg); }
    static void Debug(const std::string& msg) { Instance().Log(LogLevel::DEBUG, msg); }
    static void Warn(const std::string& msg) { Instance().Log(LogLevel::WARN, msg); }
    static void Error(const std::string& msg) { Instance().Log(LogLevel::ERR, msg); }

private:
    NPCLogger() = default;
    std::mutex mutex_;

    const char* LevelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERR:   return "ERROR";
            default:              return "UNKN ";
        }
    }
};

} // namespace NPCInference
