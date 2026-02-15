#pragma once

#include <string>
#include <map>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace NPCInference {

/**
 * Centralized Error Logging System
 * Provides consistent logging across all components
 */
class ErrorLogger {
public:
    enum class Severity {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        FATAL
    };
    
    /**
     * Log a message with severity and component
     */
    static void Log(Severity level, const std::string& component, const std::string& message) {
        std::string timestamp = GetTimestamp();
        std::string severity_str = SeverityToString(level);
        
        std::cerr << "[" << timestamp << "] [" << severity_str << "] [" << component << "] " 
                  << message << std::endl;
    }
    
    /**
     * Log a message with additional context
     */
    static void LogWithContext(Severity level, const std::string& component, 
                               const std::string& message,
                               const std::map<std::string, std::string>& context) {
        std::string timestamp = GetTimestamp();
        std::string severity_str = SeverityToString(level);
        
        std::cerr << "[" << timestamp << "] [" << severity_str << "] [" << component << "] " 
                  << message;
        
        if (!context.empty()) {
            std::cerr << " | Context: {";
            bool first = true;
            for (const auto& [key, value] : context) {
                if (!first) std::cerr << ", ";
                std::cerr << key << ": " << value;
                first = false;
            }
            std::cerr << "}";
        }
        std::cerr << std::endl;
    }
    
    /**
     * Convenience methods
     */
    static void Debug(const std::string& component, const std::string& message) {
        Log(Severity::DEBUG, component, message);
    }
    
    static void Info(const std::string& component, const std::string& message) {
        Log(Severity::INFO, component, message);
    }
    
    static void Warning(const std::string& component, const std::string& message) {
        Log(Severity::WARNING, component, message);
    }
    
    static void Error(const std::string& component, const std::string& message) {
        Log(Severity::ERROR, component, message);
    }
    
    static void Fatal(const std::string& component, const std::string& message) {
        Log(Severity::FATAL, component, message);
    }

private:
    static std::string GetTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
    
    static std::string SeverityToString(Severity level) {
        switch (level) {
            case Severity::DEBUG:   return "DEBUG";
            case Severity::INFO:    return "INFO ";
            case Severity::WARNING: return "WARN ";
            case Severity::ERROR:   return "ERROR";
            case Severity::FATAL:   return "FATAL";
            default:                return "UNKNOWN";
        }
    }
};

} // namespace NPCInference
