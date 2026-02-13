// PythonBridge.h - Manages the Python inference process and communication
#pragma once

#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef GetCurrentTime
#endif
#include <nlohmann/json.hpp>

namespace NPCInference {

/**
 * Handles communication with the Python NPC CLI via pipes.
 * Provides a memory-efficient fallback by using the Python 4-bit backend.
 */
class PythonBridge {
public:
    PythonBridge();
    ~PythonBridge();

    /**
     * Start the Python process
     * @param python_path Path to python.exe
     * @param script_path Path to npc_cli.py
     * @param model_path Path to the model (merged or base+adapter)
     * @return true if started successfully
     */
    bool Start(const std::string& python_path, const std::string& script_path, const std::string& model_path);

    /**
     * Stop the Python process
     */
    void Stop();

    /**
     * Send a request and wait for a response
     * @param request JSON request object
     * @return JSON response object
     */
    nlohmann::json SendRequest(const nlohmann::json& request);

    /**
     * Check if the process is still alive and ready
     */
    bool IsAlive() const { return hProcess != NULL; }

private:
    HANDLE hProcess = NULL;
    HANDLE hStdInWrite = NULL;
    HANDLE hStdOutRead = NULL;

    // Helper to read a line from a pipe
    std::string ReadLine();
    
    // Helper to write a string to a pipe
    void WriteString(const std::string& str);
};

} // namespace NPCInference
