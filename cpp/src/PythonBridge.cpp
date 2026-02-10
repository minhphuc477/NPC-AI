// PythonBridge.cpp - Windows implementation of Python process bridge
#include "PythonBridge.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace NPCInference {

PythonBridge::PythonBridge() = default;

PythonBridge::~PythonBridge() {
    Stop();
}

bool PythonBridge::Start(const std::string& python_path, const std::string& script_path, const std::string& model_path) {
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    // Create pipes for stdout and stdin
    HANDLE hStdOutWrite = NULL;
    if (!CreatePipe(&hStdOutRead, &hStdOutWrite, &saAttr, 0)) return false;
    if (!SetHandleInformation(hStdOutRead, HANDLE_FLAG_INHERIT, 0)) return false;

    HANDLE hStdInRead = NULL;
    if (!CreatePipe(&hStdInRead, &hStdInWrite, &saAttr, 0)) return false;
    if (!SetHandleInformation(hStdInWrite, HANDLE_FLAG_INHERIT, 0)) return false;

    // Build command line
    std::stringstream ss;
    ss << "\"" << python_path << "\" \"" << script_path << "\" \"" << model_path << "\"";
    std::string cmd = ss.str();

    STARTUPINFOA siStartInfo;
    PROCESS_INFORMATION piProcInfo;
    ZeroMemory(&siStartInfo, sizeof(STARTUPINFOA));
    siStartInfo.cb = sizeof(STARTUPINFOA);
    siStartInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    siStartInfo.hStdOutput = hStdOutWrite;
    siStartInfo.hStdInput = hStdInRead;
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

    if (!CreateProcessA(NULL, (LPSTR)cmd.c_str(), NULL, NULL, TRUE, 0, NULL, NULL, &siStartInfo, &piProcInfo)) {
        CloseHandle(hStdOutRead);
        CloseHandle(hStdOutWrite);
        CloseHandle(hStdInRead);
        CloseHandle(hStdInWrite);
        return false;
    }

    hProcess = piProcInfo.hProcess;
    CloseHandle(piProcInfo.hThread);
    CloseHandle(hStdOutWrite); // Close our copy of the write end of stdout
    CloseHandle(hStdInRead);   // Close our copy of the read end of stdin
    
    // Check if process started and is waiting for "READY"
    std::cerr << "Waiting for Python bridge to initialize..." << std::endl;
    std::string line;
    while (true) {
        line = ReadLine();
        if (line.empty()) break;
        std::cerr << "Python: " << line << std::endl;
        if (line.find("READY") != std::string::npos) {
            std::cerr << "Bridge connected!" << std::endl;
            return true;
        }
    }

    Stop();
    return false;
}

void PythonBridge::Stop() {
    if (hProcess) {
        TerminateProcess(hProcess, 0);
        CloseHandle(hProcess);
        hProcess = NULL;
    }
    if (hStdInWrite) {
        CloseHandle(hStdInWrite);
        hStdInWrite = NULL;
    }
    if (hStdOutRead) {
        CloseHandle(hStdOutRead);
        hStdOutRead = NULL;
    }
}

nlohmann::json PythonBridge::SendRequest(const nlohmann::json& request) {
    if (!hProcess) throw std::runtime_error("Python process not running");

    WriteString(request.dump() + "\n");
    
    std::string line = ReadLine();
    if (line.empty()) throw std::runtime_error("Connection lost to Python process");

    return nlohmann::json::parse(line);
}

std::string PythonBridge::ReadLine() {
    std::string result;
    char ch;
    DWORD dwRead;
    while (ReadFile(hStdOutRead, &ch, 1, &dwRead, NULL) && dwRead > 0) {
        if (ch == '\n') break;
        if (ch != '\r') result += ch;
    }
    return result;
}

void PythonBridge::WriteString(const std::string& str) {
    DWORD dwWritten;
    WriteFile(hStdInWrite, str.c_str(), (DWORD)str.length(), &dwWritten, NULL);
}

} // namespace NPCInference
