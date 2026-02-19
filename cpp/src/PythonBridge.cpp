// PythonBridge.cpp - Cross-platform implementation of Python process bridge
#include "PythonBridge.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#endif

namespace NPCInference {

PythonBridge::PythonBridge() = default;

PythonBridge::~PythonBridge() {
    Stop();
}

bool PythonBridge::Start(const std::string& python_path, const std::string& script_path, const std::string& model_path) {
#ifdef _WIN32
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
#else
    int pipe_stdin[2];
    int pipe_stdout[2];

    if (pipe(pipe_stdin) == -1) return false;
    if (pipe(pipe_stdout) == -1) {
        close(pipe_stdin[0]);
        close(pipe_stdin[1]);
        return false;
    }

    pid_t pid = fork();
    if (pid == -1) {
        close(pipe_stdin[0]); close(pipe_stdin[1]);
        close(pipe_stdout[0]); close(pipe_stdout[1]);
        return false;
    }

    if (pid == 0) {
        // Child process
        dup2(pipe_stdin[0], STDIN_FILENO);
        dup2(pipe_stdout[1], STDOUT_FILENO);
        dup2(pipe_stdout[1], STDERR_FILENO);

        close(pipe_stdin[0]); close(pipe_stdin[1]);
        close(pipe_stdout[0]); close(pipe_stdout[1]);

        char* args[] = {(char*)python_path.c_str(), (char*)script_path.c_str(), (char*)model_path.c_str(), NULL};
        execvp(python_path.c_str(), args);
        exit(1);
    } else {
        // Parent process
        hProcess = pid;
        hStdInWrite = pipe_stdin[1];
        hStdOutRead = pipe_stdout[0];
        close(pipe_stdin[0]);
        close(pipe_stdout[1]);
    }
#endif
    
    // Check if process started and is waiting for "READY"
    std::cerr << "Waiting for Python bridge to initialize..." << std::endl;
    std::string line;
    for(int i=0; i<50; ++i) { // Limit retry
        line = ReadLine();
        if (line.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
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
#ifdef _WIN32
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
#else
    if (hProcess > 0) {
        kill(hProcess, SIGKILL);
        waitpid(hProcess, NULL, 0);
        hProcess = -1;
    }
    if (hStdInWrite != -1) {
        close(hStdInWrite);
        hStdInWrite = -1;
    }
    if (hStdOutRead != -1) {
        close(hStdOutRead);
        hStdOutRead = -1;
    }
#endif
}

nlohmann::json PythonBridge::SendRequest(const nlohmann::json& request) {
    if (!IsAlive()) throw std::runtime_error("Python process not running");

    WriteString(request.dump() + "\n");
    
    std::string line = ReadLine();
    if (line.empty()) throw std::runtime_error("Connection lost to Python process");

    return nlohmann::json::parse(line);
}

std::string PythonBridge::ReadLine() {
    std::string result;
    char ch;
#ifdef _WIN32
    DWORD dwRead;
    while (ReadFile(hStdOutRead, &ch, 1, &dwRead, NULL) && dwRead > 0) {
        if (ch == '\n') break;
        if (ch != '\r') result += ch;
    }
#else
    ssize_t n;
    while ((n = read(hStdOutRead, &ch, 1)) > 0) {
        if (ch == '\n') break;
        if (ch != '\r') result += ch;
    }
#endif
    return result;
}

void PythonBridge::WriteString(const std::string& str) {
#ifdef _WIN32
    DWORD dwWritten;
    WriteFile(hStdInWrite, str.c_str(), (DWORD)str.length(), &dwWritten, NULL);
#else
    write(hStdInWrite, str.c_str(), str.length());
#endif
}

} // namespace NPCInference
