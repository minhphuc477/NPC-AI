#include "OllamaClient.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

// Windows-specific includes for HTTP
#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#endif

namespace NPCInference {

OllamaClient::OllamaClient(const std::string& model_name, const std::string& base_url)
    : model_name_(model_name), base_url_(base_url) {
}

std::string OllamaClient::Generate(const std::string& prompt, int max_tokens, float temperature) {
    // Build JSON request
    nlohmann::json request;
    request["model"] = model_name_;
    request["prompt"] = prompt;
    request["stream"] = false;
    request["options"]["num_predict"] = max_tokens;
    request["options"]["temperature"] = temperature;
    
    std::string json_str = request.dump();
    
    // Call Ollama API
    std::string response = HttpPost(base_url_ + "/api/generate", json_str, &cancel_flag_);
    
    // Soft reset cancel flag afterwards
    cancel_flag_ = false;
    
    if (response.empty()) {
        return "[Error: Failed to connect to Ollama. Make sure 'ollama serve' is running.]";
    }
    
    try {
        nlohmann::json response_json = nlohmann::json::parse(response);
        if (response_json.contains("response")) {
            return response_json["response"].get<std::string>();
        } else if (response_json.contains("error")) {
            return "[Ollama Error: " + response_json["error"].get<std::string>() + "]";
        }
    } catch (const std::exception& e) {
        return "[Error parsing Ollama response: " + std::string(e.what()) + "]";
    }
    
    return "[Error: Unexpected response from Ollama]";
}

std::future<std::string> OllamaClient::GenerateAsync(const std::string& prompt, int max_tokens, float temperature) {
    return std::async(std::launch::async, [this, prompt, max_tokens, temperature]() {
        return this->Generate(prompt, max_tokens, temperature);
    });
}

std::future<std::string> OllamaClient::GenerateStreamAsync(
    const std::string& prompt,
    std::function<void(const std::string&)> on_token_callback,
    std::function<void(const std::string&)> on_action_callback,
    int max_tokens,
    float temperature
) {
    return std::async(std::launch::async, [this, prompt, max_tokens, temperature, on_token_callback, on_action_callback]() {
        nlohmann::json request;
        request["model"] = model_name_;
        request["prompt"] = prompt;
        request["stream"] = true; // Enable streaming
        request["options"]["num_predict"] = max_tokens;
        request["options"]["temperature"] = temperature;
        
        std::string json_str = request.dump();
        
        std::string full_response = "";
        std::string action_buffer = "";
        bool in_action = false;
        
        auto chunk_parser = [&](const std::string& chunk_raw) {
            // Ollama sends chunks of NDJSON (Newline Delimited JSON)
            std::stringstream ss(chunk_raw);
            std::string line;
            while (std::getline(ss, line)) {
                if (line.empty()) continue;
                try {
                    nlohmann::json j = nlohmann::json::parse(line);
                    if (j.contains("response")) {
                        std::string token = j["response"].get<std::string>();
                        full_response += token;
                        
                        // Action Streaming FSM: Look for *action*
                        for (char c : token) {
                            if (c == '*') {
                                if (in_action) {
                                    // End of action
                                    in_action = false;
                                    if (on_action_callback && !action_buffer.empty()) {
                                        on_action_callback(action_buffer);
                                    }
                                    action_buffer.clear();
                                } else {
                                    // Start of action
                                    in_action = true;
                                    action_buffer.clear();
                                }
                            } else if (in_action) {
                                action_buffer += c;
                            }
                        }
                        
                        if (on_token_callback && !in_action && token.find('*') == std::string::npos) {
                            on_token_callback(token);
                        }
                    }
                } catch (...) {
                    // Ignore partial json lines if they occur
                }
            }
        };

        HttpPost(base_url_ + "/api/generate", json_str, &cancel_flag_, chunk_parser);
        cancel_flag_ = false;

        return full_response;
    });
}

bool OllamaClient::IsReady() {
    try {
        std::string response = HttpPost(base_url_ + "/api/tags", "{}");
        return !response.empty();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return false;
    }
}

#ifdef _WIN32
// RAII wrapper for WinHTTP HINTERNET handles
struct WinHttpDeleter {
    void operator()(void* handle) const {
        if (handle) {
            WinHttpCloseHandle(handle);
        }
    }
};
using unique_hinternet = std::unique_ptr<void, WinHttpDeleter>;

std::string OllamaClient::HttpPost(const std::string& url, const std::string& json_data, std::atomic<bool>* local_cancel, std::function<void(const std::string&)> on_chunk_received) {
    // Parse URL
    std::wstring wurl(url.begin(), url.end());
    
    URL_COMPONENTS urlComp;
    ZeroMemory(&urlComp, sizeof(urlComp));
    urlComp.dwStructSize = sizeof(urlComp);
    
    wchar_t hostname[256];
    wchar_t path[1024];
    urlComp.lpszHostName = hostname;
    urlComp.dwHostNameLength = sizeof(hostname) / sizeof(wchar_t);
    urlComp.lpszUrlPath = path;
    urlComp.dwUrlPathLength = sizeof(path) / sizeof(wchar_t);
    
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &urlComp)) {
        return "";
    }
    
    // Initialize WinHTTP
    unique_hinternet hSession(WinHttpOpen(
        L"OllamaClient/1.0",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        0
    ));
    
    if (!hSession) return "";
    
    // Set Timeouts: Resolve=10s, Connect=10s, Send=30s, Receive=120s (2 minutes)
    WinHttpSetTimeouts(hSession.get(), 10000, 10000, 30000, 120000);
    
    // Connect
    unique_hinternet hConnect(WinHttpConnect(
        hSession.get(),
        hostname,
        urlComp.nPort,
        0
    ));
    
    if (!hConnect) {
        return "";
    }
    
    // Open request
    unique_hinternet hRequest(WinHttpOpenRequest(
        hConnect.get(),
        L"POST",
        path,
        NULL,
        WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES,
        0
    ));
    
    if (!hRequest) {
        return "";
    }

    // Safely store handle for asynchronous cancellation
    {
        std::lock_guard<std::mutex> lock(handle_mutex_);
        active_request_handle_ = hRequest.get();
    }
    
    // Set headers
    std::wstring headers = L"Content-Type: application/json\r\n";
    
    // Send request
    BOOL result = WinHttpSendRequest(
        hRequest.get(),
        headers.c_str(),
        -1,
        (LPVOID)json_data.c_str(),
        json_data.length(),
        json_data.length(),
        0
    );
    
    if (!result) {
        return "";
    }
    
    // Receive response
    result = WinHttpReceiveResponse(hRequest.get(), NULL);
    if (!result) {
        return "";
    }
    
    // Read data
    std::string response_data;
    DWORD bytesAvailable = 0;
    DWORD bytesRead = 0;
    char buffer[4096];
    
    do {
        if (local_cancel && local_cancel->load()) {
            response_data = "[Cancelled by Player]";
            break;
        }

        bytesAvailable = 0;
        if (!WinHttpQueryDataAvailable(hRequest.get(), &bytesAvailable)) {
            break;
        }
        
        if (bytesAvailable > 0) {
            DWORD toRead = (std::min)(bytesAvailable, (DWORD)sizeof(buffer));
            if (WinHttpReadData(hRequest.get(), buffer, toRead, &bytesRead)) {
                if (on_chunk_received && bytesRead > 0) {
                    on_chunk_received(std::string(buffer, bytesRead));
                }
                response_data.append(buffer, bytesRead);
            }
        }
    } while (bytesAvailable > 0);
    
    // Clear the active handle lock
    {
        std::lock_guard<std::mutex> lock(handle_mutex_);
        active_request_handle_ = nullptr;
    }

    // Resources are automatically cleaned up by unique_hinternet
    
    return response_data;
}
#else
// Linux/Mac implementation using libcurl
#include <curl/curl.h>

// Callback function to write response data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string OllamaClient::HttpPost(const std::string& url, const std::string& json_data, std::atomic<bool>* local_cancel, std::function<void(const std::string&)> on_chunk_received) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize libcurl" << std::endl;
        return "";
    }
    
    std::string response_data;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);  // 30 second timeout
    
    // We don't fully support live callbacks in libcurl in this simplified build yet,
    // but the functionality is wrapped here for cross-platform compatibility.
    
    CURLcode res = curl_easy_perform(curl);
    
    
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return (res == CURLE_OK) ? response_data : "";
}
#endif

void OllamaClient::Cancel() {
    cancel_flag_ = true;

#ifdef _WIN32
    // If we are actively blocking on a WinHTTP request, we force close it from this thread 
    // to instantly unblock the HTTP Receiver.
    std::lock_guard<std::mutex> lock(handle_mutex_);
    if (active_request_handle_) {
        WinHttpCloseHandle(active_request_handle_);
        active_request_handle_ = nullptr;
    }
#endif
}

} // namespace NPCInference
