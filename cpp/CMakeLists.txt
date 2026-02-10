cmake_minimum_required(VERSION 3.15)
project(NPCInference VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project options
option(USE_CUDA "Build with CUDA support" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_CLI "Build standalone CLI" ON)
option(DOWNLOAD_ONNXRUNTIME "Download ONNX Runtime automatically" ON)

# Include FetchContent for dependency management
include(FetchContent)

# Fetch nlohmann/json for JSON parsing
message(STATUS "Fetching nlohmann/json...")
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# Fetch SentencePiece for tokenization
message(STATUS "Fetching SentencePiece...")
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.1.99
    GIT_SHALLOW TRUE
)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "Build shared library" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Enable TCMalloc" FORCE)
FetchContent_MakeAvailable(sentencepiece)

# Fetch unum-cloud/usearch for Vector Search
message(STATUS "Fetching usearch...")
FetchContent_Declare(
    usearch
    GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
    GIT_TAG v2.11.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(usearch)

# Find or download ONNX Runtime
if(DOWNLOAD_ONNXRUNTIME)
    message(STATUS "Downloading ONNX Runtime...")
    
    if(WIN32)
        set(ONNXRUNTIME_VERSION "1.17.0")
        if(USE_CUDA)
            set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")
            set(ONNXRUNTIME_DIRNAME "onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}")
        else()
            set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
            set(ONNXRUNTIME_DIRNAME "onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
        endif()
        
        FetchContent_Declare(
            onnxruntime
            URL ${ONNXRUNTIME_URL}
            SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime
        )
        FetchContent_MakeAvailable(onnxruntime)
        
        set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/include")
        set(ONNXRUNTIME_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib")
        find_library(ONNXRUNTIME_LIBRARY
            NAMES onnxruntime
            PATHS ${ONNXRUNTIME_LIB_DIR}
            NO_DEFAULT_PATH
            REQUIRED
        )
    else()
        message(FATAL_ERROR "Automatic ONNX Runtime download only supported on Windows currently")
    endif()
else()
    # Manual ONNX Runtime path
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS 
            $ENV{ONNXRUNTIME_ROOT}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime/include
        REQUIRED
    )

    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS 
            $ENV{ONNXRUNTIME_ROOT}/lib
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime/lib
        REQUIRED
    )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# Optional: SentencePiece for tokenization
# Commented out for now - will implement in next iteration
# find_package(sentencepiece QUIET)

# Library target
add_library(npc_inference STATIC
    src/PromptFormatter.cpp
    src/NPCInference.cpp
    src/ModelLoader.cpp
    src/PythonBridge.cpp
    src/BehaviorTree.cpp
    src/PromptBuilder.cpp
    src/Tokenizer.cpp
    src/VectorStore.cpp
    src/EmbeddingModel.cpp
    src/KVCacheManager.cpp
    src/BM25Retriever.cpp
    src/HybridRetriever.cpp
    src/SemanticCache.cpp
    src/PerformanceProfiler.cpp
    src/ToolRegistry.cpp
)

target_include_directories(npc_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${sentencepiece_SOURCE_DIR}/src
        ${usearch_SOURCE_DIR}/include
)

target_link_libraries(npc_inference
    PUBLIC
        nlohmann_json::nlohmann_json
    PRIVATE
        ${ONNXRUNTIME_LIBRARY}
        sentencepiece-static
)

# Standalone CLI executable
if(BUILD_CLI)
    add_executable(npc_cli
        src/main_cli.cpp
    )
    
    target_link_libraries(npc_cli
        PRIVATE
            npc_inference
    )
    
    # Copy ONNX Runtime DLL on Windows
    if(WIN32 AND ONNXRUNTIME_LIB_DIR)
        add_custom_command(TARGET npc_cli POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
                $<TARGET_FILE_DIR:npc_cli>
            COMMENT "Copying onnxruntime.dll to output directory"
        )
    endif()
endif()

# Tests
if(BUILD_TESTS)
    add_executable(test_inference
        tests/test_inference.cpp
    )
    
    target_link_libraries(test_inference
        PRIVATE
            npc_inference
    )
endif()

# Installation
install(TARGETS npc_inference
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

if(BUILD_CLI)
    install(TARGETS npc_cli
        RUNTIME DESTINATION bin
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== NPC Inference Configuration ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Support: ${USE_CUDA}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "Build CLI: ${BUILD_CLI}")
message(STATUS "===================================")
message(STATUS "")

