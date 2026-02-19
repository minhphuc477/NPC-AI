cmake_minimum_required(VERSION 3.15)
project(NPCInference VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project options
option(USE_CUDA "Build with CUDA support" OFF)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_CLI "Build standalone CLI" ON)
option(DOWNLOAD_ONNXRUNTIME "Download ONNX Runtime automatically" ON)
option(NPC_USE_MOCKS "Build with mock implementations for testing" OFF)

if(MSVC)
    add_compile_options(/wd4715) # Suppress "not all control paths return a value"
endif()

# Find libcurl for Unix platforms (needed for OllamaClient HTTP)
if(UNIX)
    find_package(CURL REQUIRED)
    if(CURL_FOUND)
        message(STATUS "Found libcurl: ${CURL_LIBRARIES}")
    else()
        message(FATAL_ERROR "libcurl is required for Linux/Mac builds. Install with: apt-get install libcurl4-openssl-dev")
    endif()
endif()

# Include FetchContent for dependency management
include(FetchContent)

# Fetch nlohmann/json for JSON parsing
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

# Populate SentencePiece first to patch it
FetchContent_GetProperties(sentencepiece)
if(NOT sentencepiece_POPULATED)
    FetchContent_Populate(sentencepiece)
    
    # Patch SentencePiece CMakeLists.txt to fix CMake version compatibility
    file(READ "${sentencepiece_SOURCE_DIR}/CMakeLists.txt" SPM_CMAKE_CONTENT)
    # Replace various version formats
    string(REGEX REPLACE "cmake_minimum_required[ ]*\\([ ]*VERSION[ ]+[0-9]+\\.[0-9]+[^)]*\\)" 
                         "cmake_minimum_required(VERSION 3.15)" 
                         SPM_CMAKE_CONTENT "${SPM_CMAKE_CONTENT}")
    file(WRITE "${sentencepiece_SOURCE_DIR}/CMakeLists.txt" "${SPM_CMAKE_CONTENT}")
    
    add_subdirectory(${sentencepiece_SOURCE_DIR} ${sentencepiece_BINARY_DIR})
endif()

# Use usearch only if NOT using mocks
if(NOT NPC_USE_MOCKS)
    message(STATUS "Integrating usearch (Production Mode)...")
    FetchContent_Declare(
        usearch
        GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
        GIT_TAG v2.10.1 # Pin version
        GIT_SHALLOW TRUE
    )
    
    FetchContent_GetProperties(usearch)
    if(NOT usearch_POPULATED)
        FetchContent_Populate(usearch)
        
        # Patch usearch CMakeLists.txt to fix CMake 3.5 compatibility error
        if(EXISTS "${usearch_SOURCE_DIR}/CMakeLists.txt")
            file(READ "${usearch_SOURCE_DIR}/CMakeLists.txt" USEARCH_CMAKE_CONTENT)
            # Replace older cmake_minimum_required with 3.15
            string(REGEX REPLACE "cmake_minimum_required[ ]*\\([ ]*VERSION[ ]+[0-9]+\\.[0-9]+[^)]*\\)" 
                                 "cmake_minimum_required(VERSION 3.15)" 
                                 USEARCH_CMAKE_CONTENT "${USEARCH_CMAKE_CONTENT}")
            file(WRITE "${usearch_SOURCE_DIR}/CMakeLists.txt" "${USEARCH_CMAKE_CONTENT}")
        endif()
        
        # Ensure submodules are in include path
        include_directories("${usearch_SOURCE_DIR}/fp16/include")
        include_directories("${usearch_SOURCE_DIR}/simsimd/include")
        
        add_subdirectory(${usearch_SOURCE_DIR} ${usearch_BINARY_DIR})
    endif()
endif()

# Find or download ONNX Runtime
if(DOWNLOAD_ONNXRUNTIME)
# ... (Keep existing ONNX Runtime logic)
    message(STATUS "Downloading ONNX Runtime...")
    
    if(WIN32)
        set(ONNXRUNTIME_VERSION "1.18.0")
        if(USE_CUDA)
            set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
            set(ONNXRUNTIME_DIRNAME "onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
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
        set(ONNXRUNTIME_VERSION "1.18.0")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
        set(ONNXRUNTIME_DIRNAME "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
        
        message(STATUS "Downloading ONNX Runtime for Linux: ${ONNXRUNTIME_URL}")
        
        FetchContent_Declare(
            onnxruntime
            URL ${ONNXRUNTIME_URL}
            SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime
        )
        FetchContent_GetProperties(onnxruntime)
        if(NOT onnxruntime_POPULATED)
            FetchContent_Populate(onnxruntime)
        endif()
        
        set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/include")
        set(ONNXRUNTIME_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib")
        
        find_library(ONNXRUNTIME_LIBRARY
            NAMES onnxruntime
            PATHS ${ONNXRUNTIME_LIB_DIR}
            NO_DEFAULT_PATH
            REQUIRED
        )
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

# Library target
add_library(npc_inference STATIC
    src/PromptFormatter.cpp
    src/NPCInference.cpp
    src/ModelLoader.cpp
    src/PythonBridge.cpp
    src/BehaviorTree.cpp
    src/PromptBuilder.cpp
    src/Tokenizer.cpp
    # VectorStore added conditionally below
    src/EmbeddingModel.cpp
    src/KVCacheManager.cpp
    src/BM25Retriever.cpp
    src/HybridRetriever.cpp
    src/TemporalMemorySystem.cpp
    src/SocialFabricNetwork.cpp
    src/EmotionalContinuitySystem.cpp
    src/PlayerBehaviorModeling.cpp
    src/AmbientAwarenessSystem.cpp
    src/OllamaClient.cpp
    src/BuildSOTAContext.cpp
    src/SemanticCache.cpp
    src/PerformanceProfiler.cpp
    src/ToolRegistry.cpp
    src/SimpleGraph.cpp
    src/VisionLoader.cpp
    src/MemoryConsolidator.cpp
    src/GrammarSampler.cpp
    src/MemoryConsolidator.cpp
    src/GrammarSampler.cpp
)

target_sources(npc_inference PRIVATE src/VectorStore.cpp)
# Add Windows-specific compile definitions to fix min/max macros
if(WIN32)
    target_compile_definitions(npc_inference PRIVATE NOMINMAX)
endif()

target_include_directories(npc_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${sentencepiece_SOURCE_DIR}/src
)

if(NOT NPC_USE_MOCKS)
    target_include_directories(npc_inference PRIVATE ${usearch_SOURCE_DIR}/include)
    # If usearch provides a target, link it
    # target_link_libraries(npc_inference PRIVATE unum::usearch)
    # But for safety given previous issues, include dir is enough for header-only.
endif()

target_link_libraries(npc_inference
    PRIVATE
        ${ONNXRUNTIME_LIBRARY}
        sentencepiece-static
)

if(WIN32)
    target_link_libraries(npc_inference PRIVATE psapi)
endif()

# Link libcurl on Unix platforms for OllamaClient HTTP
if(UNIX)
    target_link_libraries(npc_inference PRIVATE CURL::libcurl)
endif()

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

    add_executable(test_final
        tests/test_final_system.cpp
    )
    
    target_link_libraries(test_final
        PRIVATE
            npc_inference
    )

    # Temporal Memory Demo: SOTA innovation demonstration
    add_executable(demo_temporal_memory
        src/demo_temporal_memory.cpp
    )
    
    target_link_libraries(demo_temporal_memory
        PRIVATE
            npc_inference
    )
    
    # Social Fabric Demo: SOTA social dynamics demonstration
    add_executable(demo_social_fabric
        src/demo_social_fabric.cpp
    )
    
    target_link_libraries(demo_social_fabric
        PRIVATE
            npc_inference
    )
    
    # Emotional Continuity Demo: SOTA emotional persistence demonstration
    add_executable(demo_emotional_continuity
        src/demo_emotional_continuity.cpp
    )
    
    target_link_libraries(demo_emotional_continuity
        PRIVATE
            npc_inference
    )
    
    # SOTA Integration Demo: All 3 systems working together
    add_executable(demo_sota_integration
        src/demo_sota_integration.cpp
    )
    
    target_link_libraries(demo_sota_integration
        PRIVATE
            npc_inference
    )
    
    # Chat Interface: Real-time NPC conversations
    add_executable(chat_interface
        src/chat_interface.cpp
    )
    
    target_link_libraries(chat_interface
        PRIVATE
            npc_inference
    )
    
    # LLM + Systems Demo: Shows LLM using all 5 NPC systems
    add_executable(demo_llm_with_systems
        src/demo_llm_with_systems.cpp
    )
    
    target_link_libraries(demo_llm_with_systems
        PRIVATE
            npc_inference
    )

    add_executable(bench_engine
        tests/bench_engine.cpp
    )
    
    target_link_libraries(bench_engine
        PRIVATE
            npc_inference
    )

    add_executable(ablation_suite
        tests/ablation_suite.cpp
    )
    
    target_link_libraries(ablation_suite
        PRIVATE
            npc_inference
    )

    add_executable(bench_memory
        tests/bench_memory.cpp
    )
    
    target_link_libraries(bench_memory
        PRIVATE
            npc_inference
    )

    add_executable(bench_retrieval
        tests/bench_retrieval.cpp
    )
    
    target_link_libraries(bench_retrieval
        PRIVATE
            npc_inference
    )

    add_executable(test_planner
        tests/test_planner.cpp
    )

    target_link_libraries(test_planner
        PRIVATE
            npc_inference
    )

    add_executable(test_reflection
        tests/test_reflection.cpp
    )

    target_link_libraries(test_reflection
        PRIVATE
            npc_inference
    )

    add_executable(test_truth_guard
        tests/test_truth_guard.cpp
    )

    target_link_libraries(test_truth_guard
        PRIVATE
            npc_inference
    )

    # New tests for Phase 1 improvements
    add_executable(test_async_init
        tests/test_async_init.cpp
    )

    target_link_libraries(test_async_init
        PRIVATE
            npc_inference
    )

    add_executable(test_state_persistence
        tests/test_state_persistence.cpp
    )

    target_link_libraries(test_state_persistence
        PRIVATE
            npc_inference
    )

    add_executable(test_tool_execution
        tests/test_tool_execution.cpp
    )

    target_link_libraries(test_tool_execution
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

