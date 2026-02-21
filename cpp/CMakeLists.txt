cmake_minimum_required(VERSION 3.15)
project(NPCInference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuration Options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_CLI "Build CLI tool" ON)
option(NPC_USE_MOCKS "Enable mock implementations for testing" ON)

# External Dependencies
include(FetchContent)

# sentencepiece
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.0
)
set(SPM_BUILD_TESTS OFF CACHE BOOL "Build sentencepiece tests" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "Build sentencepiece shared" FORCE)
FetchContent_MakeAvailable(sentencepiece)

# json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# usearch
FetchContent_Declare(
    usearch
    GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
    GIT_TAG v2.10.0
)
FetchContent_MakeAvailable(usearch)

# Download ONNX Runtime
if(WIN32)
    set(ONNXRUNTIME_VERSION "1.18.0")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
    set(ONNXRUNTIME_ZIP "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.zip")
    
    if(NOT EXISTS "${ONNXRUNTIME_ZIP}")
        message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION}...")
        file(DOWNLOAD "${ONNXRUNTIME_URL}" "${ONNXRUNTIME_ZIP}" SHOW_PROGRESS)
    endif()

    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")
        message(STATUS "Extracting ONNX Runtime...")
        file(ARCHIVE_EXTRACT INPUT "${ONNXRUNTIME_ZIP}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")
        # Releases extract into a subfolder like onnxruntime-win-x64-1.18.0
        file(GLOB EXTRACTED_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/onnxruntime-win-x64-*")
        if(EXTRACTED_DIR)
            file(RENAME "${EXTRACTED_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/include")
            file(RENAME "${EXTRACTED_DIR}/lib" "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib")
        endif()
    endif()

    set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/include")
    set(ONNXRUNTIME_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib")
    
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_LIB_DIR}
        NO_DEFAULT_PATH
        REQUIRED
    )
else()
    # Path for Linux/other
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime REQUIRED)
endif()

# Library Target
add_library(npc_inference STATIC
    src/PromptFormatter.cpp
    src/NPCInference.cpp
    src/ModelLoader.cpp
    src/PythonBridge.cpp
    src/BehaviorTree.cpp
    src/PromptBuilder.cpp
    src/Tokenizer.cpp
    src/EmbeddingModel.cpp
    src/KVCacheManager.cpp
    src/VectorStore.cpp
    src/VisionLoader.cpp
    src/SocialFabricNetwork.cpp
    src/TemporalMemorySystem.cpp
    src/EmotionalContinuitySystem.cpp
    src/AmbientAwarenessSystem.cpp
    src/OrtEnvironmentManager.cpp
    src/SimpleGraph.cpp
    src/BM25Retriever.cpp
    src/HybridRetriever.cpp
    src/SemanticCache.cpp
    src/ToolRegistry.cpp
    src/GrammarSampler.cpp
    src/PerformanceProfiler.cpp
    src/MemoryConsolidator.cpp
    src/PlayerBehaviorModeling.cpp
    src/OllamaClient.cpp
    src/BuildSOTAContext.cpp
)

target_include_directories(npc_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
    PRIVATE
        ${sentencepiece_SOURCE_DIR}/src
        ${usearch_SOURCE_DIR}/include
)

target_link_libraries(npc_inference
    PRIVATE
        ${ONNXRUNTIME_LIBRARY}
        sentencepiece-static
        nlohmann_json::nlohmann_json
)

if(WIN32)
    target_compile_definitions(npc_inference PRIVATE NOMINMAX USEARCH_USE_FP16LIB=0)
    target_link_libraries(npc_inference PRIVATE psapi)
endif()

# Function to deploy runtime dependencies (DLLs)
function(npc_deploy_runtime_deps TARGET_NAME)
    if(WIN32 AND ONNXRUNTIME_LIB_DIR)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
                $<TARGET_FILE_DIR:${TARGET_NAME}>
            COMMENT "Deploying onnxruntime.dll for ${TARGET_NAME}"
        )
    endif()
endfunction()

# Executables
if(BUILD_CLI)
    # add_executable(npc_cli src/main_cli.cpp)
    # target_link_libraries(npc_cli PRIVATE npc_inference)
    # npc_deploy_runtime_deps(npc_cli)
endif()

if(BUILD_TESTS)
    macro(npc_add_test TEST_NAME SRC)
        add_executable(${TEST_NAME} ${SRC})
        target_link_libraries(${TEST_NAME} PRIVATE npc_inference)
        npc_deploy_runtime_deps(${TEST_NAME})
    endmacro()

    npc_add_test(test_inference tests/test_inference.cpp)
    npc_add_test(test_final tests/test_final_system.cpp)
    npc_add_test(test_reasoning_loop tests/test_reasoning_loop.cpp)
    npc_add_test(test_multithreading tests/test_multithreading.cpp)
    npc_add_test(test_truth_guard tests/test_truth_guard.cpp)
    npc_add_test(demo_temporal_memory src/demo_temporal_memory.cpp)
    npc_add_test(demo_social_fabric src/demo_social_fabric.cpp)
    npc_add_test(demo_emotional_continuity src/demo_emotional_continuity.cpp)
    npc_add_test(demo_sota_integration src/demo_sota_integration.cpp)
    npc_add_test(chat_interface src/chat_interface.cpp)
    npc_add_test(demo_llm_with_systems src/demo_llm_with_systems.cpp)
    npc_add_test(bench_engine tests/bench_engine.cpp)
    npc_add_test(bench_memory tests/bench_memory.cpp)
    npc_add_test(bench_retrieval tests/bench_retrieval.cpp)
    npc_add_test(ablation_suite tests/ablation_suite.cpp)
endif()
