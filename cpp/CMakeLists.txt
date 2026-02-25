cmake_minimum_required(VERSION 3.15)
project(NPCInference VERSION 1.0.0 LANGUAGES CXX)

# CMake 4.x removes compatibility with very old minimum versions used by some
# third-party CMakeLists (e.g., sentencepiece). Set a policy floor explicitly.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    set(CMAKE_POLICY_VERSION_MINIMUM "3.5")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuration Options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_CLI "Build CLI tool" ON)
option(NPC_USE_MOCKS "Enable mock implementations for testing" ON)
if(WIN32)
    set(NPC_USE_USEARCH_DEFAULT OFF)
else()
    set(NPC_USE_USEARCH_DEFAULT ON)
endif()
option(NPC_USE_USEARCH "Enable usearch backend for VectorStore" ${NPC_USE_USEARCH_DEFAULT})

if(NPC_USE_MOCKS)
    message(STATUS "NPC_USE_MOCKS=ON (compile-time mock mode enabled)")
    add_compile_definitions(NPC_USE_MOCKS=1)
else()
    message(STATUS "NPC_USE_MOCKS=OFF (production mode)")
    add_compile_definitions(NPC_USE_MOCKS=0)
endif()

if(NPC_USE_USEARCH)
    message(STATUS "NPC_USE_USEARCH=ON (VectorStore uses usearch backend)")
    add_compile_definitions(NPC_USE_USEARCH=1)
else()
    message(STATUS "NPC_USE_USEARCH=OFF (VectorStore fallback backend)")
    add_compile_definitions(NPC_USE_USEARCH=0)
endif()

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

if(NPC_USE_USEARCH)
    FetchContent_Declare(
        usearch
        GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
        GIT_TAG v2.10.0
    )
    FetchContent_MakeAvailable(usearch)
endif()

# Download ONNX Runtime
set(ONNXRUNTIME_VERSION "1.18.0")
if(WIN32)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
    set(ONNXRUNTIME_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.zip")
elseif(APPLE)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
    set(ONNXRUNTIME_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.tgz")
else()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
    set(ONNXRUNTIME_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.tgz")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ARCHIVE}")
    message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION}...")
    file(DOWNLOAD "${ONNXRUNTIME_URL}" "${ONNXRUNTIME_ARCHIVE}" SHOW_PROGRESS)
endif()

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")
    message(STATUS "Extracting ONNX Runtime...")
    file(ARCHIVE_EXTRACT INPUT "${ONNXRUNTIME_ARCHIVE}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")
    # Releases extract into a subfolder like onnxruntime-win-x64-1.18.0 or onnxruntime-linux-x64-1.18.0
    file(GLOB EXTRACTED_DIRS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/onnxruntime-*")
    if(EXTRACTED_DIRS)
        list(GET EXTRACTED_DIRS 0 EXTRACTED_DIR)
        if(EXISTS "${EXTRACTED_DIR}/include" AND NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/include")
            file(RENAME "${EXTRACTED_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/include")
        endif()
        if(EXISTS "${EXTRACTED_DIR}/lib" AND NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib")
            file(RENAME "${EXTRACTED_DIR}/lib" "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib")
        endif()
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

# Library Target
if(NPC_USE_USEARCH)
    set(VECTOR_STORE_SOURCE src/VectorStore.cpp)
else()
    set(VECTOR_STORE_SOURCE src/VectorStoreFallback.cpp)
endif()

add_library(npc_inference STATIC
    src/PromptFormatter.cpp
    src/NPCInference.cpp
    src/ResponseController.cpp
    src/ModelLoader.cpp
    src/PythonBridge.cpp
    src/BehaviorTree.cpp
    src/PromptBuilder.cpp
    src/Tokenizer.cpp
    src/EmbeddingModel.cpp
    src/KVCacheManager.cpp
    ${VECTOR_STORE_SOURCE}
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
    src/BuildAdvancedContext.cpp
)

target_include_directories(npc_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
    PRIVATE
        ${sentencepiece_SOURCE_DIR}/src
)

if(NPC_USE_USEARCH)
    target_include_directories(npc_inference PRIVATE ${usearch_SOURCE_DIR}/include)
endif()

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
    npc_add_test(test_tool_registry tests/test_tool_registry.cpp)
    npc_add_test(test_retrieval_guard tests/test_retrieval_guard.cpp)
    npc_add_test(test_response_controller tests/test_response_controller.cpp)
    npc_add_test(test_multithreading tests/test_multithreading.cpp)
    npc_add_test(test_truth_guard tests/test_truth_guard.cpp)
    npc_add_test(demo_temporal_memory src/demo_temporal_memory.cpp)
    npc_add_test(demo_social_fabric src/demo_social_fabric.cpp)
    npc_add_test(demo_emotional_continuity src/demo_emotional_continuity.cpp)
    npc_add_test(demo_advanced_integration src/demo_integration.cpp)
    npc_add_test(chat_interface src/chat_interface.cpp)
    npc_add_test(demo_llm_with_systems src/demo_llm_with_systems.cpp)
    npc_add_test(bench_engine tests/bench_engine.cpp)
    npc_add_test(bench_memory tests/bench_memory.cpp)
    npc_add_test(bench_retrieval tests/bench_retrieval.cpp)
    npc_add_test(bench_retrieval_security tests/bench_retrieval_security.cpp)
    npc_add_test(ablation_suite tests/ablation_suite.cpp)
endif()
