cmake_minimum_required(VERSION 3.10)
project(test_fix)

set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)

# Include directories (Pointing to where the main project built dependencies)
include_directories(../include)
include_directories(../build/_deps/json-src/single_include)
include_directories(../build/_deps/json-src/include)
include_directories(../build/onnxruntime/include)
include_directories(../build/_deps/usearch-src/include)
include_directories(../build/_deps/usearch-src/fp16/include)
include_directories(../build/_deps/usearch-src/simsimd/include)
include_directories(../build/_deps/sentencepiece-src/src)

# Source files
file(GLOB SOURCES 
    "../src/HybridRetriever.cpp"
    "../src/VectorStore.cpp"
    "../src/BM25Retriever.cpp"
    "../src/EmbeddingModel.cpp"
    "../src/ModelLoader.cpp"
    "../src/KVCacheManager.cpp"
    "../src/Tokenizer.cpp"
    "test_retriever_fix.cpp"
)

# Create executable for retriever fix
add_executable(test_fix ${SOURCES})

# Create executable for grammar test
add_executable(test_grammar 
    "../src/GrammarSampler.cpp"
    "../src/Tokenizer.cpp"
    "test_grammar.cpp"
)

# Create executable for integration test
add_executable(test_integration
    "../src/NPCInference.cpp"
    "../src/ModelLoader.cpp"
    "../src/Tokenizer.cpp"
    "../src/EmbeddingModel.cpp"
    "../src/VectorStore.cpp"
    "../src/HybridRetriever.cpp"
    "../src/BM25Retriever.cpp"
    "../src/SemanticCache.cpp"
    "../src/SimpleGraph.cpp"
    "../src/MemoryConsolidator.cpp"
    "../src/VisionLoader.cpp"
    "../src/GrammarSampler.cpp"
    "../src/PromptBuilder.cpp"
    "../src/PromptFormatter.cpp"
    "../src/BehaviorTree.cpp"
    "../src/PythonBridge.cpp" 
    "../src/KVCacheManager.cpp"
    "../src/ToolRegistry.cpp" 
    "../src/PerformanceProfiler.cpp"
    "test_integration.cpp"
)

# Create executable for graph rag loop test
add_executable(test_graph_rag_loop
    "../src/NPCInference.cpp"
    "../src/ModelLoader.cpp"
    "../src/Tokenizer.cpp"
    "../src/EmbeddingModel.cpp"
    "../src/VectorStore.cpp"
    "../src/HybridRetriever.cpp"
    "../src/BM25Retriever.cpp"
    "../src/SemanticCache.cpp"
    "../src/SimpleGraph.cpp"
    "../src/MemoryConsolidator.cpp"
    "../src/VisionLoader.cpp"
    "../src/GrammarSampler.cpp"
    "../src/PromptBuilder.cpp"
    "../src/PromptFormatter.cpp"
    "../src/BehaviorTree.cpp"
    "../src/PythonBridge.cpp" 
    "../src/KVCacheManager.cpp"
    "../src/ToolRegistry.cpp" 
    "../src/PerformanceProfiler.cpp"
    "test_graph_rag_loop.cpp"
)

# Link libraries
if(WIN32)
    target_link_libraries(test_fix 
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/onnxruntime/lib/onnxruntime.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/sentencepiece-build/src/Release/sentencepiece.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/sentencepiece-build/src/Release/sentencepiece_train.lib
        shlwapi
    )
    target_link_libraries(test_grammar
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/sentencepiece-build/src/Release/sentencepiece.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/sentencepiece-build/src/Release/sentencepiece_train.lib
    )
    target_link_libraries(test_integration
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/onnxruntime/lib/onnxruntime.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/sentencepiece-build/src/Release/sentencepiece.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/sentencepiece-build/src/Release/sentencepiece_train.lib
        shlwapi
    )
    target_compile_definitions(test_integration PRIVATE NOMINMAX NOGDI)

    target_link_libraries(test_graph_rag_loop
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/onnxruntime/lib/onnxruntime.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/sentencepiece-build/src/Release/sentencepiece.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/sentencepiece-build/src/Release/sentencepiece_train.lib
        shlwapi
    )
    target_compile_definitions(test_graph_rag_loop PRIVATE NOMINMAX NOGDI)
else()
    target_link_libraries(test_fix onnxruntime sentencepiece)
endif()
