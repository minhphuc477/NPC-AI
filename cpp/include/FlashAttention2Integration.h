// FlashAttention2Integration.h - Flash Attention 2 support for faster inference

#pragma once

#include <string>
#include <memory>

namespace NPCInference {

/**
 * Flash Attention 2 Integration
 * 
 * Provides 2-4x faster attention computation with O(N) memory complexity
 * instead of standard O(N²) attention.
 * 
 * Requirements:
 * - CUDA-capable GPU
 * - ONNX Runtime with Flash Attention support OR
 * - TensorRT-LLM backend
 * 
 * Benefits:
 * - 2.5x speedup on attention layers
 * - Reduced memory usage for long contexts
 * - Better scaling for large batch sizes
 */
class FlashAttention2Config {
public:
    bool enable_flash_attention = false;
    bool use_tensorrt = false;  // Use TensorRT-LLM instead of ONNX Runtime
    int max_sequence_length = 4096;
    int num_attention_heads = 32;
    int head_dimension = 128;
    
    // Flash Attention specific parameters
    bool use_causal_mask = true;
    float softmax_scale = 0.0f;  // Auto-calculate if 0
    
    /**
     * Check if Flash Attention 2 is available
     */
    static bool IsAvailable();
    
    /**
     * Get recommended configuration for current hardware
     */
    static FlashAttention2Config GetRecommendedConfig();
};

/**
 * Flash Attention 2 Optimizer
 * 
 * Optimizes ONNX models to use Flash Attention 2 kernels
 */
class FlashAttention2Optimizer {
public:
    /**
     * Optimize ONNX model to use Flash Attention 2
     * 
     * @param input_model Path to original ONNX model
     * @param output_model Path to save optimized model
     * @param config Flash Attention configuration
     * @return true if optimization successful
     */
    static bool OptimizeModel(
        const std::string& input_model,
        const std::string& output_model,
        const FlashAttention2Config& config
    );
    
    /**
     * Benchmark attention performance
     * 
     * @param model_path Path to ONNX model
     * @param use_flash_attention Whether to use Flash Attention
     * @return Latency in milliseconds
     */
    static double BenchmarkAttention(
        const std::string& model_path,
        bool use_flash_attention
    );
};

} // namespace NPCInference
