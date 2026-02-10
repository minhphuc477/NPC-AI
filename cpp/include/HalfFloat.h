#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <cstring>

namespace NPCInference {

    // Simple IEEE 754 Half-Precision Floating Point Implementation
    // Based on public domain implementations (e.g., Ryg's)
    struct HalfFloat {
        uint16_t bits;

        HalfFloat() = default;

        // Constructor from float
        HalfFloat(float f) {
            bits = FloatToHalf(f);
        }

        // Template constructor for other types (bool, int, double)
        template <typename T>
        HalfFloat(T v) : HalfFloat(static_cast<float>(v)) {}

        // Conversion to float
        operator float() const {
            return HalfToFloat(bits);
        }

        // Static conversion methods
        static uint16_t FloatToHalf(float f) {
            uint32_t f32i;
            std::memcpy(&f32i, &f, 4);

            uint32_t s = (f32i >> 31) & 0x00000001; // Sign
            uint32_t e = (f32i >> 23) & 0x000000FF; // Exponent
            uint32_t m = f32i & 0x007FFFFF;         // Mantissa

            uint16_t h_s = (uint16_t)(s << 15);
            uint16_t h_e;
            uint16_t h_m;

            if (e == 0) {
                // Zero or Denormal
                // We'll treat denormals as zero for simplicity in this fallback
                return h_s;
            } else if (e == 255) {
                // Inf or NaN
                h_e = 31;
                h_m = m ? 0x200 : 0; // Preserve NaN-ness roughly
                return h_s | (h_e << 10) | h_m;
            } else {
                // Normalized
                int new_e = e - 127 + 15;
                if (new_e >= 31) {
                    // Overflow to Inf
                    return h_s | 0x7C00; 
                } else if (new_e <= 0) {
                    // Underflow to zero (signed)
                    return h_s;
                } else {
                    h_e = (uint16_t)new_e;
                    h_m = (uint16_t)(m >> 13);
                    return h_s | (h_e << 10) | h_m;
                }
            }
        }

        static float HalfToFloat(uint16_t h) {
            uint32_t s = (h >> 15) & 0x00000001;
            uint32_t e = (h >> 10) & 0x0000001F;
            uint32_t m = h & 0x000003FF;

            uint32_t f_s = s << 31;
            uint32_t f_e;
            uint32_t f_m;

            if (e == 0) {
                if (m == 0) {
                    // Zero
                    uint32_t result = f_s;
                    float f;
                    std::memcpy(&f, &result, 4);
                    return f;
                } else {
                    // Denormal (treat as zero for this lightweight impl)
                    // Or we could normalize it, but that's expensive.
                    // For embeddings, typically normalized, usually > 1e-4.
                    return 0.0f; 
                }
            } else if (e == 31) {
                // Inf or NaN
                f_e = 255;
                f_m = m ? 0x00400000 : 0; 
            } else {
                // Normalized
                f_e = e - 15 + 127;
                f_m = m << 13;
            }

            uint32_t result = f_s | (f_e << 23) | f_m;
            float f;
            std::memcpy(&f, &result, 4);
            return f;
        }
    };

} // namespace NPCInference
