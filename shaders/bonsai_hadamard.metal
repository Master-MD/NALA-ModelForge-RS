#include <metal_stdlib>
using namespace metal;

// NALA-ModelForge-RS (RSI-MF-01) — Apple Silicon Metal Compute Shader
// Fused Randomized Walsh-Hadamard Transform (H * D * x) + Bonsai 1-Bit g128 Bit-Packing.
// Executes 128-element blocks entirely in on-chip threadgroup SRAM (zero global atomics).

struct Bonsai1BitBlock128 {
    float scale;
    uint32_t bits[4]; // 128 packed sign bits = 16 bytes
};

inline uint64_t lcg_step(uint64_t s) {
    return s * 6364136223846793005ULL + 1442695040888963407ULL;
}

kernel void bonsai_hadamard_g128_pack(
    device const float* input_weights [[buffer(0)]],
    device Bonsai1BitBlock128* output_blocks [[buffer(1)]],
    constant uint32_t& total_blocks [[buffer(2)]],
    constant uint64_t& rht_seed [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (gid >= total_blocks || tid >= 128) return;

    threadgroup float buf[128];
    threadgroup float abs_buf[128];

    // 1. Load weight & apply deterministic Rademacher sign flip D in {-1, +1}
    uint base_idx = gid * 128 + tid;
    float w = input_weights[base_idx];
    uint64_t h = lcg_step(rht_seed ^ (uint64_t(tid) * 0x9E3779B97F4A7C15ULL));
    if ((h >> 63) == 1ULL) w = -w;
    buf[tid] = w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. 7-stage in-SRAM Fast Walsh-Hadamard Transform (128 = 2^7)
    for (uint stride = 1; stride < 128; stride <<= 1) {
        uint pair_group = tid / stride;
        uint lane = tid % stride;
        uint i0 = pair_group * (stride << 1) + lane;
        if (i0 + stride < 128) {
            float x = buf[i0];
            float y = buf[i0 + stride];
            buf[i0] = x + y;
            buf[i0 + stride] = x - y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize by 1/sqrt(128)
    float rotated = buf[tid] * 0.088388347648f;
    buf[tid] = rotated;
    abs_buf[tid] = fabs(rotated);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Parallel tree reduction for absmean scale alpha = mean(|w_rot|)
    for (uint s = 64; s > 0; s >>= 1) {
        if (tid < s) {
            abs_buf[tid] += abs_buf[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Pack 128 sign bits into 4 x uint32 (lanes 0..3)
    if (tid < 4) {
        uint32_t word = 0;
        uint offset = tid * 32;
        for (uint b = 0; b < 32; ++b) {
            if (buf[offset + b] >= 0.0f) {
                word |= (1u << b);
            }
        }
        output_blocks[gid].bits[tid] = word;
        if (tid == 0) {
            output_blocks[gid].scale = max(abs_buf[0] / 128.0f, 1e-8f);
        }
    }
}
