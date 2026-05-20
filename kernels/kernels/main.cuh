#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <vector>
#include <cstdio>

// ====== Configuration Enums ======

enum class MemoryStrategy {
    CONSTANT_MEMORY,   // Fast, read-only, limited size (~64KB)
    DEVICE_MEMORY      // Flexible, larger, slightly slower access
};

enum class DataType {
    FLOAT32,           // 32-bit float (faster)
    FLOAT64            // 64-bit double (higher precision)
};

enum class Algorithm {
    NAIVE,             // Direct computation, no optimizations
    SHARED_MEMORY      // Uses shared memory optimization
};

// ====== Main GPU processing functions ======

// Super wrapper with all options (recommended for advanced usage)
void runPlaneSweepingGPU_Advanced(const uint8_t* ref_image, int width, int height,
    const std::vector<uint8_t*>& sensor_images,
    const std::vector<std::vector<double>>& cam_params,
    float* cost_cube, float ZNear, float ZFar, int ZPlanes,
    MemoryStrategy memory_strategy = MemoryStrategy::CONSTANT_MEMORY,
    DataType data_type = DataType::FLOAT32,
    Algorithm algorithm = Algorithm::NAIVE);

// Wrapper with memory strategy selection (backward compatible)
void runPlaneSweepingGPU(const uint8_t* ref_image, int width, int height,
    const std::vector<uint8_t*>& sensor_images,
    const std::vector<std::vector<double>>& cam_params,
    float* cost_cube, float ZNear, float ZFar, int ZPlanes,
    MemoryStrategy strategy = MemoryStrategy::CONSTANT_MEMORY);

// Specific implementations (use these if you need explicit control)
void runPlaneSweepingGPU_ConstantMemory(const uint8_t* ref_image, int width, int height,
    const std::vector<uint8_t*>& sensor_images,
    const std::vector<std::vector<double>>& cam_params,
    float* cost_cube, float ZNear, float ZFar, int ZPlanes);

void runPlaneSweepingGPU_DeviceMemory(const uint8_t* ref_image, int width, int height,
    const std::vector<uint8_t*>& sensor_images,
    const std::vector<std::vector<double>>& cam_params,
    float* cost_cube, float ZNear, float ZFar, int ZPlanes);