#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h> // Per uint8_t
#include <vector>   // Per std::vector
#include <cstdio>



// This is the public interface of our cuda function, called directly in main.cpp
void runPlaneSweepingGPU(const uint8_t* ref_image, int width, int height,
    const std::vector<uint8_t*>& sensor_images,
    const std::vector<std::vector<double>>& cam_params,
    float* cost_cube, float ZNear, float ZFar, int ZPlanes);