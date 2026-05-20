#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <new>
#include <memory>
#include <chrono>
#include <stdint.h>
#include <vector>
#include <iostream>



// macro for indexing 2D arrays stored in 1D
#define MI(r, c, width) ((r) * (width) + (c))
#define BLOCKSIZE 16
#define RAD 1

// select precision: 0 = float, 1 = double
#define USE_DOUBLE 0


#if USE_DOUBLE
typedef double Real;
#else
typedef float Real;
#endif

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                cudaGetErrorString((code)), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// Dynamycally allocated constant memory for camera parameters 
__constant__ Real c_invK[9];
__constant__ Real c_R_inv[9];
__constant__ Real c_t_inv[3];
__constant__ Real c_K_proj[9];
__constant__ Real c_R_proj[9];
__constant__ Real c_t_proj[3];

//__device__ Real c_invK[9];
//__device__ Real c_R_inv[9];
//__device__ Real c_t_inv[3];
//__device__ Real c_K_proj[9];
//__device__ Real c_R_proj[9];
//__device__ Real c_t_proj[3];

__global__ void warmup(float* A, float* B, int w, int h) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= w || j >= h) return;

    int idx = MI(j, i, w);
    A[idx] = B[idx];
}

__global__ void shared_kernel(uint8_t* refY, uint8_t* sensY, float* costCube,
    int img_w, int img_h, float ZNear, float ZFar, int ZPlanes)
{

    // Declaration of shared memory for the tile + halo (size: (BLOCKSIZE + 2*RAD) x (BLOCKSIZE + 2*RAD))
    extern __shared__ uint8_t tmp[];
    // Shared memory width (including halo) for indexing
    const int s_width = BLOCKSIZE + 2 * RAD;


    // Global indexing for the pixel (i, j) in the reference image
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Extractio of the Z coordinate directly from the CUDA grid (pure 3D architecture)
    int zi = blockIdx.z;

    // Thread indices within the block for shared memory access
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Loading of Shared Menory (Executed once per block)

    // Loading center of the tile
    if (i < img_w && j < img_h)
        tmp[MI(ty + RAD, tx + RAD, s_width)] = refY[MI(j, i, img_w)];
    else
        tmp[MI(ty + RAD, tx + RAD, s_width)] = 0;

    // Loading of the Left Side
    if (tx < RAD) {
        int gx = i - RAD;
        int gy = j;
        tmp[MI(ty + RAD, tx, s_width)] = (gx >= 0 && gy >= 0 && gy < img_h) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // Loading of the Right Side
    if (tx >= BLOCKSIZE - RAD) {
        int gx = i + RAD;
        int gy = j;
        tmp[MI(ty + RAD, tx + 2 * RAD, s_width)] = (gx < img_w && gy >= 0 && gy < img_h) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // Loading of the Top Side
    if (ty < RAD) {
        int gx = i;
        int gy = j - RAD;
        tmp[MI(ty, tx + RAD, s_width)] = (gy >= 0 && gx >= 0 && gx < img_w) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // Loading of the Bottom Side
    if (ty >= BLOCKSIZE - RAD) {
        int gx = i;
        int gy = j + RAD;
        tmp[MI(ty + 2 * RAD, tx + RAD, s_width)] = (gy < img_h && gx >= 0 && gx < img_w) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // Loading of the Top-Left Corner
    if (tx < RAD && ty < RAD) {
        int gx = i - RAD;
        int gy = j - RAD;
        tmp[MI(ty, tx, s_width)] = (gx >= 0 && gy >= 0) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // Loading of the Top-Right Corner
    if (tx >= BLOCKSIZE - RAD && ty < RAD) {
        int gx = i + RAD;
        int gy = j - RAD;
        tmp[MI(ty, tx + 2 * RAD, s_width)] = (gx < img_w && gy >= 0) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // Loading of the Bottom-Left Corner
    if (tx < RAD && ty >= BLOCKSIZE - RAD) {
        int gx = i - RAD;
        int gy = j + RAD;
        tmp[MI(ty + 2 * RAD, tx, s_width)] = (gx >= 0 && gy < img_h) ? refY[MI(gy, gx, img_w)] : 0;
    }

    // Loading of the Bottom-Right Corner
    if (tx >= BLOCKSIZE - RAD && ty >= BLOCKSIZE - RAD) {
        int gx = i + RAD;
        int gy = j + RAD;
        tmp[MI(ty + 2 * RAD, tx + 2 * RAD, s_width)] = (gx < img_w && gy < img_h) ? refY[MI(gy, gx, img_w)] : 0;
    }


    // sync to ensure all threads have loaded the shared memory before any thread accesses it
    __syncthreads();

    // Geometry and SAD computation for the current pixel (i, j) and depth plane zi

    // Check memory bounds extended to the third dimension (ZPlanes)
    if (i >= img_w || j >= img_h || zi >= ZPlanes) return;

    // estimation of depth for this specific plane
    Real z = (Real)ZNear * (Real)ZFar / ((Real)ZNear + (((Real)zi / (Real)ZPlanes) * ((Real)ZFar - (Real)ZNear)));

    // estimation of the 3D point in the reference camera coordinate system
    Real X_ref = (c_invK[0] * i + c_invK[1] * j + c_invK[2]) * z;
    Real Y_ref = (c_invK[3] * i + c_invK[4] * j + c_invK[5]) * z;
    Real Z_ref = (c_invK[6] * i + c_invK[7] * j + c_invK[8]) * z;

    // estimation of the 3D point in the world coordinate system (using inverse extrinsics of the reference camera)
    Real Xw = c_R_inv[0] * X_ref + c_R_inv[1] * Y_ref + c_R_inv[2] * Z_ref - c_t_inv[0];
    Real Yw = c_R_inv[3] * X_ref + c_R_inv[4] * Y_ref + c_R_inv[5] * Z_ref - c_t_inv[1];
    Real Zw = c_R_inv[6] * X_ref + c_R_inv[7] * Y_ref + c_R_inv[8] * Z_ref - c_t_inv[2];

    // estimation of the 3D point in the projector camera coordinate system (using extrinsics of the projector camera)
    Real X_p = c_R_proj[0] * Xw + c_R_proj[1] * Yw + c_R_proj[2] * Zw - c_t_proj[0];
    Real Y_p = c_R_proj[3] * Xw + c_R_proj[4] * Yw + c_R_proj[5] * Zw - c_t_proj[1];
    Real Z_p = c_R_proj[6] * Xw + c_R_proj[7] * Yw + c_R_proj[8] * Zw - c_t_proj[2];


    // projection of the 3D point onto the projector image plane (using intrinsics of the projector camera)
    float x_proj = (float)(c_K_proj[0] * X_p / Z_p + c_K_proj[1] * Y_p / Z_p + c_K_proj[2]);
    float y_proj = (float)(c_K_proj[3] * X_p / Z_p + c_K_proj[4] * Y_p / Z_p + c_K_proj[5]);


    // initialization of cost and count for the SAD computation
    float cost = 0.0f;
    float count = 0.0f;

    // sad computation using the tile in shared memory (tmp) for the reference image and direct access to sensY for the sensed image

    for (int ki = -RAD; ki <= RAD; ++ki) {
        for (int kj = -RAD; kj <= RAD; ++kj) {
            uint8_t valRef = tmp[MI(ty + RAD + kj, tx + RAD + ki, s_width)];

            int sx = roundf(x_proj) + ki;
            int sy = roundf(y_proj) + kj;

            if (sx >= 0 && sx < img_w && sy >= 0 && sy < img_h) {
                cost += fabsf((float)valRef - (float)sensY[MI(sy, sx, img_w)]);
                count += 1.0f;
            }
        }
    }

    if (count > 0) {

        // indexing of the output cost cube extended to the third dimension (ZPlanes)
        int out_idx = zi * (img_w * img_h) + (j * img_w + i);
        costCube[out_idx] = fminf(cost / count, costCube[out_idx]);
    }
}



__global__ void naive_kernel(uint8_t* refY, uint8_t* sensY, float* costCube,
    int img_w, int img_h, float ZNear, float ZFar, int ZPlanes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 1. Estrazione della coordinata Z direttamente dalla griglia CUDA
    int zi = blockIdx.z;

    // 2. Controllo dei limiti di memoria esteso alla terza dimensione
    if (i >= img_w || j >= img_h || zi >= ZPlanes) return;

    // 3. Calcolo della profondità per questo specifico piano
    Real z = (Real)ZNear * (Real)ZFar / ((Real)ZNear + (((Real)zi / (Real)ZPlanes) * ((Real)ZFar - (Real)ZNear)));

    Real X_ref = (c_invK[0] * i + c_invK[1] * j + c_invK[2]) * z;
    Real Y_ref = (c_invK[3] * i + c_invK[4] * j + c_invK[5]) * z;
    Real Z_ref = (c_invK[6] * i + c_invK[7] * j + c_invK[8]) * z;

    Real Xw = c_R_inv[0] * X_ref + c_R_inv[1] * Y_ref + c_R_inv[2] * Z_ref - c_t_inv[0];
    Real Yw = c_R_inv[3] * X_ref + c_R_inv[4] * Y_ref + c_R_inv[5] * Z_ref - c_t_inv[1];
    Real Zw = c_R_inv[6] * X_ref + c_R_inv[7] * Y_ref + c_R_inv[8] * Z_ref - c_t_inv[2];

    Real X_p = c_R_proj[0] * Xw + c_R_proj[1] * Yw + c_R_proj[2] * Zw - c_t_proj[0];
    Real Y_p = c_R_proj[3] * Xw + c_R_proj[4] * Yw + c_R_proj[5] * Zw - c_t_proj[1];
    Real Z_p = c_R_proj[6] * Xw + c_R_proj[7] * Yw + c_R_proj[8] * Zw - c_t_proj[2];

    float x_proj = (float)(c_K_proj[0] * X_p / Z_p + c_K_proj[1] * Y_p / Z_p + c_K_proj[2]);
    float y_proj = (float)(c_K_proj[3] * X_p / Z_p + c_K_proj[4] * Y_p / Z_p + c_K_proj[5]);

    float cost = 0.0f;
    float count = 0.0f;

    // Il SAD avviene esattamente come prima, leggendo direttamente dalla VRAM (refY e sensY)
    for (int ki = -RAD; ki <= RAD; ++ki) {
        for (int kj = -RAD; kj <= RAD; ++kj) {
            int ref_x = i + ki;
            int ref_y = j + kj;

            int sx = roundf(x_proj) + ki;
            int sy = roundf(y_proj) + kj;

            if (ref_x >= 0 && ref_x < img_w && ref_y >= 0 && ref_y < img_h &&
                sx >= 0 && sx < img_w && sy >= 0 && sy < img_h)
            {
                uint8_t valRef = refY[MI(ref_y, ref_x, img_w)];
                uint8_t valSens = sensY[MI(sy, sx, img_w)];

                cost += fabsf((float)valRef - (float)valSens);
                count += 1.0f;
            }
        }
    }

    if (count > 0) {
        int out_idx = zi * (img_w * img_h) + (j * img_w + i);
        costCube[out_idx] = fminf(cost / count, costCube[out_idx]);
    }
}

// Kernel for initializing the cost cube with a specific value (e.g., 255.0f)
__global__ void initCostCube(float* cube, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) cube[idx] = val;
}

// Interface for main application to run the plane sweeping algorithm on the GPU
void runPlaneSweepingGPU(const uint8_t* ref_image, int width, int height,
    const std::vector<uint8_t*>& sensor_images,
    const std::vector<std::vector<double>>& cam_params,
    float* cost_cube, float ZNear, float ZFar, int ZPlanes)
{
    const int imgSize = width * height;
    size_t plane = width * height * sizeof(float);

    //float* warmupA; float* warmupB;
    //CHK(cudaMalloc(&warmupA, plane));
    //CHK(cudaMalloc(&warmupB, plane));

    //dim3 warmupblock(BLOCKSIZE, BLOCKSIZE);
    //dim3 warmupgrid(((width + BLOCKSIZE - 1) / BLOCKSIZE), ((height + BLOCKSIZE - 1) / BLOCKSIZE));
    //warmup << <warmupgrid, warmupblock >> > (warmupA, warmupB, width, height);
    //CHK(cudaGetLastError());

    auto total_start = std::chrono::high_resolution_clock::now();


    // allocation of device memory for reference image, sensed image, and cost cube
    uint8_t* d_ref, * d_sens;
    float* d_costCube;

    CHK(cudaMalloc(&d_ref, imgSize));
    CHK(cudaMalloc(&d_sens, imgSize));
    CHK(cudaMalloc(&d_costCube, imgSize * ZPlanes * sizeof(float)));

    int totalElements = imgSize * ZPlanes;
    initCostCube << < (totalElements + 255) / 256, 256 >> > (d_costCube, totalElements, 255.0f);
    CHK(cudaDeviceSynchronize());

    CHK(cudaMemcpy(d_ref, ref_image, imgSize, cudaMemcpyHostToDevice));

    // Extraction of camera parameters for the reference camera and copying them to constant memory
    const auto& ref_params = cam_params[0];
    std::vector<Real> h_invK(ref_params.begin(), ref_params.begin() + 9);
    std::vector<Real> h_R_inv(ref_params.begin() + 9, ref_params.begin() + 18);
    std::vector<Real> h_t_inv(ref_params.begin() + 18, ref_params.end());

    CHK(cudaMemcpyToSymbol(c_invK, h_invK.data(), 9 * sizeof(Real)));
    CHK(cudaMemcpyToSymbol(c_R_inv, h_R_inv.data(), 9 * sizeof(Real)));
    CHK(cudaMemcpyToSymbol(c_t_inv, h_t_inv.data(), 3 * sizeof(Real)));

    // Definition of block and grid dimensions for the kernel launch
    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid_3D((width + BLOCKSIZE - 1) / BLOCKSIZE, (height + BLOCKSIZE - 1) / BLOCKSIZE, ZPlanes);

    // definition of shared memory size for the shared kernel (if used)
    size_t sharedMemBytes = (BLOCKSIZE + 2 * RAD) * (BLOCKSIZE + 2 * RAD) * sizeof(uint8_t);

    auto kernel_start = std::chrono::high_resolution_clock::now();


    // loop over the sensed images (projector views)
    for (size_t c = 0; c < sensor_images.size(); c++) {
        const auto& sens_params = cam_params[c + 1];

        std::vector<Real> h_K_proj(sens_params.begin(), sens_params.begin() + 9);
        std::vector<Real> h_R_proj(sens_params.begin() + 9, sens_params.begin() + 18);
        std::vector<Real> h_t_proj(sens_params.begin() + 18, sens_params.end());

        CHK(cudaMemcpyToSymbol(c_K_proj, h_K_proj.data(), 9 * sizeof(Real)));
        CHK(cudaMemcpyToSymbol(c_R_proj, h_R_proj.data(), 9 * sizeof(Real)));
        CHK(cudaMemcpyToSymbol(c_t_proj, h_t_proj.data(), 3 * sizeof(Real)));

        CHK(cudaMemcpy(d_sens, sensor_images[c], imgSize, cudaMemcpyHostToDevice));

        // launch of the kernel for this sensed image (projector view) - shared memory version
        naive_kernel << <grid_3D, block, sharedMemBytes >> > (
            d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);

        //shared_kernel << <grid_3D, block, sharedMemBytes >> > (
        //    d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);

        CHK(cudaGetLastError());
    }

    auto kernel_end = std::chrono::high_resolution_clock::now();
    auto total_end = std::chrono::high_resolution_clock::now();

    double kernel_ms = std::chrono::duration<double>(kernel_end - kernel_start).count();

    double total_ms = std::chrono::duration<double>(total_end - total_start).count();

    printf("Kernel time: %.3f s\n", kernel_ms);
    printf("Total time: %.3f s \n", total_ms);

    CHK(cudaDeviceSynchronize());

    CHK(cudaMemcpy(cost_cube, d_costCube, imgSize * ZPlanes * sizeof(float), cudaMemcpyDeviceToHost));

    CHK(cudaFree(d_ref)); CHK(cudaFree(d_sens)); CHK(cudaFree(d_costCube));
    //CHK(cudaFree(warmupA)); CHK(cudaFree(warmupB));
}