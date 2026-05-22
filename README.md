# CUDA Plane Sweeping — Depth Estimation via SAD

GPU implementation in CUDA of a **Plane Sweeping** algorithm for depth estimation from multi-camera images, with support for multiple memory optimization strategies and parallelism approaches.

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Theoretical Background](#theoretical-background)
   - [Plane Sweeping](#plane-sweeping)
   - [SAD — Sum of Absolute Differences](#sad--sum-of-absolute-differences)
   - [Projection Geometry](#projection-geometry)
3. [Code Architecture](#code-architecture)
4. [Implemented Optimization Strategies](#implemented-optimization-strategies)
   - [Naive 3D Kernel](#naive-3d-kernel)
   - [Shared Memory Kernel](#shared-memory-kernel)
   - [Constant Memory](#constant-memory)
   - [Precision: Float vs Double](#precision-float-vs-double)
5. [How to Configure and Select Optimizations](#how-to-configure-and-select-optimizations)
6. [Main Parameters](#main-parameters)
7. [Output and Performance Metrics](#output-and-performance-metrics)
8. [Requirements](#requirements)

---

## Problem Overview

Given a set of images captured from multiple cameras with calibrated positions and orientations, the algorithm estimates for each pixel of the **reference camera** its **depth** (distance from the camera). The result is a dense **depth map**.

This kind of processing is fundamental in:
- 3D reconstruction of real scenes
- Multi-baseline stereo vision
- Robotics and autonomous navigation

---

## Theoretical Background

### Plane Sweeping

The core idea is to "sweep" the scene with a series of virtual planes at increasing depths (from `ZNear` to `ZFar`), project each pixel of the reference camera onto each of these planes, and verify how photometrically consistent the projection onto the secondary cameras is.

For each Z plane, a matching cost is computed: the plane that yields the minimum cost for a given pixel corresponds to the estimated depth.

Plane depths are not distributed linearly but in an **inverse** (disparity-space) fashion:

```
z(zi) = ZNear * ZFar / (ZNear + (zi / ZPlanes) * (ZFar - ZNear))
```

This ensures a denser distribution of planes close to the camera, where disparity variation is largest.

### SAD — Sum of Absolute Differences

The matching cost is computed via **SAD (Sum of Absolute Differences)** over a window of size `(2*RAD+1) x (2*RAD+1)`:

```
SAD(x, y, z) = Σ_{ki,kj} | I_ref(x+ki, y+kj) - I_sens(x'_proj+ki, y'_proj+kj) |
```

A low value indicates high photometric similarity → the estimated depth is correct.

The cost is normalized by the number of valid pixels in the window (those that fall within image boundaries) and compared against the current value in the cost cube via `fminf`, keeping the minimum across all sensors.

### Projection Geometry

For each pixel `(i, j)` of the reference camera, at depth plane `z`, the geometric pipeline is:

1. **Back-projection** into reference camera space: `X_ref = K_ref^{-1} * [i, j, 1]^T * z`
2. **World transformation**: `X_world = R_ref^{-1} * X_ref - t_ref`
3. **Projection onto the sensor camera**: `X_sens = R_sens * X_world + t_sens`
4. **2D projection**: `[x', y'] = K_sens * X_sens / Z_sens`

Each camera's parameters are passed as a vector of 21 values:
- `[0..8]` → `K_inv` (inverse of the 3x3 intrinsic matrix, for the reference) or `K` (for sensors)
- `[9..17]` → rotation matrix `R` (3x3)
- `[18..20]` → translation vector `t` (3 elements)

---

## Code Architecture

```
planeSweeping.cu
├── Macros and typedef
│   ├── MI(r, c, width)         → linear 2D/3D indexing
│   ├── BLOCKSIZE               → thread block size (x and y)
│   ├── RAD                     → SAD window radius
│   └── USE_DOUBLE              → precision selector
│
├── Constant Memory
│   ├── c_invK, c_R_inv, c_t_inv      → reference camera parameters
│   └── c_K_proj, c_R_proj, c_t_proj  → current sensor camera parameters
│
├── GPU Kernels
│   ├── warmup()                → stabilizes GPU clock frequencies
│   ├── initCostCube()          → initializes cost cube to 255.0f
│   ├── planeSweepingSAD_Naive_Pure3D_Kernel()   → naive version
│   └── planeSweepingSAD_Shared_Pure3D_Kernel()  → shared memory version
│
└── runPlaneSweepingGPU()       → host interface: allocate, transfer, launch kernels
```

---

## Implemented Optimization Strategies

### Naive 3D Kernel

**Function:** `planeSweepingSAD_Naive_Pure3D_Kernel`

Baseline version. Each thread reads SAD window pixels directly from **global VRAM**, for every Z plane.

- **3D grid**: `(img_w / BLOCKSIZE, img_h / BLOCKSIZE, ZPlanes)` — each Z block corresponds to one depth plane.
- **Pros**: simple to implement and debug.
- **Cons**: high memory latency, no data reuse. Each reference pixel is re-read from VRAM for every Z plane — the access pattern is not cache-friendly along the Z dimension.

### Shared Memory Kernel

**Function:** `planeSweepingSAD_Shared_Pure3D_Kernel`

Optimization via **shared memory (SHMEM)**. Before computing the SAD, each block loads a tile of the reference camera image — including the surrounding halo of width `RAD` — into shared memory. Threads then read from SHMEM instead of global VRAM.

Loading is split into regions:
- tile center
- left/right and top/bottom borders
- four corners

```
SHMEM size = (BLOCKSIZE + 2*RAD)^2 * sizeof(uint8_t)
```

- **Pros**: significantly reduces global VRAM accesses for reference pixels within the SAD window.
- **Cons**: shared memory is reloaded for every Z plane (pure 3D grid), so the gain is partial compared to an approach with an explicit Z loop.
- **Important note**: the code comment explicitly warns about this limitation: *"the GPU is forced to reload the halo from VRAM for every single Z plane"*.

### Constant Memory

All camera parameters (K, R, t matrices) are stored in **constant memory** (`__constant__`), which is a dedicated broadcast cache: when all threads in a warp read the same address, latency is equivalent to a register access.

```cpp
__constant__ Real c_invK[9];
__constant__ Real c_R_inv[9];
__constant__ Real c_t_inv[3];
__constant__ Real c_K_proj[9];
__constant__ Real c_R_proj[9];
__constant__ Real c_t_proj[3];
```

Parameters are updated between sensor cameras via `cudaMemcpyToSymbol`.

- **Pros**: uniform access from all threads, dedicated cache, no L2 bandwidth consumption.
- **Cons**: limited capacity (64 KB total on most GPUs).

### Precision: Float vs Double

The code supports both precisions via the `USE_DOUBLE` macro:

```cpp
#define USE_DOUBLE 0  // 0 = float, 1 = double
```

This selects the `Real` type used in all intermediate geometric computations (back-projection, world transformation, projection). The final cost in the cost cube is always `float`.

- **Float**: double throughput on FP operations, lower memory bandwidth pressure.
- **Double**: necessary when the scene has a very wide depth range or requires high numerical precision.

---

## How to Configure and Select Optimizations

### 1. Select the kernel (Naive vs Shared Memory)

In `planeSweeping.cu`, inside `runPlaneSweepingGPU()`, find the kernel launch block and comment/uncomment the desired version:

```cpp
// === NAIVE VERSION (no shared memory) ===
planeSweepingSAD_Naive_Pure3D_Kernel<<<grid_3D, block>>>(
    d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);

// === SHARED MEMORY VERSION ===
// planeSweepingSAD_Shared_Pure3D_Kernel<<<grid_3D, block, sharedMemBytes>>>(
//     d_ref, d_sens, d_costCube, width, height, ZNear, ZFar, ZPlanes);
```

Only activate one version at a time.

### 2. Select numerical precision

At the top of the file:

```cpp
#define USE_DOUBLE 0   // Float (default, faster)
#define USE_DOUBLE 1   // Double (higher precision)
```

### 3. Configure the SAD window size

```cpp
#define RAD 1   // 3x3 window (default)
// RAD 2 → 5x5 window, more robust to noise but more expensive
// RAD 3 → 7x7 window
```

Increasing `RAD` improves matching robustness but raises operation count quadratically: `(2*RAD+1)^2`.

### 4. Configure the CUDA block size

```cpp
#define BLOCKSIZE 16   // 16x16 threads per block (default)
// Typical values: 8, 16, 32
```

Higher values increase theoretical occupancy but reduce the shared memory available per block. With `RAD=1` and `BLOCKSIZE=16`, SHMEM usage is `(16+2)^2 = 324 bytes` — well below the hardware limit.

### 5. Enable/disable GPU warmup

The warmup is commented out in the code. To re-enable it (useful for more accurate benchmarks on GPUs with dynamic frequency boost), uncomment the relevant lines:

```cpp
// float* warmupA; float* warmupB;
// CHK(cudaMalloc(&warmupA, plane));
// ...
// warmup<<<warmupgrid, warmupblock>>>(warmupA, warmupB, width, height);
```

---

## Main Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BLOCKSIZE` | Thread block size (x and y) | `16` |
| `RAD` | SAD window radius | `1` |
| `USE_DOUBLE` | Precision: 0=float, 1=double | `0` |
| `ZNear` | Minimum depth to estimate | scene-dependent |
| `ZFar` | Maximum depth to estimate | scene-dependent |
| `ZPlanes` | Number of depth planes | typically 64–256 |

---

## Output and Performance Metrics

At the end of execution, `end_cuda_timer` prints:

```
CONSTANT_MEMORY (Kernel):
  Processing: 0.023141 (s), GFLOPS: 12.47
```

The FLOP count is estimated as:
```
FLOP = 2 * (2*RAD+1)^2 * img_w * img_h * ZPlanes * num_sensors
```

The factor of 2 approximates the subtraction and accumulation operations per SAD pixel. The geometric pipeline cost is not included, so this should be treated as a lower bound.

The resulting **cost cube** has dimensions `[ZPlanes × img_h × img_w]` in float. To extract the depth map, select for each pixel `(i, j)` the plane `zi` with the minimum cost and convert it to depth using the inverse formula.

---

## Requirements

- CUDA Toolkit ≥ 11.0
- GPU with Compute Capability ≥ 6.0 (Pascal or newer)
- C++14-compatible compiler (`nvcc`)

Example compilation:

```bash
nvcc -O3 -arch=sm_86 planeSweeping.cu -o planeSweeping
```

Replace `sm_86` with your GPU's architecture (`sm_75` for Turing, `sm_80` for Ampere A100, etc.).
