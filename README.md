# CUDA Plane Sweeping Stereo

This project implements a high-performance **Plane Sweeping** algorithm for 3D reconstruction on NVIDIA GPUs using **CUDA**. It is designed to compute a 3D cost volume (Cost Cube) by projecting reference and sensor images onto depth planes, using the **SAD (Sum of Absolute Differences)** metric to find stereo correspondences.

---

## Table of Contents
1. [Theoretical Context](#theoretical-context)
2. [Key Features](#key-features)
3. [Prerequisites](#prerequisites)
4. [Compilation](#compilation)
5. [Configuration](#configuration)
6. [Integration Guide](#integration-guide)
7. [Performance Monitoring](#performance-monitoring)

---

## 1. Theoretical Context
The Plane Sweeping algorithm creates a cost volume by iteratively sampling the depth space ($Z$):
*   **Depth Planes:** The 3D space is divided into $N$ planes ranging from $Z_{near}$ to $Z_{far}$.
*   **Warping:** For each plane, images from secondary sensors are projected onto the reference camera's view.
*   **SAD Matching:** The difference in intensity between the reference image and the projected sensor view is computed.
*   **Result:** The final output is a 3D tensor (`costCube`) where the minimum value along the depth axis represents the most likely depth for each pixel.

## 2. Key Features
*   **Constant Memory:** Exploits `__constant__` memory for camera parameters ($K, R, t$), minimizing latency for coordinate transformations.
*   **Shared Memory Optimization:** The `planeSweepingSAD_Shared_Pure3D_Kernel` caches image neighborhoods to reduce redundant Global Memory (VRAM) transactions.
*   **3D Grid Parallelism:** Utilizes a 3D CUDA grid $(x, y, z)$ to process depth planes concurrently.

## 3. Prerequisites
*   **NVIDIA GPU:** Compute Capability 3.0 or higher.
*   **CUDA Toolkit:** Installed and added to your `PATH`.
*   **Compiler:** A C++ compiler (e.g., `g++` or `msvc`) compatible with `nvcc`.

## 4. Compilation
Use the NVIDIA CUDA Compiler (`nvcc`) to build the project. Ensure you set the `-arch` flag to match your GPU's Compute Capability.

```bash
# Example for a GPU with Compute Capability 8.6 (e.g., RTX 30-series)
nvcc -O3 --ptxas-options=-v -arch=sm_86 plane_sweeping.cu -o plane_sweeping
