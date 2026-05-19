cpp
#include <cuda_runtime.h>
#include <cuda_triton.h>

#define BLOCK_SIZE 64
#define TILE_SIZE 16

__device__ float load_float(const float* ptr, int offset) {
    return ptr[offset];
}

__device__ void store_float(float* ptr, int offset, float value) {
    ptr[offset] = value;
}

__device__ float add_float(float a, float b) {
    return a + b;
}

__device__ float sub_float(float a, float b) {
    return a - b;
}

__device__ float mul_float(float a, float b) {
    return a * b;
}

__device__ float div_float(float a, float b) {
    return a / b;
}

__device__ float max_float(float a, float b) {
    return fmaxf(a, b);
}

__device__ float min_float(float a, float b) {
    return fminf(a, b);
}

__device__ float pow_float(float a, float b) {
    return powf(a, b);
}

__device__ float sqrt_float(float a) {
    return sqrtf(a);
}

__device__ float abs_float(float a) {
    return fabsf(a);
}

__device__ float neg_float(float a) {
    return -a;
}

__device__ int eq_float(float a, float b) {
    return a == b;
}

__device__ int neq_float(float a, float b) {
    return a != b;
}

__device__ int lt_float(float a, float b) {
    return a < b;
}

__device__ int le_float(float a, float b) {
    return a <= b;
}

__device__ int gt_float(float a, float b) {
    return a > b;
}

__device__ int ge_float(float a, float b) {
    return a >= b;
}

__device__ float atomic_add_float(float* ptr, float value) {
    return atomicAdd(ptr, value);
}

__device__ float atomic_sub_float(float* ptr, float value) {
    return atomicAdd(ptr, -value);
}

__device__ float atomic_min_float(float* ptr, float value) {
    float old_value;
    do {
        old_value = *ptr;
    } while (old_value > value &&
             !atomicCAS(ptr, old_value, value));
    return old_value;
}

__device__ float atomic_max_float(float* ptr, float value) {
    float old_value;
    do {
        old_value = *ptr;
    } while (old_value < value &&
             !atomicCAS(ptr, old_value, value));
    return old_value;
}

__device__ void barrier() {
    __syncthreads();
}

__device__ void grid_barrier() {
    barrier();
}

__device__ int warp_size() {
    return WARP_SIZE;
}

__device__ int warp_id() {
    return threadIdx.x / WARP_SIZE;
}

__device__ int lane_id() {
    return threadIdx.x % WARP_SIZE;
}

__device__ int grid_size() {
    return gridDim.x;
}

__device__ int grid_id() {
    return blockIdx.x;
}

__device__ int block_size() {
    return blockDim.x;
}

__device__ int block_id() {
    return threadIdx.x;
}

__device__ int warp_num() {
    return blockDim.x / WARP_SIZE;
}

__device__ int block_num() {
    return gridDim.x;
}

__device__ int block_num_y() {
    return gridDim.y;
}

__device__ int block_num_z() {
    return gridDim.z;
}

__device__ int block_id_y() {
    return blockIdx.y;
}

__device__ int block_id_z() {
    return blockIdx.z;
}

__device__ int thread_id_y() {
    return threadIdx.y;
}

__device__ int thread_id_z() {
    return threadIdx.z;
}

__device__ int block_id_x() {
    return blockIdx.x;
}

__device__ int thread_id_x() {
    return threadIdx.x;
}

__device__ int warp_id_y() {
    return (blockIdx.y * blockDim.y + threadIdx.y) / WARP_SIZE;
}

__device__ int warp_id_z() {
    return (blockIdx.z * blockDim.z + threadIdx.z) / WARP_SIZE;
}

__device__ int warp_id_x() {
    return threadIdx.x / WARP_SIZE;
}

__device__ int thread_num_y() {
    return blockDim.y * block_num_y();
}

__device__ int thread_num_z() {
    return blockDim.z * block_num_z();
}

__device__ int thread_num_x() {
    return blockDim.x * block_num_x();
}

__device__ int thread_id_flat() {
    return block_id_x() * blockDim.x + thread_id_x();
}

__device__ int thread_num_flat() {
    return blockDim.x * block_num_x();
}

__device__ int warp_id_flat() {
    return warp_id_x() * WARP_SIZE + lane_id();
}

__device__ int warp_num_flat() {
    return blockDim.x / WARP_SIZE * block_num_x();
}

__device__ int block_id_flat() {
    return block_id_x();
}

__device__ int block_num_flat() {
    return block_num_x();
}

__device__ int warp_size_flat() {
    return WARP_SIZE;
}

__device__ int thread_num() {
    return blockDim.x * gridDim.x;
}

__device__ int thread_id() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ int block_num() {
    return gridDim.x;
}

__device__ int block_size() {
    return blockDim.x;
}

__device__ int grid_size() {
    return gridDim.x;
}

__device__ int warp_size() {
    return WARP_SIZE;
}

__device__ int warp_id() {
    return threadIdx.x / WARP_SIZE;
}

__device__ int lane_id() {
    return threadIdx.x % WARP_SIZE;
}

__device__ int block_id() {
    return blockIdx.x;
}

__device__ int grid_id() {
    return blockIdx.y;
}

__device__ int warp_id() {
    return threadIdx.x / WARP_SIZE;
}

__device__ int lane_id() {
    return threadIdx.x % WARP_SIZE;
}

__device__ int block_id() {
    return blockIdx.x;
}

__device__ int grid_id() {
    return blockIdx.y;
}

__device__ int warp_id() {
    return threadIdx.x / WARP_SIZE;
}

__device__ int lane_id() {
