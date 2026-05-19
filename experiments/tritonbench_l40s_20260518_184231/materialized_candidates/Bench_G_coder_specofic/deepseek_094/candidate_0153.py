c++
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <triton/language.h>

namespace cg = cooperative_groups;

__device__ float my_sin(float x) {
    // your implementation here
}

__global__ void sin_kernel(float* in_ptr0, float* out_ptr, size_t n_elements, int BLOCK_SIZE) {
    cg::grid_group grid = cg::this_grid();
    int block_id = grid.thread_rank();

    for (size_t i = block_id; i < n_elements; i += BLOCK_SIZE) {
        float value = tl.load(in_ptr0 + i);
        float result = my_sin(value);
        tl.store(out_ptr + i, result);
    }
}

void sin_triton(float* x, float* out, int BLOCK_SIZE) {
    size_t n_elements = x.size();
    dim3 grid(n_elements, 1, 1);
    sin_kernel<<<grid, BLOCK_SIZE>>>(x, out, n_elements, BLOCK_SIZE);
}
