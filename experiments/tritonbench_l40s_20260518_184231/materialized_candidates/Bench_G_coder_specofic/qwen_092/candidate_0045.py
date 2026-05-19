cpp
#include <triton/core.hh>
#include <triton/ir.hh>
#include <triton/ops.hh>

using namespace triton::core;
using namespace triton::ir;
using namespace triton::ops;

__global__ void pow_func_scalar_tensor_kernel_rank_1(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    int n) {
    __shared__ float shared_input[128];
    __shared__ float shared_output[128];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        shared_input[tid] = input[idx];
        __syncthreads();

        int shared_idx = tid;
        while (shared_idx < 128) {
            shared_output[shared_idx] = powf(shared_input[shared_idx], scalar);
            shared_idx += blockDim.x;
        }
        __syncthreads();

        if (tid < n) {
            output[idx] = shared_output[tid];
        }
    }
}
