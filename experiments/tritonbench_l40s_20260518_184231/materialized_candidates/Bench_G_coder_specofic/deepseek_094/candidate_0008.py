c++
#include <cuda_runtime.h>
#include <triton/triton.h>

__device__ void atomicMax(int* address, int val) {
    int old = *address;
    while (val > old && old != *address) {
        old = atomicCAS(address, old, val);
    }
}

__global__ void _fwd_kernel_destindex_copy_quantize_kv(
    const float* K,
    const int* DestLoc,
    int8_t* Out,
    float* Out_scale,
    int batch,
    int head,
    int head_dim,
    int group_size,
    int BLOCK_GROUP_NUM,
    int BLOCK_GROUP_DIM
) {
    int b = blockIdx.x / head;
    int h = blockIdx.x % head;
    int d = threadIdx.x;

    int group = d / BLOCK_GROUP_DIM;
    int idx = d % BLOCK_GROUP_DIM;

    int dest_index = DestLoc[b * head * head_dim + h * head_dim + idx];

    float scale = 0.0f;
    int8_t quantized_value = 0;

    if (group < BLOCK_GROUP_NUM) {
        for (int i = 0; i < group_size; i++) {
            float value = K[b * head * head_dim + h * head_dim + group * group_size + i];
            atomicMax(&scale, fabsf(value));
        }

        if (scale > 0.0f) {
            for (int i = 0; i < group_size; i++) {
                float value = K[b * head * head_dim + h * head_dim + group * group_size + i];
                quantized_value = (int8_t)(value / scale);
                Out[dest_index * head * head_dim + h * head_dim + i] = quantized_value;
            }
            Out_scale[dest_index * head + h] = scale;
        }
    }
}

void destindex_copy_quantize_kv(
    const float* K,
    const int* DestLoc,
    int8_t* Out,
    float* Out_scale,
    int batch,
    int head,
    int head_dim,
    int group_size,
    int BLOCK_GROUP_NUM,
    int BLOCK_GROUP_DIM
) {
    dim3 grid(batch * head, 1, 1);
    dim3 block(BLOCK_GROUP_NUM * BLOCK_GROUP_DIM, 1, 1);
    _fwd_kernel_destindex_copy_quantize_kv<<<grid, block>>>(
        K,
        DestLoc,
        Out,
        Out_scale,
        batch,
        head,
        head_dim,
        group_size,
        BLOCK_GROUP_NUM,
        BLOCK_GROUP_DIM
    );
}
