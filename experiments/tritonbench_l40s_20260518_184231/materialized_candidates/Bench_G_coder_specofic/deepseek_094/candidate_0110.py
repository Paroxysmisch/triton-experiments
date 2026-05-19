c++
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <triton/ir.h>

using namespace triton;

#define BLOCK_N 128
#define BLOCK_K 128

__device__ void _bgmv_expand_kernel(const half* input_ptr, const half* lora_ptr, half* out_ptr, int N, int K, const int* lora_indices, int in_stride, int lora_stride, int out_stride, bool add_input) {
    int batch = blockIdx.x;
    int lora_index = lora_indices[batch];

    if (lora_index == -1) {
        return;
    }

    int start = lora_index * K;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        half sum = 0.0f;
        for (int k = 0; k < K; k += blockDim.y) {
            int index = start + k;
            half input_val = input_ptr[i * in_stride + batch];
            half lora_val = lora_ptr[index * lora_stride];
            sum += input_val * lora_val;
        }
        if (add_input) {
            out_ptr[i * out_stride + batch] += sum;
        } else {
            out_ptr[i * out_stride + batch] = sum;
        }
    }
}

void _bgmv_expand(const Tensor& input, const Tensor& lora, Tensor& output, const Tensor& lora_indices, bool add_input) {
    input.ensure_contiguous();
    lora.ensure_contiguous();
    output.ensure_contiguous();

    int N = input.shape[0];
    int K = lora.shape[0];
    int batch_size = input.shape[1];

    dim3 grid(batch_size);
    dim3 block(BLOCK_N, BLOCK_K);

    bool cast_type = input.dtype == triton::ir::DataType::FLOAT16 && lora.dtype == triton::ir::DataType::FLOAT16;

    if (cast_type) {
        _bgmv_expand_kernel<<<grid, block>>>((const half*)input.data, (const half*)lora.data, (half*)output.data, N, K, (const int*)lora_indices.data, input.stride(1), lora.stride(0), output.stride(1), add_input);
    } else {
        _bgmv_expand_kernel<<<grid, block>>>((const float*)input.data, (const float*)lora.data, (float*)output.data, N, K, (const int*)lora_indices.data, input.stride(1), lora.stride(0), output.stride(1), add_input);
    }
}
