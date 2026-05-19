cpp
#include <cuda_runtime.h>
#include <triton/api.h>
#include <cmath>

namespace {
    __device__ bool _isfinite(double val) {
        return std::isfinite(val);
    }

    __device__ bool _isfinite(float val) {
        return std::isfinite(val);
    }

    __device__ bool isfinite_func(double val) {
        return _isfinite(val);
    }

    __device__ bool isfinite_func(float val) {
        return _isfinite(val);
    }

    __global__ void isfinite_func_kernel_rank_1(const float* in0, bool* out, size_t size) {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
            out[i] = isfinite_func(in0[i]);
        }
    }

    __global__ void isfinite_func_kernel_rank_1(const double* in0, bool* out, size_t size) {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
            out[i] = isfinite_func(in0[i]);
        }
    }
}

triton::api::Tensor isfinite_func_wrapper_rank_1(triton::api::Tensor in0, bool one_tile_per_cta) {
    const auto& in0_shape = in0.shape();
    const auto in0_size = in0.num_elements();

    auto out_shape = in0_shape;
    auto out = triton::api::Tensor(out_shape, in0.dtype(), in0.device());

    auto grid = heuristics_for_grid_size(in0_size, one_tile_per_cta);
    auto block = heuristics_for_block_size(in0_size, one_tile_per_cta);

    if (in0.dtype() == triton::api::DataType::FLOAT32) {
        isfinite_func_kernel_rank_1<<<grid, block>>>(in0.data(), out.data(), in0_size);
    } else if (in0.dtype() == triton::api::DataType::FLOAT64) {
        isfinite_func_kernel_rank_1<<<grid, block>>>(in0.data(), out.data(), in0_size);
    }

    return out;
}
