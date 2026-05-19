cpp
#include <triton/triton.h>
#include <cmath>

// Function to determine if a floating-point number is finite
template <typename T>
__device__ bool isfinite_func(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return _finitef(value);
    } else if constexpr (std::is_same_v<T, double>) {
        return _isfinited(value);
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Unsupported type");
        return false;
    }
}

// Triton kernel for checking the finiteness of tensor elements
template <typename T>
__global__ void isfinite_func_kernel_rank_1(
    T* out0,
    const T* in0,
    int32_t N,
    bool one_tile_per_cta) {
    
    using namespace triton::common;
    using namespace triton::intrinsics;
    using namespace triton::cuda;

    // Define the block and grid dimensions
    constexpr int32_t BLOCK_SIZE = 256;
    constexpr int32_t TILE_SIZE = 64;

    // Define the block and grid indices
    int32_t tid = triton::blockIdx.x * triton::blockDim.x + triton::threadIdx.x;
    int32_t stride = triton::blockDim.x * triton::gridDim.x;

    // Load input data into Triton tensor
    triton::Tensor in0_tensor(in0, triton::Shape{N}, triton::DataType::kFloat32);

    // Process the tensor in tiles
    for (int32_t start = tid; start < N; start += stride) {
        T value = in0_tensor.load(start);
        bool result = isfinite_func(value);
        out0[start] = static_cast<T>(result);
    }
}

// Wrapper function for processing input and output tensors
template <typename T>
void isfinite_func_wrapper_rank_1(
    T* out0,
    const T* in0,
    int32_t N,
    bool one_tile_per_cta) {
    
    // Define the block and grid dimensions
    constexpr int32_t BLOCK_SIZE = 256;
    constexpr int32_t TILE_SIZE = 64;

    // Define the block and grid indices
    int32_t num_ctas = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    triton::dim3 grid(num_ctas);
    triton::dim3 block(BLOCK_SIZE);

    // Launch the kernel
    isfinite_func_kernel_rank_1<<<grid, block>>>(out0, in0, N, one_tile_per_cta);
}
