cpp
#include <triton/triton.h>

// Kernel definition
__global__ void puzzle1_kernel(float* x_ptr, float* output_ptr, int N, float value) {
    // Get the block index and thread index
    int pid = blockIdx.x;
    int tid = threadIdx.x;

    // Calculate the starting index for the current block
    int block_start = pid * blockDim.x;

    // Calculate the offsets for the current block
    int offsets = block_start + tid;

    // Ensure the offsets are within the valid range
    int mask = (offsets < N);

    // Load data from x_ptr, add the value, and store the result in output_ptr
    if (mask) {
        output_ptr[offsets] = x_ptr[offsets] + value;
    }
}

// Wrapper function
triton::Tensor puzzle1(triton::Tensor x) {
    // Verify CUDA compatibility of input and output tensors
    if (x.dtype() != triton::DataType::FLOAT32) {
        throw std::runtime_error("Input tensor must be of type FLOAT32");
    }

    // Initialize output tensor with the same shape and type as x
    triton::Tensor output = triton::Tensor(x.shape(), triton::DataType::FLOAT32);

    // Calculate the total number of elements
    int N = x.numel();

    // Define the grid size
    dim3 grid((N + blockDim.x - 1) / blockDim.x, 1, 1);

    // Define the block size
    dim3 block(triton::BLOCK_SIZE, 1, 1);

    // Define the constant value to add
    float value = 10.0f;

    // Launch the kernel
    puzzle1_kernel<<<grid, block>>>(x.data<float>(), output.data<float>(), N, value);

    // Return the output tensor
    return output;
}
