cpp
__device__ void my_kernel(int* input, int* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        output[index] = input[index] * 2;
    }
}

__global__ void my_kernel_wrapper(int* input, int* output) {
    my_kernel(input, output);
}
