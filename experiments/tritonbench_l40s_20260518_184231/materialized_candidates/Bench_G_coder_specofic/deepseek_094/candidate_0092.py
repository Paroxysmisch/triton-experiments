cpp
__device__ int puzzle1_kernel(int* x_ptr, int* output_ptr, int N, int BLOCK_SIZE, int value) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_start = pid * BLOCK_SIZE;
    int offsets = block_start + threadIdx.x;
    int mask = (offsets < N);
    int data = mask * x_ptr[offsets];
    int result = data + value;
    output_ptr[offsets] = mask * result;
    return output_ptr[offsets];
}

__device__ int* puzzle1(int* x, int N, int BLOCK_SIZE, int value) {
    int* output = (int*)malloc(N * sizeof(int));
    cudaMemcpy(output, x, N * sizeof(int), cudaMemcpyDeviceToDevice);
    puzzle1_kernel<<<grid(N, BLOCK_SIZE), BLOCK_SIZE>>>(x, output, N, BLOCK_SIZE, value);
    return output;
}
