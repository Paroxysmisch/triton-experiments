cpp
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 1024
#define ALPHA 1.6732632423543772848170429916717
#define SCALE 1.0507009873554804934193349852946

__global__ void selu_kernel(float* input, float* output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = (x > 0) ? SCALE * x : SCALE * (ALPHA * (exp(x) - 1));
    }
}

extern "C" {
    void selu(float* input, float* output, int size) {
        int threads = BLOCK_SIZE;
        int blocks = (size + threads - 1) / threads;
        selu_kernel<<<blocks, threads>>>(input, output, size);
    }
}
