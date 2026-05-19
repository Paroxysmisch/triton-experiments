cpp
#include <cmath>
#include <cuda_runtime.h>
#include <triton/api.h>

void sqrt_tanh_kernel(const float* input, float* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        float sqrt_val = sqrtf(input[i]);
        output[i] = tanhf(sqrt_val);
    }
}
