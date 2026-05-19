cpp
#include <cuda_runtime.h>
#include <triton/kernel_config.h>

#define SIGN_MASK_F4 (0x80)

__device__ uint16_t convert_f4_to_bf16(uint8_t x, uint8_t s, uint16_t sign_mask) {
    uint16_t exponent_mask = 0x7C00;
    uint16_t mantissa_mask = 0x03FF;
    uint16_t result = 0;

    uint8_t sign = (x & sign_mask) >> 7;
    uint8_t exponent = (x & exponent_mask) >> 4;
    uint8_t mantissa = x & mantissa_mask;

    result = (sign << 15) | (exponent << 10) | (mantissa << 4) | (s & 0x0F);

    return result;
}

__global__ void triton_f4_to_scaled_bf16_kernel(const uint8_t* x_ptr, const uint8_t* s_ptr, uint16_t* output_ptr, size_t n_elements_in) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n_elements_in) {
        uint16_t result = convert_f4_to_bf16(x_ptr[gid], s_ptr[gid / 2], SIGN_MASK_F4);
        output_ptr[gid] = result;
    }
}
