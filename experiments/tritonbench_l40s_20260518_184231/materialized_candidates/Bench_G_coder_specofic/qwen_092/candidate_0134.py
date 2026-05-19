cpp
#include <triton/core/triton.h>

// Define constants for blocking strategy
const int BLOCK_M = 64;
const int BLOCK_DMODEL = 64;
const int BLOCK_N = 64;

// Triton kernel for scaled dot-product attention
__global__ void _fwd_kernel(
    const float* Q, const float* K, const float* V,
    float* Out, const float* sm_scale, const int* B_Start_Loc, const int* B_Seqlen,
    int B, int H, int Lq, int Lk, int Dk, int Dv, int L_out)
{
    // Thread indices
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;

    // Initialize output
    float out_val = 0.0f;

    // Iterate over the sequence length of K
    for (int k = 0; k < Lk; ++k)
    {
        // Calculate dot product of Q[m, h, :] and K[:, h, k]
        float dot_product = 0.0f;
        for (int d = 0; d < Dk; ++d)
        {
            dot_product += Q[m * H * Dk + h * Dk + d] * K[k * H * Dk + h * Dk + d];
        }

        // Scale the dot product and apply softmax
        dot_product *= sm_scale[h];
        out_val += expf(dot_product) * V[k * H * Dv + h * Dv + n];
    }

    // Write the result to the output tensor
    if (m < L_out && n < Dv)
    {
        Out[m * L_out * Dv + n] = out_val;
    }
}

// Triton wrapper function for scaled dot-product attention
__global__ void context_attention_fwd(
    float* Q, float* K, float* V, float* Out,
    const int* B_Start_Loc, const int* B_Seqlen,
    int B, int H, int Lq, int Lk, int Dk, int Dv, int L_out)
{
    // Compute the softmax scaling factor
    float sm_scale[H];
    for (int h = 0; h < H; ++h)
    {
        sm_scale[h] = 1.0f / sqrtf(Dk);
    }

    // Define the grid and block dimensions
    dim3 gridDim((Lq + BLOCK_M - 1) / BLOCK_M, (L_out + BLOCK_DMODEL - 1) / BLOCK_DMODEL, H);
    dim3 blockDim(BLOCK_M, BLOCK_DMODEL, 1);

    // Dispatch the kernel
    _fwd_kernel<<<gridDim, blockDim>>>(Q, K, V, Out, sm_scale, B_Start_Loc, B_Seqlen, B, H, Lq, Lk, Dk, Dv, L_out);
}
