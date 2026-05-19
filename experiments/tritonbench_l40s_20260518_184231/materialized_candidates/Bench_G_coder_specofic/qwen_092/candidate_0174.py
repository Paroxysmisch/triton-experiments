cpp
#include <triton/core.h>
#include <triton/language.h>

using namespace triton;
using namespace triton::language;

// Kernel function to compute attention scores between query and key tensors
__global__ void _fwd_kernel_token_att1(
    float* Q,         // Query tensor
    float* K,         // Key tensor
    float* B_Loc,     // Positional information about keys
    float* B_Start_Loc, // Sequence start indices
    float* B_Seqlen,  // Sequence lengths
    float* Att_Out,   // Output attention values
    int B,            // Batch size
    int N,            // Number of heads
    int D,            // Dimension of each head
    int max_input_len, // Maximum sequence length within a batch
    float sm_scale    // Scaling factor for dot product
) {
    const int BLOCK_N = 32; // Block size for parallel processing

    // Thread and block indices
    int b = blockIdx.x / (B * N);
    int n = blockIdx.x % (B * N);
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;

    // Indices for Q and K
    int q_idx = b * N * D + n * D + i;
    int k_idx = b * N * D + n * D + j;

    // Load query and key segments
    float q = Q[q_idx];
    float k = K[k_idx];

    // Compute dot product and scale
    float dot_product = q * k;
    dot_product *= sm_scale;

    // Store the result in the output tensor
    Att_Out[q_idx] = dot_product;
}
