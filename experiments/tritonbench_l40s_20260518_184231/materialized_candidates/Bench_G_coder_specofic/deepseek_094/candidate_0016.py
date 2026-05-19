cpp
__device__
void _fwd_kernel(float* Q, float* K, float* V, float* Out, float sm_scale, int B_Start_Loc, int B_Seqlen) {
    // Define the block dimensions
    const int BLOCK_M = 128;
    const int BLOCK_DMODEL = 64;
    const int BLOCK_N = 128;

    // Get the block and thread IDs
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // Calculate the start and end indices for the current block
    int start = B_Start_Loc + bid * BLOCK_DMODEL;
    int end = start + BLOCK_DMODEL;

    // Initialize the output to zero
    float out[BLOCK_M][BLOCK_N];
    for (int i = 0; i < BLOCK_M; i++) {
        for (int j = 0; j < BLOCK_N; j++) {
            out[i][j] = 0.0f;
        }
    }

    // Compute the attention scores
    for (int i = start; i < end; i++) {
        for (int j = 0; j < BLOCK_N; j++) {
            float qk = Q[i * BLOCK_N + j] * K[i * BLOCK_N + j];
            out[i % BLOCK_M][j] += qk * sm_scale;
        }
    }

    // Store the output
    for (int i = 0; i < BLOCK_M; i++) {
        for (int j = 0; j < BLOCK_N; j++) {
            Out[(bid * BLOCK_M + i) * BLOCK_N + j] = out[i][j];
        }
    }
}

void context_attention_fwd(float* Q, float* K, float* V, float* Out, float sm_scale, int B_Start_Loc, int B_Seqlen) {
    // Calculate the number of blocks
    int num_blocks = (B_Seqlen + BLOCK_DMODEL - 1) / BLOCK_DMODEL;

    // Invoke the Triton kernel
    _fwd_kernel<<<num_blocks, BLOCK_DMODEL>>>(Q, K, V, Out, sm_scale, B_Start_Loc, B_Seqlen);
}
