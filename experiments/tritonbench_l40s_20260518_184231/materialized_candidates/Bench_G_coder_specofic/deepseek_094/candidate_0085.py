cpp
__global__ void _fwd_kernel_flash_decode_stage2(
    const int B_Seqlen,
    const float* __restrict__ Mid_O,
    const float* __restrict__ Mid_O_LogExpSum,
    float* __restrict__ O,
    const int stride_mid_ob, const int stride_mid_oh, const int stride_mid_os, const int stride_mid_od,
    const int BLOCK_SEQ, const int BLOCK_DMODEL) {

    // Calculate the global thread IDs
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // Check if within bounds
    if (bid >= B_Seqlen || tid >= BLOCK_DMODEL) return;

    // Initialize accumulation
    float accum = 0.0f;

    // Accumulate weighted values
    for (int bs = 0; bs < BLOCK_SEQ; bs++) {
        int index = bid * BLOCK_SEQ * stride_mid_ob + bs * stride_mid_oh + tid * stride_mid_os;
        accum += Mid_O[index] * expf(Mid_O_LogExpSum[bid * BLOCK_SEQ + bs]);
    }

    // Normalize and write back
    O[bid * BLOCK_DMODEL + tid] = accum / expf(Mid_O_LogExpSum[bid * BLOCK_SEQ]);
}

void flash_decode_stage2(
    const int B_Seqlen,
    const float* Mid_O,
    const float* Mid_O_LogExpSum,
    float* O,
    const int stride_mid_ob, const int stride_mid_oh, const int stride_mid_os, const int stride_mid_od,
    const int BLOCK_SEQ, const int BLOCK_DMODEL) {

    // Set up the computation grid
    dim3 threadsPerBlock(BLOCK_DMODEL);
    dim3 numBlocks(B_Seqlen);

    // Launch the kernel
    _fwd_kernel_flash_decode_stage2<<<numBlocks, threadsPerBlock>>>(
        B_Seqlen, Mid_O, Mid_O_LogExpSum, O,
        stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
        BLOCK_SEQ, BLOCK_DMODEL);
}
