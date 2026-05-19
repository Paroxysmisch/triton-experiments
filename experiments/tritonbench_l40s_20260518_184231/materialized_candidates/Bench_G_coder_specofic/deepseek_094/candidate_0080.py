cpp
__device__ void _fwd_kernel_token_att2(
    const int32_t* __restrict__ Prob,
    const int8_t* __restrict__ V,
    int8_t* __restrict__ Out,
    const int32_t* __restrict__ Req_to_tokens,
    const int32_t cur_batch,
    const int32_t cur_head,
    const int32_t B_req_idx,
    const int32_t B_Start_Loc,
    const int32_t B_Seqlen,
    const int32_t num_warps,
    const int32_t num_stages,
    const int32_t kv_group_num,
    const int32_t BLOCK_DMODEL,
    const int32_t BLOCK_N
) {
    // Calculate offsets
    const int32_t offset_Prob = cur_batch * BLOCK_DMODEL * B_Seqlen + cur_head * BLOCK_N;
    const int32_t offset_V = cur_batch * BLOCK_DMODEL * B_Seqlen;
    const int32_t offset_Out = cur_batch * BLOCK_DMODEL * B_Seqlen;

    // Initialize accumulator
    int32_t acc[BLOCK_DMODEL];
    for (int32_t i = 0; i < BLOCK_DMODEL; i++) {
        acc[i] = 0;
    }

    // Perform attention computation
    for (int32_t block_idx = 0; block_idx < BLOCK_N; block_idx++) {
        const int32_t token_idx = B_Start_Loc + block_idx;
        const int32_t prob_idx = offset_Prob + token_idx;
        const int32_t v_idx = offset_V + token_idx * BLOCK_DMODEL;

        const int32_t prob = Prob[prob_idx];
        for (int32_t d = 0; d < BLOCK_DMODEL; d++) {
            acc[d] += prob * V[v_idx + d];
        }
    }

    // Store result
    const int32_t out_idx = offset_Out + B_req_idx * BLOCK_DMODEL;
    for (int32_t d = 0; d < BLOCK_DMODEL; d++) {
        Out[out_idx + d] = acc[d];
    }
}
