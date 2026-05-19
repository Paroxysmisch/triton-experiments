cpp
__device__
float token_softmax_reducev_fwd_kernel(float* Logics, float* V, float* Out, int B_Loc, int B_Start_Loc, int B_Seqlen, int LogicsStride0, int LogicsStride1, int VStride0, int VStride1, int OutStride0, int OutStride1, int BLOCK_DMODEL) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int l = threadIdx.x;

    float e_max = -FLT_MAX;
    float sum = 0.0f;
    float acc = 0.0f;

    for (int i = 0; i < BLOCK_DMODEL; i++) {
        float logit = tl::load(Logics + b * LogicsStride0 + h * LogicsStride1 + l * BLOCK_DMODEL + i);
        e_max = max(e_max, logit);
    }

    for (int i = 0; i < BLOCK_DMODEL; i++) {
        float logit = tl::load(Logics + b * LogicsStride0 + h * LogicsStride1 + l * BLOCK_DMODEL + i);
        float e = exp(logit - e_max);
        sum += e;
    }

    for (int i = 0; i < BLOCK_DMODEL; i++) {
        float logit = tl::load(Logics + b * LogicsStride0 + h * LogicsStride1 + l * BLOCK_DMODEL + i);
        float e = exp(logit - e_max);
        acc += e * tl::load(V + b * VStride0 + h * VStride1 + l * BLOCK_DMODEL + i);
    }

    for (int i = 0; i < BLOCK_DMODEL; i++) {
        float p = exp(logit - e_max) / sum;
        tl::store(Out + b * OutStride0 + h * OutStride1 + l * BLOCK_DMODEL + i, p);
    }

    return acc;
}
