c++
void chunk_linear_attn_fwd_kernel_h(
    float* h,
    const float* k,
    const float* v,
    const float* initial_h,
    int T,
    int K,
    int V,
    int BT,
    int BK,
    int BV,
    cudaStream_t stream) {
    // Implementation
}

void chunk_linear_attn_fwd_kernel_o(
    float* o,
    const float* q,
    const float* k,
    const float* h,
    const float* mask,
    int T,
    int K,
    int V,
    int BT,
    int BK,
    int BV,
    cudaStream_t stream) {
    // Implementation
}

void chunk_linear_attn_bwd_kernel_dh(
    float* dh,
    const float* dout,
    const float* q,
    const float* k,
    const float* v,
    const float* h,
    int T,
    int K,
    int V,
    int BT,
    int BK,
    int BV,
    cudaStream_t stream) {
    // Implementation
}

void chunk_linear_attn_bwd_kernel_dqkv(
    float* dq,
    float* dk,
    float* dv,
    const float* dout,
    const float* q,
    const float* k,
    const float* v,
    const float* h,
    int T,
    int K,
    int V,
    int BT,
    int BK,
    int BV,
    cudaStream_t stream) {
    // Implementation
}
