c
__device__
void fused_recurrent_rwkv6_fwd_kernel(
    int t,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ u,
    float* __restrict__ h,
    float* __restrict__ o,
    float scale,
    bool reverse
) {
    // Load inputs
    float q_t = q[t];
    float k_t = k[t];
    float v_t = v[t];
    float w_t = w[t];
    float u_t = u[t];

    // Recurrent update
    float h_t = h[t];
    h_t = w_t * h_t + u_t;

    // Compute output
    float o_t = scale * h_t * k_t * v_t;

    // Store outputs
    h[t] = h_t;
    o[t] = o_t;
}
