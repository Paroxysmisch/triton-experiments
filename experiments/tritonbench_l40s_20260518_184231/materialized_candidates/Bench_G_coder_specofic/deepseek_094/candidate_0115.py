c++
// Triton Kernel
__device__ void rmsnorm_triton(
    float* x_ptr, float* rms_w_ptr, float* out_ptr, 
    const int N_SIZE, const float eps, 
    const int K_start, const int K_end) {

    int batch = blockIdx.x;
    int M = blockIdx.y;
    int K = threadIdx.x;

    if (K < K_end) {
        float rms = 0.0f;
        for (int i = 0; i < N_SIZE; ++i) {
            float val = x_ptr[batch * N_SIZE * K + M * K + i];
            rms += val * val;
        }
        rms = sqrtf(rms / N_SIZE) * rms_w_ptr[M] + eps;

        for (int i = 0; i < N_SIZE; ++i) {
            float val = x_ptr[batch * N_SIZE * K + M * K + i];
            out_ptr[batch * N_SIZE * K + M * K + i] = val / rms;
        }
    }
}

// Triton Wrapper
void rmsnorm_wrapper(
    float* x, float* rms_w, float* out, 
    const int batch, const int M, const int K, 
    const int N_SIZE, const float eps) {

    dim3 grid_dim(batch, M);
    dim3 block_dim(K);

    rmsnorm_triton<<<grid_dim, block_dim>>>(
        x, rms_w, out, N_SIZE, eps, 0, K);
}
