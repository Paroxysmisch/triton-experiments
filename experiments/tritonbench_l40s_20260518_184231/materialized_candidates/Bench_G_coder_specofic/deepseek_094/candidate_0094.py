c
#define BLOCK_SIZE_M 16
#define BLOCK_SIZE_N 16
#define BLOCK_SIZE_K 16

__device__ void matmul_kernel(float* a, float* b, float* c, int M, int N, int K, int pid_m, int pid_n, int stride_am, int stride_ak, int stride_bk, int stride_bn, int stride_cm, int stride_cn, int offs_am, int offs_ak, int offs_bk, int offs_bn, int offs_cm, int offs_cn) {
    __shared__ float a_block[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float b_block[BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accumulator = 0.0f;

    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        a_block[pid_m][pid_n] = a[offs_am + k * stride_ak];
        b_block[pid_m][pid_n] = b[offs_bk + k * stride_bk];

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE_K; i++) {
            accumulator += a_block[pid_m][i] * b_block[i][pid_n];
        }

        __syncthreads();
    }

    c[offs_cm] = (float16)accumulator;
}

__global__ void matmul(float* a, float* b, float* c, int M, int N, int K) {
    int pid_m = blockIdx.y * BLOCK_SIZE_M;
    int pid_n = blockIdx.x * BLOCK_SIZE_N;

    int stride_am = M;
    int stride_ak = 1;
    int stride_bk = K;
    int stride_bn = 1;
    int stride_cm = M;
    int stride_cn = 1;

    int offs_am = pid_m * stride_am + pid_n * stride_ak;
    int offs_bk = pid_m * stride_bk + pid_n * stride_bn;
    int offs_cm = pid_m * stride_cm + pid_n * stride_cn;

    matmul_kernel(a, b, c, M, N, K, pid_m, pid_n, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, offs_am, offs_ak, offs_bk, offs_bn, offs_cm, offs_cn);
}
