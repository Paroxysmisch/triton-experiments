cpp
__device__ void spinning_lock_kernel(
    float* P,
    float* C,
    int* locks,
    int num_sms,
    int k,
    int M,
    int N,
    int stride_cm,
    int stride_cn,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int pid,
    int pid_m,
    int pid_n)
{
    int gid = pid * BLOCK_SIZE_M * BLOCK_SIZE_N + pid_m * BLOCK_SIZE_N + pid_n;
    float acc = 0.0f;
    for (int l = 0; l < 9; ++l) {
        if (pid % k == 0) {
            int lock_idx = gid * k + l;
            while (tl.atomic_cas(&locks[lock_idx], 0, 1)) {
                // Spinning
            }
            acc += P[gid * 9 + l];
            locks[lock_idx] = 0;
        } else {
            if (pid_m == 0 && pid_n == 0) {
                C[gid * 9 + l] = acc;
            }
            tl.atomic_xchg(&locks[gid * k + l], 0);
        }
    }
}
