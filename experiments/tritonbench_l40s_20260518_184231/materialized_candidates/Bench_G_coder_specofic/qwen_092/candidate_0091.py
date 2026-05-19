import triton
import triton.language as tl

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

@triton.jit
def matmul_kernel(
    Q: tl.tensor,  # quantized input matrix (M, K)
    S: tl.tensor,  # scale (K,)
    Z: tl.tensor,  # zero point (K,)
    W: tl.tensor,  # quantized weight matrix (K, N)
    B: tl.tensor,  # bias (N,)
    acc: tl.tensor,  # accumulator (M, N)
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_Q: tl.int32, stride_W: tl.int32, stride_acc: tl.int32,
    split_k: tl.int32,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    pid_m = pid // (num_pid_n * num_pid_k)
    pid_n = (pid // num_pid_k) % num_pid_n
    pid_k = pid % num_pid_k

    row_base = pid_m * BLOCK_SIZE_M
    col_base = pid_n * BLOCK_SIZE_N
    k_base = pid_k * BLOCK_SIZE_K

    acc_ptr = acc + row_base * stride_acc + col_base

    for k in range(0, K, BLOCK_SIZE_K):
        k_inner = k // BLOCK_SIZE_K
        q_ptr = Q + row_base * stride_Q + (k + k_inner * BLOCK_SIZE_K)
        w_ptr = W + (k + k_inner * BLOCK_SIZE_K) * stride_W + col_base

        q = tl.load(q_ptr, mask=(k + k_inner * BLOCK_SIZE_K < K), eviction_policy=tl.EvictionPolicy.LRU)
        w = tl.load(w_ptr, mask=(k + k_inner * BLOCK_SIZE_K < K), eviction_policy=tl.EvictionPolicy.LRU)

        scale = S[k + k_inner * BLOCK_SIZE_K]
        zero_point = Z[k + k_inner * BLOCK_SIZE_K]

        q_deq = (q * scale + zero_point).to(tl.float32)
        w_deq = (w * scale + zero_point).to(tl.float32)

        for m in range(BLOCK_SIZE_M):
            for n in range(BLOCK_SIZE_N):
                acc_mn = acc_ptr[m, n]
                acc_ptr[m, n] = acc_mn + q_deq[m] * w_deq[n]
