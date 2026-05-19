import triton
import triton.language as tl

# Triton kernel for forward attention mechanism
@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, stride_qm, stride_ql, stride_km, stride_kl, stride_vm, stride_vl, stride_om, stride_ol, N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    num_pid_in_block = num_pid_m * num_pid_n
    block_id = pid // num_pid_in_block
    pid_m = (pid % num_pid_in_block) // num_pid_n
    pid_n = (pid % num_pid_in_block) % num_pid_n

    # Block bounds
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Offsets for Q, K, V, and Out
    q_ptrs = Q + (block_id * stride_qm + offs_m[:, None]) * stride_ql
    k_ptrs = K + (block_id * stride_km + offs_n[None, :]) * stride_kl
    v_ptrs = V + (block_id * stride_vm + offs_n[None, :]) * stride_vl
    out_ptrs = Out + (block_id * stride_om + offs_m[:, None]) * stride_ol

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    # Loop over the context size
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load Q, K, V blocks
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs + start_n * stride_kl)
        v = tl.load(v_ptrs + start_n * stride_vl)

        # Compute attention scores
        qk = tl.dot(q, k, allow_tf32=True) * Q_scale * K_scale
        qk = tl.exp(qk - l_i)

        # Update accumulators
        acc += qk
        l_i += tl.sum(qk, axis=1, keepdim=True)

    # Normalize and store the result
    out = acc / l_i
    tl.store(out_ptrs, out.to(Out.dtype.element_ty))

# Helper function to call the kernel
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, BLOCK_M, BLOCK_N):
    # Get tensor shapes
    M, N_CTX = Q.shape
    _, N_CTX = K.shape
    _, N_CTX = V.shape
    _, N_CTX = Out.shape

    # Get strides
    stride_qm, stride_ql = Q.stride(0), Q.stride(1)
    stride_km, stride_kl = K.stride(0), K.stride(1)
    stride_vm, stride_vl = V.stride(0), V.stride(1)
    stride_om, stride_ol = Out.stride(0), Out.stride(1)

    # Launch the kernel
    grid = (triton.cdiv(N_CTX, BLOCK_M) * triton.cdiv(N_CTX, BLOCK_N) * M, 1, 1)
    _attn_fwd[grid](Q, K, V, Q_scale, K_scale, Out, stride_qm, stride_ql, stride_km, stride_kl, stride_vm, stride_vl, stride_om, stride_ol, N_CTX, BLOCK_M, BLOCK_N)
