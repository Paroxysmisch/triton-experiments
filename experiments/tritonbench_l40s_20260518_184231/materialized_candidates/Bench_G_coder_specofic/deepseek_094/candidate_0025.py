import triton

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, max_ptr, denom_ptr, mask_ptr,
    batch_size, num_heads, sequence_length, dimensions, sm_scale,
    BLOCK_M, BLOCK_N, IS_CAUSAL, Lk, num_warps):

    # Load inputs
    Q = triton.mem(Q_ptr + blockIdx.x * sequence_length * dimensions).to_float()
    K = triton.mem(K_ptr + blockIdx.x * sequence_length * dimensions).to_float()
    V = triton.mem(V_ptr + blockIdx.x * sequence_length * dimensions).to_float()

    # Compute dot products
    qk = triton.dot(Q, K)

    # Apply softmax scaling
    qk_exp = triton.exp(qk * sm_scale)

    # Apply causal mask if enabled
    if IS_CAUSAL:
        mask = triton.mem(mask_ptr + blockIdx.x * sequence_length).to_float()
        qk_exp = qk_exp * mask

    # Compute denominator
    denom = triton.sum(qk_exp, axis=-1)

    # Store denominator
    triton.mem[denom_ptr + blockIdx.x * sequence_length].to_float() = denom

    # Normalize and store output
    o = qk_exp / denom.reshape((-1, 1))
    triton.mem[O_ptr + blockIdx.x * sequence_length * dimensions].to_float() = o

    # Compute and store maximum
    max_val = triton.max(o, axis=-1)
    triton.mem[max_ptr + blockIdx.x * sequence_length].to_float() = max_val

def flash_attn_triton(q, k, v, o, max_o, denom, mask, sm_scale):
    # Define parameters
    batch_size, num_heads, sequence_length, dimensions = q.shape
    Lk = dimensions // num_heads
    BLOCK_M = 128 // num_heads
    BLOCK_N = 128 // num_heads
    IS_CAUSAL = False
    num_warps = 4

    # Check if Lk is supported
    assert Lk in [16, 32, 64, 128]

    # Compute grid and block dimensions
    grid = (batch_size, num_heads, sequence_length)
    block = (BLOCK_M, BLOCK_N, num_warps)

    # Call Triton kernel
    _fwd_kernel[grid, block](
        q.device_ctypes_ptr, k.device_ctypes_ptr, v.device_ctypes_ptr,
        o.device_ctypes_ptr, max_o.device_ctypes_ptr, denom.device_ctypes_ptr,
        mask.device_ctypes_ptr, batch_size, num_heads, sequence_length,
        dimensions, sm_scale, BLOCK_M, BLOCK_N, IS_CAUSAL, Lk, num_warps)
