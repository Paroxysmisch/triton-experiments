import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q,
    K,
    V,
    B0,
    sm_scale,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_b0bs,
    stride_b0h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_heads = tl.cdiv(Q.shape[1], BLOCK_DMODEL)
    head_id = pid % num_heads
    batch_id = pid // num_heads

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (batch_id * stride_qbs + head_id * stride_qh) + offs_m[:, None] * stride_qbs + offs_d[None, :]
    k_ptrs = K + (batch_id * stride_kbs + head_id * stride_kh) + offs_d[:, None] * stride_kbs + offs_n[None, :]
    v_ptrs = V + (batch_id * stride_vbs + head_id * stride_vh) + offs_n[:, None] * stride_vbs + offs_d[None, :]
    b0_ptrs = B0 + (batch_id * stride_b0bs + head_id * stride_b0h) + offs_m[:, None] * stride_b0bs + offs_n[None, :]
    out_ptrs = Out + (batch_id * stride_obs + head_id * stride_oh) + offs_m[:, None] * stride_obs + offs_d[None, :]

    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    b0 = tl.load(b0_ptrs)

    # Compute QK^T
    qk = tl.dot(q, k, allow_tf32=True)
    qk *= sm_scale

    # Add bias
    qk += b0

    # Compute softmax
    m = tl.max(qk, 1)
    qk = qk - m[:, None]
    p = tl.exp(qk)
    l = tl.sum(p, 1)
    p = p / l[:, None]

    # Compute output
    out = tl.dot(p, v, allow_tf32=True)

    # Store output
    tl.store(out_ptrs, out)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, out):
    # Validate input tensor shapes and types
    assert q.shape == k.shape == v.shape, "Q, K, V must have the same shape"
    assert q.dtype == k.dtype == v.dtype, "Q, K, V must have the same data type"
    assert q.shape[2] == rel_h_w.shape[0], "Q's third dimension must match rel_h_w's first dimension"
    assert q.shape[3] == rel_h_w.shape[1], "Q's fourth dimension must match rel_h_w's second dimension"

    # Constants
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = q.shape[3]
    OUT_DTYPE = q.dtype
    BIAS_LAST_SIZE = rel_h_w.shape[1]

    # Grid configuration
    grid = (q.shape[0] * q.shape[1],)

    # Launch the kernel
    _fwd_kernel_aligned[grid](
        q,
        k,
        v,
        rel_h_w,
        sm_scale,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        rel_h_w.stride(0),
        rel_h_w.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=1,
    )
