import triton.language as tl
from triton import Tensor

@triton.jit
def _fwd_kernel(
    Q: tl.types.handle,
    K: tl.types.handle,
    V: tl.types.handle,
    B0: tl.types.handle,
    Out: tl.types.handle,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BIAS_LAST_SIZE: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    STRIDE_Q: tl.constexpr,
    STRIDE_K: tl.constexpr,
    STRIDE_V: tl.constexpr,
    STRIDE_B0: tl.constexpr,
    STRIDE_O: tl.constexpr
):
    # Fetch program id
    batch = tl.program_id(0)
    head = tl.program_id(1)
    m = tl.program_id(2)
    
    # Fetch offset information
    off_q = batch * STRIDE_Q + head * STRIDE_Q + m * BLOCK_DMODEL
    off_k = batch * STRIDE_K + head * STRIDE_K + m * BLOCK_DMODEL
    off_v = batch * STRIDE_V + head * STRIDE_V + m * BLOCK_DMODEL
    off_b0 = head * STRIDE_B0 + m * BLOCK_DMODEL
    off_o = batch * STRIDE_O + head * STRIDE_O

    # Initialize memories
    q = tl.load(Q + off_q)
    k = tl.load(K + off_k)
    v = tl.load(V + off_v)
    b0 = tl.load(B0 + off_b0)

    # Calculate dot product
    dot_product = tl.dot(q, k)
    dot_product *= sm_scale

    # Apply bias
    dot_product += b0

    # Apply softmax
    exp_values = tl.math.exp2(dot_product)
    sum_values = tl.sum(exp_values)
    softmax_values = exp_values / sum_values

    # Apply attention
    attention = tl.dot(softmax_values, v)

    # Store result
    tl.store(Out + off_o, attention)

def attention_fwd(Q, K, V, B0, O, sm_scale, batch_size, head_num, seq_len):
    # Define constants
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = 64
    OUT_DTYPE = Q.dtype
    BIAS_LAST_SIZE = head_num * seq_len
    STRIDE_Q = Q.stride(0)
    STRIDE_K = K.stride(0)
    STRIDE_V = V.stride(0)
    STRIDE_B0 = B0.stride(0)
    STRIDE_O = O.stride(0)

    # Configure grid
    grid = (batch_size, head_num, seq_len)

    # Launch kernel
    _fwd_kernel[grid](
        Q.ptr,
        K.ptr,
        V.ptr,
        B0.ptr,
        O.ptr,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DMODEL,
        sm_scale,
        OUT_DTYPE,
        BIAS_LAST_SIZE,
        batch_size,
        head_num,
        seq_len,
        STRIDE_Q,
        STRIDE_K,
        STRIDE_V,
        STRIDE_B0,
        STRIDE_O
    )
