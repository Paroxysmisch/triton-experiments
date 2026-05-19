import torch
import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_M = 128
BLOCK_DMODEL = 64
BLOCK_N = 128

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    B_Start_Loc, B_Seqlen,
    sm_scale,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Calculate block indices
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id = tl.program_id(2)

    # Calculate starting indices for Q, K, V, and Out
    start_q = batch_id * stride_qm + head_id * stride_qd + block_id * BLOCK_M
    start_k = batch_id * stride_km + head_id * stride_kd
    start_v = batch_id * stride_vm + head_id * stride_vd
    start_o = batch_id * stride_om + head_id * stride_od + block_id * BLOCK_M

    # Load block of Q
    q = tl.load(Q + start_q + tl.arange(0, BLOCK_M)[:, None] * stride_qd)

    # Initialize output block
    o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Iterate over K and V blocks
    for i in range(0, tl.cdiv(B_Seqlen, BLOCK_N)):
        # Load block of K and V
        k = tl.load(K + start_k + i * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] * stride_kd)
        v = tl.load(V + start_v + i * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] * stride_vd)

        # Compute attention scores
        qk = tl.dot(q, k, trans_b=True) * sm_scale

        # Apply softmax
        qk = tl.softmax(qk, axis=1)

        # Compute weighted sum
        o += tl.dot(qk, v)

    # Store result in output tensor
    tl.store(Out + start_o + tl.arange(0, BLOCK_M)[:, None] * stride_od, o)

def context_attention_fwd(Q, K, V, B_Start_Loc, B_Seqlen, sm_scale):
    # Define grid dimensions
    batch_size, num_heads, seq_len, d_model = Q.shape
    grid = (batch_size, num_heads, seq_len // BLOCK_M)

    # Allocate output tensor
    Out = torch.empty_like(Q)

    # Launch Triton kernel
    _fwd_kernel[grid](
        Q, K, V, Out,
        B_Start_Loc, B_Seqlen,
        sm_scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N
    )

    return Out
