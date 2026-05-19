import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q_ptr, K_ptr, V_ptr, Out_ptr, sm_scale, stride_qm, stride_qk, stride_qd, 
                stride_km, stride_kn, stride_kd, stride_vm, stride_vn, stride_vd, 
                stride_om, stride_on, stride_od, 
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr, 
                IS_CAUSAL: tl.constexpr, USE_FP8: tl.constexpr):
    
    # Define the program IDs for blocks
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate the offsets for each block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q, K, V blocks
    Q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    K = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    V = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # Compute QK^T
    qk = tl.dot(Q, tl.trans(K)) * sm_scale

    # Apply causal mask if necessary
    if IS_CAUSAL:
        mask = tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :]
        qk = tl.where(mask, qk, float('-inf'))

    # Compute softmax along the N dimension
    qk = tl.softmax(qk, axis=1)

    # Compute output
    Out = tl.dot(qk, V)

    # Store the result
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, Out)

### Wrapper Function

def triton_fa(Q, K, V, sm_scale, BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=64, IS_CAUSAL=False, USE_FP8=False):
    # Get the shapes and strides of the input tensors
    Q_shape, K_shape, V_shape = Q.shape, K.shape, V.shape
    Q_strides, K_strides, V_strides = Q.stride(), K.stride(), V.stride()

    # Allocate output tensor
    Out = torch.empty((Q_shape[0], V_shape[1], Q_shape[2]), device=Q.device, dtype=Q.dtype)

    # Calculate grid size
    grid = (triton.cdiv(Q_shape[0], BLOCK_M), triton.cdiv(K_shape[1], BLOCK_N))

    # Launch the kernel
    _fwd_kernel[grid](
        Q, K, V, Out, sm_scale,
        Q_strides[0], Q_strides[1], Q_strides[2],
        K_strides[0], K_strides[1], K_strides[2],
        V_strides[0], V_strides[1], V_strides[2],
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=IS_CAUSAL, USE_FP8=USE_FP8
    )

    return Out
