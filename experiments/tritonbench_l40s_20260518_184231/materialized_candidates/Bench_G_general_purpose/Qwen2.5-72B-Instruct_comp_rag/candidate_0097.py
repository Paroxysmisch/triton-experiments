import triton
import triton.language as tl
import torch

# Define the rotary kernel
@triton.jit
def _rotary_kernel(
    Q, K, Cos, Sin,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_cm, stride_ch, stride_cd,
    stride_sm, stride_sh, stride_sd,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Get the program ID
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Compute the offsets for Q and K
    offset_q = pid_h * stride_qh + pid_m * stride_qm
    offset_k = pid_h * stride_kh + pid_m * stride_km

    # Compute the offsets for Cos and Sin
    offset_c = pid_m * stride_cm
    offset_s = pid_m * stride_sm

    # Load the segments of Q and K
    q = tl.load(Q + offset_q + tl.arange(0, BLOCK_DMODEL), mask=offset_q + tl.arange(0, BLOCK_DMODEL) < Q.shape[2], other=0.0)
    k = tl.load(K + offset_k + tl.arange(0, BLOCK_DMODEL), mask=offset_k + tl.arange(0, BLOCK_DMODEL) < K.shape[2], other=0.0)

    # Load the segments of Cos and Sin
    cos = tl.load(Cos + offset_c + tl.arange(0, BLOCK_DMODEL), mask=offset_c + tl.arange(0, BLOCK_DMODEL) < Cos.shape[2], other=0.0)
    sin = tl.load(Sin + offset_s + tl.arange(0, BLOCK_DMODEL) + tl.arange(0, BLOCK_DMODEL), mask=offset_s + tl.arange(0, BLOCK_DMODEL) < Sin.shape[2], other=0.0)

    # Apply the rotary transformation
    q0 = q[:BLOCK_DMODEL // 2]
    q1 = q[BLOCK_DMODEL // 2:]
    k0 = k[:BLOCK_DMODEL // 2]
    k1 = k[BLOCK_DMODEL // 2:]

    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos

    out_q = tl.cat(out0, out1)
    out_k = tl.cat(k0 * cos - k1 * sin, k0 * sin + k1 * cos)

    # Store the transformed segments back to Q and K
    tl.store(Q + offset_q + tl.arange(0, BLOCK_DMODEL), out_q, mask=offset_q + tl.arange(0, BLOCK_DMODEL) < Q.shape[2])
    tl.store(K + offset_k + tl.arange(0, BLOCK_DMODEL), out_k, mask=offset_k + tl.arange(0, BLOCK_DMODEL) < K.shape[2])

### High-Level Interface for Invoking the Kernel

def rotary_emb_fwd(Q, K, Cos, Sin, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL, num_warps=4):
    # Validate input shapes
    assert Q.shape == K.shape, "Q and K must have the same shape"
    assert Cos.shape == Sin.shape, "Cos and Sin must have the same shape"
    assert Q.shape[0] == Cos.shape[0], "Q and Cos must have the same sequence length"
    assert Q.shape[1] == Cos.shape[1], "Q and Cos must have the same head dimension"
    assert Q.shape[2] == BLOCK_DMODEL, "Q and K must have the same head dimension as BLOCK_DMODEL"

    # Calculate grid dimensions
    grid = (Q.shape[1] // BLOCK_HEAD, Q.shape[0] // BLOCK_SEQ)

    # Launch the kernel
    _rotary_kernel[grid](
        Q, K, Cos, Sin,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        Cos.stride(0), Cos.stride(1), Cos.stride(2),
        Sin.stride(0), Sin.stride(1), Sin.stride(2),
        BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL,
        num_warps=num_warps
    )

# Example usage
Q = torch.randn(128, 8, 64, device='cuda')
K = torch.randn(128, 8, 64, device='cuda')
Cos = torch.randn(128, 8, 64, device='cuda')
Sin = torch.randn(128, 8, 64, device='cuda')

BLOCK_HEAD = 8
BLOCK_SEQ = 16
BLOCK_DMODEL = 64

rotary_emb_fwd(Q, K, Cos, Sin, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL)
