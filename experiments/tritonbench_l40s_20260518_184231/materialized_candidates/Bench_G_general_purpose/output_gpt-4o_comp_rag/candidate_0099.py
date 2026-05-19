import triton
import triton.language as tl
import torch

@triton.jit
def _rotary_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    Q_stride_h, Q_stride_s, Q_stride_d,
    K_stride_h, K_stride_s, K_stride_d,
    Cos_stride_h, Cos_stride_s, Cos_stride_d,
    Sin_stride_h, Sin_stride_s, Sin_stride_d,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Calculate the base pointers for Q, K, Cos, Sin
    Q_base_ptr = Q_ptr + pid_h * Q_stride_h + pid_s * Q_stride_s
    K_base_ptr = K_ptr + pid_h * K_stride_h + pid_s * K_stride_s
    Cos_base_ptr = Cos_ptr + pid_h * Cos_stride_h + pid_s * Cos_stride_s
    Sin_base_ptr = Sin_ptr + pid_h * Sin_stride_h + pid_s * Sin_stride_s

    # Load Q and K blocks
    q = tl.load(Q_base_ptr + tl.arange(0, BLOCK_DMODEL), mask=True, other=0.0)
    k = tl.load(K_base_ptr + tl.arange(0, BLOCK_DMODEL), mask=True, other=0.0)

    # Load Cos and Sin blocks
    cos = tl.load(Cos_base_ptr + tl.arange(0, BLOCK_DMODEL), mask=True, other=0.0)
    sin = tl.load(Sin_base_ptr + tl.arange(0, BLOCK_DMODEL), mask=True, other=0.0)

    # Apply rotary transformation
    q_rot = q * cos - tl.swizzle(q, axis=0) * sin
    k_rot = k * cos - tl.swizzle(k, axis=0) * sin

    # Store the transformed Q and K back
    tl.store(Q_base_ptr + tl.arange(0, BLOCK_DMODEL), q_rot, mask=True)
    tl.store(K_base_ptr + tl.arange(0, BLOCK_DMODEL), k_rot, mask=True)

def rotary_emb_fwd(Q, K, Cos, Sin, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL):
    assert Q.shape == K.shape
    assert Q.shape == Cos.shape
    assert Q.shape == Sin.shape
    assert Q.shape[2] == BLOCK_DMODEL

    grid = (Q.shape[0] // BLOCK_HEAD, Q.shape[1] // BLOCK_SEQ)

    num_warps = 4 if BLOCK_DMODEL >= 64 else 2

    _rotary_kernel[grid](
        Q, K, Cos, Sin,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        Cos.stride(0), Cos.stride(1), Cos.stride(2),
        Sin.stride(0), Sin.stride(1), Sin.stride(2),
        BLOCK_HEAD=BLOCK_HEAD, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )

# Example usage
BLOCK_HEAD = 1
BLOCK_SEQ = 1
BLOCK_DMODEL = 64

Q = torch.randn(8, 128, 64, device='cuda', dtype=torch.float32)
K = torch.randn(8, 128, 64, device='cuda', dtype=torch.float32)
Cos = torch.randn(8, 128, 64, device='cuda', dtype=torch.float32)
Sin = torch.randn(8, 128, 64, device='cuda', dtype=torch.float32)

rotary_emb_fwd(Q, K, Cos, Sin, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL)
