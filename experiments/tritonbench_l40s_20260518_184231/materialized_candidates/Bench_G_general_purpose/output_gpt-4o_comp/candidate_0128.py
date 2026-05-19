import triton
import triton.language as tl
import torch

@triton.jit
def rotary_kernel(
    X_ptr, COS_ptr, SIN_ptr, OUT_ptr, CU_SEQLENS_ptr,
    batch_size, num_heads, head_dim, seq_len,
    stride_xb, stride_xh, stride_xm, stride_xd,
    stride_cosm, stride_cosd,
    stride_sinm, stride_sind,
    stride_outb, stride_outh, stride_outm, stride_outd,
    interleaved: tl.constexpr, conjugate: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    offset_batch = pid_batch * stride_xb
    offset_head = pid_head * stride_xh
    offset_m = pid_m * stride_xm

    if CU_SEQLENS_ptr is not None:
        cu_seqlen_start = tl.load(CU_SEQLENS_ptr + pid_batch)
        cu_seqlen_end = tl.load(CU_SEQLENS_ptr + pid_batch + 1)
        seq_len = cu_seqlen_end - cu_seqlen_start
        offset_m += cu_seqlen_start * stride_xm

    for d in range(0, head_dim, tl.numel):
        offset_d = d * stride_xd
        x_ptr = X_ptr + offset_batch + offset_head + offset_m + offset_d
        cos_ptr = COS_ptr + pid_m * stride_cosm + d * stride_cosd
        sin_ptr = SIN_ptr + pid_m * stride_sinm + d * stride_sind
        out_ptr = OUT_ptr + offset_batch + offset_head + offset_m + offset_d

        x = tl.load(x_ptr, mask=(d < head_dim))
        cos = tl.load(cos_ptr, mask=(d < head_dim))
        sin = tl.load(sin_ptr, mask=(d < head_dim))

        if conjugate:
            x = tl.complex(x.real, -x.imag)

        if interleaved:
            x_real, x_imag = x.real, x.imag
            out_real = x_real * cos - x_imag * sin
            out_imag = x_real * sin + x_imag * cos
            out = tl.complex(out_real, out_imag)
        else:
            out = x * cos + x * sin

        tl.store(out_ptr, out, mask=(d < head_dim))

def apply_rotary(x, cos, sin, cu_seqlens=None, interleaved=False, conjugate=False):
    batch_size, num_heads, seq_len, head_dim = x.shape
    out = torch.empty_like(x)

    grid = (batch_size, num_heads, seq_len)
    num_warps = 4
    num_threads = 128

    rotary_kernel[grid](
        x, cos, sin, out, cu_seqlens,
        batch_size, num_heads, head_dim, seq_len,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        interleaved=interleaved, conjugate=conjugate,
        num_warps=num_warps, num_threads=num_threads
    )

    return out
