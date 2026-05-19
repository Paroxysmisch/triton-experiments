import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X, OUT, COS, SIN, CU_SEQLENS,
    stride_xb, stride_xh, stride_xm, stride_xr,
    stride_ob, stride_oh, stride_om, stride_or,
    stride_cb, stride_ch, stride_cm,
    stride_sb, stride_sh, stride_sm,
    B, H, M, R, INTERLEAVED, CONJUGATE,
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    rm = pid_m * BLOCK_M

    if CU_SEQLENS is not None:
        start_m = tl.load(CU_SEQLENS + pid_batch)
        rm = rm + start_m

    X = X + pid_batch * stride_xb + pid_head * stride_xh + rm * stride_xm
    OUT = OUT + pid_batch * stride_ob + pid_head * stride_oh + rm * stride_om
    COS = COS + pid_batch * stride_cb + pid_head * stride_ch + rm * stride_cm
    SIN = SIN + pid_batch * stride_sb + pid_head * stride_sh + rm * stride_sm

    for m in range(BLOCK_M):
        if rm + m < M:
            for r in range(0, R, BLOCK_R):
                x_ptr = X + m * stride_xm + r * stride_xr
                o_ptr = OUT + m * stride_om + r * stride_or
                c_ptr = COS + m * stride_cm + r
                s_ptr = SIN + m * stride_sm + r

                x = tl.load(x_ptr, mask=r < R, other=0.0)
                c = tl.load(c_ptr, mask=r < R, other=1.0)
                s = tl.load(s_ptr, mask=r < R, other=0.0)

                if INTERLEAVED:
                    x_re, x_im = x[0::2], x[1::2]
                    if CONJUGATE:
                        out_re = x_re * c + x_im * s
                        out_im = x_im * c - x_re * s
                    else:
                        out_re = x_re * c - x_im * s
                        out_im = x_im * c + x_re * s
                    out = tl.where(r < R, tl.cat(out_re, out_im, dim=0), x)
                else:
                    x_top, x_bottom = x[:R//2], x[R//2:]
                    if CONJUGATE:
                        out_top = x_top * c + x_bottom * s
                        out_bottom = x_bottom * c - x_top * s
                    else:
                        out_top = x_top * c - x_bottom * s
                        out_bottom = x_bottom * c + x_top * s
                    out = tl.where(r < R, tl.cat(out_top, out_bottom, dim=0), x)

                tl.store(o_ptr, out, mask=r < R)

def apply_rotary(x, cos, sin, seqlen_offset=0, cu_seqlens=None):
    batch, seqlen, nheads, headdim = x.shape
    rotary_dim = cos.shape[-1]
    
    assert rotary_dim <= headdim
    assert cos.shape == (seqlen, rotary_dim)
    assert sin.shape == (seqlen, rotary_dim)
    assert seqlen_offset >= 0
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
    
    cos = cos[seqlen_offset : seqlen_offset + seqlen]
    sin = sin[seqlen_offset : seqlen_offset + seqlen]
    
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    output = torch.empty_like(x)
    if rotary_dim < headdim:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    
    BLOCK_M = 16
    BLOCK_R = 16
    
    def grid(META):
        return (
            batch,
            nheads,
            triton.cdiv(seqlen, META["BLOCK_M"])
        )
    
    rotary_kernel[grid](
        x, output, cos, sin, cu_seqlens,
        x.stride(0), x.stride(2), x.stride(1), x.stride(3),
        output.stride(0), output.stride(2), output.stride(1), output.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2),
        sin.stride(0), sin.stride(1), sin.stride(2),
        batch, nheads, seqlen, rotary_dim,
        x.is_complex(),
        False,  # CONJUGATE
        BLOCK_M=BLOCK_M,
        BLOCK_R=BLOCK_R,
    )
    
    return output
